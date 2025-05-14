"""
German Tax Deduction Classifier

This script analyzes unique tax combinations (occupation, family status, expense categories)
and uses LLM-based classification to determine if expenses are tax-deductible under German tax law.
It leverages OpenAI models via LiteLLM, with Instructor for structured outputs and Pydantic for validation.

Features:
- Jinja templating for tax rule context
- LiteLLM's built-in prompt caching to minimize API calls
- Structured outputs with confidence scores
- Batch processing with error handling
"""

import os
import csv
import json
import hashlib
import time
import argparse
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field, validator
import instructor
from litellm import completion, completion_cost

# Configure paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = DATA_DIR / "processed" / "tax_deduction_results"
TEMPLATES_DIR = SCRIPT_DIR

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure Jinja2 environment
jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

# Pydantic models for structured output
class DeductionReason(BaseModel):
    law_reference: str = Field(..., description="The specific paragraph or section in German tax law that applies")
    explanation: str = Field(..., description="Clear explanation of why this is deductible under German tax law")

class DeductionResult(BaseModel):
    is_deductible: bool = Field(..., description="Whether the expense is tax-deductible")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    category: Optional[str] = Field(None, description="The tax category this deduction falls under (e.g., Werbungskosten, Sonderausgaben)")
    max_deduction_percentage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Maximum percentage deductible")
    max_deduction_amount: Optional[float] = Field(None, ge=0.0, description="Maximum amount deductible in EUR, if applicable")
    reasons: Optional[List[DeductionReason]] = Field(None, description="Reasons for deductibility with legal references")
    
    @validator('confidence_score')
    def round_confidence(cls, v):
        return round(v, 2)

class TaxCombination(BaseModel):
    occupation_category: str
    family_status: str
    subcategory: str
    description: str
    income_band: str

def load_tax_rules() -> str:
    """Load German tax rules from Jinja template."""
    template = jinja_env.get_template("german_tax_rules.jinja")
    return template.render()

def generate_prompt(tax_combination: TaxCombination, tax_rules: str) -> str:
    """Generate a prompt for the LLM based on tax combination and rules."""
    try:
        template = jinja_env.get_template("tax_prompt.jinja")
        return template.render(
            occupation=tax_combination.occupation_category,
            family_status=tax_combination.family_status,
            income_band=tax_combination.income_band,
            subcategory=tax_combination.subcategory,
            description=tax_combination.description,
            tax_rules=tax_rules
        )
    except Exception as e:
        print(f"Error loading template: {e}")
        # Fallback to inline template
        prompt = f"""
        Based on German tax law, determine if the following expense is tax-deductible:
        
        Taxpayer Profile:
        - Occupation: {tax_combination.occupation_category}
        - Family Status: {tax_combination.family_status}
        - Income Band: {tax_combination.income_band}
        
        Expense Details:
        - Category: {tax_combination.subcategory}
        - Description: {tax_combination.description}
        
        German Tax Rules Reference:
        {tax_rules}
        
        Provide a detailed analysis with:
        1. Whether the expense is deductible (yes/no)
        2. The specific tax category it falls under (e.g., Werbungskosten, Sonderausgaben)
        3. The legal basis for your determination (specific paragraph references)
        4. Maximum deduction percentage or amount (if applicable)
        5. A confidence score (0.0-1.0) for your determination
        """
        return prompt

def classify_deduction(tax_combination: TaxCombination, tax_rules: str, model: str = "gpt-4o") -> DeductionResult:
    """
    Classify if a tax combination is deductible using OpenAI.
    LiteLLM's built-in prompt caching will automatically avoid redundant API calls.
    
    Note: OpenAI caching is only available for prompts containing 1024 tokens or more,
    so the tax rules are structured to exceed this threshold.
    """
    prompt = generate_prompt(tax_combination, tax_rules)
    
    # Use instructor with LiteLLM for structured output
    client = instructor.from_litellm(completion)
    
    try:
        # For OpenAI caching, the content must exceed 1024 tokens
        # We structure the messages to take advantage of caching
        system_message = "You are a German tax expert specializing in tax deductions."
        # Expand tax rules content to ensure it exceeds token threshold for caching
        expanded_tax_rules = tax_rules + "\n\n" + "Please keep all of this information in mind when making your assessment." * 10
        
        response = client.chat.completions.create(
            model=model,
            response_model=DeductionResult,
            messages=[
                # System message with instructions
                {"role": "system", "content": system_message},
                # First user message with tax rules (cacheable)
                {"role": "user", "content": expanded_tax_rules},
                # Assistant acknowledgment
                {"role": "assistant", "content": "I understand the German tax rules and will apply them to analyze tax deductions accurately."},
                # Second user message with specific tax combination to evaluate
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for more consistent responses
            max_tokens=1500
        )
        
        # Calculate cost (will handle cached tokens correctly)
        cost = completion_cost(completion_response=response, model=model)
        formatted_cost = f"${float(cost):.6f}"
        
        # Log caching info if available
        if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens_details'):
            cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0)
            print(f"Cached tokens: {cached_tokens}, Cost: {formatted_cost}")
        else:
            print(f"No caching information available, Cost: {formatted_cost}")
        
        return response
    except Exception as e:
        print(f"Error classifying deduction: {e}")
        # Return a fallback response indicating error
        return DeductionResult(
            is_deductible=False,
            confidence_score=0.0,
            category="Error",
            reasons=[DeductionReason(
                law_reference="Error",
                explanation=f"Failed to classify: {str(e)}"
            )]
        )

def process_batch(batch_df: pd.DataFrame, tax_rules: str, model: str = "gpt-4o") -> List[Dict]:
    """Process a batch of tax combinations."""
    batch_results = []
    
    for i, row in batch_df.iterrows():
        try:
            print(f"Processing item {i} in batch: {row['description']}")
            
            tax_combination = TaxCombination(
                occupation_category=row['occupation_category'],
                family_status=row['family_status'],
                subcategory=row['subcategory'],
                description=row['description'],
                income_band=row['income_band']
            )
            
            result = classify_deduction(tax_combination, tax_rules, model)
            
            # Add original tax combination data to result
            result_dict = result.model_dump()
            result_dict.update({
                "occupation_category": tax_combination.occupation_category,
                "family_status": tax_combination.family_status,
                "subcategory": tax_combination.subcategory,
                "description": tax_combination.description,
                "income_band": tax_combination.income_band
            })
            
            batch_results.append(result_dict)
            
        except Exception as e:
            print(f"Error processing item {i} in batch: {str(e)}")
            # Add error entry to results
            error_result = {
                "occupation_category": row['occupation_category'],
                "family_status": row['family_status'],
                "subcategory": row['subcategory'],
                "description": row['description'],
                "income_band": row['income_band'],
                "is_deductible": False,
                "confidence_score": 0.0,
                "category": "Error",
                "error_message": str(e)
            }
            batch_results.append(error_result)
    
    return batch_results

def process_tax_combinations(input_file: str, output_file: str, batch_size: int = 10, model: str = "gpt-4o", 
                           start_idx: int = 0, end_idx: Optional[int] = None) -> None:
    """Process tax combinations from CSV in batches and save results."""
    print(f"Loading tax combinations from {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} tax combinations")
    
    # Apply start and end index limits if specified
    if end_idx is not None:
        df = df.iloc[start_idx:end_idx]
    elif start_idx > 0:
        df = df.iloc[start_idx:]
    
    total_combinations = len(df)
    print(f"Processing {total_combinations} combinations with batch size {batch_size}")
    
    # Load tax rules once
    tax_rules = load_tax_rules()
    print("Loaded German tax rules")
    
    all_results = []
    errors = []
    
    # Process in batches
    for batch_start in range(0, total_combinations, batch_size):
        batch_end = min(batch_start + batch_size, total_combinations)
        batch_df = df.iloc[batch_start:batch_end]
        
        print(f"\n--- Processing batch {batch_start//batch_size + 1}/{(total_combinations-1)//batch_size + 1} ---")
        print(f"Items {batch_start+1}-{batch_end} of {total_combinations}")
        
        # Process this batch
        batch_results = process_batch(batch_df, tax_rules, model)
        all_results.extend(batch_results)
        
        # Save intermediate results
        intermediate_df = pd.DataFrame(all_results)
        intermediate_df.to_csv(output_file, index=False)
        print(f"Intermediate results saved to {output_file}")
        
        # Add a small delay between batches to avoid rate limiting
        if batch_end < total_combinations:
            print("Waiting between batches...")
            time.sleep(2)
    
    # Final save (should be the same as the last intermediate save)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)
    print(f"Final results saved to {output_file}")
    
    # Generate summary
    if all_results:
        deductible_count = sum(1 for r in all_results if r.get('is_deductible', False))
        error_count = sum(1 for r in all_results if r.get('category') == 'Error')
        
        print(f"\nSummary:")
        print(f"- Total processed: {len(all_results)}")
        print(f"- Deductible: {deductible_count} ({deductible_count/len(all_results)*100:.1f}%)")
        print(f"- Non-deductible: {len(all_results) - deductible_count - error_count} ({(len(all_results) - deductible_count - error_count)/len(all_results)*100:.1f}%)")
        print(f"- Errors: {error_count} ({error_count/len(all_results)*100:.1f}%)")

def main():
    """Main execution function with command-line arguments."""
    parser = argparse.ArgumentParser(description='German Tax Deduction Classifier')
    parser.add_argument('--batch-size', type=int, default=10, 
                        help='Number of tax combinations to process in each batch (default: 10)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting index in the dataset (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                        help='Ending index in the dataset (default: process until end)')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input CSV file (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output CSV file (default: auto-generate with timestamp)')
    
    args = parser.parse_args()
    
    print("Starting German Tax Deduction Classifier")
    print(f"Using OpenAI model: {args.model}")
    
    # Check for OPENAI_API_KEY
    if not os.environ.get('OPENAI_API_KEY'):
        print("WARNING: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key with: export OPENAI_API_KEY=your_api_key_here")
        exit(1)
    
    # Define input and output files
    if args.input:
        input_file = args.input
    else:
        input_file = os.path.join(DATA_DIR, "unique_tax_combinations_sorted.csv")
        # Try alternative location if needed
        if not os.path.exists(input_file):
            alt_input_file = os.path.join(SCRIPT_DIR.parent.parent, "notebooks", "data", "unique_tax_combinations_sorted.csv")
            if os.path.exists(alt_input_file):
                input_file = alt_input_file
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at {input_file}")
    
    # Define output file
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_DIR, f"tax_deduction_results_{timestamp}.csv")
    
    # Process tax combinations
    process_tax_combinations(
        input_file=input_file,
        output_file=output_file,
        batch_size=args.batch_size,
        model=args.model,
        start_idx=args.start,
        end_idx=args.end
    )
    
    print("Tax deduction classification complete!")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
