# Tax Optimization Causal Uplift Analysis

## Overview
This analysis uses causal inference to identify which expense categories have the most significant impact on tax refunds for German taxpayers. By leveraging a causal forest approach, we estimate the true causal effect of different expense categories on tax refunds, accounting for confounding factors and providing actionable, data-driven tax optimization tips.

## Process

### 1. Data Preparation

**User-Level Aggregation:**
- Transaction data is aggregated at the user-category level.
- For each user and category, we sum the user’s spending and assign their total refund only once, ensuring accurate attribution and avoiding double-counting.

**Category Summarization:**
- For each category, we sum the total spending and total refunds across all users.
- This allows us to compute the tax benefit percentage as the ratio of total refunds to total spending for each category.

**Feature Engineering:**
We calculate key metrics for each user and category, including:
- Refund per income ratio
- Deduction ratio
- Category-specific spending patterns
- Tax benefit percentage (portion of expense amount received back in tax benefits)

### 2. Causal Modeling

**Causal Forest DML:**
- We use a Causal Forest Double Machine Learning (DML) approach to estimate the causal effect of each expense category on tax refunds, controlling for confounding variables.

**Confidence Intervals:**
- The model provides confidence intervals for each estimated effect, quantifying the uncertainty in our recommendations.

**Confidence Levels:**
Each category is assigned a confidence level based on the number of unique users:
- High: 30+ users
- Medium: 10–29 users
- Low: 5–9 users
- Very Low: <5 users

### 3. Generating Tax Optimization Tips

For each category, we generate a tip that includes:
- The estimated uplift in refund as a percentage of income
- The suggested amount to spend in the category
- The expected additional refund for a reference income (€75,000)
- The typical tax benefit percentage for users in that category
- A confidence level and a clear, actionable message

## Example Results

Below are sample insights from `tax_optimization_insights.csv`:

| Category              | Uplift (% of income) | Suggested Spend (€) | Expected Refund (€) | Tax Benefit (%) | Confidence | Message |
|-----------------------|----------------------|----------------------|----------------------|------------------|------------|--------------------------------------------------------------------------------------------------------------------------------|
| Work Equipment        | 0.04%                | 432                  | 29                   | 40.6             | Medium     | ✅ Adding €432 in Work Equipment would likely increase your refund by 0.04% of your income (±0.06%). Users typically receive 40.6% of their spending back in tax benefits. |
| Child Care            | -0.04%               | 461                  | -26                  | 7.2              | Medium     | ⚠️ Adding €461 in Child Care would likely decrease your refund by 0.04% of your income (±0.06%). Users typically receive 7.2% back. |
| Transportation        | -0.03%               | 68                   | -21                  | 55.1             | Medium     | ⚠️ Adding €68 in Transportation would likely decrease your refund by 0.03% of your income (±0.21%). Users typically receive 55.1% back. |
| Medical               | 0.02%                | 104                  | 12                   | 86.5             | Medium     | ✅ Adding €104 in Medical would likely increase your refund by 0.02% of your income (±0.09%). Users typically receive 86.5% back. |
| Charitable Donations  | -0.02%               | 430                  | -11                  | 111.9            | Medium     | ⚠️ Adding €430 in Charitable Donations would likely decrease your refund by 0.02% of your income (±0.06%). Users typically receive 111.9% back. |
| Self-Employment       | 0.21%                | 1351                 | 156                  | 3.5              | Low        | 💰 Adding €1351 in Self-Employment could increase your refund by up to 0.21% of your income (±0.11%). Users typically receive 3.5% back. |
| Business Meals        | 0.20%                | 67                   | 152                  | 53.8             | Low        | 💰 Adding €67 in Business Meals could increase your refund by up to 0.20% of your income (±0.11%). Users typically receive 53.8% back. |
| Home Office           | -0.04%               | 270                  | -31                  | 80.2             | Low        | ⚠️ Adding €270 in Home Office could reduce your tax refund. Users typically receive 80.2% back. |

See the full `tax_optimization_insights.csv` for all categories and detailed tips.

## How to Use These Insights

**Strategic Planning:**
- Focus on categories with high or medium confidence and positive uplift for the most reliable tax optimization opportunities.

**Caution with Low/Very Low Confidence:**
- Treat tips with low or very low confidence as exploratory; more data is needed for robust conclusions.

**Interpret Tax Benefit Percentages Carefully:**
- The tax benefit percentage shows the average portion of spending received back in tax benefits for users in that category.
- Values over 100% may occur due to the way refunds are allocated or special tax rules and should be interpreted with caution.

## Files in this Directory

- `tax_optimization_insights.csv`: Category-level tax optimization tips and insights  
- `category_statistics.csv`: Basic statistics for each category  
- `category_performance.csv`: Model performance metrics  
- `uplift_curves/`: Visualizations of uplift effects  
- `summary_report.txt`: Comprehensive analysis summary

## Technical Details

- **Model**: Causal Forest DML with Random Forest base learners  
- **Confidence Intervals**: 90% level  
- **Data Requirements**: Transaction-level spending, tax refund, income, and category data


# Hierarchical Clustering Tax Peer Group Analysis

## Overview
This analysis uses hierarchical clustering to group users into tax peer groups based on their expense patterns, income, and demographic features. The goal is to identify how each user compares to their peers and to generate personalized tax optimization recommendations.

## Methodology
- **Data Preparation:**
  - Transaction-level data is aggregated by user and tax year.
  - Expense categories are normalized as a percentage of each user's income.
  - Demographic features (occupation, age range, family status, region) are included if available.
- **Feature Engineering:**
  - Principal Component Analysis (PCA) reduces dimensionality for clustering.
- **Clustering:**
  - Hierarchical clustering (Ward's method) is used to form peer groups.
  - The optimal number of clusters is determined using silhouette scores.
- **Gap & Recommendation Generation:**
  - For each user and category, the gap between their spending and the cluster average is calculated.
  - Recommendations are generated for all users and categories, with confidence levels reflecting the strength of the evidence.

## Cluster Explanations
Based on the current analysis, the following clusters were identified (see `hierarchical_cluster_profiles.txt`):

### Cluster 0 (3 members)
- **Income Bands:** Band 2, 4, 5 (mid to high income)
- **Occupation:** 100% Self-Employed
- **Top Expenses:**
  - Self-Employment: 21.7% of income
  - Child Care: 3.3%
  - Travel: 2.1%
- **Interpretation:**
  - This cluster represents self-employed users with significant business-related expenses and moderate child care and travel deductions.

### Cluster 1 (9 members)
- **Income Bands:** Broadly distributed (Bands 1–5)
- **Occupations:** IT Professional, Education, Manufacturing, Healthcare, Finance
- **Top Expenses:**
  - Child Care: 5.3% of income
  - Professional Development: 1.9%
  - Transportation: 1.4%
- **Interpretation:**
  - This is a diverse cluster with a mix of professions and income levels, but a strong focus on child care and professional development deductions.

### Cluster 2 (1 member)
- **Income Band:** Band 4 (upper-middle income)
- **Occupation:** Healthcare
- **Top Expenses:**
  - Medical: 3.8% of income
  - Transportation: 1.0%
- **Interpretation:**
  - A single healthcare professional with high medical and transportation expenses.

### Cluster 3 (2 members)
- **Income Band:** Band 1 (lowest income)
- **Occupation:** Retail
- **Top Expenses:**
  - Child Care: 5.5% of income
  - Work Clothing: 1.4%
  - Transportation: 1.1%
- **Interpretation:**
  - Low-income retail workers with notable child care and work clothing deductions.

### Cluster 4 (1 member)
- **Income Band:** Band 5 (highest income)
- **Occupation:** Finance
- **Top Expenses:**
  - Rental: 30.3% of income
  - Property Expenses: 3.6%
- **Interpretation:**
  - A high-income finance professional with substantial rental and property-related deductions.

## Key Findings
- **All users are included** in the recommendations, regardless of gap size.
- **Confidence levels** (Very Weak to Very Strong) indicate the strength of each recommendation.
- **Cluster profiles** reveal distinct peer groups based on occupation, income, and expense patterns.
- **Personalized recommendations** are generated for every user and category, even if the gap is small or negative.

## How to Interpret the Results
- **Recommendation Text:** Explains how your spending compares to your peer group for each category.
- **Gap %:** Positive means you spend less than your peers; negative means you spend more.
- **Confidence Level:** Higher confidence means stronger evidence for the recommendation.
- **Cluster Context:** Your peer group is defined by similar income, occupation, and expense patterns.

## Limitations
- Small clusters (with 1–2 users) may yield less reliable recommendations.
- Recommendations for categories with very low peer usage may be less actionable.
- The analysis is based on available data and may not capture all relevant user circumstances.

## Usage Recommendations
- Focus on recommendations with Moderate, Strong, or Very Strong confidence.
- Use cluster profiles to understand your peer group context.
- Treat Very Weak recommendations as exploratory insights.

## Technical Details
- **Clustering:** Hierarchical (Ward's method), PCA for dimensionality reduction
- **Features:** Expense ratios, occupation, age, family status, region
- **Outputs:**
  - `hierarchical_tax_recommendations_with_confidence.csv`: All user-category recommendations
  - `hierarchical_cluster_profiles.txt`: Cluster summaries
  - `enriched_tax_data.csv`: Original data with cluster assignments

---
For questions or further analysis, see the cluster profiles or contact the data science team.

# German Tax Deduction Classifier Analysis

## Overview
The German Tax Deduction Classifier is a sophisticated system that analyzes tax combinations based on occupation, family status, and expense categories to determine tax deductibility under German tax law. The system uses LLM-based classification with OpenAI models via LiteLLM, incorporating structured outputs and validation through Pydantic.

## Process Flow

### Data Input
Takes unique tax combinations including:
- Occupation category
- Family status
- Subcategory
- Description
- Income band

### Classification Process
- Uses Jinja templating for tax rule context
- Leverages LiteLLM's built-in prompt caching
- Processes in batches with error handling
- Generates structured outputs with confidence scores

## Output Structure
Each classification result includes:
- `is_deductible`: Boolean indicating deductibility
- `confidence_score`: Score between 0.0 and 1.0
- `category`: Tax category (e.g., Werbungskosten, Sonderausgaben)
- `max_deduction_percentage`: Maximum percentage deductible
- `max_deduction_amount`: Maximum amount in EUR
- `reasons`: Legal references and explanations

## Results Analysis

### Key Categories

#### Werbungskosten (Income-related Expenses)
- Most common category
- Includes work equipment, training costs, commuting expenses
- Often 100% deductible with specific limits

#### Sonderausgaben (Special Expenses)
- Includes child-care costs (80% up to €4,800)
- Donations (up to 20% of income)

#### Betriebsausgaben (Business Expenses)
- Primarily for self-employed individuals
- 100% deductible for business-related expenses

#### Außergewöhnliche Belastungen (Extraordinary Burdens)
- Medical expenses
- Subject to income-based thresholds

### Notable Patterns
- High confidence scores (0.8–0.95) for most classifications
- Clear distinction between employee and self-employed deductions
- Specific limits for certain categories (e.g., €4,500 for commuting)
- Comprehensive legal references in explanations

## Technical Implementation
- Uses Pydantic for data validation
- Implements batch processing for efficiency
- Includes error handling and fallback mechanisms
- Maintains detailed logging of processing steps

## Usage
The classifier can be run with various parameters:
- Batch size control
- Model selection
- Start/end index specification
- Custom input/output file paths

## Limitations
- Requires OpenAI API key
- Subject to API rate limits
- Confidence scores may vary based on input complexity
- Some edge cases may require manual review
