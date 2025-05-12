import os
import argparse
import sys
from pathlib import Path
import glob # Import glob

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.ingestion.processor import DocumentAIProcessor

def get_file_patterns(doc_type):
    """Get file patterns based on document type."""
    if doc_type.lower() == "receipt":
        # Receipts are images
        return ["*.jpg", "*.jpeg", "*.png", "*.PNG", "*.tiff", "*.tif"]
    else:
        # Income statements/payslips are PDFs
        return ["*.pdf"]

def get_output_dir(base_output_dir: str, doc_type: str) -> str:
    """Get the appropriate output directory based on document type."""
    if doc_type.lower() == "receipt":
        return os.path.join(base_output_dir, "processed_receipts")
    else:
        return os.path.join(base_output_dir, "processed_income_statements")

def process_documents(input_dir: str, output_dir: str, doc_type: str):
    """Process documents in the input directory and save results to output directory."""

    # --- Path Correction Start ---
    # Determine the correct absolute path
    if os.path.isabs(input_dir):
        input_path = Path(input_dir)
    else:
        # If path is relative, make it relative to the project root
        # and assume it should be inside 'app/' if 'app/' isn't specified
        if not input_dir.startswith("app/"):
             input_path = Path(os.getcwd()) / "app" / input_dir
        else:
             input_path = Path(os.getcwd()) / input_dir

    print(f"Looking for files in: {input_path}")

    # Check if the directory actually exists *before* trying to list contents
    if not input_path.exists() or not input_path.is_dir():
        print(f"ERROR: Directory not found or is not a directory: {input_path}")
        # Debugging: List contents of parent directory if it exists
        parent_dir = input_path.parent
        if parent_dir.exists() and parent_dir.is_dir():
            print(f"\nContents of parent directory ({parent_dir}):")
            try:
                for item in parent_dir.iterdir():
                     print(f"  - {item.name}")
            except Exception as e:
                print(f"  Error listing parent dir: {e}")
        return
    # --- Path Correction End ---


    # Create output directory based on document type
    type_specific_output = get_output_dir(output_dir, doc_type)
    output_path = Path(type_specific_output)
    if not output_path.is_absolute():
        output_path = Path(os.getcwd()) / output_path
    os.makedirs(output_path, exist_ok=True)


    # List all files in directory for debugging (now that we know it exists)
    print(f"\nAll files in directory ({input_path}):")
    try:
        found_items = False
        for item in input_path.iterdir():
            found_items = True
            print(f"  - {item.name} {'(dir)' if item.is_dir() else '(file)'}")
        if not found_items:
            print("  (Directory is empty)")
    except Exception as e:
        print(f"  Error listing directory contents: {e}")
        return


    # Get processor
    try:
        processor = DocumentAIProcessor()
    except Exception as e:
        print(f"Error initializing Document AI processor: {e}")
        print("\nMake sure you have set the following environment variables:")
        print("  - GOOGLE_CLOUD_PROJECT")
        print("  - DOCUMENT_AI_LOCATION (default: eu)")
        print("  - RECEIPT_PROCESSOR_ID")
        print("  - INCOME_STATEMENT_PROCESSOR_ID")
        return

    # Get files in the input directory based on document type
    file_patterns = get_file_patterns(doc_type)

    files = []
    print("\nSearching for files:")
    for pattern in file_patterns:
        # Use glob directly from the Path object
        print(f"  Searching with pattern: {pattern}")
        matched = list(input_path.glob(pattern))
        print(f"    Found {len(matched)} files: {[f.name for f in matched]}")
        files.extend(matched)

    if not files:
        patterns_str = ", ".join(file_patterns)
        print(f"\nNo files matching patterns [{patterns_str}] found in {input_path}")
        return

    print(f"\nFound {len(files)} files to process")
    for file in files:
        print(f"Processing {file.name}...")
        try:
            # Process the document
            result = processor.process_document(str(file), doc_type)
            print(result)
            # Save results as JSON
            # Use output_path which is now absolute
            output_file = output_path / f"{file.stem}.json"
            processor.save_as_json(result, str(output_file))
            print(f"✅ Saved results to {output_file}")

        except Exception as e:
            print(f"❌ Error processing {file.name}: {str(e)}")

# ... main function and __main__ block remain the same ...
def main():
    parser = argparse.ArgumentParser(description="Process documents using Google Document AI")
    parser.add_argument(
        "--input", "-i",
        help="Directory containing files to process (e.g., data/receipt or app/data/receipt)"
    )
    parser.add_argument(
        "--output", "-o",
        default="app/data/processed",
        help="Directory to save processed results (default: app/data/processed)"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["receipt", "income_statement", "payslip", "all"],
        help="Type of document to process (default: all)"
    )

    args = parser.parse_args()
    
    # Default to processing all types if no arguments provided
    if not args.input or not args.type:
        # Process receipts
        receipt_input = "app/data/receipt"
        process_documents(receipt_input, args.output, "receipt")
        
        # Process income statements
        income_input = "app/data/income_statement"
        process_documents(income_input, args.output, "income_statement")
    elif args.type == "all":
        # Find appropriate input directories based on the provided input path
        base_input = args.input
        if not os.path.isabs(base_input):
            if not base_input.startswith("app/"):
                base_input = os.path.join("app", base_input)
        
        # Process receipts
        receipt_input = os.path.join(os.path.dirname(base_input), "receipt")
        process_documents(receipt_input, args.output, "receipt")
        
        # Process income statements
        income_input = os.path.join(os.path.dirname(base_input), "income_statement")
        process_documents(income_input, args.output, "income_statement")
    else:
        # Process the specified type only
        process_documents(args.input, args.output, args.type)

if __name__ == "__main__":
    main()
