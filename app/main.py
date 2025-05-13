import os
import argparse
import sys
import sqlite3
from pathlib import Path
import glob # Import glob
import json # Add json for debugging
import time
import re
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.ingestion.processor import DocumentAIProcessor
from app.database.db import Database
from app.load_transactions import load_transactions, query_examples

# Import process_enriched_data functionality
from app.batch.process_enriched_data import main as process_enriched_data

# Import vector search components
from app.vector.client import get_weaviate_client
from app.vector.schema import create_schema, delete_schema
from app.vector.upsert import add_or_update_user, add_or_update_transaction
from app.vector.search import search_similar_users, search_similar_transactions, search_transactions_for_user
from app.vector.data_loader import get_user_by_id, get_transaction_by_id, fetch_users, fetch_transactions
from app.vector.search_api import TaxInsightSearchAPI

# Import semantic package instead of implementing it directly
from app.semantic import search as semantic_search
from app.semantic import run_example_semantic_queries

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
            
            # DEBUG: Check if occupation data is present in processed document
            if doc_type.lower() in ["income_statement", "income", "payslip"]:
                if "occupation_data" in result:
                    print(f"✅ [DEBUG] Found occupation data in {file.name}: {result['occupation_data']}")
                else:
                    print(f"❌ [DEBUG] No occupation data found in {file.name}")
            
            # Save results as JSON
            # Use output_path which is now absolute
            output_file = output_path / f"{file.stem}.json"
            processor.save_as_json(result, str(output_file))
            print(f"✅ Saved results to {output_file}")

        except Exception as e:
            print(f"❌ Error processing {file.name}: {str(e)}")

def process_and_load_to_database(processed_dir: str = "app/data/processed", run_examples: bool = True):
    """
    Process all document JSONs and load them into the SQLite database.
    
    Args:
        processed_dir: Directory containing processed JSON files
        run_examples: Whether to run example queries after loading
    """
    print("\n" + "="*80)
    print("LOADING DATA INTO DATABASE")
    print("="*80)
    
    # DEBUG: Check for income statement files and their occupation data
    income_stmt_dir = os.path.join(processed_dir, "processed_income_statements")
    if os.path.exists(income_stmt_dir):
        print(f"[DEBUG] Checking income statement files in {income_stmt_dir}")
        for file_name in os.listdir(income_stmt_dir):
            if file_name.endswith(".json"):
                try:
                    with open(os.path.join(income_stmt_dir, file_name), 'r') as f:
                        json_data = json.load(f)
                        if "occupation_data" in json_data:
                            print(f"[DEBUG] {file_name} has occupation data: {json_data['occupation_data']}")
                        else:
                            print(f"[DEBUG] {file_name} has NO occupation data")
                except Exception as e:
                    print(f"[DEBUG] Error reading {file_name}: {e}")
    
    # Create database directory if it doesn't exist
    db_dir = os.path.join(os.path.dirname(processed_dir), "db")
    os.makedirs(db_dir, exist_ok=True)
    
    # Set database path
    db_path = os.path.join(db_dir, "transactions.db")
    
    # DEBUG: Check database schema before loading
    print("[DEBUG] Checking database schema before loading data")
    try:
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(users)")
            columns = cursor.fetchall()
            print("[DEBUG] Users table columns:")
            for col in columns:
                print(f"[DEBUG]   {col[1]}: {col[2]}")
            conn.close()
    except Exception as e:
        print(f"[DEBUG] Error checking database schema: {e}")
    
    try:
        # Load transactions into database
        print("[DEBUG] Starting load_transactions function")
        results = load_transactions(processed_dir)
        
        print(f"\nDatabase created at: {db_path}")
        
        # DEBUG: Check database after loading
        print("[DEBUG] Checking database after loading data")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check users table
            cursor.execute("SELECT user_id, occupation_category FROM users")
            users = cursor.fetchall()
            print(f"[DEBUG] Users in database after loading: {len(users)}")
            for user in users:
                print(f"[DEBUG]   User ID: {user[0]}, Occupation: {user[1] or 'None'}")
                
            conn.close()
        except Exception as e:
            print(f"[DEBUG] Error checking database after loading: {e}")
        
        # Run example queries if requested
        if run_examples:
            print("\n" + "="*80)
            print("RUNNING DATABASE QUERIES")
            print("="*80)
            query_examples(db_path)
            
    except Exception as e:
        print(f"❌ Error loading data into database: {e}")

def process_enriched_historical_data():
    """
    Process enriched historical data to create tax_insights.db.
    This is required for the RAG system to have a knowledge base.
    """
    print("\n" + "="*80)
    print("PROCESSING ENRICHED HISTORICAL DATA")
    print("="*80)
    
    # Check if tax_insights.db already exists
    tax_insights_db = "app/data/db/tax_insights.db"
    if os.path.exists(tax_insights_db):
        print(f"Tax insights database already exists at: {tax_insights_db}")
        # Check the number of records
        try:
            if os.path.exists(tax_insights_db):
                conn = sqlite3.connect(tax_insights_db)
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM enriched_users")
                user_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM enriched_transactions")
                tx_count = cursor.fetchone()[0]
                
                print(f"Database contains {user_count} enriched users and {tx_count} enriched transactions")
                conn.close()
                
                if user_count > 0:
                    print("Database has records, skipping processing of enriched data")
                    return
                else:
                    print("Database exists but is empty, processing enriched data...")
                
        except Exception as e:
            print(f"Error checking tax_insights database: {e}")
            print("Proceeding with processing enriched data...")
    else:
        print(f"Tax insights database does not exist, creating it...")
    
    # Call the process_enriched_data function to create the database
    try:
        process_enriched_data()
        print("Successfully processed enriched historical data and created tax_insights.db")
    except Exception as e:
        print(f"Error processing enriched data: {e}")
        print("This is a CRITICAL error. The RAG system needs historical data to work!")
        print("Make sure the notebooks/data/full_joined_data.csv file exists and the paths are correct.")
        sys.exit(1)

def run_vector_search_pipeline(skip_ingestion=False):
    """
    Run the vector search pipeline to find similar users and transactions.
    This integrates the functionality from test_with_sample_data.py but processes ALL data.
    
    Args:
        skip_ingestion: If True, skip the data ingestion step and use existing data in Weaviate
    """
    print("\n" + "="*80)
    print("RUNNING VECTOR SEARCH PIPELINE WITH ALL DATA")
    print("="*80)
    
    # Check if tax_insights.db exists before proceeding
    tax_insights_db = "app/data/db/tax_insights.db"
    if not os.path.exists(tax_insights_db):
        print(f"ERROR: Tax insights database not found at {tax_insights_db}")
        print("You need to process the enriched historical data first.")
        return
    
    # Connect to Weaviate
    print("Connecting to Weaviate...")
    client = get_weaviate_client()
    if not client:
        print("Failed to connect to Weaviate!")
        return
    
    try:
        # Ingest data into Weaviate if not skipping
        if not skip_ingestion:
            # Reset schema
            try:
                print("Deleting existing schema...")
                delete_schema(client)
            except Exception as e:
                print(f"No schema to delete or error: {str(e)}")
            
            # Create schema
            print("Creating schema...")
            create_schema(client)
            
            # Load ALL historical data from tax_insights.db
            print("Loading ALL historical data from tax_insights.db...")
            historical_users = fetch_users(limit=1000000, user_id=None)  # High limit to get all users
            print(f"Loaded {len(historical_users)} historical users")
            
            historical_transactions = fetch_transactions(limit=1000000)  # High limit to get all transactions
            print(f"Loaded {len(historical_transactions)} historical transactions")
            
            # Add ALL historical users to Weaviate
            print("Adding historical users to Weaviate...")
            for i, user in enumerate(historical_users, 1):
                add_or_update_user(client, user)
                if i % 100 == 0 or i == len(historical_users):
                    print(f"Progress: Added {i}/{len(historical_users)} historical users")
            
            # Add ALL historical transactions to Weaviate
            print("Adding historical transactions to Weaviate...")
            for i, tx in enumerate(historical_transactions, 1):
                add_or_update_transaction(client, tx)
                if i % 100 == 0 or i == len(historical_transactions):
                    print(f"Progress: Added {i}/{len(historical_transactions)} historical transactions")
        else:
            print("Skipping data ingestion - using existing data in Weaviate")
            
        # Load ALL input users from transactions.db
        print("Loading ALL input users from transactions.db...")
        input_conn = sqlite3.connect("app/data/db/transactions.db")
        input_conn.row_factory = sqlite3.Row
        input_cursor = input_conn.cursor()
        
        # Get ALL users
        input_cursor.execute("SELECT * FROM users")
        input_users = [dict(row) for row in input_cursor.fetchall()]
        
        # Modify input user IDs to make them distinct for search
        # This prevents users from finding themselves with perfect similarity
        for user in input_users:
            # Store original ID for reference and for finding transactions
            user['original_user_id'] = user['user_id']
            # Create a distinct ID for search
            user['search_id'] = f"INPUT-{user['user_id']}"
        
        print(f"Loaded {len(input_users)} input users")
        
        # Get ALL transactions
        input_cursor.execute("SELECT * FROM transactions")
        input_transactions = [dict(row) for row in input_cursor.fetchall()]
        print(f"Loaded {len(input_transactions)} input transactions")
        
        input_conn.close()
        
        # Wait for indexing to complete
        print("Waiting for indexing to complete...")
        time.sleep(5)
        
        # For each input user, find similar historical users
        print("\nRUNNING SIMILARITY SEARCHES FOR ALL INPUT USERS")
        for i, input_user in enumerate(input_users, 1):
            # Print a separator for each user
            print("\n" + "="*80)
            print(f"INPUT USER {i}/{len(input_users)}")
            print("="*80)
            
            # Print input user details in the same format as test_with_sample_data.py
            print("Input User:")
            print(f"User ID: {input_user['user_id']}")
            print(f"Occupation: {input_user.get('occupation_category', 'N/A')}")
            print(f"Annual Income: {input_user.get('annualized_income', 'N/A')}")
            print(f"Annual Tax Deductions: {input_user.get('annualized_tax_deductions', 'N/A')}")
            
            print("\n" + "-"*80 + "\n")
            
            # Use a copy of the user with the search ID to prevent self-matching
            search_user = input_user.copy()
            search_user['user_id'] = search_user['search_id']
            
            print("Finding similar historical users...")
            similar_users = search_similar_users(client, user_data=search_user, limit=10)  # Get more results initially
            
            # Filter out any exact self-matches (similarity = 1.0)
            # Also filter out low-quality matches below 0.4 similarity
            # Log how many users were filtered out
            original_count = len(similar_users)
            similar_users = [u for u in similar_users if u.similarity < 0.99 and u.similarity >= 0.4]
            filtered_count = original_count - len(similar_users)
            if filtered_count > 0:
                print(f"[DEBUG] Filtered out {filtered_count} users with similarity >= 0.99 or < 0.4")
            
            # Process user results to keep only those with meaningful recommendations
            users_with_recommendations = []
            seen_recommendations = set()  # Track unique recommendations
            
            for user_result in similar_users:
                user_id = user_result.data.get("user_id")
                
                # Add diagnostic print to see the user_id being looked up
                print(f"[DEBUG] Looking up user with ID: {user_id}")
                
                user_data = get_user_by_id(user_id) if user_id else user_result.data
                
                if user_data:
                    has_cluster = user_data.get('cluster_recommendation') not in [None, '']
                    has_uplift = user_data.get('uplift_message') not in [None, '']
                    
                    print(f"[DEBUG] User {user_id} has cluster_recommendation: {has_cluster}")
                    print(f"[DEBUG] User {user_id} has uplift_message: {has_uplift}")
                    
                    # Extract recommendations
                    cluster_rec = user_data.get('cluster_recommendation', '')
                    uplift_msg = user_data.get('uplift_message', '')
                    
                    # Create a unique signature for this recommendation set
                    rec_signature = f"{cluster_rec}||{uplift_msg}"
                    
                    # Only include users with at least one recommendation AND unique recommendation signature
                    if (has_cluster or has_uplift) and rec_signature not in seen_recommendations:
                        user_data["similarity_score"] = user_result.similarity
                        users_with_recommendations.append(user_data)
                        seen_recommendations.add(rec_signature)
                        print(f"[DEBUG] Adding unique recommendation: {rec_signature[:50]}...")
                    elif rec_signature in seen_recommendations:
                        print(f"[DEBUG] Skipping duplicate recommendation for user {user_id}")
            
            # Update similar_users to only include those with recommendations
            print(f"[DEBUG] Found {len(users_with_recommendations)} users with unique recommendations out of {len(similar_users)} similar users")
            
            # Display the filtered results
            print(f"Found {len(users_with_recommendations)} similar historical users with recommendations:")
            
            for j, user_data in enumerate(users_with_recommendations, 1):
                print(f"\nSimilar Historical User {j}:")
                # Print user using similar format as test_with_sample_data.py
                print(f"User ID: {user_data.get('user_id')}")
                print(f"Occupation: {user_data.get('occupation_category', 'N/A')}")
                print(f"Total Income: {user_data.get('total_income', 'N/A')}")
                print(f"Total Tax Deductions: {user_data.get('total_deductions', 'N/A')}")
                print(f"Similarity Score: {user_data.get('similarity_score'):.4f}")
                
                # Print recommendations if available
                if user_data.get("cluster_recommendation"):
                    print(f"\nRecommendation: {user_data.get('cluster_recommendation')}")
                    if user_data.get("cluster_confidence_level"):
                        print(f"Confidence: {user_data.get('cluster_confidence_level')}")
                if user_data.get("uplift_message"):
                    print(f"\nUplift Message: {user_data.get('uplift_message')}")
                    if user_data.get("uplift_confidence_level"):
                        print(f"Confidence: {user_data.get('uplift_confidence_level')}")
            
            # Print a separator between searches
            print("\n" + "-"*80)
            
            # Find similar transactions for this user's transactions
            user_transactions = [tx for tx in input_transactions if tx.get('user_id') == input_user['original_user_id']]
            if user_transactions:
                print(f"\nFinding similar historical transactions for {len(user_transactions)} transactions of user {input_user['user_id']}...")
                
                # Keep track of all seen transaction recommendations to avoid duplicates across searches
                all_seen_tx_recommendations = set()
                
                for k, tx in enumerate(user_transactions, 1):
                    print(f"\nInput Transaction {k}:")
                    print(f"Transaction ID: {tx.get('transaction_id')}")
                    print(f"Date: {tx.get('transaction_date', 'N/A')}")
                    print(f"Amount: {tx.get('amount', 'N/A')}")
                    print(f"Category: {tx.get('category', 'N/A')}")
                    if tx.get('subcategory'):
                        print(f"Subcategory: {tx.get('subcategory')}")
                    print(f"Vendor: {tx.get('vendor', 'N/A')}")
                    
                    # Create a copy for search
                    search_tx = tx.copy()
                    search_tx['transaction_id'] = f"INPUT-{tx['transaction_id']}"
                    
                    print("\nFinding similar historical transactions...")
                    similar_txs = search_similar_transactions(client, transaction_data=search_tx, limit=10)  # Get more results initially
                    
                    # Filter out any exact self-matches (similarity = 1.0)
                    # Also filter out low-quality matches below 0.4 similarity
                    # Log how many transactions were filtered out
                    tx_original_count = len(similar_txs)
                    similar_txs = [t for t in similar_txs if t.similarity < 0.99 and t.similarity >= 0.4]
                    tx_filtered_count = tx_original_count - len(similar_txs)
                    if tx_filtered_count > 0:
                        print(f"[DEBUG] Filtered out {tx_filtered_count} transactions with similarity >= 0.99 or < 0.4")
                    
                    # Process transaction results to keep only those with deduction info
                    txs_with_recommendations = []
                    seen_tx_recommendations = set()  # Track unique recommendations
                    
                    for tx_result in similar_txs:
                        tx_id = tx_result.data.get("transaction_id")
                        tx_data = get_transaction_by_id(tx_id) if tx_id else tx_result.data
                        
                        if tx_data:
                            has_deduction_info = (
                                tx_data.get('is_deductible') is not None or
                                tx_data.get('deduction_recommendation') not in [None, ''] or
                                tx_data.get('deduction_category') not in [None, '']
                            )
                            
                            if has_deduction_info:
                                # Create a unique signature for this deduction recommendation
                                deduction_rec = str(tx_data.get('deduction_recommendation', ''))
                                deduction_cat = str(tx_data.get('deduction_category', ''))
                                is_deductible = str(tx_data.get('is_deductible', ''))
                                
                                rec_signature = f"{is_deductible}||{deduction_cat}||{deduction_rec}"
                                
                                # Only include transactions with unique deduction recommendations
                                if rec_signature not in seen_tx_recommendations and rec_signature not in all_seen_tx_recommendations:
                                    tx_data["similarity_score"] = tx_result.similarity
                                    txs_with_recommendations.append(tx_data)
                                    seen_tx_recommendations.add(rec_signature)
                                    all_seen_tx_recommendations.add(rec_signature)
                                    print(f"[DEBUG] Adding unique tx recommendation: {rec_signature[:50]}...")
                                else:
                                    print(f"[DEBUG] Skipping duplicate tx recommendation for tx {tx_id}")
                    
                    # Display the filtered results
                    print(f"Found {len(txs_with_recommendations)} similar historical transactions with unique deduction info:")
                    
                    for l, tx_data in enumerate(txs_with_recommendations, 1):
                        print(f"\nSimilar Historical Transaction {l}:")
                        # Print transaction details
                        print(f"Transaction ID: {tx_data.get('transaction_id')}")
                        print(f"User ID: {tx_data.get('user_id', 'N/A')}")
                        print(f"Date: {tx_data.get('transaction_date', 'N/A')}")
                        print(f"Amount: {tx_data.get('amount', 'N/A')}")
                        print(f"Category: {tx_data.get('category', 'N/A')}")
                        if tx_data.get('subcategory'):
                            print(f"Subcategory: {tx_data.get('subcategory')}")
                        print(f"Similarity Score: {tx_data.get('similarity_score'):.4f}")
                        
                        # Print deduction information if available
                        if tx_data.get("is_deductible"):
                            print(f"Deductible: {tx_data.get('is_deductible')}")
                            if tx_data.get("deduction_recommendation"):
                                print(f"Deduction Recommendation: {tx_data.get('deduction_recommendation')}")
                            if tx_data.get("deduction_category"):
                                print(f"Deduction Category: {tx_data.get('deduction_category')}")
                
                # Now find transactions that might be relevant based on the user profile
                print("\n" + "-"*80)
                print(f"\nFinding historical transactions similar to {input_user['user_id']}'s profile...")
                
                relevant_txs = search_transactions_for_user(client, user_data=search_user, limit=10)  # Get more results initially
                
                # Filter out any exact self-matches (similarity = 1.0)
                # Also filter out low-quality matches below 0.4 similarity
                # Log how many transactions were filtered out
                tx_original_count = len(relevant_txs)
                relevant_txs = [t for t in relevant_txs if t.similarity < 0.99 and t.similarity >= 0.4]
                tx_filtered_count = tx_original_count - len(relevant_txs)
                if tx_filtered_count > 0:
                    print(f"[DEBUG] Filtered out {tx_filtered_count} transactions with similarity >= 0.99 or < 0.4")
                
                # Process transaction results to keep only those with deduction info
                profile_txs_with_recommendations = []
                seen_profile_tx_recommendations = set()  # Track unique recommendations
                
                for tx_result in relevant_txs:
                    tx_id = tx_result.data.get("transaction_id")
                    tx_data = get_transaction_by_id(tx_id) if tx_id else tx_result.data
                    
                    if tx_data:
                        has_deduction_info = (
                            tx_data.get('is_deductible') is not None or
                            tx_data.get('deduction_recommendation') not in [None, ''] or
                            tx_data.get('deduction_category') not in [None, '']
                        )
                        
                        if has_deduction_info:
                            # Create a unique signature for this deduction recommendation
                            deduction_rec = str(tx_data.get('deduction_recommendation', ''))
                            deduction_cat = str(tx_data.get('deduction_category', ''))
                            is_deductible = str(tx_data.get('is_deductible', ''))
                            
                            rec_signature = f"{is_deductible}||{deduction_cat}||{deduction_rec}"
                            
                            # Check if we've already seen this recommendation
                            # Also check against all previously seen transaction recommendations
                            if rec_signature not in seen_profile_tx_recommendations and rec_signature not in all_seen_tx_recommendations:
                                tx_data["similarity_score"] = tx_result.similarity
                                profile_txs_with_recommendations.append(tx_data)
                                seen_profile_tx_recommendations.add(rec_signature)
                                all_seen_tx_recommendations.add(rec_signature)
                                print(f"[DEBUG] Adding unique profile-based tx recommendation: {rec_signature[:50]}...")
                            else:
                                print(f"[DEBUG] Skipping duplicate profile-based tx recommendation for tx {tx_id}")
                
                # Display the filtered results
                print(f"Found {len(profile_txs_with_recommendations)} relevant historical transactions with unique deduction info:")
                
                for m, tx_data in enumerate(profile_txs_with_recommendations, 1):
                    print(f"\nRelevant Historical Transaction {m}:")
                    # Print transaction details
                    print(f"Transaction ID: {tx_data.get('transaction_id')}")
                    print(f"User ID: {tx_data.get('user_id', 'N/A')}")
                    print(f"Date: {tx_data.get('transaction_date', 'N/A')}")
                    print(f"Amount: {tx_data.get('amount', 'N/A')}")
                    print(f"Category: {tx_data.get('category', 'N/A')}")
                    if tx_data.get('subcategory'):
                        print(f"Subcategory: {tx_data.get('subcategory')}")
                    print(f"Similarity Score: {tx_data.get('similarity_score'):.4f}")
                    
                    # Print deduction information if available
                    if tx_data.get("is_deductible"):
                        print(f"Deductible: {tx_data.get('is_deductible')}")
                        if tx_data.get("deduction_recommendation"):
                            print(f"Deduction Recommendation: {tx_data.get('deduction_recommendation')}")
                        if tx_data.get("deduction_category"):
                            print(f"Deduction Category: {tx_data.get('deduction_category')}")
        
        print("\n" + "="*80)
        print("VECTOR SEARCH PIPELINE COMPLETED")
        print("="*80)
    
    finally:
        # Ensure we properly close the Weaviate client
        if client:
            print("Closing Weaviate client connection...")
            client.close()

def main():
    parser = argparse.ArgumentParser(description="Process documents using Google Document AI and load into SQLite database")
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
    parser.add_argument(
        "--skip-db", "-s",
        action="store_true",
        help="Skip database creation and loading step"
    )
    parser.add_argument(
        "--skip-examples", "-e",
        action="store_true",
        help="Skip running example queries"
    )
    parser.add_argument(
        "--skip-vector", "-v",
        action="store_true",
        help="Skip vector search pipeline"
    )
    parser.add_argument(
        "--skip-enriched", "-r",
        action="store_true",
        help="Skip processing of enriched historical data"
    )
    parser.add_argument(
        "--skip-ingestion", "-g",
        action="store_true",
        help="Skip vector data ingestion (use existing Weaviate data)"
    )
    parser.add_argument(
        "--vector-only", "-w",
        action="store_true",
        help="Run only the vector search pipeline, skipping document processing and database creation"
    )
    parser.add_argument(
        "--semantic-only", "-m",
        action="store_true",
        help="Run only the semantic search queries, skipping document processing and database creation"
    )

    args = parser.parse_args()
    
    # If semantic-only mode is specified, skip to semantic search queries
    if args.semantic_only:
        print("Running in semantic-only mode - skipping document processing and database creation")
        run_example_semantic_queries()
        return
    
    # If vector-only mode is specified, skip to vector search pipeline
    if args.vector_only:
        print("Running in vector-only mode - skipping document processing and database creation")
        run_vector_search_pipeline(skip_ingestion=args.skip_ingestion)
        return
    
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
    
    # After processing documents, load them into the database
    if not args.skip_db:
        process_and_load_to_database(args.output, not args.skip_examples)
    
    # Process enriched historical data to create tax_insights.db
    if not args.skip_enriched:
        process_enriched_historical_data()
        
    # Run vector search pipeline
    if not args.skip_vector:
        run_vector_search_pipeline(skip_ingestion=args.skip_ingestion)

    # Run example semantic queries if not skipped
    if not args.skip_examples:
        run_example_semantic_queries()

if __name__ == "__main__":
    main()
