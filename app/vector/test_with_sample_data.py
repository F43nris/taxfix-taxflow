#!/usr/bin/env python3
"""
Test script to demonstrate the vector search functionality with sample data.
"""
import os
import sys
import json
import sqlite3
from typing import Dict, Any, List, Tuple

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.vector.client import get_weaviate_client
from app.vector.schema import create_schema, delete_schema
from app.vector.upsert import add_or_update_user, add_or_update_transaction
from app.vector.search import search_similar_users, search_similar_transactions, search_transactions_for_user
from app.vector.data_loader import get_user_by_id, get_transaction_by_id
from app.vector.search_api import TaxInsightSearchAPI

def print_separator():
    """Print a separator line."""
    print("\n" + "=" * 80 + "\n")

def print_user(user: Dict[str, Any]):
    """Print user information in a readable format."""
    print(f"User ID: {user.get('user_id')}")
    print(f"Name: {user.get('employee_name', 'N/A')}")
    print(f"Employer: {user.get('employer_name', 'N/A')}")
    print(f"Occupation: {user.get('occupation_category', 'N/A')}")
    print(f"Income Band: {user.get('income_band', 'N/A')}")
    print(f"Annual Income: {user.get('annualized_income', 'N/A')}")
    
    # Print similarity score if available
    if "similarity_score" in user:
        print(f"Similarity Score: {user.get('similarity_score'):.4f}")
    
    # Print enriched data if available
    if user.get("cluster_recommendation"):
        print(f"\nRecommendation: {user.get('cluster_recommendation')}")
    
    if user.get("uplift_message"):
        print(f"Uplift Message: {user.get('uplift_message')}")

def print_transaction(tx: Dict[str, Any]):
    """Print transaction information in a readable format."""
    print(f"Transaction ID: {tx.get('transaction_id')}")
    print(f"User ID: {tx.get('user_id', 'N/A')}")
    print(f"Date: {tx.get('transaction_date', 'N/A')}")
    print(f"Amount: {tx.get('amount', 'N/A')}")
    print(f"Category: {tx.get('category', 'N/A')}")
    print(f"Vendor: {tx.get('vendor', 'N/A')}")
    print(f"Description: {tx.get('description', 'N/A')}")
    
    # Print similarity score if available
    if "similarity_score" in tx:
        print(f"Similarity Score: {tx.get('similarity_score'):.4f}")
    
    # Print enriched data if available
    if "is_deductible" in tx:
        print(f"Deductible: {tx.get('is_deductible')}")
        if tx.get("deduction_recommendation"):
            print(f"Deduction Recommendation: {tx.get('deduction_recommendation')}")
        if tx.get("deduction_category"):
            print(f"Deduction Category: {tx.get('deduction_category')}")

def get_input_data() -> Tuple[List[Dict], List[Dict]]:
    """
    Get input data from the transactions database created by main.py.
    This represents real input data from newly processed documents.
    """
    # Path to the transactions database created by main.py
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "db", "transactions.db")
    
    if not os.path.exists(db_path):
        print(f"Input database not found at {db_path}")
        print("Make sure to run main.py first to process documents and create the database.")
        return [], []
    
    print(f"Loading input data from {db_path}")
    
    input_users = []
    input_transactions = []
    
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get users from the database
        cursor.execute("SELECT * FROM users LIMIT 3")
        user_rows = cursor.fetchall()
        for i, row in enumerate(user_rows):
            user_dict = dict(row)
            # Create a modified copy with a new user_id to ensure separation from historical data
            original_id = user_dict.get('user_id', 'Unknown ID')
            user_dict['user_id'] = f"INPUT-USER-{i+1}-FROM-{original_id}"
            user_dict['original_user_id'] = original_id  # Keep the original ID for reference
            
            input_users.append(user_dict)
            print(f"Loaded input user: {user_dict.get('user_id')} (originally {original_id})")
        
        # Get transactions from the database
        cursor.execute("SELECT * FROM transactions LIMIT 5")
        tx_rows = cursor.fetchall()
        for i, row in enumerate(tx_rows):
            tx_dict = dict(row)
            # Update transaction to use the new user ID if it belongs to one of our input users
            original_user_id = tx_dict.get('user_id')
            for user in input_users:
                if user.get('original_user_id') == original_user_id:
                    tx_dict['user_id'] = user.get('user_id')
                    break
            
            # Also assign a new transaction ID to ensure complete separation
            original_tx_id = tx_dict.get('transaction_id', 'Unknown ID')
            tx_dict['transaction_id'] = f"INPUT-TX-{i+1}-FROM-{original_tx_id}"
            tx_dict['original_transaction_id'] = original_tx_id  # Keep the original ID for reference
            
            input_transactions.append(tx_dict)
            print(f"Loaded input transaction: {tx_dict.get('transaction_id')} (originally {original_tx_id})")
        
        conn.close()
        
    except Exception as e:
        print(f"Error loading input data: {e}")
    
    return input_users, input_transactions

def get_historical_data() -> Tuple[List[Dict], List[Dict]]:
    """
    Get historical data ONLY from tax_insights.db.
    This ensures proper separation from input data in transactions.db.
    """
    # Path to the enriched database created by process_enriched_data.py
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "db", "tax_insights.db")
    
    if not os.path.exists(db_path):
        print(f"Historical database not found at {db_path}")
        print("Make sure to run process_enriched_data.py first to create the enriched database.")
        return [], []
    
    print(f"Loading historical data from {db_path}")
    
    historical_users = []
    historical_transactions = []
    
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get users from the enriched database
        cursor.execute("SELECT * FROM enriched_users LIMIT 10")
        user_rows = cursor.fetchall()
        for row in user_rows:
            user_dict = dict(row)
            historical_users.append(user_dict)
            print(f"Loaded historical user: {user_dict.get('user_id', 'Unknown ID')}")
        
        # Get transactions from the enriched database
        cursor.execute("SELECT * FROM enriched_transactions LIMIT 20")
        tx_rows = cursor.fetchall()
        for row in tx_rows:
            tx_dict = dict(row)
            historical_transactions.append(tx_dict)
            print(f"Loaded historical transaction: {tx_dict.get('transaction_id', 'Unknown ID')}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error loading historical data: {e}")
    
    return historical_users, historical_transactions

def setup_vector_db():
    """
    Set up the vector database with historical data.
    This simulates the enriched historical data in a real RAG system.
    """
    print("Setting up vector database with historical data...")
    
    # Connect to Weaviate
    client = get_weaviate_client()
    
    # Reset schema
    try:
        delete_schema(client)
        print("Deleted existing schema")
    except Exception as e:
        print(f"No schema to delete or error: {e}")
    
    # Create schema
    create_schema(client)
    print("Created schema")
    
    # Load historical data ONLY from enriched database (tax_insights.db)
    historical_users, historical_transactions = get_historical_data()
    
    print(f"Loaded {len(historical_users)} historical users and {len(historical_transactions)} historical transactions")
    
    # Add historical users to Weaviate
    for user in historical_users:
        add_or_update_user(client, user)
        print(f"Added historical user {user.get('user_id')} to Weaviate")
    
    # Add historical transactions to Weaviate
    for tx in historical_transactions:
        add_or_update_transaction(client, tx)
        print(f"Added historical transaction {tx.get('transaction_id')} to Weaviate")
    
    print("Vector database setup complete with historical data!")
    return client

def test_similarity_searches(client, input_users, input_transactions):
    """
    Test similarity searches using real input data to query historical data.
    This simulates the real-world scenario where new documents query for similar historical records.
    """
    print_separator()
    print("TESTING SIMILARITY SEARCHES WITH REAL INPUT DATA")
    print_separator()
    
    if not input_users:
        print("No input users available for testing!")
        return
    
    # 1. Find similar users using the first input user
    test_user = input_users[0]
    print("Input User:")
    print_user(test_user)
    print_separator()
    
    print("Finding similar historical users...")
    similar_users = search_similar_users(
        client,
        user_data=test_user,
        limit=3
    )
    
    print(f"Found {len(similar_users)} similar historical users:")
    for i, user_result in enumerate(similar_users):
        print(f"\nSimilar Historical User {i+1}:")
        user_id = user_result.data.get("user_id")
        user_data = get_user_by_id(user_id) if user_id else user_result.data
        if user_data:
            user_data["similarity_score"] = user_result.similarity
            print_user(user_data)
    
    print_separator()
    
    # 2. Find similar transactions using the first input transaction
    if not input_transactions:
        print("No input transactions available for testing!")
        return
    
    test_tx = input_transactions[0]
    print("Input Transaction:")
    print_transaction(test_tx)
    print_separator()
    
    print("Finding similar historical transactions...")
    similar_txs = search_similar_transactions(
        client,
        transaction_data=test_tx,
        limit=3
    )
    
    print(f"Found {len(similar_txs)} similar historical transactions:")
    for i, tx_result in enumerate(similar_txs):
        print(f"\nSimilar Historical Transaction {i+1}:")
        tx_id = tx_result.data.get("transaction_id")
        tx_data = get_transaction_by_id(tx_id) if tx_id else tx_result.data
        if tx_data:
            tx_data["similarity_score"] = tx_result.similarity
            print_transaction(tx_data)
    
    print_separator()
    
    # 3. Find transactions for user
    print("Finding historical transactions similar to input user's profile...")
    user_txs = search_transactions_for_user(
        client,
        user_data=test_user,
        limit=3
    )
    
    print(f"Found {len(user_txs)} relevant historical transactions:")
    for i, tx_result in enumerate(user_txs):
        print(f"\nRelevant Historical Transaction {i+1}:")
        tx_id = tx_result.data.get("transaction_id")
        tx_data = get_transaction_by_id(tx_id) if tx_id else tx_result.data
        if tx_data:
            tx_data["similarity_score"] = tx_result.similarity
            print_transaction(tx_data)

def main():
    """Main function."""
    print("VECTOR SEARCH TEST WITH REAL DATA FLOW")
    print_separator()
    
    # Setup vector database with historical data
    client = setup_vector_db()
    if not client:
        print("Failed to set up vector database!")
        return 1
    
    # Get real input data from database created by main.py
    input_users, input_transactions = get_input_data()
    
    if not input_users or not input_transactions:
        print("ERROR: No input data found in transactions.db")
        print("Please run main.py first to process documents and create the database.")
        return 1
    
    # Wait a bit for indexing to complete
    import time
    print("Waiting for indexing to complete...")
    time.sleep(3)
    
    # Test similarity searches with input data querying historical data
    test_similarity_searches(client, input_users, input_transactions)
    
    print_separator()
    print("Test completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 