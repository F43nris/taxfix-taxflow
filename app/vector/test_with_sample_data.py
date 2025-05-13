#!/usr/bin/env python3
"""
Test script to demonstrate the vector search functionality with sample data.
"""
import os
import sys
import json
from typing import Dict, Any, List

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.vector.client import get_weaviate_client
from app.vector.schema import create_schema, delete_schema
from app.vector.upsert import add_or_update_user, add_or_update_transaction
from app.vector.search import search_similar_users, search_similar_transactions, search_transactions_for_user
from app.vector.data_loader import get_sample_data, get_user_by_id, get_transaction_by_id
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

def setup_vector_db():
    """Set up the vector database with sample data."""
    print("Setting up vector database...")
    
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
    
    # Load sample data
    users, transactions = get_sample_data()
    
    print(f"Loaded {len(users)} users and {len(transactions)} transactions from SQLite databases")
    
    # Add users to Weaviate
    for user in users:
        add_or_update_user(client, user)
        print(f"Added user {user.get('user_id')} to Weaviate")
    
    # Add transactions to Weaviate
    for tx in transactions:
        add_or_update_transaction(client, tx)
        print(f"Added transaction {tx.get('transaction_id')} to Weaviate")
    
    print("Vector database setup complete!")
    return client

def test_similarity_searches(client):
    """Test similarity searches."""
    print_separator()
    print("TESTING SIMILARITY SEARCHES")
    print_separator()
    
    # Get the first user
    users, _ = get_sample_data()
    if not users:
        print("No users found in sample data!")
        return
    
    test_user = users[0]
    print("Test User:")
    print_user(test_user)
    print_separator()
    
    # 1. Find similar users
    print("Finding similar users...")
    similar_users = search_similar_users(
        client,
        user_data=test_user,
        limit=3
    )
    
    print(f"Found {len(similar_users)} similar users:")
    for i, user_result in enumerate(similar_users):
        print(f"\nSimilar User {i+1}:")
        user_id = user_result.data.get("user_id")
        user_data = get_user_by_id(user_id) if user_id else user_result.data
        if user_data:
            user_data["similarity_score"] = user_result.similarity
            print_user(user_data)
    
    print_separator()
    
    # 2. Find similar transactions
    _, transactions = get_sample_data()
    if not transactions:
        print("No transactions found in sample data!")
        return
    
    test_tx = transactions[0]
    print("Test Transaction:")
    print_transaction(test_tx)
    print_separator()
    
    print("Finding similar transactions...")
    similar_txs = search_similar_transactions(
        client,
        transaction_data=test_tx,
        limit=3
    )
    
    print(f"Found {len(similar_txs)} similar transactions:")
    for i, tx_result in enumerate(similar_txs):
        print(f"\nSimilar Transaction {i+1}:")
        tx_id = tx_result.data.get("transaction_id")
        tx_data = get_transaction_by_id(tx_id) if tx_id else tx_result.data
        if tx_data:
            tx_data["similarity_score"] = tx_result.similarity
            print_transaction(tx_data)
    
    print_separator()
    
    # 3. Find transactions for user
    print("Finding transactions for user...")
    user_txs = search_transactions_for_user(
        client,
        user_data=test_user,
        limit=3
    )
    
    print(f"Found {len(user_txs)} transactions for user:")
    for i, tx_result in enumerate(user_txs):
        print(f"\nTransaction {i+1} for User:")
        tx_id = tx_result.data.get("transaction_id")
        tx_data = get_transaction_by_id(tx_id) if tx_id else tx_result.data
        if tx_data:
            tx_data["similarity_score"] = tx_result.similarity
            print_transaction(tx_data)

def test_search_api():
    """Test the high-level search API."""
    print_separator()
    print("TESTING SEARCH API")
    print_separator()
    
    # Create API instance
    api = TaxInsightSearchAPI()
    
    # Get the first user
    users, _ = get_sample_data()
    if not users:
        print("No users found in sample data!")
        return
    
    test_user = users[0]
    user_id = test_user.get("user_id")
    
    print(f"Getting tax recommendations for user {user_id}...")
    recommendations = api.get_tax_recommendations(user_id)
    
    if "error" in recommendations:
        print(f"Error: {recommendations['error']}")
        return
    
    # Print user profile
    print("\nUser Profile:")
    print_user(recommendations["user_profile"])
    
    # Print transaction count
    print(f"\nTransaction Count: {recommendations['transaction_count']}")
    
    # Print similar users
    print(f"\nSimilar Users ({len(recommendations['similar_users'])}):")
    for i, user in enumerate(recommendations['similar_users'][:3]):  # Show top 3
        print(f"\nSimilar User {i+1}:")
        print_user(user)
    
    # Print potential deductions
    print(f"\nPotential Deductions ({len(recommendations['potential_deductions'])}):")
    for i, tx in enumerate(recommendations['potential_deductions'][:3]):  # Show top 3
        print(f"\nPotential Deduction {i+1}:")
        print_transaction(tx)
    
    # Print recommendations
    if recommendations.get("cluster_recommendation"):
        print(f"\nRecommendation: {recommendations.get('cluster_recommendation')}")
    
    if recommendations.get("uplift_message"):
        print(f"\nUplift Message: {recommendations.get('uplift_message')}")

def main():
    """Main function."""
    print("VECTOR SEARCH TEST WITH SAMPLE DATA")
    print_separator()
    
    # Setup vector database
    client = setup_vector_db()
    if not client:
        print("Failed to set up vector database!")
        return 1
    
    # Wait a bit for indexing to complete
    import time
    print("Waiting for indexing to complete...")
    time.sleep(3)
    
    # Test similarity searches
    test_similarity_searches(client)
    
    # Test search API
    test_search_api()
    
    print_separator()
    print("Test completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 