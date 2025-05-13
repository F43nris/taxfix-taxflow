"""
Examples and tests for Weaviate vector database integration.
"""
import time
from datetime import datetime
import json
import sqlite3
from typing import Dict, Any, List
import os

from app.vector.client import get_weaviate_client
from app.vector.schema import create_schema, delete_schema
from app.vector.upsert import add_or_update_user, add_or_update_transaction
from app.vector.search import (
    search_similar_users,
    search_similar_transactions,
    search_transactions_for_user,
    search_users_for_transaction,
    SearchResult
)

def test_connection():
    """Test connection to Weaviate."""
    print("Testing connection to Weaviate...")
    try:
        client = get_weaviate_client()
        print("✓ Successfully connected to Weaviate!")
        return client
    except Exception as e:
        print(f"✗ Failed to connect to Weaviate: {str(e)}")
        return None

def test_schema(client):
    """Test schema creation and deletion."""
    print("\nTesting schema operations...")
    
    try:
        # Delete existing schema if it exists
        delete_schema(client)
        print("✓ Schema deleted successfully")
        
        # Create schema
        create_schema(client)
        print("✓ Schema created successfully")
        
        # Get schema
        collections = client.collections.list_all()
        collection_names = [collection.name for collection in collections]
        print(f"✓ Available collections: {collection_names}")
        
        return True
    except Exception as e:
        print(f"✗ Schema test failed: {str(e)}")
        return False

def test_data_operations(client):
    """Test data operations (add, update, search)."""
    print("\nTesting data operations...")
    
    try:
        # Create test user
        user_data = {
            "user_id": "test-user-001",
            "employee_name": "Jane Doe",
            "employer_name": "Acme Corp",
            "occupation_category": "Software Engineering",
            "income_band": "80000-100000",
            "avg_gross_pay": 8500.0,
            "avg_net_pay": 6200.0,
            "avg_tax_deductions": 2300.0,
            "annualized_income": 102000.0,
            "tax_year": 2023,
            "payslip_count": 12
        }
        
        user_id = add_or_update_user(client, user_data)
        print(f"✓ Added user with ID: {user_id}")
        
        # Create test transactions
        transactions = [
            {
                "transaction_id": "test-tx-001",
                "user_id": "test-user-001",
                "receipt_id": "receipt-001",
                "transaction_date": datetime.now().isoformat(),
                "amount": 120.50,
                "category": "Software",
                "subcategory": "Development Tools",
                "description": "IDE Subscription",
                "vendor": "JetBrains",
                "year": 2023,
                "month": 6,
                "quarter": 2,
                "confidence_score": 0.95
            },
            {
                "transaction_id": "test-tx-002",
                "user_id": "test-user-001",
                "receipt_id": "receipt-002",
                "transaction_date": datetime.now().isoformat(),
                "amount": 45.99,
                "category": "Office Supplies",
                "subcategory": "Computer Accessories",
                "description": "Wireless Mouse",
                "vendor": "Logitech",
                "year": 2023,
                "month": 6,
                "quarter": 2,
                "confidence_score": 0.92
            }
        ]
        
        for tx in transactions:
            tx_id = add_or_update_transaction(client, tx)
            print(f"✓ Added transaction with ID: {tx_id}")
        
        # Wait for indexing
        print("Waiting for indexing...")
        time.sleep(2)
        
        # Test search
        print("\nTesting search operations...")
        
        # Search for similar users
        similar_users = search_similar_users(
            client,
            query_text="software engineer with high income",
            limit=5
        )
        print(f"✓ Found {len(similar_users)} similar users")
        
        # Search for similar transactions
        similar_txs = search_similar_transactions(
            client,
            query_text="computer equipment purchase",
            limit=5
        )
        print(f"✓ Found {len(similar_txs)} similar transactions")
        
        # Search for transactions for a user
        user_txs = search_transactions_for_user(
            client,
            user_data,
            limit=5
        )
        print(f"✓ Found {len(user_txs)} transactions for user")
        
        # Search for users for a transaction
        tx_users = search_users_for_transaction(
            client,
            transactions[0],
            limit=5
        )
        print(f"✓ Found {len(tx_users)} users for transaction")
        
        return True
    except Exception as e:
        print(f"✗ Data operations test failed: {str(e)}")
        return False

def format_search_result(result: SearchResult) -> Dict[str, Any]:
    """Format a search result for display."""
    return {
        "object_id": result.object_id,
        "class_name": result.class_name,
        "similarity": f"{result.similarity:.4f}",
        "data": {k: v for k, v in result.data.items() if k not in ["vector"]}
    }

def print_search_results(results: List[SearchResult], title: str):
    """Print search results in a readable format."""
    print(f"\n=== {title} ===")
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result.class_name} (Similarity: {result.similarity:.4f})")
        # Print the most relevant fields based on class
        if result.class_name == "User":
            print(f"   User ID: {result.data.get('user_id')}")
            print(f"   Employee: {result.data.get('employee_name')}")
            print(f"   Employer: {result.data.get('employer_name')}")
            print(f"   Occupation: {result.data.get('occupation_category')}")
            print(f"   Income Band: {result.data.get('income_band')}")
            print(f"   Annual Income: {result.data.get('annualized_income')}")
        else:  # Transaction
            print(f"   Transaction ID: {result.data.get('transaction_id')}")
            print(f"   User ID: {result.data.get('user_id')}")
            print(f"   Date: {result.data.get('transaction_date')}")
            print(f"   Amount: {result.data.get('amount')}")
            print(f"   Category: {result.data.get('category')}")
            print(f"   Vendor: {result.data.get('vendor')}")
            print(f"   Description: {result.data.get('description')}")
            if result.data.get('is_deductible') is not None:
                print(f"   Deductible: {result.data.get('is_deductible')}")
                print(f"   Deduction Category: {result.data.get('deduction_category')}")

def fetch_sample_data_from_db():
    """Fetch sample data from SQLite database."""
    db_path = os.path.join("app", "data", "db", "tax_insights.db")
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return None, None
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Fetch a sample user
        cursor.execute("SELECT * FROM enriched_users LIMIT 1")
        user = dict(cursor.fetchone())
        
        # Fetch sample transactions
        cursor.execute("SELECT * FROM enriched_transactions LIMIT 5")
        transactions = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return user, transactions
    except Exception as e:
        print(f"Error fetching sample data: {str(e)}")
        return None, None

def example_similar_users_search(client):
    """Example of searching for similar users."""
    # Create a sample user query
    query_user = {
        "employee_name": "John Smith",
        "employer_name": "Tech Solutions Inc.",
        "occupation_category": "Software Engineer",
        "avg_gross_pay": 6000.0,
        "avg_net_pay": 4500.0,
        "income_band": "C: 51,131-100,000 €"
    }
    
    print("\n=== Searching for similar users ===")
    print("Query user profile:")
    print(json.dumps(query_user, indent=2))
    
    # Search for similar users
    results = search_similar_users(
        client=client,
        user_data=query_user,
        limit=3
    )
    
    print_search_results(results, "Similar Users")
    return results

def example_similar_transactions_search(client):
    """Example of searching for similar transactions."""
    # Create a sample transaction query
    query_transaction = {
        "amount": 250.0,
        "category": "Office Supplies",
        "vendor": "Office Depot",
        "description": "Purchased laptop accessories and office supplies"
    }
    
    print("\n=== Searching for similar transactions ===")
    print("Query transaction:")
    print(json.dumps(query_transaction, indent=2))
    
    # Search for similar transactions
    results = search_similar_transactions(
        client=client,
        transaction_data=query_transaction,
        limit=3
    )
    
    print_search_results(results, "Similar Transactions")
    return results

def example_transactions_for_user(client, user_data):
    """Example of searching for transactions related to a user profile."""
    print("\n=== Searching for transactions related to user profile ===")
    print("User profile:")
    print(f"Name: {user_data.get('employee_name')}")
    print(f"Employer: {user_data.get('employer_name')}")
    print(f"Occupation: {user_data.get('occupation_category')}")
    print(f"Income Band: {user_data.get('income_band')}")
    
    # Search for transactions related to this user
    results = search_transactions_for_user(
        client=client,
        user_data=user_data,
        limit=5
    )
    
    print_search_results(results, "Transactions for User")
    return results

def example_users_for_transaction(client, transaction_data):
    """Example of searching for users related to a transaction."""
    print("\n=== Searching for users related to transaction ===")
    print("Transaction:")
    print(f"Amount: {transaction_data.get('amount')}")
    print(f"Category: {transaction_data.get('category')}")
    print(f"Vendor: {transaction_data.get('vendor')}")
    print(f"Description: {transaction_data.get('description')}")
    
    # Search for users related to this transaction
    results = search_users_for_transaction(
        client=client,
        transaction_data=transaction_data,
        limit=3
    )
    
    print_search_results(results, "Users for Transaction")
    return results

def example_filtered_search(client):
    """Example of searching with filters."""
    # Search for high-income users
    print("\n=== Searching for high-income users ===")
    filters = {
        "income_band": "D: 100,001+ €"
    }
    
    results = search_similar_users(
        client=client,
        query_text="high income professionals",
        filters=filters,
        limit=3
    )
    
    print_search_results(results, "High-Income Users")
    
    # Search for deductible transactions
    print("\n=== Searching for deductible transactions ===")
    filters = {
        "is_deductible": True
    }
    
    results = search_similar_transactions(
        client=client,
        query_text="tax deductible expenses",
        filters=filters,
        limit=3
    )
    
    print_search_results(results, "Deductible Transactions")
    return results

def run_examples():
    """Run all examples."""
    client = test_connection()
    if not client:
        return
    
    try:
        print("\n=== Running Vector Search Examples ===")
        
        # Fetch sample data
        user, transactions = fetch_sample_data_from_db()
        
        # Run examples with sample data if available
        if user and transactions:
            example_transactions_for_user(client, user)
            example_users_for_transaction(client, transactions[0])
        
        # Run examples with synthetic data
        example_similar_users_search(client)
        example_similar_transactions_search(client)
        example_filtered_search(client)
        
        print("\n=== Examples completed successfully ===")
    except Exception as e:
        print(f"Error running examples: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    run_examples() 