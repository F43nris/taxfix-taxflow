"""
Examples and tests for Weaviate vector database integration.
"""
import time
from datetime import datetime

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

def print_search_results(results):
    """Print search results in a readable format."""
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result.class_name} (Similarity: {result.similarity:.4f})")
        for key, value in result.data.items():
            if isinstance(value, (str, int, float)) and not key.startswith("_"):
                print(f"   {key}: {value}")

def run_all_tests():
    """Run all tests."""
    print("=== WEAVIATE INTEGRATION TESTS ===\n")
    
    # Test connection
    client = test_connection()
    if not client:
        return False
    
    try:
        # Test schema
        if not test_schema(client):
            return False
        
        # Test data operations
        if not test_data_operations(client):
            return False
        
        print("\n=== ALL TESTS PASSED ===")
        return True
    finally:
        # Close the client connection
        if client:
            client.close()
            print("\nWeaviate client connection closed.")

if __name__ == "__main__":
    run_all_tests() 