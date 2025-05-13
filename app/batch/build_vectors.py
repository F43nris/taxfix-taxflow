#!/usr/bin/env python3
"""
Script to build and upload vector embeddings for users and transactions to Weaviate.
"""

import os
import sys
import argparse
import sqlite3
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import time

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.vector.client import get_weaviate_client
from app.vector.schema import create_schema, delete_schema
from app.vector.upsert import batch_add_users, batch_add_transactions
from app.vector.settings import USER_CLASS, TRANSACTION_CLASS, WEAVIATE_URL

def fetch_users_from_db(db_path: str) -> List[Dict[str, Any]]:
    """
    Fetch all users from the SQLite database.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        List of user dictionaries
    """
    users = []
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Query all users
        cursor.execute("""
        SELECT 
            user_id, employee_name, employee_name_confidence, employer_name, employer_name_confidence,
            occupation_category, occupation_category_confidence, filing_id, tax_year,
            filing_date, avg_gross_pay, gross_pay_confidence, avg_net_pay, net_pay_confidence,
            avg_tax_deductions, income_band, annualized_income, annualized_net_pay,
            annualized_tax_deductions, payslip_count, gross_pay_count, net_pay_count
        FROM users
        """)
        
        # Process results
        for row in cursor.fetchall():
            user = dict(row)
            
            # Convert filing_date to datetime if it's not None
            if user["filing_date"]:
                try:
                    user["filing_date"] = datetime.fromisoformat(user["filing_date"].replace("Z", "+00:00"))
                except:
                    pass
            
            users.append(user)
        
        # Close connection
        conn.close()
        
        print(f"Fetched {len(users)} users from database")
        return users
    except Exception as e:
        print(f"Error fetching users from database: {str(e)}")
        return []

def fetch_transactions_from_db(db_path: str) -> List[Dict[str, Any]]:
    """
    Fetch all transactions from the SQLite database.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        List of transaction dictionaries
    """
    transactions = []
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Query all transactions
        cursor.execute("""
        SELECT 
            transaction_id, user_id, transaction_date, amount, category, subcategory,
            description, vendor, receipt_id, confidence_score, amount_confidence,
            date_confidence, vendor_confidence, category_confidence,
            transaction_month, quarter, year
        FROM transactions
        """)
        
        # Process results
        for row in cursor.fetchall():
            transaction = dict(row)
            
            # Convert transaction_date to datetime if it's not None
            if transaction["transaction_date"]:
                try:
                    transaction["transaction_date"] = datetime.fromisoformat(transaction["transaction_date"].replace("Z", "+00:00"))
                except:
                    pass
            
            transactions.append(transaction)
        
        # Close connection
        conn.close()
        
        print(f"Fetched {len(transactions)} transactions from database")
        return transactions
    except Exception as e:
        print(f"Error fetching transactions from database: {str(e)}")
        return []

def build_and_upload_vectors(db_path: str, reset: bool = False):
    """
    Build and upload vector embeddings for users and transactions to Weaviate.
    
    Args:
        db_path: Path to the SQLite database
        reset: Whether to reset the Weaviate schema
    """
    print(f"\n{'=' * 80}")
    print(f"BUILDING AND UPLOADING VECTORS TO WEAVIATE")
    print(f"{'=' * 80}")
    
    # Initialize Weaviate client
    client = get_weaviate_client()
    print(f"Connected to Weaviate at {WEAVIATE_URL}")
    
    try:
        # Reset schema if requested
        if reset:
            print("Resetting Weaviate schema...")
            delete_schema(client)
        
        # Create schema
        create_schema(client)
        
        # Fetch users and transactions from database
        users = fetch_users_from_db(db_path)
        transactions = fetch_transactions_from_db(db_path)
        
        if not users and not transactions:
            print("No data to upload")
            return
        
        # Upload users
        if users:
            print(f"\nUploading {len(users)} users to Weaviate...")
            start_time = time.time()
            user_ids = batch_add_users(client, users)
            elapsed_time = time.time() - start_time
            print(f"Uploaded {len(user_ids)} users in {elapsed_time:.2f} seconds")
        
        # Upload transactions
        if transactions:
            print(f"\nUploading {len(transactions)} transactions to Weaviate...")
            start_time = time.time()
            transaction_ids = batch_add_transactions(client, transactions)
            elapsed_time = time.time() - start_time
            print(f"Uploaded {len(transaction_ids)} transactions in {elapsed_time:.2f} seconds")
        
        print(f"\nVector database build completed!")
    finally:
        # Close the client connection
        client.close()

def main():
    parser = argparse.ArgumentParser(description="Build and upload vector embeddings to Weaviate")
    parser.add_argument(
        "--db", "-d",
        default="app/data/db/transactions.db",
        help="Path to the SQLite database (default: app/data/db/transactions.db)"
    )
    parser.add_argument(
        "--reset", "-r",
        action="store_true",
        help="Reset Weaviate schema before uploading"
    )
    
    args = parser.parse_args()
    
    # Build and upload vectors
    build_and_upload_vectors(args.db, args.reset)

if __name__ == "__main__":
    main() 