#!/usr/bin/env python3
"""
Script to load processed document data into SQLite database.
"""

import os
import argparse
from pathlib import Path

from app.database.db import Database
from app.database.models import DocumentType, User


def load_transactions(processed_dir: str = "app/data/processed", user_id: str = None) -> dict:
    """
    Load all processed documents into the transactions database.
    
    Args:
        processed_dir: Directory containing processed JSON files
        user_id: User ID to associate with transactions (optional)
        
    Returns:
        Dictionary with counts of processed files
    """
    # Create and initialize the database
    db_dir = os.path.join(os.path.dirname(processed_dir), "db")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "transactions.db")
    
    with Database(db_path) as db:
        print(f"Connected to database: {db_path}")
        
        # Create tables if they don't exist
        db.create_tables()
        print("Database tables created/verified")
        
        # Process all documents
        results = db.process_all_documents(processed_dir, user_id)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"  Receipts processed: {results['receipts']}")
        print(f"  Income statements processed: {results['income_statements']}")
        if results['errors'] > 0:
            print(f"  Errors encountered: {results['errors']}")
        
        # Get total transactions count
        db.cursor.execute("SELECT COUNT(*) FROM transactions")
        total_transactions = db.cursor.fetchone()[0]
        print(f"\nTotal transactions in database: {total_transactions}")
        
        # Get total users count
        db.cursor.execute("SELECT COUNT(*) FROM users")
        total_users = db.cursor.fetchone()[0]
        print(f"Total users in database: {total_users}")
        
        return results


def query_examples(db_path: str = "app/data/db/transactions.db"):
    """
    Run some example queries to demonstrate the database functionality.
    
    Args:
        db_path: Path to the SQLite database
    """
    with Database(db_path) as db:
        # Show database summary
        print("\n=== DATABASE SUMMARY ===")
        # Users table count
        db.cursor.execute("SELECT COUNT(*) as count FROM users")
        user_count = db.cursor.fetchone()['count']
        print(f"Users table: {user_count} rows")
        
        # Transactions table count
        db.cursor.execute("SELECT COUNT(*) as count FROM transactions")
        transaction_count = db.cursor.fetchone()['count']
        print(f"Transactions table: {transaction_count} rows")
        
        # Remove queries to non-existent tables
        print("===========================\n")
        
        # Get transactions by category
        print("\nTransactions by Category:")
        db.cursor.execute("""
        SELECT category, COUNT(*) as count, SUM(amount) as total_amount 
        FROM transactions 
        GROUP BY category
        """)
        for row in db.cursor.fetchall():
            print(f"  {row['category']}: {row['count']} transactions, total: €{row['total_amount']:.2f}")
        
        # Get average confidence score
        print("\nAverage Confidence Score:")
        db.cursor.execute("SELECT AVG(confidence_score) as avg_confidence FROM transactions")
        avg_confidence = db.cursor.fetchone()['avg_confidence']
        if avg_confidence:
            print(f"  {avg_confidence:.2%}")
        else:
            print("  No transactions with confidence scores")
        
        # Get transaction details with more focus on confidence
        print("\nRecent Transaction Information (with confidence details):")
        db.cursor.execute("""
        SELECT 
            transaction_id,
            vendor,
            transaction_date,
            amount,
            category,
            subcategory,
            description,
            confidence_score
        FROM transactions
        ORDER BY transaction_date DESC
        LIMIT 5
        """)
        transactions = db.cursor.fetchall()
        for tx in transactions:
            print(f"  Transaction {tx['transaction_id']}:")
            print(f"    Vendor: {tx['vendor'] or 'Unknown'}")
            print(f"    Date: {tx['transaction_date']}")
            print(f"    Amount: €{tx['amount']:.2f}")
            print(f"    Category: {tx['category']}{' / ' + tx['subcategory'] if tx['subcategory'] else ''}")
            print(f"    Confidence: {tx['confidence_score']:.2%}")
            if tx['description']:
                print(f"    Details: {tx['description']}")
            
        # Get user information with annualized income instead of name
        print("\nUser Information (with annualized income):")
        db.cursor.execute("""
        SELECT 
            u.user_id,
            u.employer_name,
            u.occupation_category, 
            u.age_range, 
            u.family_status, 
            u.region,
            u.total_income,
            u.annualized_income,
            u.payslip_count,
            u.income_band,
            COUNT(t.transaction_id) as transaction_count
        FROM users u
        LEFT JOIN transactions t ON u.user_id = t.user_id
        GROUP BY u.user_id
        """)
        users = db.cursor.fetchall()
        for user in users:
            print(f"  User {user['user_id']}:")
            print(f"    Employer: {user['employer_name'] or 'Unknown'}")
            print(f"    Profile: {user['occupation_category']}, {user['age_range']}, {user['family_status']}")
            print(f"    Region: {user['region']}")
            
            # Display income information
            if user['total_income']:
                print(f"    Accumulated Income (from {user['payslip_count']} payslips): €{user['total_income']:.2f}")
                if user['annualized_income']:
                    print(f"    Annualized Income (projected): €{user['annualized_income']:.2f}")
            else:
                print("    Income: Not available")
                
            print(f"    Income Band: {user['income_band'] or 'Not categorized'}")
            print(f"    Transactions: {user['transaction_count']}")


def main():
    parser = argparse.ArgumentParser(description="Load processed documents into SQLite database")
    parser.add_argument(
        "--input", "-i",
        default="app/data/processed",
        help="Directory containing processed JSON files (default: app/data/processed)"
    )
    parser.add_argument(
        "--user", "-u",
        help="User ID to associate with transactions (optional, will create new user if not provided)"
    )
    parser.add_argument(
        "--examples", "-e",
        action="store_true",
        help="Run example queries after loading data"
    )
    
    args = parser.parse_args()
    
    # Load transactions
    results = load_transactions(args.input, args.user)
    
    # Run example queries if requested
    if args.examples and (results['receipts'] > 0 or results['income_statements'] > 0):
        db_dir = os.path.join(os.path.dirname(args.input), "db")
        db_path = os.path.join(db_dir, "transactions.db")
        query_examples(db_path)


if __name__ == "__main__":
    main() 