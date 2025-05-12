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
        
        # Get average confidence score by field
        print("\nAverage Confidence Scores by Field:")
        db.cursor.execute("""
        SELECT 
            AVG(confidence_score) as avg_overall_confidence,
            AVG(amount_confidence) as avg_amount_confidence,
            AVG(date_confidence) as avg_date_confidence,
            AVG(vendor_confidence) as avg_vendor_confidence,
            AVG(category_confidence) as avg_category_confidence 
        FROM transactions
        """)
        conf = db.cursor.fetchone()
        if conf:
            print(f"  Overall: {conf['avg_overall_confidence']:.2%} confidence")
            print(f"  Amount field: {conf['avg_amount_confidence']:.2%} confidence" if conf['avg_amount_confidence'] else "  Amount field: No data")
            print(f"  Date field: {conf['avg_date_confidence']:.2%} confidence" if conf['avg_date_confidence'] else "  Date field: No data")
            print(f"  Vendor field: {conf['avg_vendor_confidence']:.2%} confidence" if conf['avg_vendor_confidence'] else "  Vendor field: No data")
            print(f"  Category field: {conf['avg_category_confidence']:.2%} confidence" if conf['avg_category_confidence'] else "  Category field: No data")
        else:
            print("  No transactions with confidence scores")
        
        # Get transaction details with more focus on confidence
        print("\nRecent Transaction Information (with detailed confidence):")
        db.cursor.execute("""
        SELECT 
            transaction_id,
            vendor,
            transaction_date,
            amount,
            category,
            subcategory,
            description,
            confidence_score,
            amount_confidence,
            date_confidence,
            vendor_confidence, 
            category_confidence
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
            print(f"    Overall Confidence: {tx['confidence_score']:.2%}")
            print(f"    Field Confidences:")
            print(f"      Amount: {tx['amount_confidence']:.2%}" if tx['amount_confidence'] else "      Amount: No data")
            print(f"      Date: {tx['date_confidence']:.2%}" if tx['date_confidence'] else "      Date: No data")
            print(f"      Vendor: {tx['vendor_confidence']:.2%}" if tx['vendor_confidence'] else "      Vendor: No data")
            print(f"      Category: {tx['category_confidence']:.2%}" if tx['category_confidence'] else "      Category: No data")
            
        # Get user information
        print("\nUser Income Information (with averages and projections):")
        db.cursor.execute("""
        SELECT 
            u.user_id,
            u.employer_name,
            u.employer_name_confidence,
            u.occupation_category,
            u.occupation_category_confidence,
            
            u.avg_gross_pay,
            u.gross_pay_confidence,
            u.avg_net_pay,
            u.net_pay_confidence,
            
            u.avg_tax_deductions,
            
            u.annualized_income,
            u.annualized_net_pay,
            u.annualized_tax_deductions,
            
            u.payslip_count,
            u.gross_pay_count,
            u.net_pay_count,
            
            u.income_band,
            COUNT(t.transaction_id) as transaction_count
        FROM users u
        LEFT JOIN transactions t ON u.user_id = t.user_id
        GROUP BY u.user_id
        """)
        users = db.cursor.fetchall()
        for user in users:
            print(f"  User {user['user_id']}:")
            if user['employer_name']:
                print(f"    Employer: {user['employer_name']} " + 
                      f"(confidence: {user['employer_name_confidence']:.2%})" if user['employer_name_confidence'] else "")
            
            # Display occupation category if available
            if user['occupation_category']:
                print(f"    Occupation: {user['occupation_category']} " +
                      f"(confidence: {user['occupation_category_confidence']:.2%})" if user['occupation_category_confidence'] else "")
            
            # Display payslip counts
            print(f"    Analyzed {user['payslip_count']} payslip(s):")
            print(f"      • {user['gross_pay_count']} with gross pay, {user['net_pay_count']} with net pay")
            
            # Display average income information
            if user['avg_gross_pay']:
                print(f"    Average Monthly Gross Pay: €{user['avg_gross_pay']:.2f} " + 
                     f"(confidence: {user['gross_pay_confidence']:.2%})" if user['gross_pay_confidence'] else "")
            if user['avg_net_pay']:
                print(f"    Average Monthly Net Pay: €{user['avg_net_pay']:.2f} " + 
                     f"(confidence: {user['net_pay_confidence']:.2%})" if user['net_pay_confidence'] else "")
            
            # Display average tax deductions
            if user['avg_tax_deductions']:
                print(f"    Average Monthly Tax Deductions: €{user['avg_tax_deductions']:.2f}")
            
            # Display projected/annualized income
            if user['annualized_income']:
                print(f"    Projected Annual Gross Income: €{user['annualized_income']:.2f}")
            if user['annualized_net_pay']:
                print(f"    Projected Annual Net Income: €{user['annualized_net_pay']:.2f}")
            if user['annualized_tax_deductions']:
                print(f"    Projected Annual Tax Deductions: €{user['annualized_tax_deductions']:.2f}")
                
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