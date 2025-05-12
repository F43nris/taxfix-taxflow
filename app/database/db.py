import os
import sqlite3
from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime
from pathlib import Path

from app.database.models import Receipt, IncomeStatement, Transaction, User


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class Database:
    """SQLite database manager for document processing results"""
    
    def __init__(self, db_path: str = "app/data/db/transactions.db"):
        """Initialize the database connection"""
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Connect to the SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")
        # Configure connection to return rows as dictionaries
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        return self.conn
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.conn:
            if exc_type:
                self.conn.rollback()
            else:
                self.conn.commit()
            self.close()
    
    def create_tables(self):
        """Create necessary database tables if they don't exist"""
        # Create users table - From payslips
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            employee_name TEXT,
            employee_name_confidence REAL,
            employer_name TEXT,
            employer_name_confidence REAL,
            occupation_category TEXT,
            age_range TEXT,
            family_status TEXT,
            region TEXT,
            filing_id TEXT,
            tax_year INTEGER,
            filing_date TEXT,
            total_income REAL,
            total_income_confidence REAL,
            total_deductions REAL,
            refund_amount REAL,
            income_band TEXT,
            annualized_income REAL,
            payslip_count INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create transactions table - From receipts
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            transaction_date TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT,
            description TEXT,
            vendor TEXT,
            receipt_id TEXT,
            confidence_score REAL,
            transaction_month INTEGER,
            quarter INTEGER,
            year INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # Create indices for better query performance
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_income_band ON users(income_band)')
        
        self.conn.commit()
    
    def insert_user(self, user: User) -> str:
        """
        Insert a user into the database.
        
        Args:
            user: User model object
            
        Returns:
            ID of the inserted user
        """
        # Check if user already exists
        self.cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user.user_id,))
        existing = self.cursor.fetchone()
        if existing:
            # Update existing user
            # Convert datetime objects to ISO format strings
            filing_date = user.filing_date.isoformat() if user.filing_date else None
            
            self.cursor.execute('''
            UPDATE users SET
                employee_name = COALESCE(?, employee_name),
                employee_name_confidence = COALESCE(?, employee_name_confidence),
                employer_name = COALESCE(?, employer_name),
                employer_name_confidence = COALESCE(?, employer_name_confidence),
                occupation_category = COALESCE(?, occupation_category),
                age_range = COALESCE(?, age_range),
                family_status = COALESCE(?, family_status),
                region = COALESCE(?, region),
                filing_id = COALESCE(?, filing_id),
                tax_year = COALESCE(?, tax_year),
                filing_date = COALESCE(?, filing_date),
                total_income = COALESCE(?, total_income),
                total_income_confidence = COALESCE(?, total_income_confidence),
                total_deductions = COALESCE(?, total_deductions),
                refund_amount = COALESCE(?, refund_amount),
                income_band = COALESCE(?, income_band),
                annualized_income = COALESCE(?, annualized_income),
                payslip_count = COALESCE(?, payslip_count)
            WHERE user_id = ?
            ''', (
                user.employee_name, user.employee_name_confidence,
                user.employer_name, user.employer_name_confidence,
                user.occupation_category, user.age_range, user.family_status, user.region,
                user.filing_id, user.tax_year, filing_date,
                user.total_income, user.total_income_confidence,
                user.total_deductions, user.refund_amount, user.income_band,
                user.annualized_income, user.payslip_count,
                user.user_id
            ))
        else:
            # Insert new user
            # Convert datetime objects to ISO format strings
            filing_date = user.filing_date.isoformat() if user.filing_date else None
            
            self.cursor.execute('''
            INSERT INTO users (
                user_id, employee_name, employee_name_confidence, employer_name, employer_name_confidence,
                occupation_category, age_range, family_status, region,
                filing_id, tax_year, filing_date, total_income, total_income_confidence,
                total_deductions, refund_amount, income_band, annualized_income, payslip_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user.user_id, user.employee_name, user.employee_name_confidence,
                user.employer_name, user.employer_name_confidence,
                user.occupation_category, user.age_range, user.family_status, user.region,
                user.filing_id, user.tax_year, filing_date, user.total_income, user.total_income_confidence,
                user.total_deductions, user.refund_amount, user.income_band, user.annualized_income, user.payslip_count
            ))
        
        self.conn.commit()
        return user.user_id
    
    def get_or_create_user(self, user_id: str = None) -> str:
        """
        Get an existing user by ID or create a new one.
        
        Args:
            user_id: Existing user ID (optional)
            
        Returns:
            User ID
        """
        # If user_id provided, check if exists
        if user_id:
            self.cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            result = self.cursor.fetchone()
            if result:
                return result["user_id"]
                
        # If we have a default user, use that
        self.cursor.execute("SELECT user_id FROM users LIMIT 1")
        existing_user = self.cursor.fetchone()
        if existing_user:
            return existing_user["user_id"]
        
        # Create new user
        user = User()
        
        # If specified user_id, use it
        if user_id:
            user.user_id = user_id
            
        # Insert the user
        return self.insert_user(user)
    
    def insert_transaction(self, transaction: Transaction) -> str:
        """
        Insert a transaction into the database.
        
        Args:
            transaction: Transaction model object
            
        Returns:
            ID of the inserted transaction
        """
        # Convert datetime objects to ISO format strings
        transaction_date = transaction.transaction_date.isoformat() if transaction.transaction_date else None
        
        self.cursor.execute('''
        INSERT INTO transactions (
            transaction_id, user_id, transaction_date, amount, category, subcategory,
            description, vendor, receipt_id, confidence_score,
            transaction_month, quarter, year
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            transaction.transaction_id, transaction.user_id, transaction_date, transaction.amount,
            transaction.category, transaction.subcategory, transaction.description, transaction.vendor,
            transaction.receipt_id, transaction.confidence_score,
            transaction.transaction_month, transaction.quarter, transaction.year
        ))
        
        self.conn.commit()
        return transaction.transaction_id
    
    def process_receipt_json(self, json_file_path: str, user_id: str = None) -> str:
        """
        Process a receipt JSON file and insert it into the database as a transaction.
        
        Args:
            json_file_path: Path to the receipt JSON file
            user_id: User ID to associate with the transaction (optional)
            
        Returns:
            ID of the inserted transaction
        """
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        
        # Create Receipt model
        receipt = Receipt.from_json(json_data, json_file_path)
        
        # Get or create user
        user_id = self.get_or_create_user(user_id)
        
        # Create transaction from receipt
        transaction = receipt.to_transaction(user_id)
        
        # Insert transaction
        self.insert_transaction(transaction)
        
        return transaction.transaction_id
    
    def process_income_statement_json(self, json_file_path: str, user_id: str = None) -> str:
        """
        Process an income statement JSON file and insert/update user information.
        
        Args:
            json_file_path: Path to the income statement JSON file
            user_id: User ID to associate with (optional)
            
        Returns:
            ID of the inserted/updated user
        """
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        
        # Create IncomeStatement model
        income_stmt = IncomeStatement.from_json(json_data, json_file_path)
        
        # Check if user exists
        if user_id:
            self.cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            existing_user_data = self.cursor.fetchone()
            if existing_user_data:
                # Convert row to dict
                user_dict = dict(existing_user_data)
                
                # Convert ISO format string back to datetime
                if user_dict['filing_date']:
                    user_dict['filing_date'] = datetime.fromisoformat(user_dict['filing_date'])
                
                # Create User object from existing data
                user = User(**user_dict)
                
                # Update with new income statement data
                user.update_from_income_statement(income_stmt)
            else:
                # Create new user with income statement data
                user = income_stmt.to_user(user_id)
        else:
            # Create new user with income statement data
            user = income_stmt.to_user(user_id)
        
        # Insert or update user
        user_id = self.insert_user(user)
        
        return user_id
    
    def process_all_documents(self, processed_dir: str = "app/data/processed", user_id: str = None):
        """
        Process all document JSON files in the processed directory.
        
        Args:
            processed_dir: Directory containing processed JSON files
            user_id: User ID to associate with the transactions (optional)
        
        Returns:
            Dictionary containing counts of processed documents by type
        """
        processed_counts = {
            "receipts": 0,
            "income_statements": 0,
            "errors": 0
        }
        
        # Get or create a user first
        user_id = self.get_or_create_user(user_id)
        
        # Process income statements first to build user profile
        income_stmt_dir = os.path.join(processed_dir, "processed_income_statements")
        if os.path.exists(income_stmt_dir):
            for file_name in os.listdir(income_stmt_dir):
                if file_name.endswith(".json"):
                    try:
                        file_path = os.path.join(income_stmt_dir, file_name)
                        self.process_income_statement_json(file_path, user_id)
                        processed_counts["income_statements"] += 1
                    except Exception as e:
                        print(f"Error processing income statement {file_name}: {str(e)}")
                        processed_counts["errors"] += 1
        
        # Process receipts second to create transactions
        receipts_dir = os.path.join(processed_dir, "processed_receipts")
        if os.path.exists(receipts_dir):
            for file_name in os.listdir(receipts_dir):
                if file_name.endswith(".json"):
                    try:
                        file_path = os.path.join(receipts_dir, file_name)
                        self.process_receipt_json(file_path, user_id)
                        processed_counts["receipts"] += 1
                    except Exception as e:
                        print(f"Error processing receipt {file_name}: {str(e)}")
                        processed_counts["errors"] += 1
        
        return processed_counts 