#!/usr/bin/env python3
"""
Script to process the enriched data and upload it to the database.

This script:
1. Splits the full joined data into user and transactions tables
2. Enriches both tables with essential tax insights
3. Uploads the enriched data to SQLite database
"""

import os
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from pydantic import BaseModel, Field, validator
import uuid
import sys

# Configure logging with colors for better visibility
class ColoredFormatter(logging.Formatter):
    """Colored formatter for logging"""
    COLORS = {
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[91m', # Red
        'INFO': '\033[92m',     # Green
        'DEBUG': '\033[94m',    # Blue
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message

# Setup logging with better formatting
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [handler]

# Define paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"
DB_PATH = DB_DIR / "tax_insights.db"

# Create directories if they don't exist
DB_DIR.mkdir(exist_ok=True)


class EnrichedUser(BaseModel):
    """Model for enriched user data"""
    user_id: str
    occupation_category: Optional[str] = None
    age_range: Optional[str] = None
    family_status: Optional[str] = None
    region: Optional[str] = None
    filing_id: Optional[str] = None
    tax_year: Optional[int] = None
    filing_date: Optional[str] = None
    total_income: Optional[float] = None
    total_deductions: Optional[float] = None
    refund_amount: Optional[float] = None
    
    # Essential fields from hierarchical recommendations
    cluster_recommendation: Optional[str] = None
    cluster_confidence_level: Optional[str] = None
    
    class Config:
        from_attributes = True


class EnrichedTransaction(BaseModel):
    """Model for enriched transaction data"""
    transaction_id: str
    user_id: str
    transaction_date: str
    amount: float
    category: str
    subcategory: Optional[str] = None
    description: Optional[str] = None
    vendor: Optional[str] = None
    transaction_month: Optional[int] = None
    quarter: Optional[int] = None
    year: Optional[int] = None
    occupation_category: Optional[str] = None
    family_status: Optional[str] = None
    
    # Essential fields from tax deduction results
    is_deductible: Optional[bool] = None
    deduction_confidence_score: Optional[float] = None
    deduction_recommendation: Optional[str] = None
    deduction_category: Optional[str] = None
    
    # Essential fields from uplift insights
    uplift_message: Optional[str] = None
    uplift_confidence_level: Optional[str] = None
    uplift_pct: Optional[float] = None
    
    class Config:
        from_attributes = True


class DatabaseManager:
    """SQLite database manager for tax insights data"""
    
    def __init__(self, db_path: str = str(DB_PATH)):
        """Initialize the database connection"""
        self.db_path = db_path
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


class EnrichedDataDB(DatabaseManager):
    """Database manager for enriched data"""
    
    def create_tables(self):
        """Create necessary database tables if they don't exist"""
        # Create users table with essential enrichment fields
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS enriched_users (
            user_id TEXT PRIMARY KEY,
            occupation_category TEXT,
            age_range TEXT,
            family_status TEXT,
            region TEXT,
            filing_id TEXT,
            tax_year INTEGER,
            filing_date TEXT,
            total_income REAL,
            total_deductions REAL,
            refund_amount REAL,
            
            -- Essential fields from hierarchical recommendations
            cluster_recommendation TEXT,
            cluster_confidence_level TEXT
        )
        ''')
        
        # Create transactions table with essential enrichment fields
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS enriched_transactions (
            transaction_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            transaction_date TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT,
            description TEXT,
            vendor TEXT,
            transaction_month INTEGER,
            quarter INTEGER,
            year INTEGER,
            occupation_category TEXT,
            family_status TEXT,
            
            -- Essential fields from tax deduction results
            is_deductible BOOLEAN,
            deduction_confidence_score REAL,
            deduction_recommendation TEXT,
            deduction_category TEXT,
            
            -- Essential fields from uplift insights
            uplift_message TEXT,
            uplift_confidence_level TEXT,
            uplift_pct REAL,
            
            FOREIGN KEY (user_id) REFERENCES enriched_users (user_id)
        )
        ''')
    
    def insert_user(self, user_data):
        """Insert a single user record"""
        try:
            self.cursor.execute('''
            INSERT OR REPLACE INTO enriched_users (
                user_id, occupation_category, age_range, family_status, region,
                filing_id, tax_year, filing_date, total_income, total_deductions,
                refund_amount, cluster_recommendation, cluster_confidence_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_data.get('user_id'),
                user_data.get('occupation_category'),
                user_data.get('age_range'),
                user_data.get('family_status'),
                user_data.get('region'),
                user_data.get('filing_id'),
                user_data.get('tax_year'),
                user_data.get('filing_date'),
                user_data.get('total_income'),
                user_data.get('total_deductions'),
                user_data.get('refund_amount'),
                user_data.get('cluster_recommendation'),
                user_data.get('cluster_confidence_level')
            ))
            return True
        except Exception as e:
            logger.error(f"Error inserting user {user_data.get('user_id')}: {e}")
            return False
    
    def insert_transaction(self, transaction_data):
        """Insert a single transaction record"""
        try:
            self.cursor.execute('''
            INSERT OR REPLACE INTO enriched_transactions (
                transaction_id, user_id, transaction_date, amount, category,
                subcategory, description, vendor, transaction_month, 
                quarter, year, occupation_category, family_status, is_deductible, 
                deduction_confidence_score, deduction_recommendation, deduction_category,
                uplift_message, uplift_confidence_level, uplift_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction_data.get('transaction_id'),
                transaction_data.get('user_id'),
                transaction_data.get('transaction_date'),
                transaction_data.get('amount'),
                transaction_data.get('category'),
                transaction_data.get('subcategory'),
                transaction_data.get('description'),
                transaction_data.get('vendor'),
                transaction_data.get('transaction_month'),
                transaction_data.get('quarter'),
                transaction_data.get('year'),
                transaction_data.get('occupation_category'),
                transaction_data.get('family_status'),
                transaction_data.get('is_deductible'),
                transaction_data.get('deduction_confidence_score'),
                transaction_data.get('deduction_recommendation'),
                transaction_data.get('deduction_category'),
                transaction_data.get('uplift_message'),
                transaction_data.get('uplift_confidence_level'),
                transaction_data.get('uplift_pct')
            ))
            return True
        except Exception as e:
            logger.error(f"Error inserting transaction {transaction_data.get('transaction_id')}: {e}")
            return False
    
    def batch_insert_users(self, users_df):
        """Insert multiple user records"""
        success_count = 0
        for _, row in users_df.iterrows():
            if self.insert_user(row):
                success_count += 1
        
        logger.info(f"Inserted {success_count} out of {len(users_df)} user records")
        return success_count
    
    def batch_insert_transactions(self, transactions_df):
        """Insert multiple transaction records"""
        success_count = 0
        for _, row in transactions_df.iterrows():
            if self.insert_transaction(row):
                success_count += 1
        
        logger.info(f"Inserted {success_count} out of {len(transactions_df)} transaction records")
        return success_count


def preprocess_data():
    """Preprocess and merge all the data sources"""
    # Load the full joined data
    full_joined_path = ROOT_DIR.parent / "notebooks" / "processed_data" / "full_joined_data.csv"
    logger.info(f"Loading full joined data from {full_joined_path}")
    
    if not full_joined_path.exists():
        logger.error(f"Data file not found at {full_joined_path}")
        alternative_path = Path(os.path.abspath(os.path.join(ROOT_DIR, "..", "notebooks", "data", "full_joined_data.csv")))
        if alternative_path.exists():
            logger.info(f"Found data file at alternative path: {alternative_path}")
            full_joined_path = alternative_path
        else:
            logger.error("Full joined data file not found. Exiting.")
            return None, None
    
    full_data = pd.read_csv(full_joined_path)
    
    # Extract user data directly from full data without loading separate files
    logger.info("Extracting user data from the full joined data...")
    
    # Define base user columns to extract
    base_user_columns = ['user_id', 'occupation_category', 'age_range', 'family_status', 'region', 
                    'filing_id', 'tax_year', 'filing_date', 'total_income', 'total_deductions', 
                    'refund_amount']
    
    # Filter the available columns
    available_user_cols = [col for col in base_user_columns if col in full_data.columns]
    
    # Extract user data with deduplication
    user_df = full_data[available_user_cols].drop_duplicates(subset=["user_id"])
    logger.info(f"Extracted {len(user_df)} unique users from full joined data")
    
    # Generate transaction IDs if needed
    if 'transaction_id' not in full_data.columns:
        logger.info("Generating transaction IDs...")
        full_data['transaction_id'] = [str(uuid.uuid4()) for _ in range(len(full_data))]
    
    # Define transaction columns (exclude user-specific columns)
    transaction_columns = ['transaction_id', 'user_id', 'transaction_date', 'amount', 'category', 
                         'subcategory', 'description', 'vendor', 'transaction_month', 
                         'quarter', 'year', 'occupation_category', 'family_status']
    
    # Keep only columns that exist in the data
    transaction_cols = [col for col in transaction_columns if col in full_data.columns]
    
    # Extract transaction data
    transaction_df = full_data[transaction_cols].copy()
    logger.info(f"Extracted {len(transaction_df)} transactions")
    
    # Filter out transactions with missing essential data
    missing_data_count = 0
    if 'category' in transaction_df.columns and 'transaction_date' in transaction_df.columns and 'amount' in transaction_df.columns:
        missing_data = (
            pd.isna(transaction_df['category']) | 
            pd.isna(transaction_df['transaction_date']) | 
            pd.isna(transaction_df['amount'])
        )
        
        missing_data_count = missing_data.sum()
        if missing_data_count > 0:
            valid_transactions = transaction_df[~missing_data].copy()
            logger.warning(f"⚠️ Filtered out {missing_data_count} transactions with missing essential data")
            transaction_df = valid_transactions
    
    # Load enrichment data
    # 1. Tax deduction results
    deduction_path = os.path.join(PROCESSED_DIR, "tax_deduction_results", 
                                "tax_deduction_results_20250513_011835.csv")
    logger.info(f"Loading tax deduction results from {deduction_path}")
    deduction_df = pd.read_csv(deduction_path)
    
    # 2. Hierarchical recommendations
    hier_path = os.path.join(PROCESSED_DIR, "tax_hierarchical_results", 
                            "hierarchical_tax_recommendations_with_confidence.csv")
    logger.info(f"Loading hierarchical recommendations from {hier_path}")
    recommendations_df = pd.read_csv(hier_path)
    
    # 3. Uplift insights
    insight_path = os.path.join(PROCESSED_DIR, "tax_uplift_results", 
                              "tax_optimization_insights.csv")
    logger.info(f"Loading uplift insights from {insight_path}")
    insights_df = pd.read_csv(insight_path)
    
    # Enrich transactions with tax deduction results (simplified fields)
    logger.info("Enriching transactions with tax deduction results...")
    
    # Rename confidence_score to deduction_confidence_score and reasons to deduction_recommendation in deduction_df
    if 'confidence_score' in deduction_df.columns:
        deduction_df = deduction_df.rename(columns={'confidence_score': 'deduction_confidence_score'})
    if 'reasons' in deduction_df.columns:
        deduction_df = deduction_df.rename(columns={'reasons': 'deduction_recommendation'})
    
    # Rename 'category' to 'deduction_category' in deduction_df as it's enrichment data, not a join key
    if 'category' in deduction_df.columns:
        deduction_df = deduction_df.rename(columns={'category': 'deduction_category'})
    
    # Make sure all the required columns are in both dataframes
    # For transactions, better joins are with subcategory, description, occupation_category, family_status
    join_columns = []
    potential_join_cols = ['subcategory', 'description', 'occupation_category', 'family_status']
    
    # Debug: Print column availability in both dataframes
    logger.info("=== JOIN DEBUGGING ===")
    logger.info(f"Transaction DF columns: {transaction_df.columns.tolist()}")
    logger.info(f"Deduction DF columns: {deduction_df.columns.tolist()}")
    
    for col in potential_join_cols:
        if col in transaction_df.columns and col in deduction_df.columns:
            # Fill NAs and convert to string to ensure clean join
            transaction_df[col] = transaction_df[col].fillna('').astype(str)
            deduction_df[col] = deduction_df[col].fillna('').astype(str)
            join_columns.append(col)
            # Debug: Show value counts for this column in both dataframes
            logger.info(f"Column {col} found in both dataframes")
            logger.info(f"Transaction DF unique {col} count: {transaction_df[col].nunique()}")
            logger.info(f"Deduction DF unique {col} count: {deduction_df[col].nunique()}")
        else:
            if col not in transaction_df.columns:
                logger.warning(f"Column {col} missing in transaction dataframe")
            if col not in deduction_df.columns:
                logger.warning(f"Column {col} missing in deduction dataframe")
    
    logger.info(f"Joining transactions with deductions on columns: {join_columns}")
    
    # Print first few rows of both DataFrames in a cleaner format for debugging join keys
    join_samples = {
        "Transaction": transaction_df.iloc[0][join_columns].to_dict() if not transaction_df.empty else {},
        "Deduction": deduction_df.iloc[0][join_columns].to_dict() if not deduction_df.empty else {}
    }
    logger.info(f"Join key samples: {join_samples}")
    
    # Debug: Show a few sample rows from each dataframe to check join keys
    logger.info("Sample transaction rows for join:")
    for i in range(min(3, len(transaction_df))):
        logger.info(f"Transaction {i}: {transaction_df.iloc[i][join_columns].to_dict()}")
    
    logger.info("Sample deduction rows for join:")
    for i in range(min(3, len(deduction_df))):
        logger.info(f"Deduction {i}: {deduction_df.iloc[i][join_columns].to_dict()}")
    
    # Merge deduction data into transactions
    if all(col in deduction_df.columns for col in join_columns + ['is_deductible', 'deduction_confidence_score', 'deduction_recommendation', 'deduction_category']):
        # Count rows before join
        pre_join_count = len(transaction_df)
        
        enriched_transactions = pd.merge(
            transaction_df,
            deduction_df[join_columns + ['is_deductible', 'deduction_confidence_score', 'deduction_recommendation', 'deduction_category']],
            on=join_columns,
            how='left'
        )
        
        # Count rows after join
        post_join_count = len(enriched_transactions)
        if post_join_count != pre_join_count:
            logger.warning(f"⚠️ Row count changed after join: {pre_join_count} -> {post_join_count}")
        
        match_count = (~pd.isna(enriched_transactions['is_deductible'])).sum()
        no_match_count = pd.isna(enriched_transactions['is_deductible']).sum()
        
        if match_count == 0:
            logger.warning(f"⚠️ No transactions were enriched with deduction data. Check join columns.")
        else:
            logger.info(f"Enriched {match_count} transactions with deduction data")
            logger.info(f"Could not enrich {no_match_count} transactions")
            
        # Debug: Check for null values in deduction fields
        null_deduction_recommendation = pd.isna(enriched_transactions['deduction_recommendation']).sum()
        null_deduction_confidence = pd.isna(enriched_transactions['deduction_confidence_score']).sum()
        null_is_deductible = pd.isna(enriched_transactions['is_deductible']).sum()
        null_deduction_category = pd.isna(enriched_transactions['deduction_category']).sum()
        
        logger.info(f"Null deduction_recommendation count: {null_deduction_recommendation} ({null_deduction_recommendation/len(enriched_transactions)*100:.2f}%)")
        logger.info(f"Null deduction_confidence_score count: {null_deduction_confidence} ({null_deduction_confidence/len(enriched_transactions)*100:.2f}%)")
        logger.info(f"Null is_deductible count: {null_is_deductible} ({null_is_deductible/len(enriched_transactions)*100:.2f}%)")
        logger.info(f"Null deduction_category count: {null_deduction_category} ({null_deduction_category/len(enriched_transactions)*100:.2f}%)")
        
        # Debug: Check distribution of values for each join column
        logger.info("=== JOIN COLUMN VALUE DISTRIBUTION ===")
        for col in join_columns:
            logger.info(f"Column: {col}")
            # Show top 5 most common values in transaction_df
            logger.info(f"Top 5 values in transaction_df[{col}]:")
            for val, count in transaction_df[col].value_counts().head(5).items():
                logger.info(f"  - '{val}': {count}")
            
            # Show top 5 most common values in deduction_df
            logger.info(f"Top 5 values in deduction_df[{col}]:")
            for val, count in deduction_df[col].value_counts().head(5).items():
                logger.info(f"  - '{val}': {count}")
    else:
        logger.warning("⚠️ Missing required columns for deduction enrichment")
        enriched_transactions = transaction_df
    
    # Enrich transactions with uplift insights
    logger.info("Enriching transactions with uplift insights...")

    # Check if insights_df has the necessary columns
    if 'category' in insights_df.columns and 'message' in insights_df.columns and 'confidence_level' in insights_df.columns:
        # Rename columns in insights_df to avoid conflicts
        insights_df = insights_df.rename(columns={
            'confidence_level': 'uplift_confidence_level',
            'message': 'uplift_message'
        })
        
        # Create a mapping from category to uplift insights
        category_to_uplift = {}
        for _, row in insights_df.iterrows():
            category = row.get('category')
            if category:
                category_to_uplift[category] = {
                    'uplift_message': row.get('uplift_message', ''),
                    'uplift_confidence_level': row.get('uplift_confidence_level', ''),
                    'uplift_pct': row.get('uplift_pct', 0.0)
                }
        
        # Add uplift insights to transactions based on category
        if 'category' in enriched_transactions.columns:
            # Initialize uplift columns with None
            enriched_transactions['uplift_message'] = None
            enriched_transactions['uplift_confidence_level'] = None
            enriched_transactions['uplift_pct'] = None
            
            # Count before enrichment
            pre_uplift_count = len(enriched_transactions)
            
            # Apply uplift insights based on category
            for category, uplift_data in category_to_uplift.items():
                mask = enriched_transactions['category'] == category
                enriched_transactions.loc[mask, 'uplift_message'] = uplift_data['uplift_message']
                enriched_transactions.loc[mask, 'uplift_confidence_level'] = uplift_data['uplift_confidence_level']
                enriched_transactions.loc[mask, 'uplift_pct'] = uplift_data['uplift_pct']
            
            # Count enriched transactions
            uplift_match_count = (~pd.isna(enriched_transactions['uplift_message'])).sum()
            uplift_no_match_count = pd.isna(enriched_transactions['uplift_message']).sum()
            
            if uplift_match_count == 0:
                logger.warning("⚠️ No transactions were enriched with uplift insights. Check category matching.")
            else:
                logger.info(f"Enriched {uplift_match_count} transactions with uplift insights")
                logger.info(f"Could not enrich {uplift_no_match_count} transactions with uplift insights")
                
            # Debug: Check for null values in uplift fields
            null_uplift_message = pd.isna(enriched_transactions['uplift_message']).sum()
            null_uplift_confidence = pd.isna(enriched_transactions['uplift_confidence_level']).sum()
            null_uplift_pct = pd.isna(enriched_transactions['uplift_pct']).sum()
            
            logger.info(f"Null uplift_message count: {null_uplift_message} ({null_uplift_message/len(enriched_transactions)*100:.2f}%)")
            logger.info(f"Null uplift_confidence_level count: {null_uplift_confidence} ({null_uplift_confidence/len(enriched_transactions)*100:.2f}%)")
            logger.info(f"Null uplift_pct count: {null_uplift_pct} ({null_uplift_pct/len(enriched_transactions)*100:.2f}%)")
        else:
            logger.warning("⚠️ Missing 'category' column in transactions for uplift enrichment")
    else:
        logger.warning("⚠️ Missing required columns for uplift insight enrichment")
    
    # Enrich users with hierarchical recommendations (essential fields only)
    logger.info("Enriching users with hierarchical recommendations...")
    
    # Group recommendations by user_id - get only the required fields
    if 'user_id' in recommendations_df.columns and 'confidence_level' in recommendations_df.columns and 'recommendation' in recommendations_df.columns:
        # Rename columns in recommendations_df
        recommendations_df = recommendations_df.rename(columns={
            'confidence_level': 'cluster_confidence_level',
            'recommendation': 'cluster_recommendation'
        })
        
        # Group by user_id and combine recommendations where user already has claims (user_pct > 0)
        user_recommendations = {}
        for user_id, group in recommendations_df.groupby('user_id'):
            if not group.empty:
                # Filter for recommendations where user already has claims (user_pct > 0)
                existing_claims = group[group['user_pct'] > 0.0]
                
                if not existing_claims.empty:
                    # Combine all recommendations for existing claims into one string
                    combined_recommendation = " | ".join(existing_claims['cluster_recommendation'].tolist())
                    # Use the highest confidence level among these recommendations
                    highest_confidence = existing_claims['cluster_confidence_level'].max()
                    
                    user_recommendations[user_id] = {
                        'cluster_recommendation': combined_recommendation,
                        'cluster_confidence_level': highest_confidence
                    }
                else:
                    # If no existing claims with recommendations, use the first recommendation
                    row = group.iloc[0]
                    user_recommendations[user_id] = {
                        'cluster_recommendation': row.get('cluster_recommendation', ''),
                        'cluster_confidence_level': row.get('cluster_confidence_level', '')
                    }
        
        # Convert to DataFrame
        user_recommendations_df = pd.DataFrame([
            {'user_id': user_id, **data} 
            for user_id, data in user_recommendations.items()
        ])
        
        # Merge with user data
        enriched_users = pd.merge(
            user_df,
            user_recommendations_df,
            on='user_id',
            how='left'
        )
        
        match_count = (~pd.isna(enriched_users['cluster_recommendation'])).sum()
        logger.info(f"Enriched {match_count} users with recommendation data")
    else:
        logger.warning("⚠️ Missing required columns for hierarchical recommendation enrichment")
        enriched_users = user_df
    
    return enriched_users, enriched_transactions


def main():
    """Main function to process and upload the enriched data"""
    logger.info("Starting data processing and enrichment...")
    
    # Preprocess and enrich the data
    enriched_users, enriched_transactions = preprocess_data()
    
    if enriched_users is None or enriched_transactions is None:
        logger.error("❌ Data preprocessing failed. Exiting.")
        return
    
    # Create database and tables
    logger.info(f"Creating database at {DB_PATH}")
    db = EnrichedDataDB()
    
    try:
        db.connect()
        db.create_tables()
        
        # Insert users first (for foreign key constraints)
        logger.info(f"Inserting {len(enriched_users)} enriched user records...")
        db.batch_insert_users(enriched_users)
        db.conn.commit()
        
        # Then insert transactions
        logger.info(f"Inserting {len(enriched_transactions)} enriched transaction records...")
        db.batch_insert_transactions(enriched_transactions)
        db.conn.commit()
        
        # Print summary
        logger.info("✅ Data processing and loading completed successfully")
        
        # Get counts
        db.cursor.execute("SELECT COUNT(*) FROM enriched_users")
        user_count = db.cursor.fetchone()[0]
        
        db.cursor.execute("SELECT COUNT(*) FROM enriched_transactions")
        transaction_count = db.cursor.fetchone()[0]
        
        logger.info(f"""
        Database Summary:
        - Enriched Users: {user_count}
        - Enriched Transactions: {transaction_count}
        """)
        
        # Show sample data from the tables
        logger.info("=== Sample data from enriched_users table: ===")
        db.cursor.execute("SELECT * FROM enriched_users LIMIT 2")
        for row in db.cursor.fetchall():
            logger.info(dict(row))
        
        logger.info("=== Sample data from enriched_transactions table: ===")
        db.cursor.execute("SELECT * FROM enriched_transactions LIMIT 2")
        for row in db.cursor.fetchall():
            logger.info(dict(row))
        
    except Exception as e:
        logger.error(f"❌ Error in database operations: {e}")
        if db.conn:
            db.conn.rollback()
    finally:
        if db:
            db.close()


if __name__ == "__main__":
    main() 