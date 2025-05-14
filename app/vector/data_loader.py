"""
Data loader for combining regular and enriched data.
"""
import os
import sqlite3
from typing import Dict, Any, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database paths
REGULAR_DB_PATH = os.path.join("app", "data", "db", "transactions.db")
ENRICHED_DB_PATH = os.path.join("app", "data", "db", "tax_insights.db")

def get_db_connection(db_path: str):
    """Get a database connection."""
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database {db_path}: {str(e)}")
        return None

def fetch_users(limit: int = 10000, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch users from both regular and enriched tables and combine them.
    
    Args:
        limit: Maximum number of users to fetch
        user_id: Optional user ID to fetch a specific user
        
    Returns:
        List of combined user dictionaries
    """
    users = []
    
    # Fetch from regular users table
    conn = get_db_connection(REGULAR_DB_PATH)
    if conn:
        try:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            else:
                cursor.execute(f"SELECT * FROM users LIMIT {limit}")
            
            regular_users = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Fetched {len(regular_users)} users from regular database")
            users.extend(regular_users)
        except Exception as e:
            logger.error(f"Error fetching from regular users table: {str(e)}")
        finally:
            conn.close()
    
    # Fetch from enriched users table
    conn = get_db_connection(ENRICHED_DB_PATH)
    if conn:
        try:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("SELECT * FROM enriched_users WHERE user_id = ?", (user_id,))
            else:
                cursor.execute(f"SELECT * FROM enriched_users LIMIT {limit}")
            
            enriched_users = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Fetched {len(enriched_users)} users from enriched database")
            
            # Combine data - merge enriched data into regular users
            user_map = {user["user_id"]: user for user in users}
            
            for enriched_user in enriched_users:
                user_id = enriched_user["user_id"]
                # Add explicit debug logging for important fields
                if "cluster_recommendation" in enriched_user:
                    logger.info(f"User {user_id} has cluster_recommendation: {enriched_user.get('cluster_recommendation')}")
                if "uplift_message" in enriched_user:
                    logger.info(f"User {user_id} has uplift_message: {enriched_user.get('uplift_message')}")
                
                if user_id in user_map:
                    # Update existing user with enriched data
                    for key, value in enriched_user.items():
                        if key != "user_id" and value is not None:
                            user_map[user_id][key] = value
                else:
                    # Add new enriched user
                    users.append(enriched_user)
            
        except Exception as e:
            logger.error(f"Error fetching from enriched users table: {str(e)}")
        finally:
            conn.close()
    
    logger.info(f"Combined {len(users)} users from all sources")
    return users

def fetch_historical_users_only(limit: int = 10000, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch users ONLY from the enriched (historical) table.
    
    Args:
        limit: Maximum number of users to fetch
        user_id: Optional user ID to fetch a specific user
        
    Returns:
        List of historical user dictionaries
    """
    users = []
    conn = get_db_connection(ENRICHED_DB_PATH)
    if conn:
        try:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("SELECT * FROM enriched_users WHERE user_id = ?", (user_id,))
            else:
                cursor.execute(f"SELECT * FROM enriched_users LIMIT {limit}")
            
            users = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Fetched {len(users)} users exclusively from enriched (historical) database")
        except Exception as e:
            logger.error(f"Error fetching from enriched_users (historical only) table: {str(e)}")
        finally:
            conn.close()
    return users

def fetch_transactions(limit: int = 10000, user_id: Optional[str] = None, transaction_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch transactions from both regular and enriched tables and combine them.
    
    Args:
        limit: Maximum number of transactions to fetch
        user_id: Optional user ID to filter transactions
        transaction_id: Optional transaction ID to fetch a specific transaction
        
    Returns:
        List of combined transaction dictionaries
    """
    transactions = []
    
    # Fetch from regular transactions table
    conn = get_db_connection(REGULAR_DB_PATH)
    if conn:
        try:
            cursor = conn.cursor()
            if transaction_id:
                cursor.execute("SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,))
            elif user_id:
                cursor.execute("SELECT * FROM transactions WHERE user_id = ? LIMIT ?", (user_id, limit))
            else:
                cursor.execute(f"SELECT * FROM transactions LIMIT {limit}")
            
            regular_txs = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Fetched {len(regular_txs)} transactions from regular database")
            transactions.extend(regular_txs)
        except Exception as e:
            logger.error(f"Error fetching from regular transactions table: {str(e)}")
        finally:
            conn.close()
    
    # Fetch from enriched transactions table
    conn = get_db_connection(ENRICHED_DB_PATH)
    if conn:
        try:
            cursor = conn.cursor()
            if transaction_id:
                cursor.execute("SELECT * FROM enriched_transactions WHERE transaction_id = ?", (transaction_id,))
            elif user_id:
                cursor.execute("SELECT * FROM enriched_transactions WHERE user_id = ? LIMIT ?", (user_id, limit))
            else:
                cursor.execute(f"SELECT * FROM enriched_transactions LIMIT {limit}")
            
            enriched_txs = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Fetched {len(enriched_txs)} transactions from enriched database")
            
            # Combine data - merge enriched data into regular transactions
            tx_map = {tx["transaction_id"]: tx for tx in transactions}
            
            for enriched_tx in enriched_txs:
                tx_id = enriched_tx["transaction_id"]
                if tx_id in tx_map:
                    # Update existing transaction with enriched data
                    for key, value in enriched_tx.items():
                        if key != "transaction_id" and value is not None:
                            tx_map[tx_id][key] = value
                else:
                    # Add new enriched transaction
                    transactions.append(enriched_tx)
            
        except Exception as e:
            logger.error(f"Error fetching from enriched transactions table: {str(e)}")
        finally:
            conn.close()
    
    logger.info(f"Combined {len(transactions)} transactions from all sources")
    return transactions

def fetch_historical_transactions_only(limit: int = 10000, user_id: Optional[str] = None, transaction_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch transactions ONLY from the enriched (historical) table.
    
    Args:
        limit: Maximum number of transactions to fetch
        user_id: Optional user ID to filter transactions
        transaction_id: Optional transaction ID to fetch a specific transaction
        
    Returns:
        List of historical transaction dictionaries
    """
    transactions = []
    conn = get_db_connection(ENRICHED_DB_PATH)
    if conn:
        try:
            cursor = conn.cursor()
            if transaction_id:
                cursor.execute("SELECT * FROM enriched_transactions WHERE transaction_id = ?", (transaction_id,))
            elif user_id:
                cursor.execute("SELECT * FROM enriched_transactions WHERE user_id = ? LIMIT ?", (user_id, limit))
            else:
                cursor.execute(f"SELECT * FROM enriched_transactions LIMIT {limit}")
            
            transactions = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Fetched {len(transactions)} transactions exclusively from enriched (historical) database")
        except Exception as e:
            logger.error(f"Error fetching from enriched_transactions (historical only) table: {str(e)}")
        finally:
            conn.close()
    return transactions

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get a user by ID from both sources."""
    logger.info(f"Retrieving user with ID: {user_id}")
    users = fetch_users(user_id=user_id)
    if users:
        user = users[0]
        # Debug log important fields
        logger.info(f"Retrieved user {user_id} with keys: {list(user.keys())}")
        if 'cluster_recommendation' in user:
            logger.info(f"User {user_id} has cluster_recommendation: {user.get('cluster_recommendation')}")
        if 'uplift_message' in user:
            logger.info(f"User {user_id} has uplift_message: {user.get('uplift_message')}")
        return user
    else:
        logger.warning(f"No user found with ID: {user_id}")
        return None

def get_transaction_by_id(transaction_id: str) -> Optional[Dict[str, Any]]:
    """Get a transaction by ID from both sources."""
    transactions = fetch_transactions(transaction_id=transaction_id)
    return transactions[0] if transactions else None

def get_user_transactions(user_id: str, limit: int = 10000) -> List[Dict[str, Any]]:
    """Get all transactions for a user."""
    return fetch_transactions(user_id=user_id, limit=limit)

def get_sample_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Get sample users and transactions for testing."""
    users = fetch_users(limit=5)
    transactions = fetch_transactions(limit=10)
    return users, transactions 