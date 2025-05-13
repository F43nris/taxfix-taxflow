"""
Core semantic search functionality for the taxflow application.
"""
import os
import sqlite3
import re
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import sys

# Set up logging with more explicit configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import vector search components
from app.vector.client import get_weaviate_client
from app.vector.search_api import TaxInsightSearchAPI
from app.vector.embed import get_embedding, prepare_user_text, prepare_transaction_text

class SemanticSearch:
    """
    Main class for handling semantic search functionality across the application.
    """
    def __init__(self, client=None):
        """Initialize with optional Weaviate client."""
        self.api = TaxInsightSearchAPI()
        if client:
            self.api.client = client  # Use provided client if available
        else:
            self.api.connect()  # Create a new client connection
    
    def close(self):
        """Close the client connection if we created it."""
        if self.api and self.api.client:
            self.api.client.close()
    
    def search(self, query):
        """
        Unified semantic search function that searches across all relevant tables
        based on the query content.
        
        Args:
            query: The search query from the user
            
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            logger.debug(f"Starting semantic search for query: '{query}'")
            # Initialize results structure
            results = {
                "query": query,
                "results": {
                    "historical_transactions": [],
                    "input_users": [],
                    "input_transactions": []
                },
                "query_type": "unknown",
                "is_personal_query": False,
                "requires_historical_data": True,  # By default, we might need historical data
                "all_medical_transactions": False,
                "quarterly_breakdown": {}
            }
            
            # Determine if this is a personal query
            query_lower = query.lower()
            
            # Expanded personal terms to include list/show commands (these are implicitly personal)
            personal_terms = [
                "my", "i ", "i've", "i have", "me", "mine", "i spent", "i received", "i earn", "i paid",
                "list", "show", "find", "display", "get", "what are", "how much", "did i"
            ]
            results["is_personal_query"] = any(term in query_lower for term in personal_terms)
            
            # For purely personal questions about spending history, we don't need historical data
            pure_personal_spending = (
                results["is_personal_query"] and 
                any(term in query_lower for term in ["spend", "spent", "spending", "pay", "paid", "purchase", "bought", "list", "purchases"])
            )
            
            # For personal income queries, we don't need historical data
            pure_personal_income = (
                results["is_personal_query"] and
                any(term in query_lower for term in ["income", "salary", "earn", "received", "job", "pay"])
            )
            
            # For tax-related queries, we don't need historical data
            tax_related_query = any(term in query_lower for term in ["tax", "taxes", "withheld", "withholding", "deduct", "deduction"])
            
            # For medical/pharmacy listing queries, we don't need historical data
            medical_terms = [
                "pharmacy", "pharmacies", "apotheke", "medical", "medicine", "doctor",
                "hospital", "clinic", "healthcare", "health care", "physician", "treatment"
            ]
            medical_query = any(term in query_lower for term in medical_terms)
            
            if medical_query and ("list" in query_lower or "show" in query_lower or "find" in query_lower or 
                               "spending" in query_lower or "expenses" in query_lower):
                results["is_personal_query"] = True
                results["requires_historical_data"] = False
                results["query_type"] = "medical_spending"  # Set query type for medical expenses
                logger.debug("Query identified as medical spending query")
            
            # Determine if we need historical data at all
            need_comparison = any(term in query_lower for term in ["compare", "similar", "like", "other", "average", "normal", "typical", "should"])
            need_tax_advice = any(term in query_lower for term in ["claim", "refund"]) and not any(term in query_lower for term in ["deduct", "write off"])
            
            # If it's a pure personal spending/income query or tax query without comparison needs, skip historical
            if (pure_personal_spending or pure_personal_income or tax_related_query) and not (need_comparison or need_tax_advice):
                results["requires_historical_data"] = False
                
            # Calculate embeddings for similarity scoring of input data
            query_embedding = None
            if self.api.client:  # Only if we have a client for embedding
                try:
                    query_embedding = get_embedding(query)
                except Exception as e:
                    logger.warning(f"Could not generate embedding for query: {str(e)}")
            
            # Get all input data from transactions.db
            input_conn = sqlite3.connect("app/data/db/transactions.db")
            input_conn.row_factory = sqlite3.Row
            input_cursor = input_conn.cursor()
            
            # Get all users
            input_cursor.execute("SELECT * FROM users")
            input_users = [dict(row) for row in input_cursor.fetchall()]
            
            # Get all transactions
            input_cursor.execute("SELECT * FROM transactions")
            input_transactions = [dict(row) for row in input_cursor.fetchall()]
            logger.debug(f"Loaded {len(input_transactions)} transactions from database")
            input_conn.close()
            
            # Add additional tax information to users
            for user in input_users:
                # Calculate tax withheld if we have both gross and net
                if user.get('annualized_income') and user.get('annualized_net_pay'):
                    gross = float(user.get('annualized_income', 0))
                    net = float(user.get('annualized_net_pay', 0))
                    tax_withheld = gross - net
                    user['tax_withheld'] = tax_withheld
                    
                    # Add tax rate if we can calculate it
                    if gross > 0:
                        tax_rate = (tax_withheld / gross) * 100
                        user['tax_rate'] = tax_rate
                elif user.get('annualized_income') and user.get('annualized_tax_deductions'):
                    # Use existing tax deductions if available
                    tax_withheld = float(user.get('annualized_tax_deductions', 0))
                    gross = float(user.get('annualized_income', 0))
                    user['tax_withheld'] = tax_withheld
                    
                    # Add tax rate if possible
                    if gross > 0:
                        tax_rate = (tax_withheld / gross) * 100
                        user['tax_rate'] = tax_rate
            
            # Special handling for medical/pharmacy queries
            if results["query_type"] == "medical_spending":
                logger.debug("Processing medical spending query")
                filtered_transactions = []
                
                # Define broader medical-related terms for searching
                medical_search_terms = [
                    'pharmacy', 'pharmacie', 'apotheke', 'drugstore', 'drug',
                    'hospital', 'clinic', 'doctor', 'arzt', 'physician', 'medical',
                    'health', 'healthcare', 'medicine', 'treatment', 'therapy',
                    'krankenhaus'
                ]
                
                # Medical categories (broader match)
                medical_categories = [
                    'health', 'medical', 'healthcare', 'doctor', 'hospital'
                ]
                
                logger.debug(f"Medical search terms: {medical_search_terms}")
                logger.debug(f"Medical categories: {medical_categories}")
                
                for tx in input_transactions:
                    # Get all text fields to search
                    vendor = tx.get('vendor', '').lower() if tx.get('vendor') else ''
                    category = tx.get('category', '').lower() if tx.get('category') else ''
                    description = tx.get('description', '').lower() if tx.get('description') else ''
                    subcategory = tx.get('subcategory', '').lower() if tx.get('subcategory') else ''
                    
                    tx_id = tx.get('transaction_id', 'unknown')
                    logger.debug(f"Checking transaction {tx_id}: vendor='{vendor}', category='{category}', description='{description}', subcategory='{subcategory}'")
                    
                    # Check for medical terms in any field
                    is_medical = False
                    match_reason = []
                    
                    # Check vendor name for medical terms
                    for term in medical_search_terms:
                        if term in vendor:
                            is_medical = True
                            match_reason.append(f"Found '{term}' in vendor")
                    
                    # Check if category or subcategory contains medical terms
                    for term in medical_categories:
                        if term in category:
                            is_medical = True
                            match_reason.append(f"Found '{term}' in category")
                        if term in subcategory:
                            is_medical = True
                            match_reason.append(f"Found '{term}' in subcategory")
                    
                    # Check description for medical terms
                    for term in medical_search_terms:
                        if term in description:
                            is_medical = True
                            match_reason.append(f"Found '{term}' in description")
                    
                    # Special case: Check for hospital in vendor (case insensitive)
                    if 'hospital' in vendor:
                        is_medical = True
                        match_reason.append("Found 'hospital' in vendor")
                    
                    if is_medical:
                        logger.debug(f"Transaction {tx_id} identified as medical: {', '.join(match_reason)}")
                        filtered_transactions.append(tx)
                    else:
                        logger.debug(f"Transaction {tx_id} NOT identified as medical")
                
                logger.debug(f"Found {len(filtered_transactions)} medical transactions before time filtering")
                
                # Store all medical transactions for comprehensive summary
                all_medical_transactions = filtered_transactions.copy()
                
                # Extract time period if mentioned (Q1, Q2, etc)
                time_period = None
                for quarter in ["q1", "q2", "q3", "q4"]:
                    if quarter in query_lower:
                        time_period = quarter.upper()
                        break
                
                # Check if query is asking for a specific year without quarter
                year_only = False
                year_pattern = r'in\s+(\d{4})'
                year_match = re.search(year_pattern, query_lower)
                if year_match and not time_period:
                    year_only = True
                
                # Check if query is asking for total/all medical spending without time specification
                is_total_query = any(term in query_lower for term in ["total", "all", "overall"]) and not time_period and not year_only
                
                # If asking for total spending with no time period specified, include all transactions
                if is_total_query:
                    logger.debug("Query is asking for total medical spending without time specification")
                    # Add metadata to indicate we're showing all transactions
                    results["all_medical_transactions"] = True
                    
                    # Sort by date
                    filtered_transactions.sort(key=lambda x: x.get('transaction_date', ''), reverse=False)
                    
                    # Log final results
                    logger.debug(f"Final medical transactions count (all): {len(filtered_transactions)}")
                    for tx in filtered_transactions:
                        logger.debug(f"Final result: {tx.get('transaction_id', 'unknown')} - {tx.get('vendor', 'unknown')} - {tx.get('transaction_date', 'unknown')} - {tx.get('amount', 'unknown')}")
                    
                    results["results"]["input_transactions"] = filtered_transactions
                    
                    # Add quarterly breakdown for comprehensive summary
                    results["quarterly_breakdown"] = self._get_quarterly_breakdown(filtered_transactions)
                    return results
                        
                # Map quarters to months
                if time_period:
                    logger.debug(f"Filtering by time period: {time_period}")
                    quarter_months = {
                        "Q1": ["01", "02", "03"],
                        "Q2": ["04", "05", "06"],
                        "Q3": ["07", "08", "09"],
                        "Q4": ["10", "11", "12"]
                    }
                    
                    # Extract year if mentioned
                    year = "2025"  # Default 
                    year_match = re.search(r'20\d\d', query_lower)
                    if year_match:
                        year = year_match.group(0)
                    
                    logger.debug(f"Filtering by year: {year} and quarter months: {quarter_months[time_period]}")
                    
                    # Filter by quarter/year
                    months = quarter_months.get(time_period, [])
                    quarter_filtered = []
                    for tx in filtered_transactions:
                        # Check transaction date
                        tx_date = tx.get('transaction_date', '')
                        tx_id = tx.get('transaction_id', 'unknown')
                        logger.debug(f"Checking if transaction {tx_id} with date '{tx_date}' matches {year} and months {months}")
                        
                        if year in tx_date:
                            # Check if month is in the specified quarter
                            for month in months:
                                if f"{year}-{month}" in tx_date:
                                    logger.debug(f"Transaction {tx_id} matches time filter")
                                    quarter_filtered.append(tx)
                                    break
                    
                    logger.debug(f"After time filtering: {len(quarter_filtered)} transactions remain")
                    filtered_transactions = quarter_filtered if quarter_filtered else filtered_transactions
                else:
                    # Extract year if mentioned
                    year_pattern = r'in\s+(\d{4})'
                    year_match = re.search(year_pattern, query_lower)
                    if year_match:
                        year = year_match.group(1)
                        logger.debug(f"Filtering by year only: {year}")
                        year_filtered = []
                        for tx in filtered_transactions:
                            # Check transaction date
                            tx_date = tx.get('transaction_date', '')
                            tx_id = tx.get('transaction_id', 'unknown')
                            logger.debug(f"Checking if transaction {tx_id} with date '{tx_date}' matches year {year}")
                            
                            if year in tx_date:
                                logger.debug(f"Transaction {tx_id} matches year filter")
                                year_filtered.append(tx)
                        
                        logger.debug(f"After year filtering: {len(year_filtered)} transactions remain")
                        filtered_transactions = year_filtered if year_filtered else filtered_transactions
                
                # Sort by date
                filtered_transactions.sort(key=lambda x: x.get('transaction_date', ''), reverse=False)
                
                # Log final results
                logger.debug(f"Final medical transactions count: {len(filtered_transactions)}")
                for tx in filtered_transactions:
                    logger.debug(f"Final result: {tx.get('transaction_id', 'unknown')} - {tx.get('vendor', 'unknown')} - {tx.get('transaction_date', 'unknown')} - {tx.get('amount', 'unknown')}")
                
                # Add all medical transactions for reference even when filtering by quarter
                results["all_medical_transactions"] = all_medical_transactions
                
                # Add quarterly breakdown for comprehensive summary
                results["quarterly_breakdown"] = self._get_quarterly_breakdown(all_medical_transactions)
                
                results["results"]["input_transactions"] = filtered_transactions
                return results
                
            # Check for tax or income related queries
            if any(term in query_lower for term in ['tax', 'taxes', 'withheld', 'withholding', 'income', 'salary', 'tax rate', 'gross', 'net', 'company', 'employer']):
                if any(term in query_lower for term in ['tax', 'taxes', 'withheld', 'withholding', 'tax rate']):
                    results["query_type"] = "tax_info"
                else:
                    results["query_type"] = "income"
                
                # Special handling for company/employer queries
                if any(term in query_lower for term in ['company', 'employer']):
                    # Extract company name from query if possible
                    company_name = None
                    for company_term in ['company', 'employer']:
                        if company_term in query_lower:
                            # Try to find "Company X" pattern
                            match = re.search(f"{company_term} ([a-z0-9]+)", query_lower)
                            if match:
                                company_name = match.group(1)
                    
                    # If company name found, filter input users
                    if company_name:
                        filtered_users = []
                        for user in input_users:
                            employer = user.get('employer', '')
                            if not employer:
                                # Try alternate fields that might contain employer info
                                employer = user.get('employer_name', '') or user.get('occupation_description', '')
                            
                            if employer and company_name.lower() in employer.lower():
                                filtered_users.append(user)
                        
                        # If we found matching users, only keep those
                        if filtered_users:
                            input_users = filtered_users
                
                # For personal income queries, we should mark this as personal
                if "i" in query_lower or "my" in query_lower:
                    results["is_personal_query"] = True
                    
                # Extract year if mentioned to filter income data
                year_pattern = r'in\s+(\d{4})'
                year_match = re.search(year_pattern, query_lower)
                year = None
                if year_match:
                    year = year_match.group(1)
                
                # Tax-related queries don't need historical data
                if tax_related_query:
                    results["requires_historical_data"] = False
                
                # Always assign input users for income/tax queries
                results["results"]["input_users"] = input_users
                
                # Calculate similarity scores if possible
                if query_embedding:
                    for user in results["results"]["input_users"]:
                        try:
                            user_text = prepare_user_text(user)
                            user_embedding = get_embedding(user_text)
                            similarity = cosine_similarity(
                                np.array(query_embedding).reshape(1, -1),
                                np.array(user_embedding).reshape(1, -1)
                            )[0][0]
                            user["similarity_score"] = float(similarity)
                        except Exception as e:
                            user["similarity_score"] = 0.0
                
                # Only get historical transactions if explicitly needed for comparison
                if results["requires_historical_data"] and need_comparison:
                    historical_transactions = self.api.find_similar_transactions(
                        query_text=query,
                        limit=10  # Reduced limit for personal queries
                    )
                    results["results"]["historical_transactions"] = historical_transactions
                
                return results
            
            # Check for transaction/receipt/spending related queries
            if any(term in query_lower for term in ['spent', 'spending', 'purchase', 'receipt', 'transaction', 
                                                'store', 'medical', 'expenses', 'deductible', 'pharmacy', 
                                                'apotheke', 'medicine', 'q1', 'q2', 'q3', 'q4']):
                results["query_type"] = "transaction"
                
                # For personal queries, prioritize input transactions
                if results["is_personal_query"]:
                    # Calculate similarity scores if possible
                    if query_embedding:
                        for tx in input_transactions:
                            try:
                                tx_text = prepare_transaction_text(tx)
                                tx_embedding = get_embedding(tx_text)
                                similarity = cosine_similarity(
                                    np.array(query_embedding).reshape(1, -1),
                                    np.array(tx_embedding).reshape(1, -1)
                                )[0][0]
                                tx["similarity_score"] = float(similarity)
                            except Exception as e:
                                tx["similarity_score"] = 0.0
                    
                    # Sort by similarity if scores available
                    if any("similarity_score" in tx for tx in input_transactions):
                        input_transactions.sort(
                            key=lambda x: x.get("similarity_score", 0), 
                            reverse=True
                        )
                    
                    results["results"]["input_transactions"] = input_transactions
                    
                    # For personal queries, get historical transactions only if needed for comparison
                    if results["requires_historical_data"] and (need_comparison or need_tax_advice):
                        historical_transactions = self.api.find_similar_transactions(
                            query_text=query,
                            limit=10  # Reduced limit for personal queries
                        )
                        results["results"]["historical_transactions"] = historical_transactions
                else:
                    # For non-personal queries, focus more on historical data
                    historical_transactions = self.api.find_similar_transactions(
                        query_text=query,
                        limit=25
                    )
                    results["results"]["historical_transactions"] = historical_transactions
                    results["results"]["input_transactions"] = input_transactions
            
            # If query type is still unknown, do both searches
            if results["query_type"] == "unknown":
                results["query_type"] = "hybrid"
                
                # For personal queries, prioritize input data
                if results["is_personal_query"]:
                    results["results"]["input_users"] = input_users
                    results["results"]["input_transactions"] = input_transactions
                    
                    # Calculate similarity scores if possible
                    if query_embedding:
                        for user in results["results"]["input_users"]:
                            try:
                                user_text = prepare_user_text(user)
                                user_embedding = get_embedding(user_text)
                                similarity = cosine_similarity(
                                    np.array(query_embedding).reshape(1, -1),
                                    np.array(user_embedding).reshape(1, -1)
                                )[0][0]
                                user["similarity_score"] = float(similarity)
                            except Exception as e:
                                user["similarity_score"] = 0.0
                        
                        for tx in results["results"]["input_transactions"]:
                            try:
                                tx_text = prepare_transaction_text(tx)
                                tx_embedding = get_embedding(tx_text)
                                similarity = cosine_similarity(
                                    np.array(query_embedding).reshape(1, -1),
                                    np.array(tx_embedding).reshape(1, -1)
                                )[0][0]
                                tx["similarity_score"] = float(similarity)
                            except Exception as e:
                                tx["similarity_score"] = 0.0
                    
                    # Sort by similarity if scores available
                    if any("similarity_score" in user for user in results["results"]["input_users"]):
                        results["results"]["input_users"].sort(
                            key=lambda x: x.get("similarity_score", 0), 
                            reverse=True
                        )
                    
                    if any("similarity_score" in tx for tx in results["results"]["input_transactions"]):
                        results["results"]["input_transactions"].sort(
                            key=lambda x: x.get("similarity_score", 0), 
                            reverse=True
                        )
                    
                    # Get some historical data for comparison but only if needed
                    if results["requires_historical_data"] and (need_comparison or need_tax_advice):
                        historical_transactions = self.api.find_similar_transactions(
                            query_text=query,
                            limit=10
                        )
                        results["results"]["historical_transactions"] = historical_transactions
                else:
                    # For non-personal queries, focus on historical transaction data only
                    historical_transactions = self.api.find_similar_transactions(
                        query_text=query,
                        limit=15
                    )
                    results["results"]["historical_transactions"] = historical_transactions
                    results["results"]["input_users"] = input_users
                    results["results"]["input_transactions"] = input_transactions
            
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            raise

    def _get_quarterly_breakdown(self, transactions):
        """
        Helper method to break down transactions by quarter.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Dictionary with quarterly breakdown of transactions and amounts
        """
        quarterly = {
            "Q1": {"transactions": [], "total": 0},
            "Q2": {"transactions": [], "total": 0},
            "Q3": {"transactions": [], "total": 0},
            "Q4": {"transactions": [], "total": 0}
        }
        
        for tx in transactions:
            tx_date = tx.get('transaction_date', '')
            amount = float(tx.get('amount', 0))
            
            # Extract month from date string (format: YYYY-MM-DD...)
            if len(tx_date) >= 7:  # Make sure we have at least YYYY-MM
                month = tx_date[5:7]  # Get the month part
                
                # Assign to quarter
                if month in ["01", "02", "03"]:
                    quarterly["Q1"]["transactions"].append(tx)
                    quarterly["Q1"]["total"] += amount
                elif month in ["04", "05", "06"]:
                    quarterly["Q2"]["transactions"].append(tx)
                    quarterly["Q2"]["total"] += amount
                elif month in ["07", "08", "09"]:
                    quarterly["Q3"]["transactions"].append(tx)
                    quarterly["Q3"]["total"] += amount
                elif month in ["10", "11", "12"]:
                    quarterly["Q4"]["transactions"].append(tx)
                    quarterly["Q4"]["total"] += amount
        
        return quarterly

def search(query, client=None):
    """
    Helper function to run semantic search without explicitly creating a SemanticSearch object.
    
    Args:
        query: The search query from the user
        client: Optional Weaviate client instance (will create one if not provided)
        
    Returns:
        Dictionary containing search results and metadata
    """
    searcher = SemanticSearch(client)
    try:
        return searcher.search(query)
    finally:
        searcher.close() 