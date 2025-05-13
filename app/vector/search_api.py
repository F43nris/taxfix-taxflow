"""
Unified search API for tax insights.

This module provides a high-level API for searching across both regular and enriched data
using vector similarity search.
"""
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import json

from app.vector.client import get_weaviate_client
from app.vector.search import (
    search_similar_users,
    search_similar_transactions,
    search_transactions_for_user,
    search_users_for_transaction,
    SearchResult
)
from app.vector.data_loader import (
    get_user_by_id,
    get_transaction_by_id,
    get_user_transactions,
    fetch_users,
    fetch_transactions
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaxInsightSearchAPI:
    """Unified search API for tax insights."""
    
    def __init__(self):
        """Initialize the search API."""
        self.client = None
    
    def connect(self):
        """Connect to the vector database."""
        if not self.client:
            self.client = get_weaviate_client()
        return self.client is not None
    
    def close(self):
        """Close the vector database connection."""
        if self.client:
            self.client.close()
            self.client = None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def find_similar_users(self, 
                          query_text: Optional[str] = None,
                          user_data: Optional[Dict[str, Any]] = None,
                          user_id: Optional[str] = None,
                          filters: Optional[Dict[str, Any]] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find users similar to the query.
        
        Args:
            query_text: Text query to search for
            user_data: User data to search for (alternative to query_text)
            user_id: User ID to search for (alternative to query_text and user_data)
            filters: Optional filters to apply to the search
            limit: Maximum number of results to return
            
        Returns:
            List of similar users with combined data
        """
        if not self.connect():
            logger.error("Failed to connect to vector database")
            return []
        
        # If user_id is provided, get the user data
        if user_id and not user_data:
            user_data = get_user_by_id(user_id)
            if not user_data:
                logger.warning(f"User with ID {user_id} not found")
                return []
        
        # Search for similar users
        results = search_similar_users(
            client=self.client,
            query_text=query_text,
            user_data=user_data,
            filters=filters,
            limit=limit
        )
        
        # Enhance results with data from both sources
        enhanced_results = []
        for result in results:
            user_id = result.data.get("user_id")
            if user_id:
                # Get combined user data
                user_data = get_user_by_id(user_id)
                if user_data:
                    # Add similarity score to the user data
                    user_data["similarity_score"] = result.similarity
                    enhanced_results.append(user_data)
        
        return enhanced_results
    
    def find_similar_transactions(self,
                                query_text: Optional[str] = None,
                                transaction_data: Optional[Dict[str, Any]] = None,
                                transaction_id: Optional[str] = None,
                                filters: Optional[Dict[str, Any]] = None,
                                limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find transactions similar to the query.
        
        Args:
            query_text: Text query to search for
            transaction_data: Transaction data to search for (alternative to query_text)
            transaction_id: Transaction ID to search for (alternative to query_text and transaction_data)
            filters: Optional filters to apply to the search
            limit: Maximum number of results to return
            
        Returns:
            List of similar transactions with combined data
        """
        if not self.connect():
            logger.error("Failed to connect to vector database")
            return []
        
        # If transaction_id is provided, get the transaction data
        if transaction_id and not transaction_data:
            transaction_data = get_transaction_by_id(transaction_id)
            if not transaction_data:
                logger.warning(f"Transaction with ID {transaction_id} not found")
                return []
        
        # Search for similar transactions
        results = search_similar_transactions(
            client=self.client,
            query_text=query_text,
            transaction_data=transaction_data,
            filters=filters,
            limit=limit
        )
        
        # Enhance results with data from both sources
        enhanced_results = []
        for result in results:
            tx_id = result.data.get("transaction_id")
            if tx_id:
                # Get combined transaction data
                tx_data = get_transaction_by_id(tx_id)
                if tx_data:
                    # Add similarity score to the transaction data
                    tx_data["similarity_score"] = result.similarity
                    enhanced_results.append(tx_data)
        
        return enhanced_results
    
    def find_transactions_for_user(self,
                                 user_id: Optional[str] = None,
                                 user_data: Optional[Dict[str, Any]] = None,
                                 filters: Optional[Dict[str, Any]] = None,
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find transactions that are semantically similar to a user profile.
        
        Args:
            user_id: User ID to search for
            user_data: User data to search for (alternative to user_id)
            filters: Optional filters to apply to the search
            limit: Maximum number of results to return
            
        Returns:
            List of transactions with combined data
        """
        if not self.connect():
            logger.error("Failed to connect to vector database")
            return []
        
        # If user_id is provided, get the user data
        if user_id and not user_data:
            user_data = get_user_by_id(user_id)
            if not user_data:
                logger.warning(f"User with ID {user_id} not found")
                return []
        
        # Search for transactions for the user
        results = search_transactions_for_user(
            client=self.client,
            user_data=user_data,
            filters=filters,
            limit=limit
        )
        
        # Enhance results with data from both sources
        enhanced_results = []
        for result in results:
            tx_id = result.data.get("transaction_id")
            if tx_id:
                # Get combined transaction data
                tx_data = get_transaction_by_id(tx_id)
                if tx_data:
                    # Add similarity score to the transaction data
                    tx_data["similarity_score"] = result.similarity
                    enhanced_results.append(tx_data)
        
        return enhanced_results
    
    def find_users_for_transaction(self,
                                transaction_id: Optional[str] = None,
                                transaction_data: Optional[Dict[str, Any]] = None,
                                filters: Optional[Dict[str, Any]] = None,
                                limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find users that are semantically similar to a transaction.
        
        Args:
            transaction_id: Transaction ID to search for
            transaction_data: Transaction data to search for (alternative to transaction_id)
            filters: Optional filters to apply to the search
            limit: Maximum number of results to return
            
        Returns:
            List of users with combined data
        """
        if not self.connect():
            logger.error("Failed to connect to vector database")
            return []
        
        # If transaction_id is provided, get the transaction data
        if transaction_id and not transaction_data:
            transaction_data = get_transaction_by_id(transaction_id)
            if not transaction_data:
                logger.warning(f"Transaction with ID {transaction_id} not found")
                return []
        
        # Search for users for the transaction
        results = search_users_for_transaction(
            client=self.client,
            transaction_data=transaction_data,
            filters=filters,
            limit=limit
        )
        
        # Enhance results with data from both sources
        enhanced_results = []
        for result in results:
            user_id = result.data.get("user_id")
            if user_id:
                # Get combined user data
                user_data = get_user_by_id(user_id)
                if user_data:
                    # Add similarity score to the user data
                    user_data["similarity_score"] = result.similarity
                    enhanced_results.append(user_data)
        
        return enhanced_results
    
    def find_deductible_transactions(self, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find tax-deductible transactions for a user.
        
        Args:
            user_id: Optional user ID to filter transactions
            limit: Maximum number of results to return
            
        Returns:
            List of tax-deductible transactions
        """
        # Apply filter for deductible transactions
        filters = {"is_deductible": True}
        
        if user_id:
            # Get user data
            user_data = get_user_by_id(user_id)
            if user_data:
                # Find transactions for this user that are deductible
                return self.find_transactions_for_user(
                    user_data=user_data,
                    filters=filters,
                    limit=limit
                )
        
        # If no user_id or user not found, search for all deductible transactions
        return self.find_similar_transactions(
            query_text="tax deductible expenses",
            filters=filters,
            limit=limit
        )
    
    def get_tax_recommendations(self, user_id: str) -> Dict[str, Any]:
        """
        Get tax recommendations for a user based on their profile and transactions.
        
        Args:
            user_id: User ID to get recommendations for
            
        Returns:
            Dictionary with recommendations
        """
        # Get user data
        user_data = get_user_by_id(user_id)
        if not user_data:
            logger.warning(f"User with ID {user_id} not found")
            return {"error": "User not found"}
        
        # Get user's transactions
        user_transactions = get_user_transactions(user_id)
        
        # Find similar users
        similar_users = self.find_similar_users(user_data=user_data, limit=5)
        
        # Find potential deductions
        deductible_transactions = self.find_deductible_transactions(user_id=user_id)
        
        # Compile recommendations
        recommendations = {
            "user_profile": user_data,
            "transaction_count": len(user_transactions),
            "similar_users": similar_users,
            "potential_deductions": deductible_transactions,
            "cluster_recommendation": user_data.get("cluster_recommendation"),
            "uplift_message": user_data.get("uplift_message")
        }
        
        return recommendations

# Create a singleton instance
search_api = TaxInsightSearchAPI()

def get_search_api() -> TaxInsightSearchAPI:
    """Get the search API singleton instance."""
    return search_api

# Example usage
if __name__ == "__main__":
    with TaxInsightSearchAPI() as api:
        # Example: Find similar users
        similar_users = api.find_similar_users(query_text="software engineer with high income")
        print(f"Found {len(similar_users)} similar users")
        
        # Example: Find tax recommendations
        if similar_users:
            user_id = similar_users[0].get("user_id")
            recommendations = api.get_tax_recommendations(user_id)
            print(f"Tax recommendations for user {user_id}:")
            print(json.dumps(recommendations, indent=2)) 