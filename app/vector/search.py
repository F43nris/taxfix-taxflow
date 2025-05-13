"""
Search functionality for Weaviate vector database.
"""
from typing import List, Dict, Any, Optional, Union
import uuid
from weaviate.classes.query import Filter, MetadataQuery

from app.vector.settings import USER_CLASS, TRANSACTION_CLASS
from app.vector.embed import get_embedding, prepare_user_text, prepare_transaction_text

class SearchResult:
    """Class to represent a search result with its metadata and similarity score."""
    
    def __init__(self, object_id: str, class_name: str, data: Dict[str, Any], distance: float):
        self.object_id = object_id
        self.class_name = class_name
        self.data = data
        self.distance = distance
        self.similarity = 1.0 - distance  # Convert distance to similarity score
    
    def __repr__(self) -> str:
        return f"SearchResult(id={self.object_id}, class={self.class_name}, similarity={self.similarity:.4f})"

def search_similar_users(
    client,
    query_text: Optional[str] = None,
    query_vector: Optional[List[float]] = None,
    user_data: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10
) -> List[SearchResult]:
    """
    Search for similar users based on a query.
    
    Args:
        client: Weaviate client instance
        query_text: Text query to search for (will be converted to embedding)
        query_vector: Vector to search for (alternative to query_text)
        user_data: User data to search for (alternative to query_text and query_vector)
        filters: Optional filters to apply to the search
        limit: Maximum number of results to return
        
    Returns:
        List of SearchResult objects
    """
    # Validate input
    if not query_text and not query_vector and not user_data:
        raise ValueError("One of query_text, query_vector, or user_data must be provided")
    
    # Generate vector from user data if provided
    if user_data and not query_vector and not query_text:
        query_text = prepare_user_text(user_data)
    
    # Generate vector from text if provided
    if query_text and not query_vector:
        query_vector = get_embedding(query_text)
    
    # Get the collection
    collection = client.collections.get(USER_CLASS)
    
    # Build where filter if filters are provided
    where_filter = build_where_filter(filters) if filters else None
    
    # Execute query
    try:
        from weaviate.classes.query import MetadataQuery
        
        # Use the v4 API for near_vector search - direct query
        results = collection.query.near_vector(
            near_vector=query_vector,
            filters=where_filter,  # pass filters directly
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )
        
        # Process results
        search_results = [
            SearchResult(
                object_id=item.uuid,
                class_name=USER_CLASS,
                data=item.properties,
                distance=item.metadata.distance
            )
            for item in results.objects
        ]
        
        return search_results
    except Exception as e:
        raise Exception(f"Error searching for similar users: {str(e)}")

def search_similar_transactions(
    client,
    query_text: Optional[str] = None,
    query_vector: Optional[List[float]] = None,
    transaction_data: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10
) -> List[SearchResult]:
    """
    Search for similar transactions based on a query.
    
    Args:
        client: Weaviate client instance
        query_text: Text query to search for (will be converted to embedding)
        query_vector: Vector to search for (alternative to query_text)
        transaction_data: Transaction data to search for (alternative to query_text and query_vector)
        filters: Optional filters to apply to the search
        limit: Maximum number of results to return
        
    Returns:
        List of SearchResult objects
    """
    # Validate input
    if not query_text and not query_vector and not transaction_data:
        raise ValueError("One of query_text, query_vector, or transaction_data must be provided")
    
    # Generate vector from transaction data if provided
    if transaction_data and not query_vector and not query_text:
        query_text = prepare_transaction_text(transaction_data)
    
    # Generate vector from text if provided
    if query_text and not query_vector:
        query_vector = get_embedding(query_text)
    
    # Get the collection
    collection = client.collections.get(TRANSACTION_CLASS)
    
    # Build where filter if filters are provided
    where_filter = build_where_filter(filters) if filters else None
    
    # Execute query
    try:
        from weaviate.classes.query import MetadataQuery
        
        # Use the v4 API for near_vector search - direct query
        results = collection.query.near_vector(
            near_vector=query_vector,
            filters=where_filter,  # pass filters directly
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )
        
        # Process results
        search_results = [
            SearchResult(
                object_id=item.uuid,
                class_name=TRANSACTION_CLASS,
                data=item.properties,
                distance=item.metadata.distance
            )
            for item in results.objects
        ]
        
        return search_results
    except Exception as e:
        raise Exception(f"Error searching for similar transactions: {str(e)}")

def search_transactions_for_user(
    client,
    user_data: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10
) -> List[SearchResult]:
    """
    Search for transactions that are semantically similar to a user profile.
    This is the main function for connecting users/payslips with transactions/receipts.
    
    Args:
        client: Weaviate client instance
        user_data: User data to find matching transactions for
        filters: Optional filters to apply to the search
        limit: Maximum number of results to return
        
    Returns:
        List of SearchResult objects
    """
    # Generate text representation of user data
    user_text = prepare_user_text(user_data)
    
    # Generate embedding
    user_vector = get_embedding(user_text)
    
    # Search for similar transactions
    return search_similar_transactions(
        client=client,
        query_vector=user_vector,
        filters=filters,
        limit=limit
    )

def search_users_for_transaction(
    client,
    transaction_data: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10
) -> List[SearchResult]:
    """
    Search for users that are semantically similar to a transaction.
    This is the main function for connecting transactions/receipts with users/payslips.
    
    Args:
        client: Weaviate client instance
        transaction_data: Transaction data to find matching users for
        filters: Optional filters to apply to the search
        limit: Maximum number of results to return
        
    Returns:
        List of SearchResult objects
    """
    # Generate text representation of transaction data
    transaction_text = prepare_transaction_text(transaction_data)
    
    # Generate embedding
    transaction_vector = get_embedding(transaction_text)
    
    # Search for similar users
    return search_similar_users(
        client=client,
        query_vector=transaction_vector,
        filters=filters,
        limit=limit
    )

def build_where_filter(filters: Dict[str, Any]) -> Filter:
    """
    Build a Weaviate where filter from a dictionary of filters.
    
    Args:
        filters: Dictionary of filters
        
    Returns:
        Weaviate Filter object
    """
    if not filters:
        return None
    
    # Convert simple filters to Weaviate v4 format
    combined_filter = None
    
    for field, value in filters.items():
        if isinstance(value, list):
            # For lists, use "ContainsAny" operator
            field_filter = Filter.by_property(name=field).contains_any(value)
        elif value is None:
            # For None values, use "IsNull" operator
            field_filter = Filter.by_property(name=field).is_null()
        elif isinstance(value, bool):
            # Convert boolean values to strings
            field_filter = Filter.by_property(name=field).equal(str(value).lower())
        else:
            # For simple values, use "Equal" operator
            field_filter = Filter.by_property(name=field).equal(value)
        
        # Combine filters with AND operator
        if combined_filter is None:
            combined_filter = field_filter
        else:
            combined_filter = Filter.and_filter(combined_filter, field_filter)
    
    return combined_filter