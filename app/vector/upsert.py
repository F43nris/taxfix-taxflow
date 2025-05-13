"""
Add or update data in Weaviate.
"""
import uuid
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
from weaviate.classes.data import DataObject

from app.vector.settings import (
    USER_CLASS,
    TRANSACTION_CLASS,
    EMBEDDING_MODEL,
    WEAVIATE_BATCH_SIZE
)
from app.vector.embed import (
    get_embedding,
    prepare_user_text,
    prepare_transaction_text
)

def add_or_update_user(client, user_data: Dict[str, Any], user_id: Optional[str] = None) -> str:
    """
    Add or update a user in Weaviate.
    
    Args:
        client: Weaviate client instance
        user_data: User data dictionary
        user_id: Optional user ID (if not provided, will be extracted from user_data)
        
    Returns:
        Weaviate object ID
    """
    # Extract user ID from data if not provided
    if not user_id:
        user_id = user_data.get("user_id")
        if not user_id:
            raise ValueError("User ID not provided and not found in user_data")
    
    # Prepare text for embedding
    text = prepare_user_text(user_data)
    
    # Generate embedding
    vector = get_embedding(text)
    
    # Add embedding model information
    user_data["embedding_model"] = EMBEDDING_MODEL
    
    # Generate a UUID based on user_id for consistent object IDs
    object_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user-{user_id}"))
    
    # Get the collection
    collection = client.collections.get(USER_CLASS)
    
    # Create DataObject for v4 API
    data_object = DataObject(
        properties=user_data,
        uuid=object_id,
        vector=vector
    )
    
    # Check if object already exists
    try:
        existing = collection.query.fetch_object_by_id(object_id)
        if existing:
            # Update existing object
            collection.data.replace(data_object)
            print(f"Updated user {user_id} in Weaviate")
            return object_id
    except Exception:
        # Object doesn't exist, continue to create
        pass
    
    # Create new object
    try:
        collection.data.insert(data_object)
        print(f"Added user {user_id} to Weaviate")
        return object_id
    except Exception as e:
        raise Exception(f"Error adding user to Weaviate: {str(e)}")

def add_or_update_transaction(client, transaction_data: Dict[str, Any], transaction_id: Optional[str] = None) -> str:
    """
    Add or update a transaction in Weaviate.
    
    Args:
        client: Weaviate client instance
        transaction_data: Transaction data dictionary
        transaction_id: Optional transaction ID (if not provided, will be extracted from transaction_data)
        
    Returns:
        Weaviate object ID
    """
    # Create a clean copy of the data to avoid modifying the original
    clean_data = transaction_data.copy()
    
    # Extract transaction ID from data if not provided
    if not transaction_id:
        transaction_id = clean_data.get("transaction_id")
        if not transaction_id:
            raise ValueError("Transaction ID not provided and not found in transaction_data")
    
    # Handle date conversion for Weaviate
    if "transaction_date" in clean_data:
        if isinstance(clean_data["transaction_date"], datetime):
            # Format date as RFC3339 with timezone
            clean_data["transaction_date"] = clean_data["transaction_date"].replace(microsecond=0).isoformat() + "Z"
    
    # Convert confidence values to strings if they're defined as text in the schema
    # or remove them if they're not defined in the schema
    if "date_confidence" in clean_data:
        if isinstance(clean_data["date_confidence"], (float, int)):
            # Remove it as it's not in our schema
            del clean_data["date_confidence"]
    
    # Convert other confidence fields to strings if needed
    for field in ["amount_confidence", "vendor_confidence", "category_confidence"]:
        if field in clean_data and isinstance(clean_data[field], (float, int)):
            # Remove these as they're not in our schema
            del clean_data[field]
    
    # Prepare text for embedding
    text = prepare_transaction_text(clean_data)
    
    # Generate embedding
    vector = get_embedding(text)
    
    # Add embedding model information
    clean_data["embedding_model"] = EMBEDDING_MODEL
    
    # Generate a UUID based on transaction_id for consistent object IDs
    object_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"transaction-{transaction_id}"))
    
    # Get the collection
    collection = client.collections.get(TRANSACTION_CLASS)
    
    # Create DataObject for v4 API
    data_object = DataObject(
        properties=clean_data,
        uuid=object_id,
        vector=vector
    )
    
    # Check if object already exists
    try:
        existing = collection.query.fetch_object_by_id(object_id)
        if existing:
            # Update existing object
            collection.data.replace(data_object)
            print(f"Updated transaction {transaction_id} in Weaviate")
            return object_id
    except Exception:
        # Object doesn't exist, continue to create
        pass
    
    # Create new object
    try:
        collection.data.insert(data_object)
        print(f"Added transaction {transaction_id} to Weaviate")
        return object_id
    except Exception as e:
        raise Exception(f"Error adding transaction to Weaviate: {str(e)}")

def batch_add_users(client, users_data: List[Dict[str, Any]]) -> List[str]:
    """
    Add multiple users to Weaviate in batches.
    
    Args:
        client: Weaviate client instance
        users_data: List of user data dictionaries
        
    Returns:
        List of Weaviate object IDs
    """
    object_ids = []
    
    # Get the collection
    collection = client.collections.get(USER_CLASS)
    
    # Process in batches
    batch_size = WEAVIATE_BATCH_SIZE
    for i in range(0, len(users_data), batch_size):
        batch = users_data[i:i+batch_size]
        batch_objects = []
        
        for user_data in batch:
            # Extract user ID
            user_id = user_data.get("user_id")
            if not user_id:
                continue
            
            # Prepare text for embedding
            text = prepare_user_text(user_data)
            
            # Generate embedding
            vector = get_embedding(text)
            
            # Add embedding model information
            user_data["embedding_model"] = EMBEDDING_MODEL
            
            # Generate a UUID based on user_id for consistent object IDs
            object_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user-{user_id}"))
            object_ids.append(object_id)
            
            # Create DataObject for v4 API
            data_object = DataObject(
                properties=user_data,
                uuid=object_id,
                vector=vector
            )
            
            # Add to batch
            batch_objects.append(data_object)
        
        # Insert batch
        if batch_objects:
            try:
                collection.data.insert_many(batch_objects)
                print(f"Added {len(batch_objects)} users to Weaviate")
            except Exception as e:
                print(f"Error adding users batch to Weaviate: {str(e)}")
    
    return object_ids

def batch_add_transactions(client, transactions_data: List[Dict[str, Any]]) -> List[str]:
    """
    Add multiple transactions to Weaviate in batches.
    
    Args:
        client: Weaviate client instance
        transactions_data: List of transaction data dictionaries
        
    Returns:
        List of Weaviate object IDs
    """
    object_ids = []
    
    # Get the collection
    collection = client.collections.get(TRANSACTION_CLASS)
    
    # Process in batches
    batch_size = WEAVIATE_BATCH_SIZE
    for i in range(0, len(transactions_data), batch_size):
        batch = transactions_data[i:i+batch_size]
        batch_objects = []
        
        for transaction_data in batch:
            # Create a clean copy of the data to avoid modifying the original
            clean_data = transaction_data.copy()
            
            # Extract transaction ID
            transaction_id = clean_data.get("transaction_id")
            if not transaction_id:
                continue
            
            # Handle date conversion for Weaviate
            if "transaction_date" in clean_data:
                if isinstance(clean_data["transaction_date"], datetime):
                    # Format date as RFC3339 with timezone
                    clean_data["transaction_date"] = clean_data["transaction_date"].replace(microsecond=0).isoformat() + "Z"
            
            # Convert confidence values to strings if they're defined as text in the schema
            # or remove them if they're not defined in the schema
            if "date_confidence" in clean_data:
                if isinstance(clean_data["date_confidence"], (float, int)):
                    # Remove it as it's not in our schema
                    del clean_data["date_confidence"]
            
            # Convert other confidence fields to strings if needed
            for field in ["amount_confidence", "vendor_confidence", "category_confidence"]:
                if field in clean_data and isinstance(clean_data[field], (float, int)):
                    # Remove these as they're not in our schema
                    del clean_data[field]
            
            # Prepare text for embedding
            text = prepare_transaction_text(clean_data)
            
            # Generate embedding
            vector = get_embedding(text)
            
            # Add embedding model information
            clean_data["embedding_model"] = EMBEDDING_MODEL
            
            # Generate a UUID based on transaction_id for consistent object IDs
            object_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"transaction-{transaction_id}"))
            object_ids.append(object_id)
            
            # Create DataObject for v4 API
            data_object = DataObject(
                properties=clean_data,
                uuid=object_id,
                vector=vector
            )
            
            # Add to batch
            batch_objects.append(data_object)
        
        # Insert batch
        if batch_objects:
            try:
                collection.data.insert_many(batch_objects)
                print(f"Added {len(batch_objects)} transactions to Weaviate")
            except Exception as e:
                print(f"Error adding transactions batch to Weaviate: {str(e)}")
    
    return object_ids 