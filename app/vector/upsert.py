"""
Add or update data in Weaviate.
"""
import uuid
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

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
    # Create a clean copy of the data to avoid modifying the original
    clean_data = user_data.copy()
    
    # Extract user ID from data if not provided
    if not user_id:
        user_id = clean_data.get("user_id")
        if not user_id:
            raise ValueError("User ID not provided and not found in user_data")
    
    # Handle date conversion for Weaviate
    if "filing_date" in clean_data:
        if isinstance(clean_data["filing_date"], datetime):
            # Format date as RFC3339 with timezone
            clean_data["filing_date"] = clean_data["filing_date"].replace(microsecond=0).isoformat() + "Z"
        elif isinstance(clean_data["filing_date"], str):
            try:
                # Try to parse the date string
                if "T" not in clean_data["filing_date"]:
                    # If it's just a date without time, add time component
                    date_obj = datetime.fromisoformat(clean_data["filing_date"].replace("Z", ""))
                    clean_data["filing_date"] = f"{date_obj.strftime('%Y-%m-%dT00:00:00')}Z"
                elif not clean_data["filing_date"].endswith("Z"):
                    # If it has time but no timezone, add Z
                    clean_data["filing_date"] = clean_data["filing_date"] + "Z"
            except ValueError:
                # If parsing fails, remove the field to avoid errors
                print(f"Warning: Could not parse filing_date '{clean_data['filing_date']}' for user {user_id}, removing field")
                del clean_data["filing_date"]
    
    # Prepare text for embedding
    text = prepare_user_text(clean_data)
    
    # Generate embedding
    vector = get_embedding(text)
    
    # Add embedding model information
    clean_data["embedding_model"] = EMBEDDING_MODEL
    
    # Generate a UUID based on user_id for consistent object IDs
    object_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user-{user_id}"))
    
    # Get the collection
    collection = client.collections.get(USER_CLASS)
    
    # Check if object already exists
    try:
        existing = collection.query.fetch_object_by_id(object_id)
        if existing:
            # Update existing object
            collection.data.replace(
                uuid=object_id,
                properties=clean_data,
                vector=vector
            )
            print(f"Updated user {user_id} in Weaviate")
            return object_id
    except Exception:
        # Object doesn't exist, continue to create
        pass
    
    # Create new object
    try:
        collection.data.insert(
            properties=clean_data,
            uuid=object_id,
            vector=vector
        )
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
        elif isinstance(clean_data["transaction_date"], str):
            try:
                # Try to parse the date string
                if "T" not in clean_data["transaction_date"]:
                    # If it's just a date without time, add time component
                    date_obj = datetime.fromisoformat(clean_data["transaction_date"].replace("Z", ""))
                    clean_data["transaction_date"] = f"{date_obj.strftime('%Y-%m-%dT00:00:00')}Z"
                elif not clean_data["transaction_date"].endswith("Z"):
                    # If it has time but no timezone, add Z
                    clean_data["transaction_date"] = clean_data["transaction_date"] + "Z"
            except ValueError:
                # If parsing fails, remove the field to avoid errors
                print(f"Warning: Could not parse transaction_date '{clean_data['transaction_date']}' for transaction {transaction_id}, removing field")
                del clean_data["transaction_date"]
    
    # Convert boolean fields to strings
    if "is_deductible" in clean_data:
        if isinstance(clean_data["is_deductible"], (bool, int, float)):
            clean_data["is_deductible"] = str(clean_data["is_deductible"]).lower()
    
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
    
    # Check if object already exists
    try:
        existing = collection.query.fetch_object_by_id(object_id)
        if existing:
            # Update existing object
            collection.data.replace(
                uuid=object_id,
                properties=clean_data,
                vector=vector
            )
            print(f"Updated transaction {transaction_id} in Weaviate")
            return object_id
    except Exception:
        # Object doesn't exist, continue to create
        pass
    
    # Create new object
    try:
        collection.data.insert(
            properties=clean_data,
            uuid=object_id,
            vector=vector
        )
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
            # Create a clean copy of the data to avoid modifying the original
            clean_data = user_data.copy()
            
            # Extract user ID
            user_id = clean_data.get("user_id")
            if not user_id:
                continue
            
            # Handle date conversion for Weaviate
            if "filing_date" in clean_data:
                if isinstance(clean_data["filing_date"], datetime):
                    # Format date as RFC3339 with timezone
                    clean_data["filing_date"] = clean_data["filing_date"].replace(microsecond=0).isoformat() + "Z"
                elif isinstance(clean_data["filing_date"], str):
                    try:
                        # Try to parse the date string
                        if "T" not in clean_data["filing_date"]:
                            # If it's just a date without time, add time component
                            date_obj = datetime.fromisoformat(clean_data["filing_date"].replace("Z", ""))
                            clean_data["filing_date"] = f"{date_obj.strftime('%Y-%m-%dT00:00:00')}Z"
                        elif not clean_data["filing_date"].endswith("Z"):
                            # If it has time but no timezone, add Z
                            clean_data["filing_date"] = clean_data["filing_date"] + "Z"
                    except ValueError:
                        # If parsing fails, remove the field to avoid errors
                        print(f"Warning: Could not parse filing_date '{clean_data['filing_date']}' for user {user_id}, removing field")
                        del clean_data["filing_date"]
            
            # Prepare text for embedding
            text = prepare_user_text(clean_data)
            
            # Generate embedding
            vector = get_embedding(text)
            
            # Add embedding model information
            clean_data["embedding_model"] = EMBEDDING_MODEL
            
            # Generate a UUID based on user_id for consistent object IDs
            object_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user-{user_id}"))
            object_ids.append(object_id)
            
            # Add to batch using dictionary format
            batch_objects.append({
                "properties": clean_data,
                "uuid": object_id,
                "vector": vector
            })
        
        # Insert batch
        if batch_objects:
            try:
                for obj in batch_objects:
                    collection.data.insert(
                        properties=obj["properties"],
                        uuid=obj["uuid"],
                        vector=obj["vector"]
                    )
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
                elif isinstance(clean_data["transaction_date"], str):
                    try:
                        # Try to parse the date string
                        if "T" not in clean_data["transaction_date"]:
                            # If it's just a date without time, add time component
                            date_obj = datetime.fromisoformat(clean_data["transaction_date"].replace("Z", ""))
                            clean_data["transaction_date"] = f"{date_obj.strftime('%Y-%m-%dT00:00:00')}Z"
                        elif not clean_data["transaction_date"].endswith("Z"):
                            # If it has time but no timezone, add Z
                            clean_data["transaction_date"] = clean_data["transaction_date"] + "Z"
                    except ValueError:
                        # If parsing fails, remove the field to avoid errors
                        print(f"Warning: Could not parse transaction_date '{clean_data['transaction_date']}' for transaction {transaction_id}, removing field")
                        del clean_data["transaction_date"]
            
            # Convert boolean fields to strings
            if "is_deductible" in clean_data:
                if isinstance(clean_data["is_deductible"], (bool, int, float)):
                    clean_data["is_deductible"] = str(clean_data["is_deductible"]).lower()
            
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
            
            # Add to batch using dictionary format
            batch_objects.append({
                "properties": clean_data,
                "uuid": object_id,
                "vector": vector
            })
        
        # Insert batch
        if batch_objects:
            try:
                for obj in batch_objects:
                    collection.data.insert(
                        properties=obj["properties"],
                        uuid=obj["uuid"],
                        vector=obj["vector"]
                    )
                print(f"Added {len(batch_objects)} transactions to Weaviate")
            except Exception as e:
                print(f"Error adding transactions batch to Weaviate: {str(e)}")
    
    return object_ids 