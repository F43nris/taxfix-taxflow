"""
Generate embeddings for text using OpenAI or Vertex AI.
"""
import numpy as np
from typing import List, Union, Dict, Any
import json
from datetime import datetime

from app.vector.settings import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    get_embedding_api_key
)

def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding for the given text using the configured provider.
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        Embedding vector as a list of floats
    """
    if EMBEDDING_PROVIDER == "openai":
        return get_openai_embedding(text)
    elif EMBEDDING_PROVIDER == "vertex":
        return get_vertex_embedding(text)
    else:
        raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")

def get_openai_embedding(text: str) -> List[float]:
    """
    Generate an embedding using OpenAI API (v1.0+).
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        Embedding vector as a list of floats
    """
    try:
        from openai import OpenAI
        
        # Initialize client with API key
        client = OpenAI(api_key=get_embedding_api_key())
        
        # Get embedding using the new API
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        
        # Extract embedding vector
        embedding = response.data[0].embedding
        
        return embedding
    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai>=1.0.0")
    except Exception as e:
        raise Exception(f"Error generating OpenAI embedding: {str(e)}")

def get_vertex_embedding(text: str) -> List[float]:
    """
    Generate an embedding using Vertex AI.
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        Embedding vector as a list of floats
    """
    try:
        from google.cloud import aiplatform
        from vertexai.language_models import TextEmbeddingModel
        
        # Initialize Vertex AI
        aiplatform.init()
        
        # Load the embedding model
        model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        
        # Generate embedding
        embeddings = model.get_embeddings([text])
        
        # Extract embedding vector
        embedding = embeddings[0].values
        
        return embedding
    except ImportError:
        raise ImportError("Vertex AI packages not installed. Install with: pip install google-cloud-aiplatform vertexai")
    except Exception as e:
        raise Exception(f"Error generating Vertex AI embedding: {str(e)}")

def prepare_user_text(user_data: Dict[str, Any]) -> str:
    """
    Prepare user data for embedding by converting it to a text representation.
    
    Args:
        user_data: User data dictionary
        
    Returns:
        Text representation of the user data
    """
    text_parts = []
    
    # Add user information
    if user_data.get("employee_name"):
        text_parts.append(f"Employee name: {user_data['employee_name']}")
    
    if user_data.get("employer_name"):
        text_parts.append(f"Employer: {user_data['employer_name']}")
    
    if user_data.get("occupation_category"):
        text_parts.append(f"Occupation: {user_data['occupation_category']}")
    
    # Add income information
    if user_data.get("avg_gross_pay"):
        text_parts.append(f"Monthly gross pay: {user_data['avg_gross_pay']:.2f}")
    
    if user_data.get("avg_net_pay"):
        text_parts.append(f"Monthly net pay: {user_data['avg_net_pay']:.2f}")
    
    if user_data.get("avg_tax_deductions"):
        text_parts.append(f"Monthly tax deductions: {user_data['avg_tax_deductions']:.2f}")
    
    if user_data.get("annualized_income"):
        text_parts.append(f"Annual income: {user_data['annualized_income']:.2f}")
    
    if user_data.get("income_band"):
        text_parts.append(f"Income band: {user_data['income_band']}")
    
    # Join all parts with newlines
    return "\n".join(text_parts)

def prepare_transaction_text(transaction_data: Dict[str, Any]) -> str:
    """
    Prepare transaction data for embedding by converting it to a text representation.
    
    Args:
        transaction_data: Transaction data dictionary
        
    Returns:
        Text representation of the transaction data
    """
    text_parts = []
    
    # Add transaction information
    if transaction_data.get("vendor"):
        text_parts.append(f"Vendor: {transaction_data['vendor']}")
    
    if transaction_data.get("category"):
        category_text = transaction_data['category']
        if transaction_data.get("subcategory"):
            category_text += f" / {transaction_data['subcategory']}"
        text_parts.append(f"Category: {category_text}")
    
    if transaction_data.get("description"):
        text_parts.append(f"Description: {transaction_data['description']}")
    
    if transaction_data.get("amount"):
        text_parts.append(f"Amount: {transaction_data['amount']:.2f}")
    
    if transaction_data.get("transaction_date"):
        # Convert string date to datetime if needed
        if isinstance(transaction_data["transaction_date"], str):
            try:
                date = datetime.fromisoformat(transaction_data["transaction_date"].replace("Z", "+00:00"))
                text_parts.append(f"Date: {date.strftime('%Y-%m-%d')}")
            except:
                text_parts.append(f"Date: {transaction_data['transaction_date']}")
        else:
            text_parts.append(f"Date: {transaction_data['transaction_date'].strftime('%Y-%m-%d')}")
    
    # Join all parts with newlines
    return "\n".join(text_parts) 