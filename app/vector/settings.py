"""
Settings for Weaviate vector database connection.
"""
import os
from typing import Optional

# Weaviate connection settings
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")  # Not needed for local dev with Docker Compose
WEAVIATE_BATCH_SIZE = int(os.environ.get("WEAVIATE_BATCH_SIZE", "100"))

# Embedding model settings
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai").lower()  # openai or vertex
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

# Schema class names
USER_CLASS = "User"
TRANSACTION_CLASS = "Transaction"

def get_embedding_api_key() -> Optional[str]:
    """Get the appropriate API key based on the embedding provider."""
    if EMBEDDING_PROVIDER == "openai":
        return OPENAI_API_KEY
    elif EMBEDDING_PROVIDER == "vertex":
        return None  # Vertex uses service account credentials
    return None 