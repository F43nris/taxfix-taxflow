"""
Weaviate client wrapper.
"""
import weaviate
from weaviate.classes.init import Auth

from app.vector.settings import (
    WEAVIATE_URL,
    WEAVIATE_API_KEY
)

def get_weaviate_client():
    """
    Get a Weaviate client instance.
    
    Returns:
        Weaviate client instance
    """
    # Configure authentication
    auth_config = None
    if WEAVIATE_API_KEY:
        auth_config = Auth.api_key(api_key=WEAVIATE_API_KEY)
    
    # For local development with Docker Compose, use the simple helper
    if WEAVIATE_URL == "http://localhost:8080":
        client = weaviate.connect_to_local()
    else:
        # For custom URLs, use the connect_to_custom helper
        # Parse URL components
        import urllib.parse
        url_parts = urllib.parse.urlparse(WEAVIATE_URL)
        is_secure = url_parts.scheme == "https"
        host = url_parts.hostname
        port = url_parts.port or (443 if is_secure else 80)
        
        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=is_secure,
            grpc_host=host,
            grpc_port=50051,  # Default gRPC port
            grpc_secure=is_secure,
            auth_credentials=auth_config
        )
    
    # Check if Weaviate is ready
    if not client.is_ready():
        raise RuntimeError(f"Weaviate at {WEAVIATE_URL} is not ready. Make sure it's running with 'docker compose up -d'")
    
    return client 