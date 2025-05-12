import os
from typing import Dict, Any, Optional, List
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
import json
import datetime
import mimetypes
from google.auth import default

class DocumentAIProcessor:
    def __init__(self):
        """Initialize the Document AI processor client."""
        # The location has a comment in it that's causing the error
        location = os.environ.get('DOCUMENT_AI_LOCATION', 'eu')
        
        # Use explicit authentication with default credentials
        credentials, project_id = default()
        
        # Use project_id from environment if not provided by credentials
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        # Set quota project via client options
        client_options = ClientOptions(
            api_endpoint=f"{location}-documentai.googleapis.com",
            quota_project_id=self.project_id
        )
        
        self.client = documentai.DocumentProcessorServiceClient(
            client_options=client_options,
            credentials=credentials
        )
        
        self.location = location
        
        # Pre-trained processor IDs
        self.receipt_processor_id = os.environ.get("RECEIPT_PROCESSOR_ID")
        self.income_statement_processor_id = os.environ.get("INCOME_STATEMENT_PROCESSOR_ID")
        
    def _get_processor_name(self, processor_id: str) -> str:
        """Get the fully qualified processor name."""
        return self.client.processor_path(
            self.project_id, self.location, processor_id
        )
    
    def _get_mime_type(self, file_path: str) -> str:
        """Determine MIME type based on file extension."""
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            # Default to application/pdf if unable to determine
            extension = os.path.splitext(file_path)[1].lower()
            if extension in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif extension == '.png':
                mime_type = 'image/png'
            elif extension in ['.tif', '.tiff']:
                mime_type = 'image/tiff'
            else:
                mime_type = 'application/pdf'
        return mime_type
    
    def _serialize_document_entities(self, document: documentai.Document) -> Dict[str, Any]:
        """Convert Document AI entities to a serializable dictionary."""
        result = {}
        
        for entity in document.entities:
            entity_data = {
                "confidence": entity.confidence,
                "mention_text": entity.mention_text,
                "normalized_value": entity.normalized_value.text if entity.normalized_value else None
            }
            
            # Handle nested entities if they exist
            if entity.properties:
                entity_data["properties"] = {}
                for prop in entity.properties:
                    entity_data["properties"][prop.type_] = {
                        "confidence": prop.confidence,
                        "mention_text": prop.mention_text,
                        "normalized_value": prop.normalized_value.text if prop.normalized_value else None
                    }
            
            # Use entity type as key or append to list if multiple entities of same type
            if entity.type_ in result:
                if not isinstance(result[entity.type_], list):
                    result[entity.type_] = [result[entity.type_]]
                result[entity.type_].append(entity_data)
            else:
                result[entity.type_] = entity_data
                
        return result
    
    def process_document(self, file_path: str, document_type: str = "receipt") -> Dict[str, Any]:
        """
        Process a document with the appropriate processor.
        
        Args:
            file_path: Path to the document file
            document_type: Type of document - "receipt" or "income_statement"
            
        Returns:
            Dictionary containing the extracted data
        """
        # Select the appropriate processor ID based on document type
        if document_type.lower() == "receipt":
            processor_id = self.receipt_processor_id
        elif document_type.lower() in ["income_statement", "income", "payslip"]:
            processor_id = self.income_statement_processor_id
        else:
            raise ValueError(f"Unsupported document type: {document_type}")
        
        if not processor_id:
            raise ValueError(f"Processor ID for {document_type} is not set in environment variables")
            
        # Read the file
        with open(file_path, "rb") as file:
            file_content = file.read()
        
        # Determine MIME type
        mime_type = self._get_mime_type(file_path)
        
        # Configure the process request
        request = documentai.ProcessRequest(
            name=self._get_processor_name(processor_id),
            raw_document=documentai.RawDocument(
                content=file_content, mime_type=mime_type
            )
        )
        
        # Process the document
        processor_name = self._get_processor_name(processor_id)
        print(f"Calling Document AI endpoint: {processor_name}")
        response = self.client.process_document(request=request, timeout=5)
        document = response.document
        
        # Extract structured data
        extracted_data = self._serialize_document_entities(document)
        
        # Add metadata
        extracted_data["_metadata"] = {
            "processed_at": datetime.datetime.now().isoformat(),
            "document_type": document_type,
            "mime_type": mime_type,
            "page_count": len(document.pages)
        }
        
        return extracted_data
    
    def save_as_json(self, data: Dict[str, Any], output_path: str) -> str:
        """
        Save extracted document data as JSON.
        
        Args:
            data: The extracted document data
            output_path: Path where the JSON file should be saved
            
        Returns:
            Path to the saved JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        return output_path