"""
Weaviate schema definitions for User and Transaction classes.
"""
from typing import Dict, Any, List
from weaviate.classes.config import Property, DataType

from app.vector.settings import USER_CLASS, TRANSACTION_CLASS

def get_user_class_definition() -> Dict[str, Any]:
    """
    Define the User class schema for Weaviate.
    This represents users with their payslip information.
    """
    return {
        "class": USER_CLASS,
        "description": "User profile with payslip information",
        "vectorizer": "none",  # We'll provide our own vectors
        "properties": [
            # Basic user identification
            {
                "name": "user_id",
                "dataType": ["string"],
                "description": "Unique identifier for the user",
                "indexInverted": True,
            },
            # Profile information
            {
                "name": "employee_name",
                "dataType": ["string"],
                "description": "Name of the employee",
                "indexInverted": True,
            },
            {
                "name": "employee_name_confidence",
                "dataType": ["number"],
                "description": "Confidence score for employee name extraction",
            },
            {
                "name": "employer_name",
                "dataType": ["string"],
                "description": "Name of the employer",
                "indexInverted": True,
            },
            {
                "name": "employer_name_confidence",
                "dataType": ["number"],
                "description": "Confidence score for employer name extraction",
            },
            {
                "name": "occupation_category",
                "dataType": ["string"],
                "description": "Occupation category of the user",
                "indexInverted": True,
            },
            {
                "name": "occupation_category_confidence",
                "dataType": ["number"],
                "description": "Confidence score for occupation category",
            },
            # Filing information
            {
                "name": "filing_id",
                "dataType": ["string"],
                "description": "Filing identifier",
                "indexInverted": True,
            },
            {
                "name": "tax_year",
                "dataType": ["int"],
                "description": "Tax year for the filing",
                "indexInverted": True,
            },
            {
                "name": "filing_date",
                "dataType": ["date"],
                "description": "Date of filing",
                "indexInverted": True,
            },
            # Income information
            {
                "name": "avg_gross_pay",
                "dataType": ["number"],
                "description": "Average monthly gross pay",
            },
            {
                "name": "gross_pay_confidence",
                "dataType": ["number"],
                "description": "Confidence score for gross pay extraction",
            },
            {
                "name": "avg_net_pay",
                "dataType": ["number"],
                "description": "Average monthly net pay",
            },
            {
                "name": "net_pay_confidence",
                "dataType": ["number"],
                "description": "Confidence score for net pay extraction",
            },
            {
                "name": "avg_tax_deductions",
                "dataType": ["number"],
                "description": "Average monthly tax deductions",
            },
            # Calculated fields
            {
                "name": "income_band",
                "dataType": ["string"],
                "description": "Income band of the user",
                "indexInverted": True,
            },
            {
                "name": "annualized_income",
                "dataType": ["number"],
                "description": "Projected annual gross income",
            },
            {
                "name": "annualized_net_pay",
                "dataType": ["number"],
                "description": "Projected annual net income",
            },
            {
                "name": "annualized_tax_deductions",
                "dataType": ["number"],
                "description": "Projected annual tax deductions",
            },
            {
                "name": "payslip_count",
                "dataType": ["int"],
                "description": "Number of payslips processed",
            },
            {
                "name": "gross_pay_count",
                "dataType": ["int"],
                "description": "Number of payslips with valid gross pay",
            },
            {
                "name": "net_pay_count",
                "dataType": ["int"],
                "description": "Number of payslips with valid net pay",
            },
            # Enriched fields from EnrichedUser
            {
                "name": "age_range",
                "dataType": ["string"],
                "description": "Age range of the user",
                "indexInverted": True,
            },
            {
                "name": "family_status",
                "dataType": ["string"],
                "description": "Family status of the user",
                "indexInverted": True,
            },
            {
                "name": "region",
                "dataType": ["string"],
                "description": "Geographic region of the user",
                "indexInverted": True,
            },
            {
                "name": "total_income",
                "dataType": ["number"],
                "description": "Total income for the tax year",
            },
            {
                "name": "total_deductions",
                "dataType": ["number"],
                "description": "Total deductions for the tax year",
            },
            {
                "name": "refund_amount",
                "dataType": ["number"],
                "description": "Tax refund amount",
            },
            # Recommendation fields
            {
                "name": "cluster_recommendation",
                "dataType": ["string"],
                "description": "Tax recommendation based on user cluster",
                "indexInverted": True,
            },
            {
                "name": "cluster_confidence_level",
                "dataType": ["string"],
                "description": "Confidence level for cluster recommendation",
                "indexInverted": True,
            },
            {
                "name": "uplift_message",
                "dataType": ["string"],
                "description": "Uplift insight message for the user",
                "indexInverted": True,
            },
            {
                "name": "uplift_confidence_level",
                "dataType": ["string"],
                "description": "Confidence level for uplift insight",
                "indexInverted": True,
            },
            # Vector metadata
            {
                "name": "embedding_model",
                "dataType": ["string"],
                "description": "Model used to generate the embedding",
                "indexInverted": True,
            }
        ],
    }

def get_transaction_class_definition() -> Dict[str, Any]:
    """
    Define the Transaction class schema for Weaviate.
    This represents transactions from receipts.
    """
    return {
        "class": TRANSACTION_CLASS,
        "description": "Transaction data from receipts",
        "vectorizer": "none",  # We'll provide our own vectors
        "properties": [
            # Basic transaction identification
            {
                "name": "transaction_id",
                "dataType": ["string"],
                "description": "Unique identifier for the transaction",
                "indexInverted": True,
            },
            {
                "name": "user_id",
                "dataType": ["string"],
                "description": "User ID associated with the transaction",
                "indexInverted": True,
            },
            {
                "name": "receipt_id",
                "dataType": ["string"],
                "description": "ID of the source receipt",
                "indexInverted": True,
            },
            # Transaction details
            {
                "name": "transaction_date",
                "dataType": ["date"],
                "description": "Date of the transaction",
                "indexInverted": True,
            },
            {
                "name": "amount",
                "dataType": ["number"],
                "description": "Transaction amount",
            },
            {
                "name": "category",
                "dataType": ["string"],
                "description": "Transaction category",
                "indexInverted": True,
            },
            {
                "name": "subcategory",
                "dataType": ["string"],
                "description": "Transaction subcategory",
                "indexInverted": True,
            },
            {
                "name": "description",
                "dataType": ["text"],
                "description": "Transaction description",
                "indexInverted": True,
            },
            {
                "name": "vendor",
                "dataType": ["string"],
                "description": "Vendor name",
                "indexInverted": True,
            },
            # Confidence scores
            {
                "name": "confidence_score",
                "dataType": ["number"],
                "description": "Overall confidence score for the transaction",
            },
            {
                "name": "amount_confidence",
                "dataType": ["number"],
                "description": "Confidence score for amount extraction",
            },
            {
                "name": "vendor_confidence",
                "dataType": ["number"],
                "description": "Confidence score for vendor extraction",
            },
            {
                "name": "category_confidence",
                "dataType": ["number"],
                "description": "Confidence score for category extraction",
            },
            # Time-based fields
            {
                "name": "year",
                "dataType": ["int"],
                "description": "Year of the transaction",
                "indexInverted": True,
            },
            {
                "name": "month",
                "dataType": ["int"],
                "description": "Month of the transaction",
                "indexInverted": True,
            },
            {
                "name": "quarter",
                "dataType": ["int"],
                "description": "Quarter of the transaction",
                "indexInverted": True,
            },
            # Enriched fields from EnrichedTransaction
            {
                "name": "occupation_category",
                "dataType": ["string"],
                "description": "Occupation category of the user",
                "indexInverted": True,
            },
            {
                "name": "family_status",
                "dataType": ["string"],
                "description": "Family status of the user",
                "indexInverted": True,
            },
            {
                "name": "is_deductible",
                "dataType": ["boolean"],
                "description": "Whether the transaction is tax deductible",
                "indexInverted": True,
            },
            {
                "name": "deduction_confidence_score",
                "dataType": ["number"],
                "description": "Confidence score for deduction classification",
            },
            {
                "name": "deduction_recommendation",
                "dataType": ["string"],
                "description": "Recommendation for tax deduction",
                "indexInverted": True,
            },
            {
                "name": "deduction_category",
                "dataType": ["string"],
                "description": "Category of tax deduction",
                "indexInverted": True,
            },
            # Vector metadata
            {
                "name": "embedding_model",
                "dataType": ["string"],
                "description": "Model used to generate the embedding",
                "indexInverted": True,
            }
        ],
    }

def create_schema(client) -> None:
    """
    Create the Weaviate schema if it doesn't exist.
    
    Args:
        client: Weaviate client instance
    """
    # Get schema collections
    collections = client.collections.list_all()
    
    # In v4 API, collections.list_all() returns a list of collection names as strings
    collection_names = collections  # collections is already a list of strings in v4
    
    # Create User class if it doesn't exist
    if USER_CLASS not in collection_names:
        # In v4 API, we create the collection first
        user_collection = client.collections.create(
            name=USER_CLASS,
            description="User profile with payslip information",
            vectorizer_config=None  # We'll provide our own vectors
        )
        
        # Add properties using the v4 API
        user_props = get_user_class_definition()["properties"]
        for prop in user_props:
            prop_name = prop["name"]
            data_type = prop["dataType"][0]
            description = prop.get("description", "")
            index_filterable = prop.get("indexInverted", False)
            
            # Only set indexSearchable for text data types
            index_searchable = index_filterable if data_type in ["text", "string"] else None
            
            # Create property with the correct parameter names for v4 API
            property_obj = Property(
                name=prop_name,
                data_type=getattr(DataType, data_type.upper(), DataType.TEXT),
                description=description,
                skip_vectorization=True,  # Since we're providing our own vectors
                index_filterable=index_filterable,
                index_searchable=index_searchable
            )
            
            # Add the property to the collection
            user_collection.config.add_property(property_obj)
        
        print(f"Created {USER_CLASS} collection in Weaviate")
    
    # Create Transaction class if it doesn't exist
    if TRANSACTION_CLASS not in collection_names:
        # In v4 API, we create the collection first
        transaction_collection = client.collections.create(
            name=TRANSACTION_CLASS,
            description="Transaction data from receipts",
            vectorizer_config=None  # We'll provide our own vectors
        )
        
        # Add properties using the v4 API
        tx_props = get_transaction_class_definition()["properties"]
        for prop in tx_props:
            prop_name = prop["name"]
            data_type = prop["dataType"][0]
            description = prop.get("description", "")
            index_filterable = prop.get("indexInverted", False)
            
            # Only set indexSearchable for text data types
            index_searchable = index_filterable if data_type in ["text", "string"] else None
            
            # Create property with the correct parameter names for v4 API
            property_obj = Property(
                name=prop_name,
                data_type=getattr(DataType, data_type.upper(), DataType.TEXT),
                description=description,
                skip_vectorization=True,  # Since we're providing our own vectors
                index_filterable=index_filterable,
                index_searchable=index_searchable
            )
            
            # Add the property to the collection
            transaction_collection.config.add_property(property_obj)
        
        print(f"Created {TRANSACTION_CLASS} collection in Weaviate")

def delete_schema(client) -> None:
    """
    Delete the schema classes if they exist.
    
    Args:
        client: Weaviate client instance
    """
    try:
        client.collections.delete(USER_CLASS)
        print(f"Deleted {USER_CLASS} collection from Weaviate")
    except Exception as e:
        print(f"Failed to delete {USER_CLASS}: {str(e)}")
    
    try:
        client.collections.delete(TRANSACTION_CLASS)
        print(f"Deleted {TRANSACTION_CLASS} collection from Weaviate")
    except Exception as e:
        print(f"Failed to delete {TRANSACTION_CLASS}: {str(e)}") 