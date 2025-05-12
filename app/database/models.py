from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid
import re


class DocumentType(str, Enum):
    RECEIPT = "receipt"
    INCOME_STATEMENT = "income_statement"


class ConfidenceScore(BaseModel):
    """Model to store confidence scores for extracted fields"""
    value: float
    field_name: str


class Metadata(BaseModel):
    """Model for document metadata"""
    processed_at: datetime
    document_type: DocumentType
    mime_type: str
    page_count: int


class TaxItem(BaseModel):
    """Model for tax items from income statements"""
    tax_type: str
    tax_this_period: Union[str, float]
    tax_this_period_confidence: float
    tax_ytd: Optional[Union[str, float]] = None
    tax_ytd_confidence: Optional[float] = None


class LineItem(BaseModel):
    """Model for line items in receipts"""
    quantity: Optional[int] = None
    quantity_confidence: Optional[float] = None
    amount: float
    amount_confidence: float
    description: Optional[str] = None


class User(BaseModel):
    """Model for user and filing information extracted from payslips"""
    # User ID
    user_id: str = Field(default_factory=lambda: f"U{uuid.uuid4().hex[:4].upper()}")
    
    # Profile information
    employee_name: Optional[str] = None
    employee_name_confidence: Optional[float] = None
    employer_name: Optional[str] = None
    employer_name_confidence: Optional[float] = None
    
    # Filing information
    filing_id: str = Field(default_factory=lambda: f"F{uuid.uuid4().hex[:5].upper()}")
    tax_year: int = datetime.now().year - 1
    filing_date: Optional[datetime] = None
    
    # Income and pay information - focusing on averages only
    avg_gross_pay: Optional[float] = None  # Average monthly gross pay
    gross_pay_confidence: Optional[float] = None
    avg_net_pay: Optional[float] = None    # Average monthly net pay
    net_pay_confidence: Optional[float] = None
    
    # Calculated tax deductions (difference between gross and net)
    avg_tax_deductions: Optional[float] = None
    
    # Additional calculated fields
    income_band: Optional[str] = None
    annualized_income: Optional[float] = None  # Projected annual gross income
    annualized_net_pay: Optional[float] = None  # Projected annual net income
    annualized_tax_deductions: Optional[float] = None  # Projected annual tax deductions
    payslip_count: int = 1  # Number of payslips processed
    
    # Tracking for average calculations
    gross_pay_count: int = 0  # Number of payslips with valid gross pay
    net_pay_count: int = 0    # Number of payslips with valid net pay
    
    @validator('income_band', always=True)
    def set_income_band(cls, v, values):
        if v is None and 'annualized_income' in values and values['annualized_income']:
            income = values['annualized_income']
            if income < 20000:
                return "A: 0-20,000 €"
            elif income < 50000:
                return "B: 20,001-50,000 €"
            elif income < 100000:
                return "C: 51,131-100,000 €"
            else:
                return "D: 100,001+ €"
        return v
    
    @validator('avg_tax_deductions', always=True)
    def calculate_tax_deductions(cls, v, values):
        """Calculate average tax deductions as difference between gross and net pay"""
        if v is None and 'avg_gross_pay' in values and values['avg_gross_pay'] and 'avg_net_pay' in values and values['avg_net_pay']:
            # Only calculate if we have both values
            return values['avg_gross_pay'] - values['avg_net_pay']
        return v
        
    @validator('annualized_income', always=True)
    def calculate_annualized_income(cls, v, values):
        # Use average gross pay to calculate annual income if available
        if v is None and 'avg_gross_pay' in values and values['avg_gross_pay']:
            return values['avg_gross_pay'] * 12
        return v
    
    @validator('annualized_net_pay', always=True)
    def calculate_annualized_net_pay(cls, v, values):
        # Use average net pay to calculate annual net income if available
        if v is None and 'avg_net_pay' in values and values['avg_net_pay']:
            return values['avg_net_pay'] * 12
        return v
    
    @validator('annualized_tax_deductions', always=True)
    def calculate_annualized_tax_deductions(cls, v, values):
        # Use average tax deductions to calculate annual tax deductions if available
        if v is None and 'avg_tax_deductions' in values and values['avg_tax_deductions']:
            return values['avg_tax_deductions'] * 12
        return v
    
    def update_from_income_statement(self, income_statement):
        """Update user with data from a new income statement"""
        self.payslip_count += 1
        
        # Update gross pay information
        if income_statement.gross_earnings is not None:
            self.gross_pay_count += 1
            
            # Update average gross pay
            if self.avg_gross_pay:
                # Recalculate the average
                total_gross = self.avg_gross_pay * (self.gross_pay_count - 1)
                total_gross += income_statement.gross_earnings
                self.avg_gross_pay = total_gross / self.gross_pay_count
            else:
                self.avg_gross_pay = income_statement.gross_earnings
            
            # Update gross pay confidence
            if income_statement.gross_earnings_confidence:
                if self.gross_pay_confidence:
                    self.gross_pay_confidence = (self.gross_pay_confidence + income_statement.gross_earnings_confidence) / 2
                else:
                    self.gross_pay_confidence = income_statement.gross_earnings_confidence
        
        # Update net pay information
        if income_statement.net_pay is not None:
            self.net_pay_count += 1
            
            # Update average net pay
            if self.avg_net_pay:
                # Recalculate the average
                total_net = self.avg_net_pay * (self.net_pay_count - 1)
                total_net += income_statement.net_pay
                self.avg_net_pay = total_net / self.net_pay_count
            else:
                self.avg_net_pay = income_statement.net_pay
            
            # Update net pay confidence
            if income_statement.net_pay_confidence:
                if self.net_pay_confidence:
                    self.net_pay_confidence = (self.net_pay_confidence + income_statement.net_pay_confidence) / 2
                else:
                    self.net_pay_confidence = income_statement.net_pay_confidence
            
        # Update fields if they are not set yet
        if not self.employee_name and income_statement.employee_name:
            self.employee_name = income_statement.employee_name
            self.employee_name_confidence = income_statement.employee_name_confidence
            
        if not self.employer_name and income_statement.employer_name:
            self.employer_name = income_statement.employer_name
            self.employer_name_confidence = income_statement.employer_name_confidence
            
        if not self.filing_date and income_statement.pay_date:
            self.filing_date = income_statement.pay_date
            
        # Recalculate derived fields
        self.annualized_income = self.calculate_annualized_income(None, {
            'avg_gross_pay': self.avg_gross_pay,
        })
        self.annualized_net_pay = self.calculate_annualized_net_pay(None, {
            'avg_net_pay': self.avg_net_pay,
        })
        self.income_band = self.set_income_band(None, {'annualized_income': self.annualized_income})
        self.avg_tax_deductions = self.calculate_tax_deductions(None, {
            'avg_gross_pay': self.avg_gross_pay,
            'avg_net_pay': self.avg_net_pay
        })
        self.annualized_tax_deductions = self.calculate_annualized_tax_deductions(None, {
            'avg_tax_deductions': self.avg_tax_deductions
        })


class Transaction(BaseModel):
    """Model for financial transactions from receipts"""
    transaction_id: str = Field(default_factory=lambda: f"T{uuid.uuid4().hex[:5].upper()}")
    user_id: str
    transaction_date: datetime
    amount: float
    category: str
    subcategory: Optional[str] = None
    description: Optional[str] = None
    vendor: Optional[str] = None
    
    # Source data details
    receipt_id: str  # ID of source receipt
    confidence_score: float  # Average confidence score
    
    # Individual field confidence scores
    amount_confidence: Optional[float] = None
    date_confidence: Optional[float] = None
    vendor_confidence: Optional[float] = None
    category_confidence: Optional[float] = None
    
    # Derived fields for analytics
    transaction_month: Optional[int] = None
    quarter: Optional[int] = None
    year: Optional[int] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set date fields from transaction_date
        if self.transaction_date:
            if not self.transaction_month:
                self.transaction_month = self.transaction_date.month
            if not self.quarter:
                self.quarter = (self.transaction_date.month - 1) // 3 + 1
            if not self.year:
                self.year = self.transaction_date.year


class Receipt(BaseModel):
    """Model for receipts"""
    id: str = Field(default_factory=lambda: f"RC{uuid.uuid4().hex[:8].upper()}")
    file_path: str
    invoice_type: Optional[str] = None
    invoice_type_confidence: Optional[float] = None
    total_amount: Optional[float] = None
    total_amount_confidence: Optional[float] = None
    net_amount: Optional[float] = None
    net_amount_confidence: Optional[float] = None
    total_tax_amount: Optional[float] = None
    total_tax_amount_confidence: Optional[float] = None
    currency: Optional[str] = None
    currency_confidence: Optional[float] = None
    invoice_date: Optional[datetime] = None
    invoice_date_confidence: Optional[float] = None
    supplier_name: Optional[str] = None
    supplier_name_confidence: Optional[float] = None
    supplier_address: Optional[Union[str, float]] = None
    supplier_address_confidence: Optional[float] = None
    supplier_phone: Optional[str] = None
    supplier_phone_confidence: Optional[float] = None
    line_items: List[LineItem] = []
    metadata: Metadata

    @classmethod
    def from_json(cls, json_data: Dict[str, Any], file_path: str) -> "Receipt":
        """Create a Receipt instance from JSON data"""
        # Extract basic fields
        data = {
            "file_path": file_path,
            "metadata": Metadata(
                processed_at=datetime.fromisoformat(json_data["_metadata"]["processed_at"]),
                document_type=json_data["_metadata"]["document_type"],
                mime_type=json_data["_metadata"]["mime_type"],
                page_count=json_data["_metadata"]["page_count"]
            )
        }
        
        # Extract fields with confidence
        for field in ["invoice_type", "supplier_name", "supplier_address", "supplier_phone", 
                     "currency", "total_amount", "net_amount", "total_tax_amount"]:
            if field in json_data:
                data[field] = cls._extract_value(json_data[field])
                data[f"{field}_confidence"] = json_data[field]["confidence"]
        
        # Handle invoice_date separately due to datetime conversion
        if "invoice_date" in json_data:
            if json_data["invoice_date"]["normalized_value"]:
                data["invoice_date"] = datetime.strptime(json_data["invoice_date"]["normalized_value"], "%Y-%m-%d")
            data["invoice_date_confidence"] = json_data["invoice_date"]["confidence"]
        
        # Extract line items
        if "line_item" in json_data:
            line_items = []
            line_data = json_data["line_item"]
            if not isinstance(line_data, list):
                line_data = [line_data]
                
            for item in line_data:
                if "properties" in item:
                    props = item["properties"]
                    # Get quantity
                    quantity = cls._extract_value(props.get("line_item/quantity", {}))
                    if quantity is not None:
                        try:
                            quantity = int(quantity)
                        except (ValueError, TypeError):
                            quantity = None
                    
                    # Get amount
                    amount = cls._extract_value(props.get("line_item/amount", {}))
                    if amount is not None:
                        if isinstance(amount, str):
                            amount = float(amount.replace(',', '.'))
                        elif isinstance(amount, (int, float)):
                            amount = float(amount)
                        else:
                            amount = 0.0
                    else:
                        amount = 0.0
                        
                    line_items.append(LineItem(
                        quantity=quantity,
                        quantity_confidence=props.get("line_item/quantity", {}).get("confidence", 0.0),
                        amount=amount,
                        amount_confidence=props.get("line_item/amount", {}).get("confidence", 0.0),
                        description=cls._extract_value(props.get("line_item/description", {}))
                    ))
            
            data["line_items"] = line_items
            
        return cls(**data)
    
    @staticmethod
    def _extract_value(field_data: Dict[str, Any]) -> Any:
        """Extract normalized or mentioned value from field data"""
        if not field_data:
            return None
            
        # Prefer normalized value if available
        if field_data.get("normalized_value") is not None:
            value = field_data["normalized_value"]
            # Try to convert currency amounts to float
            if isinstance(value, str) and re.search(r'\d+[\.,]?\d*\s*[€$A-Z]{0,3}', value):
                # Extract just the numeric part
                numeric_part = re.search(r'(\d+[\.,]?\d*)', value)
                if numeric_part:
                    # Convert to float, handling European format
                    try:
                        return float(numeric_part.group(1).replace(',', '.'))
                    except ValueError:
                        return value
            return value
            
        return field_data.get("mention_text")
    
    def to_transaction(self, user_id: str) -> Transaction:
        """Convert receipt to a transaction"""
        # Calculate confidence score from all used fields
        confidence_fields = [
            ('total_amount', self.total_amount_confidence),
            ('invoice_date', self.invoice_date_confidence),
            ('supplier_name', self.supplier_name_confidence),
            ('currency', self.currency_confidence),
            ('invoice_type', self.invoice_type_confidence)
        ]
        
        # Filter out None values
        valid_confidence = [(name, score) for name, score in confidence_fields if score is not None]
        
        # Calculate average confidence if any valid scores exist
        if valid_confidence:
            scores_sum = sum(score for _, score in valid_confidence)
            avg_confidence = scores_sum / len(valid_confidence)
            # Also track which fields contributed to confidence score
            confidence_field_names = ", ".join(name for name, _ in valid_confidence)
        else:
            avg_confidence = 0
            confidence_field_names = "none"
            
        # Set transaction date from invoice date
        transaction_date = self.invoice_date or datetime.now()
            
        # Determine category and subcategory based on invoice type
        category = "Uncategorized"
        subcategory = None
        if self.invoice_type:
            if "restaurant" in str(self.invoice_type).lower():
                category = "Food & Dining"
                subcategory = "Restaurant"
            elif "grocery" in str(self.invoice_type).lower():
                category = "Food & Dining"
                subcategory = "Groceries"
        
        # Get specific field confidences
        amount_confidence = self.total_amount_confidence
        date_confidence = self.invoice_date_confidence
        vendor_confidence = self.supplier_name_confidence
        category_confidence = self.invoice_type_confidence
        
        # Format confidence details for description
        confidence_details = []
        for field_name, confidence in valid_confidence:
            if confidence is not None:
                confidence_details.append(f"{field_name}: {confidence:.2%}")
        
        description = f"Fields: {confidence_field_names}\nConfidence details: {', '.join(confidence_details)}"
        
        # Create transaction
        return Transaction(
            user_id=user_id,
            transaction_date=transaction_date,
            amount=self.total_amount or 0,
            category=category,
            subcategory=subcategory,
            description=description,
            vendor=self.supplier_name,
            receipt_id=self.id,
            confidence_score=avg_confidence,
            # Individual confidences
            amount_confidence=amount_confidence,
            date_confidence=date_confidence,
            vendor_confidence=vendor_confidence,
            category_confidence=category_confidence
        )


class IncomeStatement(BaseModel):
    """Model for income statements/payslips"""
    id: str = Field(default_factory=lambda: f"IS{uuid.uuid4().hex[:8].upper()}")
    file_path: str
    employee_name: Optional[str] = None
    employee_name_confidence: Optional[float] = None
    employee_address: Optional[str] = None
    employee_address_confidence: Optional[float] = None
    employer_name: Optional[str] = None
    employer_name_confidence: Optional[float] = None
    pay_date: Optional[datetime] = None
    pay_date_confidence: Optional[float] = None
    gross_earnings: Optional[float] = None
    gross_earnings_confidence: Optional[float] = None
    gross_earnings_ytd: Optional[float] = None
    gross_earnings_ytd_confidence: Optional[float] = None
    net_pay: Optional[float] = None
    net_pay_confidence: Optional[float] = None
    net_pay_ytd: Optional[float] = None
    net_pay_ytd_confidence: Optional[float] = None
    tax_items: List[TaxItem] = []
    metadata: Metadata

    @classmethod
    def from_json(cls, json_data: Dict[str, Any], file_path: str) -> "IncomeStatement":
        """Create an IncomeStatement instance from JSON data"""
        # Extract basic fields
        data = {
            "file_path": file_path,
            "metadata": Metadata(
                processed_at=datetime.fromisoformat(json_data["_metadata"]["processed_at"]),
                document_type=json_data["_metadata"]["document_type"],
                mime_type=json_data["_metadata"]["mime_type"],
                page_count=json_data["_metadata"]["page_count"]
            )
        }
        
        # Extract fields with confidence
        for field in ["employee_name", "employee_address", "employer_name", "gross_earnings", 
                     "gross_earnings_ytd", "net_pay", "net_pay_ytd"]:
            if field in json_data:
                data[field] = cls._extract_value(json_data[field])
                data[f"{field}_confidence"] = json_data[field]["confidence"]
        
        # Handle pay_date separately due to datetime conversion
        if "pay_date" in json_data:
            if json_data["pay_date"]["normalized_value"]:
                data["pay_date"] = datetime.strptime(json_data["pay_date"]["normalized_value"], "%Y-%m-%d")
            data["pay_date_confidence"] = json_data["pay_date"]["confidence"]
        
        # Extract tax items
        if "tax_item" in json_data:
            tax_items = []
            tax_data = json_data["tax_item"]
            if not isinstance(tax_data, list):
                tax_data = [tax_data]
                
            for tax_item in tax_data:
                if "properties" in tax_item:
                    props = tax_item["properties"]
                    tax_items.append(TaxItem(
                        tax_type=cls._extract_value(props.get("tax_type", {})),
                        tax_this_period=cls._extract_value(props.get("tax_this_period", {})),
                        tax_this_period_confidence=props.get("tax_this_period", {}).get("confidence", 0.0),
                        tax_ytd=cls._extract_value(props.get("tax_ytd", {})),
                        tax_ytd_confidence=props.get("tax_ytd", {}).get("confidence", 0.0)
                    ))
            
            data["tax_items"] = tax_items
            
        return cls(**data)
    
    @staticmethod
    def _extract_value(field_data: Dict[str, Any]) -> Any:
        """Extract normalized or mentioned value from field data"""
        if not field_data:
            return None
            
        # Prefer normalized value if available
        if field_data.get("normalized_value") is not None:
            value = field_data["normalized_value"]
            # Try to convert currency amounts to float
            if isinstance(value, str) and re.search(r'\d+[\.,]?\d*\s*[€$A-Z]{0,3}', value):
                # Extract just the numeric part
                numeric_part = re.search(r'(\d+[\.,]?\d*)', value)
                if numeric_part:
                    # Convert to float, handling European format
                    try:
                        return float(numeric_part.group(1).replace(',', '.'))
                    except ValueError:
                        return value
            return value
            
        return field_data.get("mention_text")
    
    def to_user(self, existing_user_id: Optional[str] = None) -> User:
        """Convert income statement to user information"""
        # Default tax year is based on pay date
        tax_year = self.pay_date.year if self.pay_date else datetime.now().year
        
        # Determine which values are valid
        gross_pay_count = 1 if self.gross_earnings is not None else 0
        net_pay_count = 1 if self.net_pay is not None else 0
        
        # Create user data
        user_data = {
            "employee_name": self.employee_name,
            "employee_name_confidence": self.employee_name_confidence,
            "employer_name": self.employer_name,
            "employer_name_confidence": self.employer_name_confidence,
            "avg_gross_pay": self.gross_earnings if gross_pay_count > 0 else None,
            "gross_pay_confidence": self.gross_earnings_confidence,
            "avg_net_pay": self.net_pay if net_pay_count > 0 else None,
            "net_pay_confidence": self.net_pay_confidence,
            "filing_date": self.pay_date or datetime.now(),
            "tax_year": tax_year,
            "payslip_count": 1,  # Each income statement represents one payslip
            "gross_pay_count": gross_pay_count,
            "net_pay_count": net_pay_count
        }
        
        # Use existing user_id if provided
        if existing_user_id:
            user_data["user_id"] = existing_user_id
        
        return User(**user_data) 