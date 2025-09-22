from dataclasses import dataclass
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from datetime import datetime

@dataclass
class ProcessingResult:
    success: bool
    message: str
    records_processed: int
    data: Optional[Dict[str, Any]] = None
    sheet_results: Optional[List[Dict[str, Any]]] = None

@dataclass
class Project:
    project_name: str
    project_description: str
    project_id: Optional[int] = None
    total_cost: Optional[float] = None
    project_date: Optional[datetime] = None
    project_category: Optional[str] = None

@dataclass
class Section:
    section_id: int
    section_name: str
    project_id: int
    cost: Optional[float] = None
    discipline: Optional[str] = None
    
@dataclass
class BOQItem:
    bill_id: int
    project_id: int
    section_id: int
    bill_no: Optional[str] = None
    bill_description: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    rate: Optional[float] = None
    total: Optional[float] = None

@dataclass
class BOQSchema:
    """Define the expected BOQ schema"""
    REQUIRED_FIELDS = {
        'Quantity': ['quantity', 'qty', 'amount', 'no', 'nos', 'units'],
        'UnitRate': ['rate', 'unit_rate', 'price', 'unit_price', 'cost', 'unit_cost']
    }
    
    OPTIONAL_FIELDS = {
        'ItemNumber': ['item_no', 'item_number', 'sl_no', 'sr_no', 'no', 'item', 'code'],
        'Description': ['description', 'desc', 'work_description', 'item_description', 'particulars', 'details'],
        'Unit': ['unit', 'uom', 'unit_of_measure', 'measure'],
        'TotalCost': ['total', 'total_cost', 'total_amount', 'amount', 'total_price'],
        'Discipline': ['discipline', 'trade', 'category', 'type', 'work_type'],
    }
    
@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # "system", "user", "assistant"
    content: str