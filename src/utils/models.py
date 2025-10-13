"""
Pydantic models for request/response validation
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class ActionIntent(BaseModel):
    """Parsed action intent from LLM"""
    intent: str = Field(..., description="The action intent type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    
    @field_validator('intent')
    @classmethod
    def validate_intent(cls, v):
        from config.settings import settings
        if v not in settings.SUPPORTED_INTENTS:
            raise ValueError(f"Unsupported intent: {v}")
        return v


class CompareAveragesParams(BaseModel):
    """Parameters for compare_averages intent"""
    column: str
    file1_id: Optional[str] = None
    file2_id: Optional[str] = None
    group_by: Optional[str] = None


class FilterThresholdParams(BaseModel):
    """Parameters for filter_threshold intent"""
    column: str
    operator: str = Field(..., description="Operator: >, <, >=, <=, ==")
    value: float
    file_id: Optional[str] = None
    
    @field_validator('operator')
    @classmethod
    def validate_operator(cls, v):
        if v not in ['>', '<', '>=', '<=', '==', '!=']:
            raise ValueError(f"Invalid operator: {v}")
        return v


class SortParams(BaseModel):
    """Parameters for sort intent"""
    column: str
    ascending: bool = True
    file_id: Optional[str] = None
    limit: Optional[int] = None


class TopNParams(BaseModel):
    """Parameters for top_n intent"""
    column: str
    n: int = Field(..., gt=0, description="Number of top items")
    ascending: bool = False  # False = highest values first
    file_id: Optional[str] = None


class CompareTopParams(BaseModel):
    """Parameters for compare_top intent"""
    column: str
    n: int = Field(..., gt=0)
    file1_id: Optional[str] = None
    file2_id: Optional[str] = None


class ExplainRowParams(BaseModel):
    """Parameters for explain_row intent"""
    query: str = Field(..., description="Semantic query to find row")
    file_id: Optional[str] = None
    top_k: int = Field(default=1, ge=1, le=10)


class DocumentQueryParams(BaseModel):
    """Parameters for document_query intent"""
    query: str = Field(..., description="Question about document content")
    file_id: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=10)


class AnalysisResult(BaseModel):
    """Result from pandas analysis"""
    result_table: List[Dict[str, Any]] = Field(default_factory=list, description="Raw data rows")
    numbers: Dict[str, Any] = Field(default_factory=dict, description="Computed numbers (can include lists)")
    narrative: str = Field(default="", description="Human-readable explanation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    supported_intents: Optional[List[str]] = None
    details: Optional[str] = None


class FileMetadata(BaseModel):
    """Metadata for an ingested CSV file"""
    file_id: str
    filename: str
    num_rows: int
    num_columns: int
    columns: List[str]
    column_types: Dict[str, str]
    numeric_columns: List[str]
    text_columns: List[str]
    ingestion_timestamp: str


class DocumentMetadata(BaseModel):
    """Metadata for an ingested document (TXT/DOCX)"""
    file_id: str
    filename: str
    document_type: str  # 'txt', 'docx', 'pdf'
    num_characters: int
    num_chunks: int
    num_qa_pairs: int
    ingestion_timestamp: str
    has_questions: bool = False


class VectorMetadata(BaseModel):
    """Metadata for a vector in the database (CSV rows/columns)"""
    file_id: str
    row_idx: int
    column_name: Optional[str] = None  # For column summaries
    is_row_vector: bool = True
    original_text: str


class DocumentChunkMetadata(BaseModel):
    """Metadata for a document chunk vector"""
    file_id: str
    chunk_idx: int
    chunk_type: str  # 'text', 'qa_pair', 'table'
    original_text: str
    question_text: Optional[str] = None
    answer_text: Optional[str] = None
    analysis_text: Optional[str] = None