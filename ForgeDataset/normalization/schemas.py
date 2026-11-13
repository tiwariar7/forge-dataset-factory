"""Pydantic Canonical Document, Chunk, QA, Audit, and Export Schemas."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime, timezone


class ExtractionEngine(str, Enum):
    DOCLING = "docling"
    MARKITDOWN = "markitdown"
    OPENPYXL = "openpyxl"
    OCR = "ocr"
    NATIVE = "native"


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    SLIDE = "slide"
    CODE = "code"
    LIST_ITEM = "list_item"
    METADATA = "metadata"


class QAStatus(str, Enum):
    APPROVED = "approved"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class QACategory(str, Enum):
    DEFINITION = "definition"
    CONCEPTUAL = "conceptual"
    APPLICATION = "application"
    ANALYTICAL = "analytical"


class DocumentElement(BaseModel):
    element_id: str = Field(default_factory=lambda: f"elem_{uuid.uuid4().hex[:8]}")
    type: ElementType
    content: str
    heading_path: List[str] = Field(default_factory=list)
    page: Optional[int] = None
    level: Optional[int] = None  # Heading level (1, 2, 3...)
    coordinates: Optional[Dict[str, Any]] = None  # BBox or cell coordinates
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TableData(BaseModel):
    table_id: str = Field(default_factory=lambda: f"tbl_{uuid.uuid4().hex[:8]}")
    title: Optional[str] = None
    sheet_name: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    cell_ranges: Optional[str] = None
    formulas: Dict[str, str] = Field(default_factory=dict)
    page: Optional[int] = None


class CanonicalDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:10]}")
    filename: str
    file_type: str
    file_size_bytes: int
    source_hash: str
    extracted_by: ExtractionEngine
    extraction_quality_score: float = 1.0
    elements: List[DocumentElement] = Field(default_factory=list)
    tables: List[TableData] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:8]}")
    doc_id: str
    source_filename: str
    content: str
    token_count: int
    heading_path: List[str] = Field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    element_ids: List[str] = Field(default_factory=list)
    chunk_type: str = "prose"  # prose, table, slide, code
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    text: str
    source_element_id: Optional[str] = None
    page: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class VerificationResult(BaseModel):
    grounded: bool
    status: QAStatus
    score: float = 0.0
    flags: List[str] = Field(default_factory=list)
    numeric_checked: bool = False
    numeric_passed: bool = True
    dedup_checked: bool = False
    dedup_passed: bool = True
    reviewer_notes: Optional[str] = None


class QAPair(BaseModel):
    qa_id: str = Field(default_factory=lambda: f"qa_{uuid.uuid4().hex[:8]}")
    chunk_id: str
    doc_id: str
    qa_type: QACategory
    question: str
    answer: str
    evidence: List[Evidence] = Field(default_factory=list)
    generator_model: str = "ollama-local"
    evaluator_model: Optional[str] = None
    groundedness_score: float = 0.0
    numeric_valid: Optional[bool] = None
    status: QAStatus = QAStatus.NEEDS_REVIEW
    rejection_reason: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    verification: Optional[VerificationResult] = None


# Specific audit output format matching specification
class AuditRecord(BaseModel):
    sample_id: str
    source: str
    page: Optional[int] = None
    chunk_id: str
    qa_type: str
    evidence: List[Dict[str, Any]]
    verification: Dict[str, Any]


# Alpaca training format
class AlpacaRecord(BaseModel):
    instruction: str
    input: str
    output: str


# ShareGPT training format
class ShareGPTMessage(BaseModel):
    from_role: str = Field(alias="from")
    value: str

    model_config = {"populate_by_name": True}


class ShareGPTRecord(BaseModel):
    conversations: List[ShareGPTMessage]

