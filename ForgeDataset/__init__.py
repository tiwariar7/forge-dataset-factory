"""ForgeDataset: 100% Local, Hybrid Document-to-SFT Dataset Factory Python Library.

Build high-quality Supervised Fine-Tuning (SFT) datasets directly from raw documents
(PDF, DOCX, XLSX, PPTX, Markdown, CSV, Code) with strict evidence provenance,
intelligent routing, and multi-engine LLM support (Ollama, llama.cpp GGUF, vLLM / OpenAI).
"""

from .normalization.schemas import (
    CanonicalDocument,
    DocumentElement,
    TableData,
    Chunk,
    QAPair,
    Evidence,
    AuditRecord,
    QAStatus,
    QACategory,
    ExtractionEngine,
)
from .ingestion.router import IngestionRouter
from .chunking.hybrid_chunker import HybridChunker
from .generation.sft_generator import SFTGenerator, QAGenerator
from .evaluation.quality_gate import QualityGate
from .evaluation.groundedness import GroundednessVerifier
from .evaluation.numeric_validator import NumericValidator
from .evaluation.deduplication import SemanticDeduplicator
from .export.exporter import DatasetExporter
from .storage.db import DatabaseManager
from .engines import (
    LLMEngine,
    get_engine,
    OllamaEngine,
    LlamaCppEngine,
    OpenAICompatibleEngine,
    HeuristicEngine,
)

__version__ = "0.1.0"

__all__ = [
    # Core Pipeline
    "IngestionRouter",
    "HybridChunker",
    "SFTGenerator",
    "QAGenerator",
    "QualityGate",
    "DatasetExporter",
    "DatabaseManager",
    
    # Schemas
    "CanonicalDocument",
    "DocumentElement",
    "TableData",
    "Chunk",
    "QAPair",
    "Evidence",
    "AuditRecord",
    "QAStatus",
    "QACategory",
    "ExtractionEngine",
    
    # Evaluation
    "GroundednessVerifier",
    "NumericValidator",
    "SemanticDeduplicator",
    
    # Engines
    "LLMEngine",
    "get_engine",
    "OllamaEngine",
    "LlamaCppEngine",
    "OpenAICompatibleEngine",
    "HeuristicEngine",
    
    # Metadata
    "__version__",
]
