"""FastAPI Endpoints for Document Ingestion, Chunking, QA Generation, Review, and Export."""

import os
import shutil
import yaml
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..ingestion.router import IngestionRouter
from ..chunking.hybrid_chunker import HybridChunker
from ..generation.sft_generator import SFTGenerator
from ..evaluation.quality_gate import QualityGate
from ..export.exporter import DatasetExporter
from ..storage.db import DatabaseManager
from ..normalization.schemas import QAStatus, Chunk, QAPair

router = APIRouter(prefix="/api")

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
config: Dict[str, Any] = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

configured_db = config.get("app", {}).get("database_url", ":memory:").replace("sqlite:///", "")
if not configured_db or "dataset-factory" in configured_db:
    configured_db = ":memory:"
db = DatabaseManager(configured_db)
ingestion_router = IngestionRouter(config)
hybrid_chunker = HybridChunker(
    max_source_context=config.get("chunking", {}).get("max_source_context", 600),
    min_merge_size=config.get("chunking", {}).get("min_merge_size", 150),
    merge_peers=config.get("chunking", {}).get("merge_peers", True),
    repeat_table_headers=config.get("chunking", {}).get("repeat_table_headers", True),
    overlap=config.get("chunking", {}).get("overlap", 0)
)
qa_generator = SFTGenerator(config)
sft_generator = qa_generator
quality_gate = QualityGate(config)
exporter = DatasetExporter(config.get("app", {}).get("output_dir", "dataset-factory/output"))


class ReviewActionRequest(BaseModel):
    status: str  # approved, rejected, needs_review
    notes: Optional[str] = None
    edited_answer: Optional[str] = None


class ExportRequest(BaseModel):
    format: str = "alpaca"  # alpaca, sharegpt, raw_jsonl
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@router.get("/health")
def health_check():
    import os
    skip = os.environ.get("OLLAMA_SKIP_CHECK", "").lower() in ("1", "true", "yes")
    ollama_ready = False if skip else qa_generator.client.is_available()
    models = qa_generator.client.list_models() if ollama_ready else []
    return {
        "status": "healthy",
        "ollama_connected": ollama_ready,
        "available_models": [m.get("name") for m in models],
        "configured_generator": qa_generator.generator_model,
        "configured_evaluator": quality_gate.groundedness_verifier.evaluator_model
    }


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    input_dir = config.get("app", {}).get("input_dir", "dataset-factory/input")
    os.makedirs(input_dir, exist_ok=True)
    target_path = os.path.join(input_dir, file.filename)

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Route through intelligent extraction
    canonical_doc = ingestion_router.route(target_path)
    db.save_document(canonical_doc)

    return {
        "message": "File parsed successfully into Canonical Document",
        "doc_id": canonical_doc.doc_id,
        "filename": canonical_doc.filename,
        "file_type": canonical_doc.file_type,
        "extracted_by": canonical_doc.extracted_by.value,
        "quality_score": canonical_doc.extraction_quality_score,
        "elements_count": len(canonical_doc.elements),
        "tables_count": len(canonical_doc.tables)
    }


@router.post("/process/{doc_id}")
def process_document(doc_id: str):
    """Run full pipeline: Chunking -> Ollama QA Generation -> Verification -> Quality Gate."""
    docs = [d for d in db.list_documents() if d["doc_id"] == doc_id]
    if not docs:
        raise HTTPException(status_code=404, detail="Document not found in registry")

    doc_meta = docs[0]
    input_dir = config.get("app", {}).get("input_dir", "dataset-factory/input")
    target_path = os.path.join(input_dir, doc_meta["filename"])

    # Re-parse or load canonical doc
    canonical_doc = ingestion_router.route(target_path)
    canonical_doc.doc_id = doc_id

    # 1. Hybrid Chunking
    chunks = hybrid_chunker.chunk_document(canonical_doc)
    db.save_chunks(chunks)

    # 2. Sequential QA Generation and Quality Gate
    all_evaluated_qa: List[QAPair] = []
    for chunk in chunks:
        candidates = qa_generator.generate_from_chunk(chunk)
        for cand in candidates:
            evaluated = quality_gate.evaluate(cand, chunk)
            all_evaluated_qa.append(evaluated)

    # Unload generator model to free VRAM for evaluation/other tasks
    qa_generator.unload_generator()

    db.save_qa_pairs(all_evaluated_qa)

    approved = [q for q in all_evaluated_qa if q.status == QAStatus.APPROVED]
    review = [q for q in all_evaluated_qa if q.status == QAStatus.NEEDS_REVIEW]
    rejected = [q for q in all_evaluated_qa if q.status == QAStatus.REJECTED]

    return {
        "doc_id": doc_id,
        "chunks_created": len(chunks),
        "qa_pairs_generated": len(all_evaluated_qa),
        "approved": len(approved),
        "needs_review": len(review),
        "rejected": len(rejected)
    }


@router.get("/documents")
def list_documents():
    return db.list_documents()


@router.get("/chunks/{doc_id}")
def get_chunks(doc_id: str):
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE doc_id = ?", (doc_id,))
        return [dict(r) for r in cursor.fetchall()]


@router.get("/qa")
def list_qa_pairs(status: Optional[str] = None, doc_id: Optional[str] = None):
    return db.list_qa_pairs(status=status, doc_id=doc_id)


@router.post("/qa/{qa_id}/review")
def review_qa_pair(qa_id: str, payload: ReviewActionRequest):
    valid_statuses = [s.value for s in QAStatus]
    if payload.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of {valid_statuses}")

    db.update_qa_status(
        qa_id=qa_id,
        new_status=payload.status,
        notes=payload.notes,
        edited_answer=payload.edited_answer
    )
    return {"message": "Review status updated successfully", "qa_id": qa_id, "status": payload.status}


@router.post("/export")
def export_dataset(payload: ExportRequest):
    raw_qa_rows = db.list_qa_pairs()
    pairs = []
    for r in raw_qa_rows:
        chunk_dict = db.get_chunk(r["chunk_id"])
        if not chunk_dict:
            continue
        chunk = Chunk(
            chunk_id=chunk_dict["chunk_id"],
            doc_id=chunk_dict["doc_id"],
            source_filename=chunk_dict["source_filename"],
            content=chunk_dict["content"],
            token_count=chunk_dict["token_count"],
            heading_path=chunk_dict["heading_path"],
            page_start=chunk_dict["page_start"],
            page_end=chunk_dict["page_end"],
            chunk_type=chunk_dict["chunk_type"],
            metadata=chunk_dict["metadata"]
        )
        qa = QAPair(
            qa_id=r["qa_id"],
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            qa_type=r["qa_type"],
            question=r["question"],
            answer=r["answer"],
            evidence=r["evidence"],
            generator_model=r["generator_model"] or "",
            evaluator_model=r["evaluator_model"] or "",
            groundedness_score=r["groundedness_score"] or 0.0,
            numeric_valid=bool(r["numeric_valid"]),
            status=QAStatus(r["status"]),
            rejection_reason=r["rejection_reason"]
        )
        pairs.append((qa, chunk))

    res = exporter.export(
        pairs=pairs,
        split_ratios=(payload.train_ratio, payload.val_ratio, payload.test_ratio),
        format_type=payload.format
    )
    return res


@router.get("/benchmarks")
def get_benchmarks():
    return db.get_benchmarks()

