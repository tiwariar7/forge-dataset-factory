import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from ForgeDataset.normalization.schemas import (
    CanonicalDocument, DocumentElement, TableData, Chunk, QAPair,
    QACategory, QAStatus, ExtractionEngine, ElementType, AlpacaRecord, ShareGPTRecord, ShareGPTMessage
)
from ForgeDataset.normalization.provenance import (
    compute_file_hash, locate_evidence_in_content, create_audit_record
)


def test_canonical_document_schema():
    doc = CanonicalDocument(
        filename="test.md",
        file_type="md",
        file_size_bytes=1024,
        source_hash="dummyhash123",
        extracted_by=ExtractionEngine.DOCLING,
        elements=[
            DocumentElement(
                type=ElementType.HEADING,
                content="Chapter 1: Neural Networks",
                heading_path=["Chapter 1: Neural Networks"],
                level=1
            ),
            DocumentElement(
                type=ElementType.PARAGRAPH,
                content="An artificial neuron receives several inputs and produces an output.",
                heading_path=["Chapter 1: Neural Networks", "1.1 Neurons"]
            )
        ]
    )
    assert len(doc.elements) == 2
    assert doc.elements[0].type == ElementType.HEADING
    assert doc.doc_id.startswith("doc_")


def test_traceable_evidence_provenance():
    source = "SQL window functions perform calculations across related rows without collapsing rows."
    ev_text = "perform calculations across related rows"
    found, start, end = locate_evidence_in_content(ev_text, source)
    assert found is True
    assert start is not None
    assert source[start:end] == ev_text

    # Missing evidence should fail
    found_missing, _, _ = locate_evidence_in_content("Kafka distributed log", source)
    assert found_missing is False


def test_audit_record_creation():
    chunk = Chunk(
        doc_id="doc_123",
        source_filename="database.pdf",
        content="SQL window functions perform calculations.",
        token_count=15,
        heading_path=["SQL", "Window Functions"],
        page_start=24
    )
    qa = QAPair(
        chunk_id=chunk.chunk_id,
        doc_id="doc_123",
        qa_type=QACategory.DEFINITION,
        question="What are window functions?",
        answer="They perform calculations.",
        evidence=[{"text": "SQL window functions perform calculations."}],
        groundedness_score=0.95,
        status=QAStatus.APPROVED
    )
    audit = create_audit_record(qa, chunk)
    assert audit.source == "database.pdf"
    assert audit.page == 24
    assert audit.qa_type == "definition"
    assert audit.verification["grounded"] is True
    assert audit.verification["status"] == "approved"
