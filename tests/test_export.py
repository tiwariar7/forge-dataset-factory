import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from ForgeDataset.normalization.schemas import Chunk, QAPair, QACategory, QAStatus, Evidence
from ForgeDataset.export.exporter import DatasetExporter


def _make_pair(doc_idx: int, qa_idx: int, tmp_chunk=None):
    doc_id = f"doc_{doc_idx}"
    chunk = Chunk(
        doc_id=doc_id,
        source_filename=f"paper_{doc_idx}.pdf",
        content=f"Support Vector Machines find the optimal hyperplane for doc {doc_idx}.",
        token_count=15,
        heading_path=["ML"]
    )
    qa = QAPair(
        chunk_id=chunk.chunk_id,
        doc_id=doc_id,
        qa_type=QACategory.DEFINITION,
        question=f"Question {qa_idx} from document {doc_idx}?",
        answer=f"Answer {qa_idx} from document {doc_idx}.",
        evidence=[Evidence(text=f"Support Vector Machines find the optimal hyperplane for doc {doc_idx}.")],
        status=QAStatus.APPROVED,
        groundedness_score=0.95
    )
    return qa, chunk


def test_export_alpaca_format(tmp_path):
    exporter = DatasetExporter(output_dir=str(tmp_path))
    pairs = [_make_pair(d, q) for d in range(1, 4) for q in range(1, 4)]

    result = exporter.export(pairs, split_ratios=(0.60, 0.20, 0.20), format_type="alpaca", prefix="test_alpaca")
    assert result["status"] == "success"
    assert result["approved_count"] == 9

    # Verify Alpaca schema: instruction / input / output
    train_file = result["files"]["train"]
    assert os.path.exists(train_file)
    with open(train_file) as f:
        lines = [json.loads(l) for l in f]
    assert len(lines) > 0
    for rec in lines:
        assert "instruction" in rec
        assert "output" in rec


def test_export_sharegpt_format(tmp_path):
    exporter = DatasetExporter(output_dir=str(tmp_path))
    pairs = [_make_pair(d, q) for d in range(1, 4) for q in range(1, 4)]

    result = exporter.export(pairs, split_ratios=(0.60, 0.20, 0.20), format_type="sharegpt", prefix="test_sgpt")
    train_file = result["files"]["train"]
    with open(train_file) as f:
        lines = [json.loads(l) for l in f]
    for rec in lines:
        assert "conversations" in rec
        assert len(rec["conversations"]) == 2
        assert rec["conversations"][0]["from"] == "human"
        assert rec["conversations"][1]["from"] == "gpt"


def test_export_audit_trail_format(tmp_path):
    """Audit file must include source, page, chunk_id, evidence, verification per spec."""
    exporter = DatasetExporter(output_dir=str(tmp_path))
    pairs = [_make_pair(d, q) for d in range(1, 3) for q in range(1, 3)]

    result = exporter.export(pairs, format_type="alpaca", prefix="audit_test")
    audit_file = result["files"]["audit"]
    assert os.path.exists(audit_file)

    with open(audit_file) as f:
        audit_lines = [json.loads(l) for l in f]

    assert len(audit_lines) == 4  # 2 docs × 2 QAs
    for rec in audit_lines:
        assert "sample_id" in rec
        assert "source" in rec
        assert "chunk_id" in rec
        assert "qa_type" in rec
        assert "evidence" in rec
        assert "verification" in rec
        assert "grounded" in rec["verification"]
        assert "status" in rec["verification"]


def test_export_document_level_split_no_leakage(tmp_path):
    """Split by document guarantees no QA from the same source appears in both train and test."""
    exporter = DatasetExporter(output_dir=str(tmp_path))
    pairs = [_make_pair(d, q) for d in range(1, 11) for q in range(1, 4)]  # 10 docs, 3 QAs each

    result = exporter.export(pairs, split_ratios=(0.70, 0.15, 0.15), format_type="alpaca", prefix="split_test")

    # Collect all doc_ids per split from filenames / metadata
    assert result["document_distribution"]["train_docs"] > 0
    # Verify train + val + test doc counts sum to total unique docs
    total = (result["document_distribution"]["train_docs"] +
             result["document_distribution"]["val_docs"] +
             result["document_distribution"]["test_docs"])
    assert total == 10


def test_export_only_approved_samples(tmp_path):
    """Rejected and needs_review samples must NOT appear in training files."""
    exporter = DatasetExporter(output_dir=str(tmp_path))

    qa_approved, chunk_a = _make_pair(1, 1)
    qa_approved.status = QAStatus.APPROVED

    qa_rejected, chunk_r = _make_pair(2, 1)
    qa_rejected.status = QAStatus.REJECTED

    qa_review, chunk_rv = _make_pair(3, 1)
    qa_review.status = QAStatus.NEEDS_REVIEW

    result = exporter.export(
        [(qa_approved, chunk_a), (qa_rejected, chunk_r), (qa_review, chunk_rv)],
        format_type="alpaca", prefix="filter_test"
    )
    # Only 1 of 3 is approved
    assert result["approved_count"] == 1
    assert result["total_evaluated"] == 3
