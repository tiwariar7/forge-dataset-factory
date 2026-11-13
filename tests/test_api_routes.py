"""API route unit tests testing endpoint functions directly."""

import sys
import os
import io
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi import UploadFile, HTTPException
from ForgeDataset.api.routes import (
    health_check,
    get_benchmarks,
    list_documents,
    list_qa_pairs,
    review_qa_pair,
    export_dataset,
    upload_document,
    ReviewActionRequest,
    ExportRequest,
)


def test_health_endpoint():
    res = health_check()
    assert res["status"] == "healthy"
    assert "ollama_connected" in res
    assert "configured_generator" in res
    assert "configured_evaluator" in res


def test_benchmarks_endpoint():
    b = get_benchmarks()
    assert "total_generated" in b
    assert "evidence_support_rate" in b
    assert "numerical_accuracy" in b
    assert "human_acceptance_rate" in b


def test_documents_list():
    docs = list_documents()
    assert isinstance(docs, list)


def test_qa_list():
    qas = list_qa_pairs()
    assert isinstance(qas, list)


def test_qa_filter_by_status():
    for status in ["approved", "rejected", "needs_review"]:
        qas = list_qa_pairs(status=status)
        assert isinstance(qas, list)


def test_upload_markdown_file():
    content = b"# Neural Networks\n\nArtificial neurons are basic computational units.\n\n## Perceptrons\n\nInvented in 1958."
    file_obj = UploadFile(
        file=io.BytesIO(content),
        filename="test_upload.md",
        headers={"content-type": "text/markdown"}
    )
    res = upload_document(file=file_obj)
    assert res["filename"] == "test_upload.md"
    assert res["file_type"] == "md"
    assert res["elements_count"] >= 2
    assert res["quality_score"] > 0


def test_review_invalid_status_raises_400():
    req = ReviewActionRequest(status="definitely_invalid")
    with pytest.raises(HTTPException) as exc_info:
        review_qa_pair("dummy_qa_id", req)
    assert exc_info.value.status_code == 400


def test_export_endpoint():
    req = ExportRequest(format="alpaca", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    res = export_dataset(req)
    assert res["status"] == "success"
    assert "files" in res
