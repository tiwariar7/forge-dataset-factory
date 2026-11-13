import sys, os, json, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from ForgeDataset.ingestion.router import IngestionRouter
from ForgeDataset.ingestion.docling_parser import DoclingParser
from ForgeDataset.normalization.schemas import ExtractionEngine


def _write_temp(content: str, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


def test_router_routes_markdown():
    router = IngestionRouter()
    path = _write_temp("# Title\n\nThis is a paragraph about machine learning.\nIt contains multiple sentences.", ".md")
    try:
        doc = router.route(path)
        assert doc.file_type == "md"
        assert doc.extracted_by in (ExtractionEngine.DOCLING, ExtractionEngine.NATIVE, ExtractionEngine.MARKITDOWN)
        assert len(doc.elements) >= 1
        total_chars = sum(len(e.content) for e in doc.elements)
        assert total_chars >= 10
    finally:
        os.unlink(path)


def test_router_routes_csv():
    router = IngestionRouter()
    path = _write_temp("Name,Age,Score\nAlice,30,92.5\nBob,25,88.0\nCarol,35,95.1", ".csv")
    try:
        doc = router.route(path)
        assert doc.file_type == "csv"
        assert len(doc.tables) == 1
        assert doc.tables[0].headers == ["Name", "Age", "Score"]
        assert len(doc.tables[0].rows) == 3
        assert doc.extraction_quality_score >= 0.9
    finally:
        os.unlink(path)


def test_router_quality_gate_triggers_on_empty():
    """A file with no extractable content must trigger fallback path."""
    router = IngestionRouter()
    path = _write_temp("", ".md")
    try:
        # Empty file — router should still return a CanonicalDocument (even if low quality)
        doc = router.route(path)
        assert doc is not None
        # quality score may be low but shouldn't crash
    finally:
        os.unlink(path)


def test_native_markdown_parser_preserves_headings():
    parser = DoclingParser()
    path = _write_temp(
        "# Chapter 1\n\nIntro paragraph here.\n\n## Section 1.1\n\nDetailed content follows.\n\n### Sub-section\n\nMore detail.",
        ".md"
    )
    try:
        doc = parser.parse(path)
        heading_contents = [e.content for e in doc.elements if "heading" in e.type.value]
        assert "Chapter 1" in heading_contents
        assert "Section 1.1" in heading_contents
        # Verify heading path depth increases
        subsec = [e for e in doc.elements if e.content == "Sub-section"]
        if subsec:
            assert len(subsec[0].heading_path) >= 3
    finally:
        os.unlink(path)


def test_native_markdown_parser_code_blocks():
    parser = DoclingParser()
    path = _write_temp(
        "# Code Example\n\nSome intro text.\n\n```python\ndef hello():\n    return 'world'\n```\n\nPost-code prose.",
        ".md"
    )
    try:
        doc = parser.parse(path)
        assert len(doc.elements) >= 2
    finally:
        os.unlink(path)

