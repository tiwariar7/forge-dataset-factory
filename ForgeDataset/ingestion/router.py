"""Intelligent Ingestion Router.
Routes files to specialized parsers and triggers fallback/OCR upon quality checks.
"""

import os
import csv
from typing import Dict, Any, List
from .docling_parser import DoclingParser
from .excel_parser import ExcelParser
from .markitdown_fallback import MarkItDownFallbackParser
from .ocr import OCREngine
from ..normalization.schemas import (
    CanonicalDocument, DocumentElement, TableData, ElementType, ExtractionEngine
)
from ..normalization.provenance import compute_file_hash


class IngestionRouter:
    """Intelligent Router selecting the best extraction strategy per document type and quality."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        ingestion_cfg = self.config.get("ingestion", {})
        self.min_chars = ingestion_cfg.get("quality_check", {}).get("min_extracted_chars", 50)
        self.docling_parser = DoclingParser()
        self.excel_parser = ExcelParser()
        self.fallback_parser = MarkItDownFallbackParser()
        self.ocr_engine = OCREngine()

    def route(self, file_path: str) -> CanonicalDocument:
        """Route file to optimal parser and evaluate extraction quality."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        filename = os.path.basename(file_path)

        # 1. Specialized Excel & Spreadsheet Routing
        if ext in ["xlsx", "xls"]:
            doc = self.excel_parser.parse(file_path)
            return doc

        if ext == "csv":
            return self._parse_csv(file_path)

        # 2. Image / Scanned Formats
        if ext in ["png", "jpg", "jpeg", "tiff", "bmp"]:
            return self.ocr_engine.extract(file_path)

        # 3. Primary Docling Parsing for PDF, DOCX, PPTX, Markdown, Text
        doc = self.docling_parser.parse(file_path)

        # 4. Extraction Quality Check
        is_good_quality = self._evaluate_quality(doc)

        if not is_good_quality:
            # Fallback path
            if ext == "pdf":
                # Scanned or non-searchable PDF -> OCR fallback
                ocr_doc = self.ocr_engine.extract(file_path)
                if self._evaluate_quality(ocr_doc):
                    return ocr_doc

            # Alternative conversion path: MarkItDown fallback
            fallback_doc = self.fallback_parser.parse(file_path)
            if self._evaluate_quality(fallback_doc):
                return fallback_doc

        return doc

    def _evaluate_quality(self, doc: CanonicalDocument) -> bool:
        """Evaluate extraction quality check based on text length and element structure."""
        total_chars = sum(len(e.content) for e in doc.elements)
        if total_chars < self.min_chars:
            return False
        if not doc.elements and not doc.tables:
            return False
        return True

    def _parse_csv(self, file_path: str) -> CanonicalDocument:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            rows = list(reader)

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        filename = os.path.basename(file_path)
        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []

        table = TableData(
            sheet_name="CSV",
            headers=headers,
            rows=data_rows
        )

        elements = [
            DocumentElement(
                type=ElementType.HEADING,
                content=f"Dataset: {filename}",
                heading_path=["CSV", filename],
                level=1
            ),
            DocumentElement(
                type=ElementType.TABLE,
                content=f"CSV table with headers: {', '.join(headers)} and {len(data_rows)} rows",
                heading_path=["CSV", filename],
                metadata={"headers": headers, "rows": len(data_rows)}
            )
        ]

        return CanonicalDocument(
            filename=filename,
            file_type="csv",
            file_size_bytes=len(file_bytes),
            source_hash=compute_file_hash(file_bytes),
            extracted_by=ExtractionEngine.NATIVE,
            extraction_quality_score=0.99,
            elements=elements,
            tables=[table]
        )

