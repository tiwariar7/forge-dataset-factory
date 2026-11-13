"""Docling Primary Parser.
Extracts PDF, DOCX, PPTX, Markdown, and other formats into CanonicalDocument.
Includes native structured fallback parser to guarantee zero runtime crashes.
"""

import os
import re
from typing import List, Dict, Any, Optional
from ..normalization.schemas import (
    CanonicalDocument, DocumentElement, TableData, ElementType, ExtractionEngine
)
from ..normalization.provenance import compute_file_hash


class DoclingParser:
    """Primary unified structured extractor wrapper around Docling with resilient native fallback."""

    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        self._docling_available = False
        try:
            import docling  # type: ignore
            self._docling_available = True
        except ImportError:
            self._docling_available = False

    def parse(self, file_path: str) -> CanonicalDocument:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        file_hash = compute_file_hash(file_bytes)
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        file_size = len(file_bytes)

        if self._docling_available:
            try:
                return self._parse_with_docling(file_path, filename, ext, file_size, file_hash)
            except Exception as e:
                # Fallback to native structured parsing
                pass

        return self._parse_with_native(file_path, file_bytes, filename, ext, file_size, file_hash)

    def _parse_with_docling(self, file_path: str, filename: str, ext: str, size: int, f_hash: str) -> CanonicalDocument:
        from docling.document_converter import DocumentConverter  # type: ignore
        converter = DocumentConverter()
        result = converter.convert(file_path)
        doc = result.document

        elements: List[DocumentElement] = []
        tables: List[TableData] = []
        current_heading_path: List[str] = []

        # Iterate Docling items
        for item in doc.iterate_items():
            label = getattr(item, "label", "text").lower()
            text = getattr(item, "text", "") or ""
            page = getattr(item, "page_no", 1)

            if "heading" in label or "title" in label or "section" in label:
                level = getattr(item, "level", 1)
                # Adjust heading path according to level
                if level <= len(current_heading_path):
                    current_heading_path = current_heading_path[:level-1]
                current_heading_path.append(text.strip())

                elements.append(DocumentElement(
                    type=ElementType.HEADING,
                    content=text.strip(),
                    heading_path=list(current_heading_path),
                    page=page,
                    level=level
                ))
            elif "table" in label:
                # Extract table headers & rows if available
                t_data = getattr(item, "data", None)
                headers = []
                rows = []
                if t_data and hasattr(t_data, "columns"):
                    headers = [str(c) for c in t_data.columns]
                    rows = [[str(v) for v in row] for row in t_data.values]
                
                table_obj = TableData(headers=headers, rows=rows, page=page)
                tables.append(table_obj)
                elements.append(DocumentElement(
                    type=ElementType.TABLE,
                    content=text or f"Table with {len(headers)} columns",
                    heading_path=list(current_heading_path),
                    page=page
                ))
            elif "code" in label:
                elements.append(DocumentElement(
                    type=ElementType.CODE,
                    content=text,
                    heading_path=list(current_heading_path),
                    page=page
                ))
            else:
                if text.strip():
                    elements.append(DocumentElement(
                        type=ElementType.PARAGRAPH,
                        content=text.strip(),
                        heading_path=list(current_heading_path),
                        page=page
                    ))

        return CanonicalDocument(
            filename=filename,
            file_type=ext,
            file_size_bytes=size,
            source_hash=f_hash,
            extracted_by=ExtractionEngine.DOCLING,
            extraction_quality_score=0.96,
            elements=elements,
            tables=tables,
            metadata={"total_items": len(elements)}
        )

    def _parse_with_native(self, file_path: str, file_bytes: bytes, filename: str, ext: str, size: int, f_hash: str) -> CanonicalDocument:
        """Native parser preserving document hierarchy for Markdown, text, and common formats."""
        elements: List[DocumentElement] = []
        tables: List[TableData] = []

        if ext in ["md", "markdown", "txt"]:
            text_content = file_bytes.decode("utf-8", errors="replace")
            elements = self._parse_markdown_prose(text_content)
        elif ext == "docx":
            elements, tables = self._parse_docx_native(file_bytes)
        elif ext == "pptx":
            elements = self._parse_pptx_native(file_bytes)
        elif ext == "pdf":
            elements = self._parse_pdf_native(file_bytes)
        else:
            text_content = file_bytes.decode("utf-8", errors="replace")
            elements.append(DocumentElement(
                type=ElementType.PARAGRAPH,
                content=text_content,
                heading_path=["Document Root"],
                page=1
            ))

        total_chars = sum(len(e.content) for e in elements)
        quality = 0.95 if total_chars > 100 else 0.40

        return CanonicalDocument(
            filename=filename,
            file_type=ext,
            file_size_bytes=size,
            source_hash=f_hash,
            extracted_by=ExtractionEngine.NATIVE,
            extraction_quality_score=quality,
            elements=elements,
            tables=tables,
            metadata={"native_fallback": True, "total_elements": len(elements)}
        )

    def _parse_markdown_prose(self, text: str) -> List[DocumentElement]:
        elements: List[DocumentElement] = []
        lines = text.split("\n")
        current_heading_path: List[str] = ["Document"]
        buffer: List[str] = []

        def flush_buffer():
            if buffer:
                content = "\n".join(buffer).strip()
                if content:
                    elements.append(DocumentElement(
                        type=ElementType.PARAGRAPH,
                        content=content,
                        heading_path=list(current_heading_path),
                        page=1
                    ))
                buffer.clear()

        for line in lines:
            h_match = re.match(r'^(#{1,6})\s+(.*)', line)
            if h_match:
                flush_buffer()
                level = len(h_match.group(1))
                title = h_match.group(2).strip()
                if level <= len(current_heading_path):
                    current_heading_path = current_heading_path[:level]
                current_heading_path.append(title)
                elements.append(DocumentElement(
                    type=ElementType.HEADING,
                    content=title,
                    heading_path=list(current_heading_path),
                    level=level,
                    page=1
                ))
            elif line.startswith("```"):
                flush_buffer()
                buffer.append(line)
            else:
                if line.strip():
                    buffer.append(line)
                else:
                    flush_buffer()
        flush_buffer()
        return elements

    def _parse_docx_native(self, file_bytes: bytes) -> tuple[List[DocumentElement], List[TableData]]:
        """Native DOCX parser using zipfile + word/document.xml."""
        import io
        import zipfile
        import xml.etree.ElementTree as ET

        elements: List[DocumentElement] = []
        tables: List[TableData] = []
        current_heading_path: List[str] = ["Document"]

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as z:
                if "word/document.xml" in z.namelist():
                    tree = ET.fromstring(z.read("word/document.xml"))
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

                    for p in tree.findall(".//w:p", ns):
                        # Extract full text
                        texts = [t.text for t in p.findall(".//w:t", ns) if t.text]
                        para_text = "".join(texts).strip()
                        if not para_text:
                            continue

                        # Check if heading
                        p_style = p.find(".//w:pStyle", ns)
                        style_val = p_style.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "") if p_style is not None else ""

                        if "heading" in style_val.lower() or "title" in style_val.lower():
                            level_match = re.search(r'\d+', style_val)
                            level = int(level_match.group()) if level_match else 1
                            if level <= len(current_heading_path):
                                current_heading_path = current_heading_path[:level]
                            current_heading_path.append(para_text)
                            elements.append(DocumentElement(
                                type=ElementType.HEADING,
                                content=para_text,
                                heading_path=list(current_heading_path),
                                level=level
                            ))
                        else:
                            elements.append(DocumentElement(
                                type=ElementType.PARAGRAPH,
                                content=para_text,
                                heading_path=list(current_heading_path)
                            ))
        except Exception as e:
            elements.append(DocumentElement(
                type=ElementType.PARAGRAPH,
                content=f"DOCX native parser note: {str(e)}",
                heading_path=["Document"]
            ))

        return elements, tables

    def _parse_pptx_native(self, file_bytes: bytes) -> List[DocumentElement]:
        """Native PPTX parser extracting slide-by-slide text boxes."""
        import io
        import zipfile
        import xml.etree.ElementTree as ET

        elements: List[DocumentElement] = []
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as z:
                slide_files = sorted([n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")])
                for idx, slide_path in enumerate(slide_files, start=1):
                    tree = ET.fromstring(z.read(slide_path))
                    texts = [t.text for t in tree.findall(".//{http://schemas.openxmlformats.org/drawingml/2006/main}t") if t.text]
                    slide_text = " ".join(texts).strip()
                    title = texts[0] if texts else f"Slide {idx}"

                    elements.append(DocumentElement(
                        type=ElementType.SLIDE,
                        content=slide_text,
                        heading_path=["Presentation", title],
                        page=idx,
                        metadata={"slide_number": idx, "title": title}
                    ))
        except Exception:
            pass
        return elements

    def _parse_pdf_native(self, file_bytes: bytes) -> List[DocumentElement]:
        """Native text-layer extraction for PDF."""
        elements: List[DocumentElement] = []
        # Extract text objects (/BT ... /ET) or readable strings from stream
        import re
        try:
            stream_matches = re.findall(rb'BT\s+(.*?)\s+ET', file_bytes, re.DOTALL)
            extracted_lines = []
            for sm in stream_matches:
                t_matches = re.findall(rb'\((.*?)\)\s*T[jJ]', sm)
                for tm in t_matches:
                    decoded = tm.decode("latin1", errors="replace")
                    if decoded.strip():
                        extracted_lines.append(decoded.strip())

            full_text = " ".join(extracted_lines)
            if full_text.strip():
                elements.append(DocumentElement(
                    type=ElementType.PARAGRAPH,
                    content=full_text,
                    heading_path=["Document", "PDF Stream Content"],
                    page=1
                ))
            else:
                # If pure binary / scanned, record empty text for router OCR check
                elements.append(DocumentElement(
                    type=ElementType.PARAGRAPH,
                    content="",
                    heading_path=["Document"],
                    page=1
                ))
        except Exception:
            pass
        return elements

