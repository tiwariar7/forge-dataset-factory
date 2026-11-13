"""MarkItDown Fallback Parser.
Alternative conversion path when Docling extraction needs checking or yields poor quality.
"""

import os
from typing import List
from ..normalization.schemas import CanonicalDocument, DocumentElement, ElementType, ExtractionEngine
from ..normalization.provenance import compute_file_hash


class MarkItDownFallbackParser:
    """Alternative parser when primary extraction fails or needs verification."""

    def __init__(self):
        self._markitdown_available = False
        try:
            import markitdown  # type: ignore
            self._markitdown_available = True
        except ImportError:
            self._markitdown_available = False

    def parse(self, file_path: str) -> CanonicalDocument:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        file_hash = compute_file_hash(file_bytes)
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        file_size = len(file_bytes)

        markdown_text = ""
        if self._markitdown_available:
            try:
                from markitdown import MarkItDown  # type: ignore
                md = MarkItDown()
                result = md.convert(file_path)
                markdown_text = result.text_content
            except Exception:
                markdown_text = file_bytes.decode("utf-8", errors="replace")
        else:
            markdown_text = file_bytes.decode("utf-8", errors="replace")

        elements = self._to_elements(markdown_text)

        return CanonicalDocument(
            filename=filename,
            file_type=ext,
            file_size_bytes=file_size,
            source_hash=file_hash,
            extracted_by=ExtractionEngine.MARKITDOWN,
            extraction_quality_score=0.90 if len(markdown_text) > 100 else 0.40,
            elements=elements,
            tables=[],
            metadata={"fallback_used": True}
        )

    def _to_elements(self, markdown_text: str) -> List[DocumentElement]:
        elements: List[DocumentElement] = []
        lines = markdown_text.split("\n")
        current_heading = ["Document Root"]
        buf = []

        for line in lines:
            if line.startswith("#"):
                if buf:
                    elements.append(DocumentElement(
                        type=ElementType.PARAGRAPH,
                        content="\n".join(buf).strip(),
                        heading_path=list(current_heading)
                    ))
                    buf = []
                title = line.lstrip("#").strip()
                current_heading = ["Document Root", title]
                elements.append(DocumentElement(
                    type=ElementType.HEADING,
                    content=title,
                    heading_path=list(current_heading)
                ))
            else:
                if line.strip():
                    buf.append(line)
                elif buf:
                    elements.append(DocumentElement(
                        type=ElementType.PARAGRAPH,
                        content="\n".join(buf).strip(),
                        heading_path=list(current_heading)
                    ))
                    buf = []

        if buf:
            elements.append(DocumentElement(
                type=ElementType.PARAGRAPH,
                content="\n".join(buf).strip(),
                heading_path=list(current_heading)
            ))

        return elements

