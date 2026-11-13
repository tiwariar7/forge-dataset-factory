"""Structural + Token-aware Hybrid Chunking Engine.
Combines hierarchical document structure with token-bounded chunking and peer merging.
Zero overlap by default to prevent duplicate question generation.
"""

from typing import List, Dict, Any, Optional
import re
from ..normalization.schemas import CanonicalDocument, DocumentElement, Chunk, ElementType
from .table_chunker import TableChunker
from .slide_chunker import SlideChunker
from .code_chunker import CodeChunker


def estimate_token_count(text: str) -> int:
    """Accurately count tokens using tiktoken if available, else standard tokenization estimate."""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # 1 token is roughly 4 characters or ~0.75 words
        words = len(text.split())
        chars = len(text)
        return max(1, int(chars / 3.8))


class HybridChunker:
    """Primary Hybrid Chunker respecting document hierarchy and token constraints."""

    def __init__(
        self,
        max_source_context: int = 600,
        min_merge_size: int = 150,
        merge_peers: bool = True,
        repeat_table_headers: bool = True,
        overlap: int = 0
    ):
        self.max_tokens = max_source_context
        self.min_merge_tokens = min_merge_size
        self.merge_peers = merge_peers
        self.repeat_table_headers = repeat_table_headers
        self.overlap = overlap

        # Specialized sub-chunkers
        self.table_chunker = TableChunker(max_tokens=max_source_context, repeat_headers=repeat_table_headers)
        self.slide_chunker = SlideChunker(max_tokens=max_source_context)
        self.code_chunker = CodeChunker(max_tokens=max_source_context)

    def chunk_document(self, doc: CanonicalDocument) -> List[Chunk]:
        """Route elements to structural chunkers and merge peers within heading boundaries."""
        chunks: List[Chunk] = []

        # 1. Specialized table chunking for structured tables
        if doc.tables:
            for table in doc.tables:
                t_chunks = self.table_chunker.chunk_table(table, doc)
                chunks.extend(t_chunks)

        # 2. Specialized slide chunking if document is presentation
        if doc.file_type == "pptx" or any(e.type == ElementType.SLIDE for e in doc.elements):
            s_chunks = self.slide_chunker.chunk_slides(doc.elements, doc)
            chunks.extend(s_chunks)
            return chunks

        # 3. Structural + Token-aware grouping for prose and code
        curr_heading_path: List[str] = []
        curr_buffer_text: List[str] = []
        curr_elem_ids: List[str] = []
        curr_page_start: Optional[int] = None
        curr_page_end: Optional[int] = None

        def flush_chunk():
            nonlocal curr_buffer_text, curr_elem_ids, curr_page_start, curr_page_end
            if not curr_buffer_text:
                return
            full_text = "\n\n".join(curr_buffer_text).strip()
            if not full_text:
                return

            token_count = estimate_token_count(full_text)
            
            # If oversized, refine by splitting at sentence boundaries
            if token_count > self.max_tokens:
                sub_chunks = self._split_oversized_text(full_text, curr_heading_path, doc, curr_elem_ids, curr_page_start, curr_page_end)
                chunks.extend(sub_chunks)
            else:
                chunk = Chunk(
                    doc_id=doc.doc_id,
                    source_filename=doc.filename,
                    content=full_text,
                    token_count=token_count,
                    heading_path=list(curr_heading_path),
                    page_start=curr_page_start,
                    page_end=curr_page_end,
                    element_ids=list(curr_elem_ids),
                    chunk_type="prose",
                    metadata={"heading_context": " > ".join(curr_heading_path)}
                )
                chunks.append(chunk)

            curr_buffer_text = []
            curr_elem_ids = []
            curr_page_start = None
            curr_page_end = None

        for elem in doc.elements:
            if elem.type == ElementType.TABLE and doc.tables:
                # Handled separately above
                continue

            # Route code elements to code chunker if standalone code
            if elem.type == ElementType.CODE:
                flush_chunk()
                code_chunks = self.code_chunker.chunk_code_element(elem, doc)
                chunks.extend(code_chunks)
                continue

            # Heading element triggers boundary check
            if elem.type == ElementType.HEADING:
                flush_chunk()
                curr_heading_path = list(elem.heading_path)
                curr_page_start = elem.page
                continue

            # Prose / Paragraph element
            elem_tokens = estimate_token_count(elem.content)
            curr_buffer_tokens = estimate_token_count("\n\n".join(curr_buffer_text))

            # Check if adding this element exceeds budget or if heading path diverged
            if self.merge_peers and (curr_buffer_tokens + elem_tokens <= self.max_tokens):
                curr_buffer_text.append(elem.content)
                curr_elem_ids.append(elem.element_id)
                if curr_page_start is None:
                    curr_page_start = elem.page
                curr_page_end = elem.page
            else:
                flush_chunk()
                curr_buffer_text.append(elem.content)
                curr_elem_ids.append(elem.element_id)
                curr_page_start = elem.page
                curr_page_end = elem.page

        flush_chunk()
        return chunks

    def _split_oversized_text(
        self,
        text: str,
        heading_path: List[str],
        doc: CanonicalDocument,
        elem_ids: List[str],
        page_start: Optional[int],
        page_end: Optional[int]
    ) -> List[Chunk]:
        """Split text that exceeds max_tokens at sentence boundaries with zero overlap."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result_chunks: List[Chunk] = []

        sub_buffer: List[str] = []
        for s in sentences:
            test_text = " ".join(sub_buffer + [s])
            if estimate_token_count(test_text) > self.max_tokens and sub_buffer:
                chunk_body = " ".join(sub_buffer)
                result_chunks.append(Chunk(
                    doc_id=doc.doc_id,
                    source_filename=doc.filename,
                    content=chunk_body,
                    token_count=estimate_token_count(chunk_body),
                    heading_path=list(heading_path),
                    page_start=page_start,
                    page_end=page_end,
                    element_ids=list(elem_ids),
                    chunk_type="prose",
                    metadata={"heading_context": " > ".join(heading_path), "split_segment": True}
                ))
                sub_buffer = [s]
            else:
                sub_buffer.append(s)

        if sub_buffer:
            chunk_body = " ".join(sub_buffer)
            result_chunks.append(Chunk(
                doc_id=doc.doc_id,
                source_filename=doc.filename,
                content=chunk_body,
                token_count=estimate_token_count(chunk_body),
                heading_path=list(heading_path),
                page_start=page_start,
                page_end=page_end,
                element_ids=list(elem_ids),
                chunk_type="prose",
                metadata={"heading_context": " > ".join(heading_path)}
            ))

        return result_chunks

