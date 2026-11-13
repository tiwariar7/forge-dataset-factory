"""Structure-aware and line-based token chunker for code, lists, and procedures.
Preserves block integrity and logical line boundaries.
"""

from typing import List
from ..normalization.schemas import DocumentElement, Chunk, CanonicalDocument


class CodeChunker:
    """Structure-aware line-based chunker for code blocks and procedural definitions."""

    def __init__(self, max_tokens: int = 600):
        self.max_tokens = max_tokens

    def chunk_code_element(self, elem: DocumentElement, doc: CanonicalDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        code_text = elem.content

        def count_tokens(text: str) -> int:
            return max(1, int(len(text) / 3.8))

        if count_tokens(code_text) <= self.max_tokens:
            chunks.append(Chunk(
                doc_id=doc.doc_id,
                source_filename=doc.filename,
                content=code_text,
                token_count=count_tokens(code_text),
                heading_path=list(elem.heading_path),
                page_start=elem.page,
                page_end=elem.page,
                element_ids=[elem.element_id],
                chunk_type="code",
                metadata={"language": elem.metadata.get("language", "plaintext")}
            ))
            return chunks

        # Oversized code block: split by logical lines / functions
        lines = code_text.split("\n")
        buffer: List[str] = []

        for line in lines:
            test_block = "\n".join(buffer + [line])
            if count_tokens(test_block) > self.max_tokens and buffer:
                content = "\n".join(buffer)
                chunks.append(Chunk(
                    doc_id=doc.doc_id,
                    source_filename=doc.filename,
                    content=content,
                    token_count=count_tokens(content),
                    heading_path=list(elem.heading_path),
                    page_start=elem.page,
                    page_end=elem.page,
                    element_ids=[elem.element_id],
                    chunk_type="code",
                    metadata={"split_code": True}
                ))
                buffer = [line]
            else:
                buffer.append(line)

        if buffer:
            content = "\n".join(buffer)
            chunks.append(Chunk(
                doc_id=doc.doc_id,
                source_filename=doc.filename,
                content=content,
                token_count=count_tokens(content),
                heading_path=list(elem.heading_path),
                page_start=elem.page,
                page_end=elem.page,
                element_ids=[elem.element_id],
                chunk_type="code",
                metadata={"split_code": True}
            ))

        return chunks

