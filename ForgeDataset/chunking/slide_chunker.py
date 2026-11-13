"""Slide-aware chunking engine for presentations.
One slide is the starting unit; merges adjacent slides only when sharing topic within token budget.
"""

from typing import List
from ..normalization.schemas import DocumentElement, Chunk, CanonicalDocument


class SlideChunker:
    """Slide-aware chunker respecting presentation slide boundaries."""

    def __init__(self, max_tokens: int = 600):
        self.max_tokens = max_tokens

    def chunk_slides(self, elements: List[DocumentElement], doc: CanonicalDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        slide_elements = [e for e in elements if e.type.value == "slide" or "slide" in e.heading_path[0].lower()]

        if not slide_elements:
            slide_elements = elements

        curr_title: str = ""
        curr_slide_texts: List[str] = []
        curr_slide_nums: List[int] = []

        def count_tokens(text: str) -> int:
            return max(1, int(len(text) / 3.8))

        def flush_slide_chunk():
            nonlocal curr_title, curr_slide_texts, curr_slide_nums
            if not curr_slide_texts:
                return
            combined_text = "\n\n---\n\n".join(curr_slide_texts)
            chunks.append(Chunk(
                doc_id=doc.doc_id,
                source_filename=doc.filename,
                content=combined_text,
                token_count=count_tokens(combined_text),
                heading_path=["Presentation", curr_title or f"Slide {curr_slide_nums[0]}"],
                page_start=curr_slide_nums[0] if curr_slide_nums else 1,
                page_end=curr_slide_nums[-1] if curr_slide_nums else 1,
                chunk_type="slide",
                metadata={"slides": list(curr_slide_nums), "title": curr_title}
            ))
            curr_slide_texts = []
            curr_slide_nums = []

        for elem in slide_elements:
            slide_no = elem.page or elem.metadata.get("slide_number", 1)
            title = elem.metadata.get("title", elem.heading_path[-1] if elem.heading_path else "")

            candidate_text = f"Slide {slide_no}: {title}\n{elem.content}"
            elem_tokens = count_tokens(candidate_text)

            # Check if same topic/title and fits in token budget
            can_merge = (
                len(curr_slide_texts) > 0 and
                (title == curr_title or not curr_title) and
                (count_tokens("\n\n".join(curr_slide_texts)) + elem_tokens <= self.max_tokens)
            )

            if can_merge:
                curr_slide_texts.append(candidate_text)
                curr_slide_nums.append(slide_no)
            else:
                flush_slide_chunk()
                curr_title = title
                curr_slide_texts = [candidate_text]
                curr_slide_nums = [slide_no]

        flush_slide_chunk()
        return chunks

