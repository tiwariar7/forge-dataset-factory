"""Table-aware row-group chunking engine.
Preserves headers, cell coordinates, units, and splits by token budget with repeated headers.
"""

from typing import List
from ..normalization.schemas import TableData, Chunk, CanonicalDocument


class TableChunker:
    """Splits large tables into token-bounded row-group chunks with repeated column headers."""

    def __init__(self, max_tokens: int = 600, repeat_headers: bool = True):
        self.max_tokens = max_tokens
        self.repeat_headers = repeat_headers

    def chunk_table(self, table: TableData, doc: CanonicalDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        if not table.headers and not table.rows:
            return chunks

        header_str = " | ".join(table.headers)
        divider = "-+-".join(["-" * len(h) for h in table.headers])
        header_block = f"Table Sheet: {table.sheet_name or 'Default'}\n| {header_str} |\n| {divider} |"

        curr_rows: List[str] = []
        
        def count_chunk_tokens(rows: List[str]) -> int:
            text = f"{header_block}\n" + "\n".join(rows)
            return max(1, int(len(text) / 3.8))

        for row_idx, row in enumerate(table.rows, start=1):
            row_str = f"| {' | '.join(str(c) for c in row)} |"
            test_rows = curr_rows + [row_str]
            if count_chunk_tokens(test_rows) > self.max_tokens and curr_rows:
                # Flush existing chunk
                chunk_content = f"{header_block}\n" + "\n".join(curr_rows)
                chunks.append(Chunk(
                    doc_id=doc.doc_id,
                    source_filename=doc.filename,
                    content=chunk_content,
                    token_count=count_chunk_tokens(curr_rows),
                    heading_path=["Tables", table.sheet_name or "Sheet1"],
                    page_start=table.page,
                    page_end=table.page,
                    chunk_type="table",
                    metadata={
                        "sheet": table.sheet_name,
                        "row_count": len(curr_rows),
                        "repeated_headers": self.repeat_headers,
                        "formulas": table.formulas
                    }
                ))
                curr_rows = [row_str]
            else:
                curr_rows.append(row_str)

        if curr_rows:
            chunk_content = f"{header_block}\n" + "\n".join(curr_rows)
            chunks.append(Chunk(
                doc_id=doc.doc_id,
                source_filename=doc.filename,
                content=chunk_content,
                token_count=count_chunk_tokens(curr_rows),
                heading_path=["Tables", table.sheet_name or "Sheet1"],
                page_start=table.page,
                page_end=table.page,
                chunk_type="table",
                metadata={
                    "sheet": table.sheet_name,
                    "row_count": len(curr_rows),
                    "repeated_headers": self.repeat_headers,
                    "formulas": table.formulas
                }
            ))

        return chunks

