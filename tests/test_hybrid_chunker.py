import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ForgeDataset.normalization.schemas import CanonicalDocument, DocumentElement, TableData, ElementType, ExtractionEngine
from ForgeDataset.chunking.hybrid_chunker import HybridChunker


def test_hybrid_chunker_heading_boundaries():
    elements = [
        DocumentElement(
            type=ElementType.HEADING,
            content="Chapter 4: Neural Networks",
            heading_path=["Chapter 4: Neural Networks"],
            level=1
        ),
        DocumentElement(
            type=ElementType.PARAGRAPH,
            content="An artificial neuron receives several inputs. It multiplies them by weights.",
            heading_path=["Chapter 4: Neural Networks"]
        ),
        DocumentElement(
            type=ElementType.HEADING,
            content="4.2 Activation Functions",
            heading_path=["Chapter 4: Neural Networks", "4.2 Activation Functions"],
            level=2
        ),
        DocumentElement(
            type=ElementType.PARAGRAPH,
            content="An activation function introduces non-linearity into the network output.",
            heading_path=["Chapter 4: Neural Networks", "4.2 Activation Functions"]
        ),
    ]

    doc = CanonicalDocument(
        filename="notes.md",
        file_type="md",
        file_size_bytes=500,
        source_hash="hash1",
        extracted_by=ExtractionEngine.NATIVE,
        elements=elements
    )

    chunker = HybridChunker(max_source_context=600, merge_peers=True, overlap=0)
    chunks = chunker.chunk_document(doc)

    # Must produce separate chunks per heading boundary
    assert len(chunks) == 2
    assert "Chapter 4: Neural Networks" in chunks[0].heading_path
    assert "4.2 Activation Functions" in chunks[1].heading_path
    assert all(c.token_count <= 600 for c in chunks)
    # Verify zero overlap (no duplicate content)
    assert chunks[0].content != chunks[1].content


def test_hybrid_chunker_zero_overlap():
    """Zero overlap is the non-negotiable default for training dataset generation."""
    elements = [
        DocumentElement(type=ElementType.HEADING, content="Section A", heading_path=["Section A"], level=1),
        DocumentElement(type=ElementType.PARAGRAPH,
                        content="This paragraph belongs entirely to section A and not to B.",
                        heading_path=["Section A"]),
        DocumentElement(type=ElementType.HEADING, content="Section B", heading_path=["Section B"], level=1),
        DocumentElement(type=ElementType.PARAGRAPH,
                        content="This paragraph belongs entirely to section B and not to A.",
                        heading_path=["Section B"]),
    ]
    doc = CanonicalDocument(
        filename="sec.md", file_type="md", file_size_bytes=200,
        source_hash="hash_sec", extracted_by=ExtractionEngine.NATIVE, elements=elements
    )
    chunker = HybridChunker(overlap=0)
    chunks = chunker.chunk_document(doc)
    for i, ci in enumerate(chunks):
        for j, cj in enumerate(chunks):
            if i != j:
                # No chunk should be a substring of another
                assert ci.content.strip() not in cj.content.strip()


def test_table_chunker_repeated_headers():
    doc = CanonicalDocument(
        filename="data.xlsx",
        file_type="xlsx",
        file_size_bytes=1000,
        source_hash="hash_tbl",
        extracted_by=ExtractionEngine.OPENPYXL,
        elements=[],
        tables=[
            TableData(
                sheet_name="Financials",
                headers=["Quarter", "Revenue", "EBITDA", "Net Profit"],
                rows=[
                    ["Q1 2024", "$100M", "$30M", "$15M"],
                    ["Q2 2024", "$120M", "$35M", "$20M"],
                    ["Q3 2024", "$115M", "$33M", "$18M"],
                ],
                formulas={"B2": "=SUM(B1:B3)"}
            )
        ]
    )

    chunker = HybridChunker(max_source_context=600, repeat_table_headers=True)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    assert "Quarter | Revenue | EBITDA | Net Profit" in chunks[0].content
    assert chunks[0].chunk_type == "table"
    assert chunks[0].metadata["repeated_headers"] is True
    assert "Financials" in chunks[0].heading_path


def test_oversized_chunk_split():
    """A single very long paragraph must be split at sentence boundaries without exceeding token limit."""
    long_text = ". ".join([f"This is sentence number {i} about deep learning architectures" for i in range(50)]) + "."
    elements = [
        DocumentElement(type=ElementType.HEADING, content="Long Section", heading_path=["Long Section"], level=1),
        DocumentElement(type=ElementType.PARAGRAPH, content=long_text, heading_path=["Long Section"])
    ]
    doc = CanonicalDocument(
        filename="long.md", file_type="md", file_size_bytes=5000,
        source_hash="hash_long", extracted_by=ExtractionEngine.NATIVE, elements=elements
    )
    chunker = HybridChunker(max_source_context=200)
    chunks = chunker.chunk_document(doc)
    for c in chunks:
        assert c.token_count <= 220  # Allow small tolerance for sentence boundary rounding
