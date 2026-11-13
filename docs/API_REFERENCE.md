# ForgeDataset API Reference

Complete Python API specification for `ForgeDataset`.

---

## 1. Ingestion (`ForgeDataset.ingestion`)

### `IngestionRouter`
Routes raw files to appropriate extraction parsers.

```python
from ForgeDataset import IngestionRouter

router = IngestionRouter(config=None)
canonical_doc = router.route("path/to/document.pdf")
```

#### Methods
- `route(file_path: str) -> CanonicalDocument`: Analyzes file type, extracts structural elements, and returns a normalized `CanonicalDocument`.

---

## 2. Chunking (`ForgeDataset.chunking`)

### `HybridChunker`
Chunks documents preserving semantic boundaries and heading hierarchies with **zero token overlap**.

```python
from ForgeDataset import HybridChunker

chunker = HybridChunker(
    max_source_context=600,
    min_merge_size=150,
    merge_peers=True,
    repeat_table_headers=True,
    overlap=0
)
chunks = chunker.chunk_document(canonical_doc)
```

#### Parameters
- `max_source_context` (int, default `600`): Maximum token budget per chunk.
- `min_merge_size` (int, default `150`): Threshold below which peer chunks are merged.
- `merge_peers` (bool, default `True`): Merge adjacent small elements under the same heading.
- `repeat_table_headers` (bool, default `True`): Duplicate table column headers when splitting large tables.
- `overlap` (int, default `0`): Token overlap (recommended `0` for SFT).

---

## 3. SFT Generation (`ForgeDataset.generation`)

### `SFTGenerator`
Generates instruction-response pairs with strictly traceable verbatim evidence.

```python
from ForgeDataset import SFTGenerator

generator = SFTGenerator(
    engine="ollama", # or "llama_cpp", "openai", "heuristic"
    model="qwen2.5:7b-instruct",
    **engine_kwargs
)
candidate_pairs = generator.generate_from_chunk(chunk)
```

#### Methods
- `generate_from_chunk(chunk: Chunk) -> List[QAPair]`: Generates candidates for a single chunk.
- `unload_generator()`: Releases model from GPU VRAM if supported.

---

## 4. Evaluation & Quality Gate (`ForgeDataset.evaluation`)

### `QualityGate`
Coordinates verification, numeric validation, and semantic deduplication.

```python
from ForgeDataset import QualityGate

gate = QualityGate(config=None)
evaluated_pair = gate.evaluate(candidate_qa_pair, chunk)
```

### Components
- `GroundednessVerifier`: Verifies exact verbatim evidence substring existence in source chunk.
- `NumericValidator`: Prevents hallucinated numbers, stats, and units.
- `SemanticDeduplicator`: Employs sentence embeddings cosine similarity to detect duplicate prompts.

---

## 5. Exporting (`ForgeDataset.export`)

### `DatasetExporter`
Splits and exports fine-tuning datasets.

```python
from ForgeDataset import DatasetExporter

exporter = DatasetExporter(output_dir="output")
results = exporter.export(
    evaluated_pairs,
    format_type="alpaca", # or "sharegpt", "raw_jsonl"
    prefix="dataset"
)
```

#### Output Structure
```
output/
├── dataset_train_alpaca.jsonl
├── dataset_val_alpaca.jsonl
├── dataset_test_alpaca.jsonl
└── dataset_audit_provenance.jsonl
```

---

## 6. Schemas (`ForgeDataset.normalization.schemas`)

- `CanonicalDocument`: Unified representation of parsed document.
- `DocumentElement`: Individual heading, paragraph, table, slide, or code block.
- `TableData`: Tabular structure with headers, rows, formulas, and cells.
- `Chunk`: Token-budgeted text slice with `heading_path` and `element_ids`.
- `QAPair`: Generated prompt/completion pair with `evidence` and `status` (`APPROVED`, `NEEDS_REVIEW`, `REJECTED`).
- `Evidence`: Character-level citation span (`start_char`, `end_char`, `page`).

