# ForgeDataset ⚡

<p align="center">
  <b>100% Local, Hybrid Document-to-SFT Dataset Factory Python Library</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue" alt="Python Versions" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/inference-llama.cpp%20%7C%20Ollama%20%7C%20vLLM-orange" alt="Inference Engines" />
  <img src="https://img.shields.io/badge/provenance-strict%20verbatim-purple" alt="Provenance" />
</p>

<p align="center">
  <img src="assets/architecture.png" alt="ForgeDataset Architecture" width="460" />
</p>

**ForgeDataset** transforms complex, uncurated enterprise documents (PDF, DOCX, XLSX, PPTX, Markdown, CSV, Code) into production-grade Supervised Fine-Tuning (SFT) datasets for training LLMs. 

Unlike naive parsers that dump raw chunks into generic prompt templates, ForgeDataset implements **intelligent document routing**, **zero-token-overlap hybrid chunking**, **pluggable local LLM engines (llama.cpp, Ollama, vLLM)**, and **strict character-level evidence verification**.

---

## 🌟 Key Features

- **Multi-Engine LLM Support**: Choose whatever inference backend fits your setup:
  - 🦙 **`llama.cpp`**: Direct in-process GGUF model execution (no background server needed).
  - 🦙 **`Ollama`**: Local REST daemon with automatic VRAM purging (`keep_alive: 0`).
  - 🚀 **`vLLM` / `LM Studio` / `OpenAI-Compatible`**: High-throughput `/v1/chat/completions`.
  - ⚡ **`Heuristic`**: Instant zero-dependency rule-based engine for CI/CD and offline tests.
- **Intelligent Ingestion Routing**: Chooses the best parser per format (Docling, OpenPyXL, native Markdown/CSV, OCR fallback).
- **Structural Hybrid Chunking**: Respects document hierarchy (`heading_path`) with **zero token overlap** to eliminate data bleed.
- **Non-Negotiable Evidence Grounding**: Every prompt/completion pair must cite exact verbatim substrings from the source document.
- **Symbolic & Numeric Validation**: Verifies numbers, percentages, and units against the source chunk to eliminate numerical hallucinations.
- **Semantic Deduplication**: Uses `sentence-transformers` embeddings to detect and drop duplicate instructions.
- **Training-Ready Exports**: Outputs Alpaca, ShareGPT, and Audit Provenance JSONL with document-level train/val/test splits.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/tiwariar7/forge-dataset-factory.git
cd forge-dataset-factory

# Install base package
pip install -e .

# Or install with llama.cpp GGUF support
pip install -e ".[llamacpp]"

# Or install all extras (FastAPI server, llama.cpp, dev tools)
pip install -e ".[all]"
```

---

## 🚀 Quickstart: Python Library Usage

```python
from ForgeDataset import (
    IngestionRouter,
    HybridChunker,
    SFTGenerator,
    QualityGate,
    DatasetExporter,
)

# 1. Ingest document (auto-routed to Docling / OpenPyXL / Native)
router = IngestionRouter()
doc = router.route("my_document.pdf")

# 2. Hybrid Chunking (heading-aware, zero token overlap)
chunker = HybridChunker(max_source_context=600, overlap=0)
chunks = chunker.chunk_document(doc)

# 3. SFT Generation (pluggable: llama_cpp, ollama, openai, heuristic)
generator = SFTGenerator(
    engine="ollama", # or engine="llama_cpp", model="path/to/model.gguf"
    model="qwen2.5:7b-instruct"
)

# 4. Strict Quality Gate & Evidence Verification
gate = QualityGate()
verified_pairs = []

for chunk in chunks:
    candidates = generator.generate_from_chunk(chunk)
    for candidate in candidates:
        evaluated = gate.evaluate(candidate, chunk)
        if evaluated.status == "approved":
            verified_pairs.append((evaluated, chunk))

# 5. Export training splits (Alpaca or ShareGPT)
exporter = DatasetExporter(output_dir="output")
results = exporter.export(verified_pairs, format_type="alpaca", prefix="finance_sft")

print(f"Exported {results['approved_count']} high-quality SFT examples!")
```

---

## 🛠️ Command-Line Interface (CLI)

ForgeDataset provides a comprehensive CLI for running end-to-end dataset generation from the terminal:

```bash
# 1. Process a document directly into an Alpaca-format SFT dataset
forge-dataset process my_report.pdf --engine ollama --model qwen2.5:7b-instruct -o output/dataset.jsonl

# 2. Process using a local GGUF model via llama.cpp
forge-dataset process technical_spec.docx --engine llama_cpp --model models/qwen2.5-7b.gguf

# 3. Check engine availability
forge-dataset engines

# 4. Launch optional local REST API server
forge-dataset server --port 8000
```

---

## 🧠 Supported LLM Inference Engines

| Engine | Description | When to Use |
| :--- | :--- | :--- |
| **`llama_cpp`** | In-process GGUF via `llama-cpp-python` | Single GPU or Apple Silicon (Metal), zero external server needed |
| **`ollama`** | Local HTTP REST API | If you already run the Ollama desktop or daemon |
| **`openai`** | OpenAI-compatible `/v1` endpoint | High-throughput batch inference via `vLLM` or `LM Studio` |
| **`heuristic`** | Deterministic regex fallback | Rapid CI/CD testing with zero model dependencies |

*See [docs/ENGINES.md](docs/ENGINES.md) for full configuration details and benchmarks.*

---

## 📖 Detailed Documentation

- 📐 [**Architecture Deep Dive**](docs/ARCHITECTURE.md): The 6-stage extraction and verification pipeline.
- ⚙️ [**Inference Engines Guide**](docs/ENGINES.md): Setup instructions for `llama.cpp`, `Ollama`, and `vLLM`.
- 📚 [**API Reference**](docs/API_REFERENCE.md): Complete Python API specification for classes, methods, and schemas.

---

## 🧪 Testing

ForgeDataset comes with a comprehensive test suite (36 unit tests + end-to-end smoke tests):

```bash
# Run pytest suite
pytest -v

# Run end-to-end offline smoke test
python tests/smoke_test_e2e.py
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
