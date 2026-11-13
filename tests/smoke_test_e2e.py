"""End-to-end offline smoke test — no Ollama required.
Exercises: ingestion → chunking → generation (heuristic fallback) → quality gate → export.
"""

import sys, os, tempfile, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ForgeDataset.ingestion.router import IngestionRouter
from ForgeDataset.chunking.hybrid_chunker import HybridChunker
from ForgeDataset.generation.sft_generator import SFTGenerator
from ForgeDataset.evaluation.quality_gate import QualityGate
from ForgeDataset.export.exporter import DatasetExporter
from ForgeDataset.storage.db import DatabaseManager


SAMPLE_DOCUMENT = """\
# SQL Window Functions

## Overview

SQL window functions perform calculations across a set of rows related to the current row,
without collapsing the result set the way GROUP BY aggregations do.
They are defined using the OVER() clause.

## Row Number Function

The ROW_NUMBER() function assigns a unique sequential integer to each row within
a partition of a result set, starting at 1 for the first row in each partition.
It is commonly used for pagination and de-duplication strategies.

## Practical Use Cases

Window functions are ideal when you need running totals, moving averages,
rank within a category, or lead/lag access to adjacent rows.
Unlike subqueries, they execute in a single pass over the dataset improving performance.

## Numerical Example

A dataset with 1500 rows can be efficiently ranked using ROW_NUMBER() without
materializing any temporary tables. Memory overhead is proportional to the partition size.
"""


def run_smoke_test():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write sample document
        doc_path = os.path.join(tmp_dir, "sql_window_functions.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DOCUMENT)

        db_path = os.path.join(tmp_dir, "test_smoke.db")
        output_dir = os.path.join(tmp_dir, "output")

        db = DatabaseManager(db_path=db_path)
        router = IngestionRouter()
        chunker = HybridChunker(max_source_context=600, overlap=0)
        generator = SFTGenerator()  # Uses heuristic fallback when Ollama is offline
        gate = QualityGate()
        exporter = DatasetExporter(output_dir=output_dir)

        print("\n[1/6] Ingesting document via intelligent router...")
        doc = router.route(doc_path)
        db.save_document(doc)
        print(f"      Extracted {len(doc.elements)} elements via {doc.extracted_by.value}")
        assert len(doc.elements) >= 4

        print("\n[2/6] Hybrid chunking (600-token budget, zero overlap)...")
        chunks = chunker.chunk_document(doc)
        db.save_chunks(chunks)
        print(f"      Produced {len(chunks)} chunks")
        for i, c in enumerate(chunks):
            heading = " > ".join(c.heading_path)
            print(f"      Chunk {i+1}: [{c.token_count} tokens] [{c.chunk_type}] {heading}")
        assert len(chunks) >= 2
        assert all(c.token_count <= 650 for c in chunks)  # 50-token tolerance

        print("\n[3/6] Evidence-first QA generation (Ollama fallback mode)...")
        all_candidates = []
        for chunk in chunks:
            candidates = generator.generate_from_chunk(chunk)
            all_candidates.extend([(cand, chunk) for cand in candidates])
        print(f"      Generated {len(all_candidates)} candidate QA pairs")
        assert len(all_candidates) >= 1

        print("\n[4/6] Running Quality Gate (traceability + numeric + dedup)...")
        evaluated_pairs = []
        for qa, chunk in all_candidates:
            qa = gate.evaluate(qa, chunk)
            evaluated_pairs.append((qa, chunk))

        from ForgeDataset.normalization.schemas import QAStatus
        approved = [p for p in evaluated_pairs if p[0].status == QAStatus.APPROVED]
        needs_review = [p for p in evaluated_pairs if p[0].status == QAStatus.NEEDS_REVIEW]
        rejected = [p for p in evaluated_pairs if p[0].status == QAStatus.REJECTED]

        print(f"      Approved:     {len(approved)}")
        print(f"      Needs Review: {len(needs_review)}")
        print(f"      Rejected:     {len(rejected)}")
        db.save_qa_pairs([p[0] for p in evaluated_pairs])

        print("\n[5/6] Exporting Alpaca + ShareGPT JSONL + Audit Trail...")
        # Promote needs_review items to approved for export smoke test
        for qa, _ in needs_review:
            qa.status = QAStatus.APPROVED
        all_exportable = approved + needs_review

        result = exporter.export(all_exportable, format_type="alpaca", prefix="smoke")
        print(f"      Total exportable: {result['approved_count']}")
        for split, fpath in result["files"].items():
            size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
            print(f"      [{split.upper()}] {os.path.basename(fpath)} ({size} bytes)")

        print("\n[6/6] Database Benchmarks...")
        metrics = db.get_benchmarks()
        print(f"      Total generated:     {metrics['total_generated']}")
        print(f"      Evidence support:    {metrics['evidence_support_rate']*100:.1f}%")
        print(f"      Numeric accuracy:    {metrics['numerical_accuracy']*100:.1f}%")
        print(f"      Human acceptance:    {metrics['human_acceptance_rate']*100:.1f}%")

        # Close SQLite connection before temp dir cleanup (Windows file-lock)
        db._get_connection().close()
        import gc; gc.collect()

        print("\n✅  Smoke test PASSED — all 6 pipeline stages completed without error.\n")


if __name__ == "__main__":
    run_smoke_test()
