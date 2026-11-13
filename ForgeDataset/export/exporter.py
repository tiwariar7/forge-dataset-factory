"""Dataset Export Engine.
Produces clean training-ready JSONL (Alpaca/ShareGPT) and separate audit trail records.
Strictly splits train/val/test by source document to avoid data leakage.
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple
from ..normalization.schemas import QAPair, Chunk, QAStatus, AlpacaRecord, ShareGPTRecord, ShareGPTMessage
from ..normalization.provenance import create_audit_record


class DatasetExporter:
    """Exports approved dataset examples to Alpaca, ShareGPT, and Audit formats."""

    def __init__(self, output_dir: str = "dataset-factory/output", seed: int = 42):
        self.output_dir = output_dir
        self.seed = seed
        os.makedirs(self.output_dir, exist_ok=True)

    def export(
        self,
        pairs: List[Tuple[QAPair, Chunk]],
        split_ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10),
        format_type: str = "alpaca",
        prefix: str = "dataset"
    ) -> Dict[str, Any]:
        """Export dataset partitioned by source document to prevent cross-contamination."""
        # Filter for approved pairs
        approved = [(qa, chunk) for qa, chunk in pairs if qa.status == QAStatus.APPROVED]

        # Group by doc_id
        doc_groups: Dict[str, List[Tuple[QAPair, Chunk]]] = {}
        for qa, chunk in approved:
            doc_groups.setdefault(qa.doc_id, []).append((qa, chunk))

        doc_ids = sorted(list(doc_groups.keys()))
        random.Random(self.seed).shuffle(doc_ids)

        train_ratio, val_ratio, test_ratio = split_ratios
        n_docs = len(doc_ids)

        n_train = max(1, int(n_docs * train_ratio)) if n_docs > 1 else 1
        n_val = max(0, int(n_docs * val_ratio))

        train_docs = set(doc_ids[:n_train])
        val_docs = set(doc_ids[n_train:n_train + n_val])
        test_docs = set(doc_ids[n_train + n_val:])

        splits = {
            "train": [item for d in train_docs for item in doc_groups.get(d, [])],
            "val": [item for d in val_docs for item in doc_groups.get(d, [])],
            "test": [item for d in test_docs for item in doc_groups.get(d, [])]
        }

        # Export training files
        result_files = {}
        for split_name, items in splits.items():
            if not items and split_name != "train":
                continue

            split_filename = f"{prefix}_{split_name}_{format_type}.jsonl"
            split_path = os.path.join(self.output_dir, split_filename)

            with open(split_path, "w", encoding="utf-8") as f:
                for qa, chunk in items:
                    if format_type == "alpaca":
                        rec = AlpacaRecord(
                            instruction=qa.question,
                            input=chunk.content if "context" in qa.question.lower() else "",
                            output=qa.answer
                        )
                        f.write(rec.model_dump_json() + "\n")
                    elif format_type == "sharegpt":
                        rec = ShareGPTRecord(conversations=[
                            ShareGPTMessage(from_role="human", value=qa.question),
                            ShareGPTMessage(from_role="gpt", value=qa.answer)
                        ])
                        f.write(rec.model_dump_json(by_alias=True) + "\n")
                    else:
                        f.write(json.dumps({
                            "question": qa.question,
                            "answer": qa.answer,
                            "context": chunk.content
                        }) + "\n")

            result_files[split_name] = split_path

        # Export Separate Audit Trail Record
        audit_filename = f"{prefix}_audit_provenance.jsonl"
        audit_path = os.path.join(self.output_dir, audit_filename)
        with open(audit_path, "w", encoding="utf-8") as f:
            for qa, chunk in pairs:
                audit_rec = create_audit_record(qa, chunk)
                f.write(audit_rec.model_dump_json() + "\n")
        result_files["audit"] = audit_path

        return {
            "status": "success",
            "approved_count": len(approved),
            "total_evaluated": len(pairs),
            "files": result_files,
            "document_distribution": {
                "train_docs": len(train_docs),
                "val_docs": len(val_docs),
                "test_docs": len(test_docs)
            }
        }

