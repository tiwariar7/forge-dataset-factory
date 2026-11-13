"""SQLite Local Storage Layer for Documents, Chunks, QA Pairs, and Review States."""

import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from ..normalization.schemas import CanonicalDocument, Chunk, QAPair, QAStatus


class DatabaseManager:
    """Manages SQLite storage for local dataset factory jobs and provenance."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path or ":memory:"
        self._shared_conn = None
        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        else:
            abs_dir = os.path.dirname(os.path.abspath(self.db_path))
            if abs_dir:
                os.makedirs(abs_dir, exist_ok=True)
        self._init_tables()

    def _get_connection(self):
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size_bytes INTEGER,
                    source_hash TEXT,
                    extracted_by TEXT,
                    quality_score REAL,
                    element_count INTEGER,
                    table_count INTEGER,
                    created_at TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    source_filename TEXT,
                    content TEXT NOT NULL,
                    token_count INTEGER,
                    heading_path TEXT,
                    page_start INTEGER,
                    page_end INTEGER,
                    chunk_type TEXT,
                    metadata TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS qa_pairs (
                    qa_id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    qa_type TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    generator_model TEXT,
                    evaluator_model TEXT,
                    groundedness_score REAL,
                    numeric_valid INTEGER,
                    status TEXT NOT NULL,
                    rejection_reason TEXT,
                    reviewer_notes TEXT,
                    created_at TEXT,
                    FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id)
                )
            """)
            conn.commit()

    def save_document(self, doc: CanonicalDocument):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents (
                    doc_id, filename, file_type, file_size_bytes, source_hash,
                    extracted_by, quality_score, element_count, table_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.doc_id, doc.filename, doc.file_type, doc.file_size_bytes, doc.source_hash,
                doc.extracted_by.value, doc.extraction_quality_score, len(doc.elements), len(doc.tables), doc.created_at
            ))
            conn.commit()

    def save_chunks(self, chunks: List[Chunk]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for ch in chunks:
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, doc_id, source_filename, content, token_count,
                        heading_path, page_start, page_end, chunk_type, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ch.chunk_id, ch.doc_id, ch.source_filename, ch.content, ch.token_count,
                    json.dumps(ch.heading_path), ch.page_start, ch.page_end, ch.chunk_type, json.dumps(ch.metadata)
                ))
            conn.commit()

    def save_qa_pairs(self, qa_pairs: List[QAPair]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for qa in qa_pairs:
                ev_json = json.dumps([e.model_dump() for e in qa.evidence])
                cursor.execute("""
                    INSERT OR REPLACE INTO qa_pairs (
                        qa_id, chunk_id, doc_id, qa_type, question, answer, evidence,
                        generator_model, evaluator_model, groundedness_score,
                        numeric_valid, status, rejection_reason, reviewer_notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    qa.qa_id, qa.chunk_id, qa.doc_id, qa.qa_type.value if hasattr(qa.qa_type, "value") else str(qa.qa_type),
                    qa.question, qa.answer, ev_json, qa.generator_model, qa.evaluator_model,
                    qa.groundedness_score, 1 if qa.numeric_valid else 0, qa.status.value,
                    qa.rejection_reason, qa.verification.reviewer_notes if qa.verification else None,
                    qa.created_at
                ))
            conn.commit()

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
            return [dict(r) for r in cursor.fetchall()]

    def list_qa_pairs(self, status: Optional[str] = None, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM qa_pairs WHERE 1=1"
            params = []
            if status:
                query += " AND status = ?"
                params.append(status)
            if doc_id:
                query += " AND doc_id = ?"
                params.append(doc_id)
            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            rows = [dict(r) for r in cursor.fetchall()]
            for r in rows:
                r["evidence"] = json.loads(r["evidence"]) if r.get("evidence") else []
            return rows

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["heading_path"] = json.loads(d["heading_path"])
                d["metadata"] = json.loads(d["metadata"])
                return d
            return None

    def update_qa_status(self, qa_id: str, new_status: str, notes: Optional[str] = None, edited_answer: Optional[str] = None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if edited_answer:
                cursor.execute("""
                    UPDATE qa_pairs SET status = ?, reviewer_notes = ?, answer = ? WHERE qa_id = ?
                """, (new_status, notes, edited_answer, qa_id))
            else:
                cursor.execute("""
                    UPDATE qa_pairs SET status = ?, reviewer_notes = ? WHERE qa_id = ?
                """, (new_status, notes, qa_id))
            conn.commit()

    def get_benchmarks(self) -> Dict[str, Any]:
        """Compute the accuracy benchmarks enforced in Section 9."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM qa_pairs")
            total = cursor.fetchone()["total"]

            if total == 0:
                return {
                    "total_generated": 0,
                    "approved_count": 0,
                    "needs_review_count": 0,
                    "rejected_count": 0,
                    "evidence_support_rate": 0.0,
                    "numerical_accuracy": 0.0,
                    "human_acceptance_rate": 0.0,
                    "rejection_rate": 0.0
                }

            cursor.execute("SELECT COUNT(*) as cnt FROM qa_pairs WHERE status = 'approved'")
            approved = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM qa_pairs WHERE status = 'needs_review'")
            needs_review = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM qa_pairs WHERE status = 'rejected'")
            rejected = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM qa_pairs WHERE groundedness_score >= 0.65")
            grounded = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM qa_pairs WHERE numeric_valid = 1")
            num_valid = cursor.fetchone()["cnt"]

            return {
                "total_generated": total,
                "approved_count": approved,
                "needs_review_count": needs_review,
                "rejected_count": rejected,
                "evidence_support_rate": round(grounded / total, 4),
                "numerical_accuracy": round(num_valid / total, 4),
                "human_acceptance_rate": round(approved / total, 4),
                "rejection_rate": round(rejected / total, 4)
            }

