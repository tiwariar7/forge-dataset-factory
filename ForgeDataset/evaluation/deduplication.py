"""Semantic Deduplication Engine.
Combines exact text hashing with local embedding-based cosine similarity.
"""

from typing import List, Set, Tuple
import math
import re
from ..normalization.schemas import QAPair
from ..normalization.provenance import compute_text_hash


class SemanticDeduplicator:
    """Detects and flags duplicate or near-identical questions in candidate dataset."""

    def __init__(self, similarity_threshold: float = 0.88, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.similarity_threshold = similarity_threshold
        self.model_name = embedding_model_name
        self.seen_hashes: Set[str] = set()
        self.seen_questions: List[Tuple[str, QAPair]] = []
        self._st_model = None
        self._init_embedding_model()

    def _init_embedding_model(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st_model = SentenceTransformer(self.model_name)
        except Exception:
            self._st_model = None

    def check_and_record(self, qa: QAPair) -> Tuple[bool, float, str]:
        """Check if QA is a duplicate.
        
        Returns: (is_duplicate: bool, similarity_score: float, reason: str)
        """
        # 1. Exact hash check
        q_hash = compute_text_hash(qa.question)
        if q_hash in self.seen_hashes:
            return True, 1.0, f"Exact question hash duplicate"

        # 2. Semantic comparison against existing questions
        if self._st_model is not None and self.seen_questions:
            try:
                import numpy as np  # type: ignore
                existing_texts = [item[0] for item in self.seen_questions]
                q_emb = self._st_model.encode([qa.question])[0]
                ex_embs = self._st_model.encode(existing_texts)

                # Cosine similarities
                dots = np.dot(ex_embs, q_emb)
                norms = np.linalg.norm(ex_embs, axis=1) * np.linalg.norm(q_emb)
                sims = dots / np.maximum(norms, 1e-9)
                max_idx = int(np.argmax(sims))
                max_sim = float(sims[max_idx])

                if max_sim >= self.similarity_threshold:
                    matched_q = existing_texts[max_idx]
                    return True, max_sim, f"Semantic duplicate (sim={max_sim:.3f}) with: '{matched_q[:60]}...'"
            except Exception:
                pass

        # 3. Fallback: n-gram Jaccard similarity
        for ex_text, _ in self.seen_questions:
            j_sim = self._jaccard_similarity(qa.question, ex_text)
            if j_sim >= self.similarity_threshold:
                return True, j_sim, f"Lexical duplicate (jaccard={j_sim:.3f}) with: '{ex_text[:60]}...'"

        # Not a duplicate -> record
        self.seen_hashes.add(q_hash)
        self.seen_questions.append((qa.question, qa))
        return False, 0.0, ""

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        words1 = set(re.findall(r'\w+', s1.lower()))
        words2 = set(re.findall(r'\w+', s2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0

    def reset(self):
        """Reset deduplication state for a new dataset run."""
        self.seen_hashes.clear()
        self.seen_questions.clear()

