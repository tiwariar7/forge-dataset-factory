"""Quality Gate Engine.
Routes candidate QA pairs to Approved, Needs Review, or Rejected based on multi-factor validation.
"""

from typing import Dict, Any, List
from ..normalization.schemas import QAPair, Chunk, QAStatus, VerificationResult
from .groundedness import GroundednessVerifier
from .numeric_validator import NumericValidator
from .deduplication import SemanticDeduplicator


class QualityGate:
    """Enforces non-negotiable traceability, factual grounding, numerical accuracy, and deduplication."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        eval_cfg = self.config.get("evaluation", {})
        qg_cfg = eval_cfg.get("quality_gate", {})
        self.auto_approve_thresh = qg_cfg.get("auto_approve_threshold", 0.85)
        self.review_thresh = qg_cfg.get("review_threshold", 0.60)

        self.groundedness_verifier = GroundednessVerifier(config)
        self.numeric_validator = NumericValidator()
        self.deduplicator = SemanticDeduplicator(
            similarity_threshold=eval_cfg.get("deduplication", {}).get("semantic_similarity_threshold", 0.88)
        )

    def evaluate(self, qa: QAPair, chunk: Chunk) -> QAPair:
        """Run all verification stages on a candidate QA pair."""
        flags: List[str] = []

        # 1. Groundedness and Traceability Check
        grounded, score, g_flags = self.groundedness_verifier.verify(qa, chunk)
        flags.extend(g_flags)
        qa.groundedness_score = score

        # 2. Pure Python Deterministic Numeric Check
        num_passed, num_flags = self.numeric_validator.validate(qa.answer, chunk.content)
        flags.extend(num_flags)
        qa.numeric_valid = num_passed

        # 3. Semantic Deduplication Check
        is_dup, dup_score, dup_reason = self.deduplicator.check_and_record(qa)
        if is_dup:
            flags.append(dup_reason)

        # 4. Multi-factor Status Resolution
        if not grounded or not num_passed or is_dup:
            qa.status = QAStatus.REJECTED
            qa.rejection_reason = " | ".join(flags)
        elif score >= self.auto_approve_thresh:
            qa.status = QAStatus.APPROVED
            qa.rejection_reason = None
        elif score >= self.review_thresh:
            qa.status = QAStatus.NEEDS_REVIEW
            qa.rejection_reason = " | ".join(flags) if flags else "Moderate groundedness score; requires human review"
        else:
            qa.status = QAStatus.REJECTED
            qa.rejection_reason = f"Groundedness score {score:.2f} below review threshold {self.review_thresh}"

        # Attach detailed verification result
        qa.verification = VerificationResult(
            grounded=grounded,
            status=qa.status,
            score=score,
            flags=flags,
            numeric_checked=True,
            numeric_passed=num_passed,
            dedup_checked=True,
            dedup_passed=not is_dup
        )

        return qa

