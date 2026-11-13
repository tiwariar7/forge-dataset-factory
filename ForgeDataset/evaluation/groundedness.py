"""Independent Groundedness and Factual Entailment Verifier."""

from typing import Tuple, List, Dict, Any
import re
from ..normalization.schemas import QAPair, Chunk
from ..normalization.provenance import locate_evidence_in_content
from ..engines import get_engine, LLMEngine
from ..generation.prompt_templates import EVALUATOR_SYSTEM_PROMPT, EVALUATOR_USER_PROMPT


class GroundednessVerifier:
    """Evaluates whether candidate answers are supported by cited evidence and source text."""

    def __init__(self, config: Dict[str, Any] = None, engine: Any = None):
        self.config = config or {}
        gen_cfg = self.config.get("generation", {})
        models_cfg = gen_cfg.get("models", {})
        ollama_cfg = gen_cfg.get("ollama", {})

        self.evaluator_model = models_cfg.get("evaluator", "llama3.1:8b")
        self.client = get_engine(
            engine=engine or gen_cfg.get("evaluator_engine", gen_cfg.get("engine", "ollama")),
            config=self.config,
            model=self.evaluator_model
        )
        self.use_llm_evaluator = ollama_cfg.get("use_evaluator_llm", False)

    def verify(self, qa: QAPair, chunk: Chunk) -> Tuple[bool, float, List[str]]:
        """Verify candidate QA groundedness against cited evidence and parent chunk.
        
        Returns: (is_grounded: bool, score: float, flags: List[str])
        """
        flags: List[str] = []

        # 1. Non-negotiable rule: Must have traceable evidence
        if not qa.evidence or not qa.evidence[0].text.strip():
            flags.append("Non-negotiable rule violation: No traceable evidence provided.")
            return False, 0.0, flags

        primary_evidence = qa.evidence[0].text.strip()

        # 2. Traceability check: does evidence exist in chunk?
        found, start, end = locate_evidence_in_content(primary_evidence, chunk.content)
        if not found:
            flags.append("Traceability failure: Cited evidence not found in source chunk.")
            return False, 0.20, flags

        # 3. Deterministic lexical overlap check between answer and evidence/chunk
        score = self._compute_entailment_overlap(qa.answer, primary_evidence, chunk.content)

        # 4. LLM evaluator check if enabled and available
        if self.use_llm_evaluator and self.client.is_available():
            try:
                prompt = EVALUATOR_USER_PROMPT.format(
                    evidence_text=primary_evidence,
                    question=qa.question,
                    answer=qa.answer
                )
                res = self.client.generate(
                    model=self.evaluator_model,
                    prompt=prompt,
                    system=EVALUATOR_SYSTEM_PROMPT,
                    json_format=True
                )
                llm_grounded = res.get("grounded", True)
                llm_score = float(res.get("score", score))
                # Combine scores
                score = (score * 0.5) + (llm_score * 0.5)
                if not llm_grounded:
                    flags.append(f"LLM Evaluator flagged unsupported claims: {res.get('unsupported_claims', [])}")
            except Exception:
                pass

        is_grounded = score >= 0.65
        return is_grounded, round(score, 4), flags

    def _compute_entailment_overlap(self, answer: str, evidence: str, chunk_content: str) -> float:
        """Compute lexical recall of answer key terms against evidence and full chunk."""
        # Stopwords filter
        stopwords = {
            "a", "an", "the", "in", "on", "of", "to", "for", "is", "are", "was", "were",
            "and", "or", "that", "this", "it", "with", "as", "by", "from", "at"
        }
        ans_words = [w for w in re.findall(r'\b\w+\b', answer.lower()) if w not in stopwords and len(w) > 2]
        if not ans_words:
            return 0.50

        ev_lower = evidence.lower()
        chunk_lower = chunk_content.lower()

        in_evidence = sum(1 for w in ans_words if w in ev_lower)
        in_chunk = sum(1 for w in ans_words if w in chunk_lower)

        ev_ratio = in_evidence / len(ans_words)
        chunk_ratio = in_chunk / len(ans_words)

        # Weighted combination: 70% evidence weight, 30% parent chunk context
        total_score = (ev_ratio * 0.70) + (chunk_ratio * 0.30)
        return min(1.0, max(0.0, total_score))

