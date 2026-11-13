"""Deterministic heuristic engine for offline testing without LLM dependencies."""

import re
import logging
from typing import Dict, Any, Optional
from .base import LLMEngine

logger = logging.getLogger("ForgeDataset.engines.heuristic")


class HeuristicEngine(LLMEngine):
    """Generates synthetic QA/SFT pairs from chunk text using regex and lexical parsing.
    
    Used when no local or remote model is configured or available.
    """

    def is_available(self) -> bool:
        return True

    def unload(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_format: bool = True,
        temperature: float = 0.3,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Extract candidate pairs deterministically from prompt context."""
        text_match = re.search(r"Content:\s*(.*?)(?:\n\n===|\Z)", prompt, re.DOTALL)
        source_text = text_match.group(1).strip() if text_match else prompt

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source_text) if len(s.strip()) > 20]
        if not sentences:
            sentences = [source_text[:120]]

        qa_pairs = []
        # Pair 1: Conceptual / Definition
        qa_pairs.append({
            "question": f"What is described regarding {sentences[0][:40].strip()}?",
            "answer": sentences[0],
            "evidence": sentences[0],
            "qa_type": "conceptual"
        })

        # Pair 2: Numeric / Procedural if available
        num_match = re.search(r"\b\d+[\d,.]*\b", source_text)
        if num_match and len(sentences) > 1:
            num_sent = next((s for s in sentences if num_match.group(0) in s), sentences[1])
            qa_pairs.append({
                "question": f"What quantitative value is specified in this context?",
                "answer": num_sent,
                "evidence": num_sent,
                "qa_type": "numeric"
            })
        elif len(sentences) > 1:
            qa_pairs.append({
                "question": f"What key details follow from the discussion?",
                "answer": sentences[1],
                "evidence": sentences[1],
                "qa_type": "factual"
            })

        return {"qa_pairs": qa_pairs}
