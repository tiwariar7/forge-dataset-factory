"""Evidence-First SFT Generator Orchestrator.
Generates multi-category SFT (Prompt-Completion) pairs with strictly traceable verbatim evidence.
Supports multiple backends: Ollama, llama.cpp (GGUF), vLLM / OpenAI-compatible, or heuristic.
"""

from typing import List, Dict, Any, Optional, Union
import logging
from ..normalization.schemas import Chunk, QAPair, Evidence, QACategory, QAStatus
from ..normalization.provenance import locate_evidence_in_content, enrich_qa_evidence_provenance
from ..engines import get_engine, LLMEngine
from .prompt_templates import SYSTEM_PROMPT_QA_GENERATOR, QA_GENERATION_USER_PROMPT

logger = logging.getLogger("ForgeDataset.generator")


class SFTGenerator:
    """Orchestrates candidate SFT pair generation with strict evidence linkage."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        engine: Optional[Union[str, LLMEngine]] = None,
        model: Optional[str] = None,
        **engine_kwargs: Any
    ):
        self.config = config or {}
        gen_cfg = self.config.get("generation", {})
        models_cfg = gen_cfg.get("models", {})

        self.generator_model = model or models_cfg.get("generator", "qwen2.5:7b-instruct")
        
        # Multi-engine support: Ollama, llama.cpp, vLLM, OpenAI-compatible, heuristic
        self.client = get_engine(
            engine=engine or gen_cfg.get("engine", "ollama"),
            config=self.config,
            model=self.generator_model,
            **engine_kwargs
        )

        self.enforce_sequential = gen_cfg.get("ollama", {}).get("enforce_sequential_models", True)
        self.min_tokens = gen_cfg.get("qa_counts", {}).get("min_chunk_tokens_for_generation", 50)

    def generate_from_chunk(self, chunk: Chunk) -> List[QAPair]:
        """Generate 3 to 6 evidence-grounded candidate SFT pairs for a single chunk."""
        if chunk.token_count < self.min_tokens:
            logger.info(f"Skipping chunk {chunk.chunk_id}: token count {chunk.token_count} below threshold {self.min_tokens}")
            return []

        heading_context = " > ".join(chunk.heading_path) if chunk.heading_path else "General"
        prompt = QA_GENERATION_USER_PROMPT.format(
            source_filename=chunk.source_filename,
            heading_path=heading_context,
            chunk_content=chunk.content
        )

        raw_response = self.client.generate(
            model=self.generator_model,
            prompt=prompt,
            system=SYSTEM_PROMPT_QA_GENERATOR,
            json_format=True,
            unload_after=False
        )

        qa_items = raw_response.get("qa_pairs", [])
        candidates: List[QAPair] = []

        valid_categories = {c.value: c for c in QACategory}

        for item in qa_items:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            raw_evidence = item.get("evidence", "").strip()
            raw_category = item.get("qa_type", "conceptual").lower()

            if not question or not answer:
                continue

            category = valid_categories.get(raw_category, QACategory.CONCEPTUAL)

            # Stage 2: Evidence extraction & strict verbatim provenance check
            evidence_list: List[Evidence] = []
            if raw_evidence:
                found, start, end = locate_evidence_in_content(raw_evidence, chunk.content)
                evidence_list.append(Evidence(
                    text=raw_evidence,
                    source_element_id=chunk.element_ids[0] if chunk.element_ids else None,
                    page=chunk.page_start,
                    start_char=start,
                    end_char=end
                ))

            qa = QAPair(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                qa_type=category,
                question=question,
                answer=answer,
                evidence=evidence_list,
                generator_model=self.generator_model,
                status=QAStatus.NEEDS_REVIEW
            )

            enrich_qa_evidence_provenance(qa, chunk)
            candidates.append(qa)

        return candidates

    def unload_generator(self):
        """Unload generator model from GPU VRAM."""
        if hasattr(self.client, "unload"):
            self.client.unload()


# Backward compatibility alias
QAGenerator = SFTGenerator
