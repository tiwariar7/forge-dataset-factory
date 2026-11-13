"""Tests for pluggable LLM engines in ForgeDataset."""

import pytest
from ForgeDataset.engines import (
    LLMEngine,
    OllamaEngine,
    LlamaCppEngine,
    OpenAICompatibleEngine,
    HeuristicEngine,
    get_engine,
)
from ForgeDataset.generation.sft_generator import SFTGenerator
from ForgeDataset.normalization.schemas import Chunk


def test_heuristic_engine_generation():
    engine = HeuristicEngine()
    assert engine.is_available() is True
    res = engine.generate("Content: Machine learning models require clean datasets to achieve optimal accuracy.")
    assert "qa_pairs" in res
    assert len(res["qa_pairs"]) >= 1
    assert "question" in res["qa_pairs"][0]
    assert "answer" in res["qa_pairs"][0]


def test_factory_get_engine():
    e_heuristic = get_engine("heuristic")
    assert isinstance(e_heuristic, HeuristicEngine)

    e_ollama = get_engine("ollama", base_url="http://127.0.0.1:11434")
    assert isinstance(e_ollama, OllamaEngine)

    e_openai = get_engine("openai", base_url="http://localhost:8000/v1")
    assert isinstance(e_openai, OpenAICompatibleEngine)

    e_llamacpp = get_engine("llama_cpp", model_path="nonexistent.gguf")
    assert isinstance(e_llamacpp, LlamaCppEngine)


def test_sft_generator_with_heuristic_engine():
    gen = SFTGenerator(engine="heuristic")
    chunk = Chunk(
        chunk_id="chunk_test_engine",
        doc_id="doc_1",
        source_filename="test.md",
        content="Supervised fine-tuning (SFT) trains models on instruction-response pairs to specialize them for domain tasks.",
        token_count=120,
        heading_path=["Introduction", "SFT"],
        page_start=1,
        page_end=1,
        chunk_type="prose"
    )
    pairs = gen.generate_from_chunk(chunk)
    assert len(pairs) >= 1
    assert pairs[0].question
    assert pairs[0].answer
