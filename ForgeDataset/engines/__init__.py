"""Pluggable LLM engines for ForgeDataset (Ollama, llama.cpp, vLLM, OpenAI-compatible)."""

from .base import LLMEngine
from .ollama import OllamaEngine
from .llama_cpp import LlamaCppEngine
from .openai_compatible import OpenAICompatibleEngine
from .heuristic import HeuristicEngine
from .factory import get_engine

__all__ = [
    "LLMEngine",
    "OllamaEngine",
    "LlamaCppEngine",
    "OpenAICompatibleEngine",
    "HeuristicEngine",
    "get_engine",
]
