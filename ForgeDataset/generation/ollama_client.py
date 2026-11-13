"""Backward-compatible OllamaClient wrapper delegating to OllamaEngine."""

from typing import Dict, Any, Optional
from ..engines.ollama import OllamaEngine


class OllamaClient(OllamaEngine):
    """Legacy compatibility class for OllamaClient."""
    pass
