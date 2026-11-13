"""Base interface and protocols for LLM engines in ForgeDataset."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union


class LLMEngine(ABC):
    """Abstract Base Class for LLM inference engines.
    
    Any backend (Ollama, llama.cpp, vLLM, OpenAI-compatible, custom)
    must implement this interface to be used across ForgeDataset.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_format: bool = True,
        temperature: float = 0.3,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate a response given a prompt and optional system message.
        
        Args:
            prompt: The user instruction / text input.
            system: Optional system prompt.
            json_format: Whether to request/parse JSON structured output.
            temperature: Sampling temperature (0.0 = deterministic).
            **kwargs: Backend-specific arguments.
            
        Returns:
            Dict containing parsed JSON data (or {"raw_output": text}).
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is reachable / loaded."""
        pass

    def unload(self) -> bool:
        """Unload or free model memory if supported by backend."""
        return True
