"""llama.cpp in-process GGUF engine using llama-cpp-python."""

import json
import logging
from typing import Dict, Any, Optional
from .base import LLMEngine
from .heuristic import HeuristicEngine

logger = logging.getLogger("ForgeDataset.engines.llama_cpp")


class LlamaCppEngine(LLMEngine):
    """Engine executing local GGUF models directly in-process via llama-cpp-python.
    
    Zero server/daemon required. Supports CPU, Apple Silicon (Metal), and NVIDIA CUDA.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        **llama_kwargs: Any
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.llama_kwargs = llama_kwargs
        self._llm = None
        self._fallback = HeuristicEngine()

    def _ensure_loaded(self):
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python is required to use LlamaCppEngine. "
                    "Install it via: pip install llama-cpp-python"
                )
            logger.info(f"Loading GGUF model into memory from: {self.model_path}")
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
                **self.llama_kwargs
            )

    def is_available(self) -> bool:
        import os
        return os.path.exists(self.model_path)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_format: bool = True,
        temperature: float = 0.3,
        **kwargs: Any
    ) -> Dict[str, Any]:
        try:
            self._ensure_loaded()
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            call_kwargs: Dict[str, Any] = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": kwargs.get("max_tokens", 2048),
            }
            if json_format:
                call_kwargs["response_format"] = {"type": "json_object"}

            output = self._llm.create_chat_completion(**call_kwargs)
            content = output["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw_output": content}
        except Exception as e:
            logger.warning(f"LlamaCpp execution failed: {e}. Falling back to heuristic.")
            return self._fallback.generate(prompt=prompt, system=system, json_format=json_format)

    def unload(self) -> bool:
        if self._llm is not None:
            del self._llm
            self._llm = None
            import gc
            gc.collect()
            return True
        return False

