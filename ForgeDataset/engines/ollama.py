"""Ollama native REST API engine."""

import json
import logging
import os
from typing import Dict, Any, Optional
import httpx
from .base import LLMEngine
from .heuristic import HeuristicEngine

logger = logging.getLogger("ForgeDataset.engines.ollama")


class OllamaEngine(LLMEngine):
    """Engine communicating with local Ollama daemon via HTTP REST API."""

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        enforce_sequential: bool = True
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.enforce_sequential = enforce_sequential
        self._fallback = HeuristicEngine()

    def is_available(self) -> bool:
        if os.environ.get("OLLAMA_SKIP_CHECK", "").lower() in ("1", "true", "yes"):
            return False
        try:
            limits = httpx.Limits(max_connections=1)
            timeout = httpx.Timeout(connect=1.0, read=2.0, write=1.0, pool=1.0)
            with httpx.Client(timeout=timeout, limits=limits) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list:
        if os.environ.get("OLLAMA_SKIP_CHECK", "").lower() in ("1", "true", "yes"):
            return []
        try:
            timeout = httpx.Timeout(connect=1.0, read=2.0, write=1.0, pool=1.0)
            with httpx.Client(timeout=timeout, limits=httpx.Limits(max_connections=1)) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    return res.json().get("models", [])
        except Exception:
            pass
        return []

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_format: bool = True,
        temperature: float = 0.3,
        unload_after: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        model_name = kwargs.get("model", self.model)
        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.9),
            }
        }
        if json_format:
            payload["format"] = "json"
        if unload_after:
            payload["keep_alive"] = 0

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw_response = data.get("response", "{}")
                try:
                    return json.loads(raw_response)
                except json.JSONDecodeError:
                    return {"raw_output": raw_response}
        except Exception as e:
            logger.warning(f"Ollama call failed or daemon offline: {e}. Using resilient local heuristic fallback.")
            return self._fallback.generate(prompt=prompt, system=system, json_format=json_format)

    def unload(self) -> bool:
        """Purge model from GPU VRAM."""
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "keep_alive": 0}
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(url, json=payload)
                return True
        except Exception:
            return False

