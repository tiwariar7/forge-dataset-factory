"""OpenAI-compatible HTTP engine (vLLM, LM Studio, Ollama /v1, LocalAI, etc.)."""

import json
import logging
from typing import Dict, Any, Optional
import httpx
from .base import LLMEngine
from .heuristic import HeuristicEngine

logger = logging.getLogger("ForgeDataset.engines.openai_compatible")


class OpenAICompatibleEngine(LLMEngine):
    """Engine supporting standard OpenAI-compatible /v1/chat/completions.
    
    Works seamlessly with:
    - vLLM (high throughput local batch inference)
    - LM Studio (local GUI model server)
    - Ollama (/v1 API)
    - LocalAI / text-generation-webui
    - Cloud APIs (DeepSeek, Groq, OpenAI) if desired.
    """

    def __init__(
        self,
        model: str = "default",
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        timeout: float = 120.0
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout
        self._fallback = HeuristicEngine()

    def is_available(self) -> bool:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/models", headers=headers)
                return res.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_format: bool = True,
        temperature: float = 0.3,
        **kwargs: Any
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
        if json_format:
            payload["response_format"] = {"type": "json_object"}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"raw_output": content}
        except Exception as e:
            logger.warning(f"OpenAI-compatible request failed: {e}. Falling back to heuristic engine.")
            return self._fallback.generate(prompt=prompt, system=system, json_format=json_format)

