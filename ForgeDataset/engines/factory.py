"""Factory for initializing LLM engines."""

from typing import Union, Dict, Any, Optional
from .base import LLMEngine
from .ollama import OllamaEngine
from .llama_cpp import LlamaCppEngine
from .openai_compatible import OpenAICompatibleEngine
from .heuristic import HeuristicEngine


def get_engine(
    engine: Optional[Union[str, LLMEngine]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> LLMEngine:
    """Instantiate an LLMEngine instance.
    
    Args:
        engine: Engine name ('ollama', 'llama_cpp', 'llama.cpp', 'openai', 'vllm', 'lmstudio', 'heuristic')
                or an already instantiated LLMEngine.
        config: Optional full configuration dictionary.
        **kwargs: Direct arguments passed to the engine constructor.
        
    Returns:
        LLMEngine instance.
    """
    if isinstance(engine, LLMEngine):
        return engine

    config = config or {}
    gen_cfg = config.get("generation", {})
    engine_name = str(engine or gen_cfg.get("engine", "ollama")).lower().replace("-", "_")

    kw = dict(kwargs)

    if engine_name in ("llama_cpp", "llama.cpp", "gguf"):
        model_path = kw.pop("model_path", gen_cfg.get("model_path", ""))
        n_ctx = kw.pop("n_ctx", gen_cfg.get("n_ctx", 4096))
        n_gpu_layers = kw.pop("n_gpu_layers", gen_cfg.get("n_gpu_layers", -1))
        return LlamaCppEngine(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, **kw)

    elif engine_name in ("openai", "openai_compatible", "vllm", "lmstudio", "lm_studio"):
        base_url = kw.pop("base_url", gen_cfg.get("base_url", "http://localhost:8000/v1"))
        model = kw.pop("model", gen_cfg.get("models", {}).get("generator", "default"))
        api_key = kw.pop("api_key", gen_cfg.get("api_key"))
        return OpenAICompatibleEngine(model=model, base_url=base_url, api_key=api_key, **kw)

    elif engine_name in ("heuristic", "offline", "mock"):
        return HeuristicEngine()

    else:
        # Default: Ollama
        ollama_cfg = gen_cfg.get("ollama", {})
        models_cfg = gen_cfg.get("models", {})
        base_url = kw.pop("base_url", ollama_cfg.get("base_url", "http://127.0.0.1:11434"))
        timeout = kw.pop("timeout", ollama_cfg.get("request_timeout", 120.0))
        model = kw.pop("model", models_cfg.get("generator", "qwen2.5:7b-instruct"))
        enforce_sequential = kw.pop("enforce_sequential", ollama_cfg.get("enforce_sequential_models", True))
        return OllamaEngine(
            model=model,
            base_url=base_url,
            timeout=timeout,
            enforce_sequential=enforce_sequential,
            **kw
        )
