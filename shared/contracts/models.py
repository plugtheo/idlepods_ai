"""
shared/contracts/models.py
===========================
Registry loader for models.yaml — the single source of truth for backend identity.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field

__all__ = ["BackendEntry", "ModelsRegistry", "load_registry", "get_backend_entry"]


class BackendEntry(BaseModel):
    served_url: str
    model_id: str
    max_model_len: int = 4096
    quantization: Optional[str] = None
    chat_template_supports_tools: bool = True
    thinking_default: bool = False
    backend_type: Literal["local_vllm", "remote_vllm"] = "local_vllm"
    auth_token: str = ""
    ssl_verify: bool = True
    # Reserved for future per-backend pre-tokenizer override — not implemented now.
    tokenizer_pre_tokenizer: Optional[Literal["bytelevel", "metaspace"]] = None


class ModelsRegistry(BaseModel):
    default_backend: str
    backends: Dict[str, BackendEntry]

    def model_post_init(self, __context) -> None:
        if self.default_backend not in self.backends:
            raise RuntimeError(
                f"models.yaml: default_backend '{self.default_backend}' "
                f"not found in backends: {list(self.backends)}"
            )


@lru_cache(maxsize=1)
def load_registry(path: str = "") -> ModelsRegistry:
    resolved = path or os.environ.get("MODELS_YAML_PATH", "/config/models.yaml")
    try:
        with open(resolved) as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        raise RuntimeError(f"models.yaml not found at '{resolved}'")
    return ModelsRegistry.model_validate(data)


def get_backend_entry(name: Optional[str] = None) -> BackendEntry:
    registry = load_registry()
    key = name if name is not None else registry.default_backend
    entry = registry.backends.get(key)
    if entry is None:
        raise ValueError(
            f"Unknown backend '{key}'. Known: {list(registry.backends)}"
        )
    return entry
