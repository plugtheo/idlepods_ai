"""
Inference service — request/response models.

Per-endpoint Pydantic shapes that are local to the inference service.
Cross-service contracts belong in shared/contracts/.
"""

from pydantic import BaseModel


class AdapterLoadRequest(BaseModel):
    backend: str
    lora_name: str
    lora_path: str


class AdapterUnloadRequest(BaseModel):
    backend: str
    lora_name: str


class AdapterSwapRequest(BaseModel):
    backend: str
    canonical_name: str
    new_path: str


class AdapterRollbackRequest(BaseModel):
    backend: str


class TokenizeRequest(BaseModel):
    backend: str
    text: str
