"""
Adapter lifecycle endpoints — runtime load/unload/swap/rollback for LoRA adapters.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from shared.contracts.models import load_registry
from ..backends.factory import get_backend

router = APIRouter(prefix="/adapters", tags=["adapters"])
logger = logging.getLogger(__name__)

# Overridable via INFERENCE__MANIFEST_PATH env; default matches container volume.
import os as _os
_MANIFEST_PATH = _os.environ.get(
    "INFERENCE__MANIFEST_PATH", "/data/lora_checkpoints/manifest.json"
)


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


def _get_backend_or_404(backend_name: str):
    registry = load_registry()
    if backend_name not in registry.backends:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend '{backend_name}'. Known: {list(registry.backends)}",
        )
    return get_backend(backend_name)


@router.post("/load")
async def load_adapter(req: AdapterLoadRequest) -> Dict[str, Any]:
    backend = _get_backend_or_404(req.backend)
    ok = await backend.load_adapter(req.lora_name, req.lora_path)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Failed to load {req.lora_name}")
    return {"status": "ok", "loaded": req.lora_name}


@router.post("/unload")
async def unload_adapter(req: AdapterUnloadRequest) -> Dict[str, Any]:
    backend = _get_backend_or_404(req.backend)
    ok = await backend.unload_adapter(req.lora_name)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Failed to unload {req.lora_name}")
    return {"status": "ok", "unloaded": req.lora_name}


@router.post("/swap")
async def swap_adapter(req: AdapterSwapRequest) -> Dict[str, Any]:
    """
    Atomic-as-possible swap: unload old canonical → load new under canonical name
    → unload __staging.  Staging adapter must already be loaded by the caller.
    Brief fallback to base model is expected during the unload/load gap.
    """
    backend = _get_backend_or_404(req.backend)
    staging_name = f"{req.canonical_name}__staging"

    old_unloaded = await backend.unload_adapter(req.canonical_name)
    if not old_unloaded:
        logger.warning("swap: could not unload old %s — proceeding", req.canonical_name)

    ok = await backend.load_adapter(req.canonical_name, req.new_path)
    if not ok:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to load {req.canonical_name} from {req.new_path}; "
                "old adapter already unloaded — manual recovery needed"
            ),
        )

    await backend.unload_adapter(staging_name)

    logger.info("swap: %s → %s", req.canonical_name, req.new_path)
    return {"status": "ok", "swapped": req.canonical_name}


@router.post("/rollback")
async def rollback_adapter(req: AdapterRollbackRequest) -> Dict[str, Any]:
    """Roll back to previous_version using the path recorded in the manifest."""
    p = Path(_MANIFEST_PATH)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")
    try:
        manifest = json.loads(p.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read manifest: {exc}")

    adapter_name: str | None = None
    adapter_entry: dict | None = None
    for name, entry in manifest.get("adapters", {}).items():
        if entry.get("backend") == req.backend:
            adapter_name = name
            adapter_entry = entry
            break

    if adapter_entry is None:
        raise HTTPException(
            status_code=404, detail=f"No adapter for backend '{req.backend}'"
        )

    previous_path = adapter_entry.get("previous_path")
    if not previous_path:
        raise HTTPException(
            status_code=400,
            detail=f"No previous_path recorded for {adapter_name} — nothing to roll back to",
        )

    backend = _get_backend_or_404(req.backend)

    await backend.unload_adapter(adapter_name)
    ok = await backend.load_adapter(adapter_name, previous_path)
    if not ok:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load previous adapter {adapter_name} from {previous_path}",
        )

    logger.info("rollback: %s → %s", adapter_name, previous_path)
    return {"status": "ok", "rolled_back": adapter_name, "previous_path": previous_path}


@router.get("")
async def list_adapters() -> Dict[str, Any]:
    """List currently loaded adapters across all backends."""
    result: Dict[str, Any] = {}
    for name in load_registry().backends:
        try:
            result[name] = await get_backend(name).list_adapters()
        except Exception as exc:
            result[name] = {"error": str(exc)}
    return result
