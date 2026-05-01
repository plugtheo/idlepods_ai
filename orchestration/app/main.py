"""
Orchestration Service — entry point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config.settings import settings
from .routes.run import router as run_router

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
)

_log = logging.getLogger(__name__)

# Fixed corpus used to measure the actual chars-per-token ratio at startup.
# Mixed English prose + Python code to reflect real workload distribution.
CHARS_PER_TOKEN_OVERSHOOT_THRESHOLD = 4.5  # warn when measured ratio exceeds this; configured ratio may be wastefully low

_CALIBRATION_CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning models tokenize text into subword units for processing. "
    "def fibonacci(n: int) -> int:\n"
    "    if n <= 1:\n"
    "        return n\n"
    "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
)


async def _startup_checks() -> None:
    from .clients.inference import InferenceClient
    # timeout=30.0 is intentional startup fail-fast; deliberately shorter than
    # settings.request_timeout (180s) so a missing inference service aborts boot quickly.
    client = InferenceClient(base_url=settings.inference_url, timeout=30.0)
    try:
        # --- Task 1: model_context_len validation ---
        try:
            model_info = await client.get_model_info()
        except Exception as exc:
            raise RuntimeError(
                f"Startup: failed to fetch model info from inference service: {exc}"
            ) from exc

        for family, served_len in model_info.items():
            if served_len != settings.model_context_len:
                raise RuntimeError(
                    f"Startup: model_context_len mismatch for {family}: "
                    f"settings={settings.model_context_len}, served={served_len}. "
                    f"Update ORCHESTRATION__MODEL_CONTEXT_LEN or --max-model-len in compose.yml."
                )
        _log.info(
            "Startup check passed: model_context_len=%d verified for all families.",
            settings.model_context_len,
        )

        # --- Task 2: chars_per_token calibration ---
        corpus_len = len(_CALIBRATION_CORPUS)
        from shared.contracts.models import load_registry as _load_reg
        for family in list(_load_reg().backends):
            try:
                token_count = await client.tokenize(family, _CALIBRATION_CORPUS)
            except Exception as exc:
                _log.warning(
                    "Startup: chars_per_token calibration skipped for %s: %s",
                    family, exc,
                )
                continue
            if token_count == 0:
                _log.warning(
                    "Startup: tokenizer returned 0 tokens for %s — skipping calibration.",
                    family,
                )
                continue
            measured = corpus_len / token_count
            _log.info(
                "Startup check: chars_per_token measured=%.2f configured=%d family=%s tokens=%d",
                measured, settings.chars_per_token, family, token_count,
            )
            
            # Check if the measured chars_per_token is significantly lower than the configured value
            # Tolerate around 2% calibration variance without masking a real misconfiguration
            if measured < settings.chars_per_token - 0.05:
                raise RuntimeError(
                    f"Startup: chars_per_token misconfigured for {family}: "
                    f"measured={measured:.2f} < configured={settings.chars_per_token}. "
                    f"Reduce ORCHESTRATION__CHARS_PER_TOKEN to {int(measured)} or lower."
                )
            if measured > CHARS_PER_TOKEN_OVERSHOOT_THRESHOLD:
                _log.warning(
                    "Startup: chars_per_token for %s measured=%.2f > 4.5 — "
                    "context budget underutilised (configured=%d). "
                    "Consider raising ORCHESTRATION__CHARS_PER_TOKEN.",
                    family, measured, settings.chars_per_token,
                )
    finally:
        await client.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup_checks()

    from pathlib import Path
    from .experience import inflight
    spool_path = Path(settings.spool_path)
    jsonl_path = Path(settings.jsonl_path)
    try:
        await inflight.replay_spool(spool_path, jsonl_path)
    except Exception as exc:
        _log.error("Spool replay failed at startup: %s", exc)

    _log.info("Orchestration Service started on port %d", settings.port)
    yield

    try:
        remaining = await inflight.drain(settings.shutdown_drain_timeout_s)
        if remaining:
            inflight.spool_pending(remaining, spool_path)
    except Exception as exc:
        _log.error("Shutdown drain/spool failed: %s", exc)

    from .clients.inference import close_inference_client
    await close_inference_client()


app = FastAPI(
    title="Orchestration Service",
    description=(
        "Drives the IdleDev multi-agent LangGraph pipeline. "
        "Receives a prompt, routes it, enriches context, runs agents iteratively "
        "until converged, stores the experience, and returns the final response."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(run_router)
