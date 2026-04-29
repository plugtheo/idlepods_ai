"""
InferenceBackend — abstract base contract
==========================================
All concrete backends implement this interface. The route layer and gRPC
server call the backend without knowing which concrete implementation is
active.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from shared.contracts.inference import GenerateRequest, GenerateResponse


class InferenceError(Exception):
    """Raised by backends on generation failure instead of leaking
    implementation-specific exceptions to callers."""


class InferenceBackend(ABC):
    """Contract every inference backend must satisfy."""

    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Run one inference call and return the model's response.

        Raises InferenceError on failure.
        """

    @abstractmethod
    async def list_adapters(self) -> list[str]:
        """Return names of currently loaded LoRA adapters."""

    @abstractmethod
    async def health(self) -> dict:
        """Return a dict with at minimum: {"status": "ok"|"degraded"|"unavailable", "backend": str}"""

    async def generate_stream(
        self, request: GenerateRequest
    ) -> AsyncGenerator[str, None]:
        """
        Yield text fragments (tokens/chunks) as the model produces them.

        Default implementation buffers via generate() and yields the full
        content as a single chunk — every backend is streamable even before a
        native streaming path is implemented.  Backends that support native
        streaming should override this.
        """
        response = await self.generate(request)
        yield response.content
