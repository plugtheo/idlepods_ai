"""
InferenceBackend — abstract base contract
==========================================
All concrete backends (local vLLM, LiteLLM/Anthropic, …) implement this
interface.  The route layer calls `.generate()` without knowing which
backend is active.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from shared.contracts.inference import GenerateRequest, GenerateResponse


class InferenceBackend(ABC):
    """Contract every inference backend must satisfy."""

    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Run one inference call and return the model's response.

        Parameters
        ----------
        request:
            Full generation request including messages, model family,
            adapter name, and sampling parameters.

        Returns
        -------
        GenerateResponse
            The generated text and metadata.
        """

    async def generate_stream(
        self, request: GenerateRequest
    ) -> AsyncGenerator[str, None]:
        """
        Yield text fragments (tokens/chunks) as the model produces them.

        Default implementation buffers via ``generate()`` and yields the
        full content as a single chunk — guarantees every backend is
        streamable even before a true streaming path is implemented.
        Backends that support native streaming should override this.
        """
        response = await self.generate(request)
        yield response.content
