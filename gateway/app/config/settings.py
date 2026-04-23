"""
Gateway Service — configuration
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY__", env_nested_delimiter="__")

    # Downstream service URLs
    orchestration_url: str = Field("http://orchestration:8001", description="Orchestration Service base URL")

    # Auth
    api_key: str = Field("", description="Bearer token required on all requests. Empty = disabled.")

    # HTTP timeouts (seconds)
    request_timeout: float = Field(300.0, description="Orchestration call timeout")

    # Service identity
    port: int = Field(8080, description="Listening port")
    debug: bool = Field(False, description="Enable debug logging")


settings = GatewaySettings()
