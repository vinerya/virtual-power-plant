"""Application settings loaded from environment variables and .env files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """VPP platform settings.

    Values are loaded from environment variables prefixed with VPP_,
    falling back to a .env file in the project root.
    """

    model_config = SettingsConfigDict(
        env_prefix="VPP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core
    env: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # Security
    secret_key: str = "change-me-to-a-real-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_key_header: str = "X-API-Key"

    # Database
    database_url: str = "sqlite+aiosqlite:///./vpp.db"

    # Redis (optional)
    redis_url: Optional[str] = None

    # Monitoring
    metrics_enabled: bool = True
    metrics_prefix: str = "vpp"

    # VPP Config
    config_path: Optional[str] = None
    default_timezone: str = "UTC"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def is_development(self) -> bool:
        return self.env == "development"

    @property
    def is_testing(self) -> bool:
        return self.env == "testing"

    @property
    def database_is_sqlite(self) -> bool:
        return "sqlite" in self.database_url

    @property
    def config_file_path(self) -> Optional[Path]:
        if self.config_path:
            return Path(self.config_path)
        return None


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
