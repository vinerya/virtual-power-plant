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
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 120
    # WebSocket (/ws, /api/v1/ws). When true (default) a socket must present
    # a valid JWT (``token`` query param, ``bearer, <jwt>`` subprotocol pair,
    # or Authorization header) or it is refused with close code 1008.
    ws_auth_required: bool = True
    # Lifetime of the short-lived tokens minted by POST /api/v1/ws/token.
    ws_token_expire_seconds: int = 60

    # Database
    database_url: str = "sqlite+aiosqlite:///./vpp.db"

    # Monitoring
    metrics_enabled: bool = True
    metrics_prefix: str = "vpp"
    # When set, GET /metrics requires "Authorization: Bearer <token>".
    metrics_bearer_token: str | None = None
    # Structured logging: JSON lines (None -> JSON only when env=production)
    # and one "vpp.access" line per HTTP request.
    log_json: bool | None = None
    access_log_enabled: bool = True

    # Alerting: evaluate persisted alert rules against RESOURCE_UPDATED
    # telemetry.  Default rules (SOC low, over-temperature, SOH degraded) are
    # inserted on startup only if the alert_rules table is empty.
    alerts_enabled: bool = True
    alerts_seed_default_rules: bool = True
    # Optional outbound webhook for fired alerts; signed with HMAC-SHA256
    # (X-VPP-Signature) when a secret is set.
    alert_webhook_url: str | None = None
    alert_webhook_secret: str | None = None
    alert_webhook_timeout_seconds: float = 5.0
    alert_webhook_max_retries: int = 3

    # Migrations / database init
    use_alembic: bool = False  # VPP_USE_ALEMBIC=1 -> run alembic upgrade head instead of create_all

    # Battery degradation periodic updater (M4)
    degradation_updater_enabled: bool = True
    degradation_updater_interval_minutes: int = 60

    # MQTT battery telemetry ingestion (M5). Disabled by default -- unlike
    # the degradation updater (DB-only), this dials out to an external
    # broker, so it must be an explicit opt-in.
    mqtt_ingestion_enabled: bool = False
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883
    mqtt_topic_prefix: str = "vpp/#"
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None

    # Modbus inverter/meter telemetry ingestion. Disabled by default --
    # dials out to physical devices. Per-device connection config (host,
    # port, device_profile, ...) lives on each resource's own `metadata`
    # under a "modbus" key, not here -- see protocols/modbus_ingestion.py.
    modbus_ingestion_enabled: bool = False

    # OCPP 1.6-J Central System (EV chargers). Disabled by default: when
    # enabled, charge points connect to ws(s)://<api>/ocpp/{charge_point_id}
    # (subprotocol "ocpp1.6"); terminate TLS at the reverse proxy.
    ocpp_enabled: bool = False
    ocpp_heartbeat_interval_s: int = 300
    ocpp_call_timeout_s: float = 30.0
    ocpp_auto_accept_boot: bool = True
    ocpp_allowed_charge_points: list[str] = Field(default_factory=list)  # empty = any id
    ocpp_basic_auth_password: str | None = None  # OCPP Security Profile 1
    ocpp_authorized_id_tags: list[str] | None = None  # None = accept every idTag
    ocpp_remote_id_tag: str = "VPP"

    # OpenADR 2.0b VEN (demand response). Disabled by default; dials out
    # to the utility's VTN over HTTPS (simple-HTTP pull model).
    openadr_enabled: bool = False
    openadr_vtn_url: str | None = None  # e.g. https://vtn.example.com/OpenADR2/Simple/2.0b
    openadr_ven_name: str = "vpp-ven"
    openadr_ven_id: str | None = None  # normally assigned by the VTN at registration
    openadr_poll_interval_s: float | None = None  # None = VTN-requested frequency
    openadr_auto_opt_in: bool = True
    openadr_timeout_s: float = 10.0
    openadr_verify_tls: bool = True
    openadr_ca_path: str | None = None
    openadr_cert_path: str | None = None
    openadr_key_path: str | None = None

    # IEEE 2030.5 (SEP 2.0) DER client. Disabled by default; requires the
    # device's client certificate (mutual TLS) in production.
    ieee2030_5_enabled: bool = False
    ieee2030_5_server_url: str | None = None  # e.g. https://utility.example.com:8443
    ieee2030_5_dcap_path: str = "/dcap"
    ieee2030_5_lfdi: str | None = None  # default: derived from the client cert
    ieee2030_5_poll_interval_s: float | None = None  # None = server pollRate
    ieee2030_5_timeout_s: float = 10.0
    ieee2030_5_verify_tls: bool = True
    ieee2030_5_ca_path: str | None = None
    ieee2030_5_cert_path: str | None = None
    ieee2030_5_key_path: str | None = None
    ieee2030_5_tls_ciphers: str | None = None  # e.g. ECDHE-ECDSA-AES128-CCM8

    # Simulated trading venue: periodically advance simulated prices, match
    # resting orders, and publish `market_data` events. Everything it emits
    # is labelled source="simulated"; no orders leave the process.
    trading_market_data_enabled: bool = True
    trading_market_data_interval_seconds: float = 5.0

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
