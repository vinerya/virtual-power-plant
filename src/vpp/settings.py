"""Application settings loaded from environment variables and .env files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from vpp.client_ip import parse_trusted_proxies

_DEFAULT_SECRET_KEY = "change-me-to-a-real-secret-key"


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
    # Loopback by default so a bare `vpp serve` isn't reachable from the
    # network by accident; containers pass --host 0.0.0.0 explicitly.
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Worker processes `vpp serve` starts. >1 is supported: singleton
    # background work runs on the holder of a DB lease (see vpp.cluster);
    # OCPP requires 1 (charge-point sockets are process-local).
    api_workers: int = Field(1, ge=1)
    # Leadership lease TTL; a crashed leader's work moves to another worker
    # within about this long. Calls forwarded to a leader (trading venue,
    # alert evaluation) wait at most cluster_call_timeout_seconds.
    cluster_lease_ttl_seconds: float = Field(15.0, gt=0)
    cluster_call_timeout_seconds: float = Field(10.0, gt=0)
    cluster_poll_interval_seconds: float = Field(0.25, gt=0)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # Security
    secret_key: str = _DEFAULT_SECRET_KEY
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_key_header: str = "X-API-Key"
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 120
    # Where the HTTP rate limiter and the login throttle keep their counters:
    # "memory" (per process, no DB round trip), "database" (shared by every
    # worker, table shared_rate_limits) or "auto" (database when
    # api_workers > 1, else memory). See vpp.auth.shared_limits.
    rate_limit_backend: Literal["auto", "memory", "database"] = "auto"
    # Reverse proxies (CIDRs or addresses, e.g. the web console container)
    # whose X-Forwarded-For / X-Real-IP headers identify the client for rate
    # limiting. Empty (default): those headers are ignored.
    trusted_proxies: list[str] = Field(default_factory=list)
    # WebSocket (/ws, /api/v1/ws). When true (default) a socket must present
    # a valid JWT (``token`` query param, ``bearer, <jwt>`` subprotocol pair,
    # or Authorization header) or it is refused with close code 1008.
    ws_auth_required: bool = True
    # Lifetime of the short-lived tokens minted by POST /api/v1/ws/token.
    ws_token_expire_seconds: int = 60
    # Accounts (see vpp.auth.passwords / vpp.auth.throttle / vpp.auth.bootstrap).
    password_min_length: int = 12
    # Per-username failed-login throttle (shared by all workers when
    # rate_limit_backend resolves to "database"); 0 disables.
    login_max_failures: int = 5
    login_lockout_seconds: int = 300
    # First-boot admin: created only while the users table is empty.
    bootstrap_admin_username: str | None = None
    bootstrap_admin_password_file: str | None = None

    # Database
    database_url: str = "sqlite+aiosqlite:///./vpp.db"

    # Monitoring
    metrics_enabled: bool = True
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
    use_alembic: bool = (
        False  # VPP_USE_ALEMBIC=1 -> run alembic upgrade head instead of create_all
    )

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
    mqtt_username: str | None = None
    mqtt_password: str | None = None

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

    # Demand-response orchestrator: OpenADR events / IEEE 2030.5 DER controls
    # -> DB-backed fleet dispatch (see vpp.dr.orchestrator). Off by default:
    # grid signals are still recorded and shown, and the VEN still answers
    # opt-in/out per openadr_auto_opt_in, but nothing is dispatched until an
    # operator turns auto-response on.
    dr_auto_response_enabled: bool = False
    dr_tick_interval_s: float = 5.0
    dr_redispatch_interval_s: float = 900.0  # re-plan an ongoing event this often
    dr_max_export_kw: float | None = None  # hard cap on any DR export target (kW)
    dr_max_import_kw: float | None = None  # hard cap on any DR absorb target (kW)
    # OpenADR SIMPLE level 0..3 -> fraction of the fleet's nominal export capacity.
    dr_simple_level_fractions: list[float] = Field(default_factory=lambda: [0.0, 0.5, 0.75, 1.0])
    # Opt out of an event when nominal fleet capability < this fraction of the request.
    dr_min_opt_in_fraction: float = 0.5
    dr_include_v2g: bool = True  # connected V2G vehicles take part (setpoints via OCPP)
    dr_ieee2030_5_set_max_w: float | None = None  # setMaxW for %-controls (default: fleet)
    v2g_max_profile_periods: int = 48  # ChargingScheduleMaxPeriods budget per profile

    # Device control (vpp.control.actuator): write dispatch setpoints to
    # batteries/inverters that opted in via metadata.modbus.control. Global
    # kill switch, off by default: nothing is written while false.
    control_enabled: bool = False
    control_watchdog_interval_s: float = 5.0  # expiry / deferred-write / keepalive check
    control_expiry_grace_s: float = (
        30.0  # setpoint lives dispatch interval + this, then falls back
    )

    # Simulated trading venue: periodically advance simulated prices, match
    # resting orders, and publish `market_data` events. Everything it emits
    # is labelled source="simulated"; no orders leave the process.
    trading_market_data_enabled: bool = True
    trading_market_data_interval_seconds: float = 5.0

    # Platform configuration document loaded at startup when none has been
    # stored with PUT /api/v1/config yet (a stored document always wins).
    config_path: str | None = None
    default_timezone: str = "UTC"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    @field_validator("rate_limit_backend", mode="before")
    @classmethod
    def normalise_rate_limit_backend(cls, v: object) -> object:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("trusted_proxies")
    @classmethod
    def validate_trusted_proxies(cls, v: list[str]) -> list[str]:
        parse_trusted_proxies(v)  # raises ValueError on an invalid CIDR/address
        return [item.strip() for item in v if item.strip()]

    @model_validator(mode="after")
    def _require_real_secret_in_production(self) -> Settings:
        # The default key is public (it's in this file), so any JWT signed
        # with it is forgeable. Refuse to boot a production instance on it.
        if self.is_production and (
            self.secret_key == _DEFAULT_SECRET_KEY or len(self.secret_key) < 32
        ):
            raise ValueError(
                "VPP_SECRET_KEY must be set to a random value of at least 32 characters "
                "when VPP_ENV=production"
            )
        return self

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
    def config_file_path(self) -> Path | None:
        if self.config_path:
            return Path(self.config_path)
        return None


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
