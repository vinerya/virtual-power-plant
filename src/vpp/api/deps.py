"""Process-wide state shared by API route handlers.

The API keeps exactly one piece of in-memory state here: the *live* VPP
configuration document (:class:`~vpp.config.VPPConfig`).

* It is persisted append-only in ``config_documents`` by
  ``PUT /api/v1/config`` (:mod:`vpp.api.routes.config`), which also swaps it
  in here.
* On startup the API lifespan re-applies the newest stored document
  (:func:`vpp.api.routes.config.apply_stored_config`), so a restart does
  not silently fall back to defaults.
* Until a document has been applied, the built-in defaults are served.

What reads it: today only the ``/api/v1/config`` endpoints. Optimization,
trading and telemetry routes take their parameters per request and read
resources from the database; they do not consult this document. (This
module used to hold a lazily-created in-memory ``VirtualPowerPlant``
singleton, ``get_vpp()``; nothing but the config routes used it, and only
for its ``.config`` attribute, so it was reduced to this holder.)
"""

from __future__ import annotations

from vpp.config import VPPConfig

_live_config: VPPConfig | None = None


def get_live_config() -> VPPConfig:
    """Return the live configuration, creating the defaults lazily."""
    global _live_config
    if _live_config is None:
        _live_config = VPPConfig()
    return _live_config


def set_live_config(config: VPPConfig) -> None:
    """Replace the live configuration (after it has been validated)."""
    global _live_config
    _live_config = config


def reset_live_config() -> None:
    """Drop the live configuration so the next read yields defaults (tests)."""
    global _live_config
    _live_config = None
