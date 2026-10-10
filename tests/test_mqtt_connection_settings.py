"""MQTT connection settings: client id, TLS and refused connections.

The adapter used a fixed client id (``vpp-mqtt-adapter``), so two
installations on one broker disconnected each other, and had no way to
configure TLS from settings. These tests drive ``MQTTAdapter.connect`` with
a fake paho client; tests/test_mqtt_e2e_broker.py checks a real broker.
"""

from __future__ import annotations

import os
import ssl
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from vpp.protocols.mqtt import MQTTAdapter, default_client_id
from vpp.settings import Settings

paho = pytest.importorskip("paho.mqtt.client")


class _FakeClient:
    instances: ClassVar[list[_FakeClient]] = []
    connack: ClassVar[Any] = 0  # reason code the fake broker answers with

    def __init__(self, api_version: Any, client_id: str = "") -> None:
        self.client_id = client_id
        self.calls: dict[str, Any] = {}
        self.on_connect: Any = None
        self.on_message: Any = None
        self.on_disconnect: Any = None
        self.loop_stopped = False
        _FakeClient.instances.append(self)

    def username_pw_set(self, username: str, password: str | None) -> None:
        self.calls["auth"] = (username, password)

    def tls_set(self, **kwargs: Any) -> None:
        self.calls["tls_set"] = kwargs

    def tls_insecure_set(self, value: bool) -> None:
        self.calls["tls_insecure_set"] = value

    def connect(self, host: str, port: int, keepalive: int) -> None:
        self.calls["connect"] = (host, port, keepalive)

    def loop_start(self) -> None:
        self.on_connect(self, None, {}, _FakeClient.connack, None)

    def loop_stop(self) -> None:
        self.loop_stopped = True

    def subscribe(self, topic: str, qos: int = 0) -> None:
        self.calls["subscribe"] = (topic, qos)

    def disconnect(self) -> None:
        pass


@pytest.fixture
def fake_paho(monkeypatch):
    _FakeClient.instances = []
    _FakeClient.connack = 0
    monkeypatch.setattr(paho, "Client", _FakeClient)
    return _FakeClient


def test_default_client_id_is_unique_per_process_and_stable():
    cid = default_client_id()
    assert cid == default_client_id()  # same for every reconnect of this process
    assert cid.startswith("vpp-")
    assert f"-{os.getpid()}-" in cid
    assert cid != "vpp-mqtt-adapter"
    assert all(c.isalnum() or c == "-" for c in cid)


def test_default_client_id_sanitises_hostname(monkeypatch):
    monkeypatch.setattr("socket.gethostname", lambda: "edge_gw.example.com" + "x" * 40)
    cid = default_client_id()
    host = cid.split("-")[1:-2]
    assert "_" not in cid and "." not in cid
    assert len("-".join(host)) <= 20


@pytest.mark.asyncio
async def test_connect_uses_default_client_id_and_no_tls(fake_paho):
    adapter = MQTTAdapter()
    adapter.configure(broker_host="broker", broker_port=1883, client_id=None)
    await adapter.connect()
    client = fake_paho.instances[-1]
    assert client.client_id == default_client_id()
    assert "tls_set" not in client.calls
    assert client.calls["subscribe"] == ("vpp/#", 1)
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_connect_uses_configured_client_id(fake_paho):
    adapter = MQTTAdapter()
    adapter.configure(client_id="vpp-site-a")
    await adapter.connect()
    assert fake_paho.instances[-1].client_id == "vpp-site-a"
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_connect_with_tls_and_client_certificate(fake_paho):
    adapter = MQTTAdapter()
    adapter.configure(
        broker_port=8883,
        use_tls=True,
        ca_certs="/etc/vpp/ca.pem",
        certfile="/etc/vpp/client.pem",
        keyfile="/etc/vpp/client.key",
        username="vpp",
        password="secret",
    )
    await adapter.connect()
    client = fake_paho.instances[-1]
    assert client.calls["tls_set"] == {
        "ca_certs": "/etc/vpp/ca.pem",
        "certfile": "/etc/vpp/client.pem",
        "keyfile": "/etc/vpp/client.key",
        "cert_reqs": ssl.CERT_REQUIRED,
    }
    assert "tls_insecure_set" not in client.calls
    assert client.calls["auth"] == ("vpp", "secret")
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_tls_without_ca_file_uses_system_trust_store(fake_paho):
    adapter = MQTTAdapter()
    adapter.configure(use_tls=True, ca_certs=None, certfile=None, keyfile=None)
    await adapter.connect()
    assert fake_paho.instances[-1].calls["tls_set"] == {
        "ca_certs": None,
        "certfile": None,
        "keyfile": None,
        "cert_reqs": ssl.CERT_REQUIRED,
    }
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_tls_insecure_disables_verification_and_warns(fake_paho, caplog):
    adapter = MQTTAdapter()
    adapter.configure(use_tls=True, tls_insecure=True)
    with caplog.at_level("WARNING", logger="vpp.protocols.mqtt"):
        await adapter.connect()
    client = fake_paho.instances[-1]
    assert client.calls["tls_set"]["cert_reqs"] == ssl.CERT_NONE
    assert client.calls["tls_insecure_set"] is True
    assert "DISABLED" in caplog.text
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_refused_connection_raises_instead_of_reporting_connected(fake_paho):
    from paho.mqtt.packettypes import PacketTypes
    from paho.mqtt.reasoncodes import ReasonCode

    fake_paho.connack = ReasonCode(PacketTypes.CONNACK, "Not authorized")
    adapter = MQTTAdapter()
    with pytest.raises(ConnectionError, match="refused the connection: Not authorized"):
        await adapter.connect()
    assert not adapter.is_connected
    client = fake_paho.instances[-1]
    assert client.loop_stopped  # paho stops retrying rejected credentials
    assert "subscribe" not in client.calls


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_mqtt_settings_defaults():
    s = Settings()
    assert s.mqtt_client_id is None
    assert s.mqtt_tls_enabled is False
    assert s.mqtt_tls_insecure is False
    assert s.mqtt_tls_ca_file is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mqtt_tls_ca_file": "/ca.pem"},
        {"mqtt_tls_certfile": "/c.pem"},
        {"mqtt_tls_insecure": True},
    ],
)
def test_tls_option_without_tls_enabled_is_rejected(kwargs):
    with pytest.raises(ValidationError, match="VPP_MQTT_TLS_ENABLED"):
        Settings(**kwargs)


def test_tls_keyfile_requires_certfile():
    with pytest.raises(ValidationError, match="VPP_MQTT_TLS_CERTFILE"):
        Settings(mqtt_tls_enabled=True, mqtt_tls_keyfile="/k.pem")


def test_tls_settings_from_environment(monkeypatch):
    monkeypatch.setenv("VPP_MQTT_TLS_ENABLED", "true")
    monkeypatch.setenv("VPP_MQTT_TLS_CA_FILE", "/etc/vpp/ca.pem")
    monkeypatch.setenv("VPP_MQTT_TLS_CERTFILE", "/etc/vpp/client.pem")
    monkeypatch.setenv("VPP_MQTT_TLS_KEYFILE", "/etc/vpp/client.key")
    monkeypatch.setenv("VPP_MQTT_CLIENT_ID", "vpp-prod")
    s = Settings()
    assert s.mqtt_tls_enabled is True
    assert s.mqtt_tls_ca_file == "/etc/vpp/ca.pem"
    assert s.mqtt_client_id == "vpp-prod"


def test_build_mqtt_adapter_passes_tls_and_client_id_settings():
    from vpp.api.app import _build_mqtt_adapter

    settings = Settings(
        mqtt_client_id="vpp-prod",
        mqtt_tls_enabled=True,
        mqtt_tls_ca_file="/ca.pem",
        mqtt_tls_certfile="/c.pem",
        mqtt_tls_keyfile="/k.pem",
    )
    config = _build_mqtt_adapter(settings)._config
    assert config["client_id"] == "vpp-prod"
    assert config["use_tls"] is True
    assert config["ca_certs"] == "/ca.pem"
    assert config["certfile"] == "/c.pem"
    assert config["keyfile"] == "/k.pem"
    assert config["tls_insecure"] is False
