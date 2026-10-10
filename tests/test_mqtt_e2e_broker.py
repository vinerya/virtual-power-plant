"""End to end against a real Mosquitto broker over mutual TLS.

Skipped unless the ``mosquitto`` binary, paho-mqtt and ``cryptography`` are
available. Starts a private broker on a free port with a throwaway CA, then
checks that the ingestion path the API runs (``MQTTAdapter`` +
``MQTTTelemetryIngestor``) connects with a client certificate, stores a
payload timestamp and updates the battery's live ``current_power``.
"""

from __future__ import annotations

import asyncio
import contextlib
import getpass
import json
import shutil
import socket
import ssl
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vpp.db.engine import get_session_factory
from vpp.db.models import ResourceModel
from vpp.db.repositories import BatteryStateRepository, ResourceRepository
from vpp.events import reset_event_bus
from vpp.protocols.mqtt import MQTTAdapter
from vpp.protocols.telemetry_ingestion import MQTTTelemetryIngestor

paho = pytest.importorskip("paho.mqtt.client")
pytest.importorskip("cryptography")

pytestmark = pytest.mark.skipif(shutil.which("mosquitto") is None, reason="needs mosquitto")


def _write_pki(d: Path) -> None:
    """CA, server cert for localhost and a client cert, all PEM."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.now(timezone.utc)

    def name(cn: str) -> x509.Name:
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])

    def save_key(key: ec.EllipticCurvePrivateKey, path: Path) -> None:
        path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (
        x509.CertificateBuilder()
        .subject_name(name("vpp-test-ca"))
        .issuer_name(name("vpp-test-ca"))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    (d / "ca.pem").write_bytes(ca.public_bytes(serialization.Encoding.PEM))

    def leaf(cn: str, usage: x509.ObjectIdentifier, san: bool) -> None:
        key = ec.generate_private_key(ec.SECP256R1())
        builder = (
            x509.CertificateBuilder()
            .subject_name(name(cn))
            .issuer_name(ca.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=1))
            .add_extension(x509.ExtendedKeyUsage([usage]), critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                critical=False,
            )
        )
        if san:
            builder = builder.add_extension(
                x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False
            )
        (d / f"{cn}.pem").write_bytes(
            builder.sign(ca_key, hashes.SHA256()).public_bytes(serialization.Encoding.PEM)
        )
        save_key(key, d / f"{cn}.key")

    leaf("server", ExtendedKeyUsageOID.SERVER_AUTH, san=True)
    leaf("client", ExtendedKeyUsageOID.CLIENT_AUTH, san=False)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def tls_broker(tmp_path):
    _write_pki(tmp_path)
    port = _free_port()
    conf = tmp_path / "mosquitto.conf"
    conf.write_text(
        f"listener {port} 127.0.0.1\n"
        f"cafile {tmp_path / 'ca.pem'}\n"
        f"certfile {tmp_path / 'server.pem'}\n"
        f"keyfile {tmp_path / 'server.key'}\n"
        "require_certificate true\n"
        "allow_anonymous true\n"
        # Started as root, mosquitto drops to user "mosquitto", which cannot
        # read pytest's private tmp dir; stay the current user instead.
        f"user {getpass.getuser()}\n"
    )
    proc = subprocess.Popen(
        ["mosquitto", "-c", str(conf)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        deadline = time.monotonic() + 10
        while True:
            with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), 0.2):
                break
            if proc.poll() is not None or time.monotonic() > deadline:
                out = proc.stdout.read().decode() if proc.stdout else ""
                pytest.skip(f"mosquitto did not start: {out}")
            time.sleep(0.05)
        yield port, tmp_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _publish_tls(port: int, pki: Path, topic: str, payload: dict) -> None:
    client = paho.Client(paho.CallbackAPIVersion.VERSION2, client_id="e2e-device")
    client.tls_set(
        ca_certs=str(pki / "ca.pem"),
        certfile=str(pki / "client.pem"),
        keyfile=str(pki / "client.key"),
    )
    client.connect("localhost", port, 30)
    client.loop_start()
    try:
        client.publish(topic, json.dumps(payload), qos=1).wait_for_publish(5)
    finally:
        client.loop_stop()
        client.disconnect()


@pytest.mark.asyncio
async def test_tls_ingestion_updates_live_power_and_uses_payload_timestamp(
    tls_broker, db_session, app
):
    port, pki = tls_broker
    reset_event_bus()
    battery = await ResourceRepository.create(
        db_session, name=f"e2e-{time.time()}", resource_type="battery", rated_power=5.0
    )
    await db_session.commit()

    adapter = MQTTAdapter()
    adapter.configure(
        broker_host="localhost",
        broker_port=port,
        use_tls=True,
        ca_certs=str(pki / "ca.pem"),
        certfile=str(pki / "client.pem"),
        keyfile=str(pki / "client.key"),
    )
    await adapter.connect()
    task = asyncio.create_task(MQTTTelemetryIngestor(adapter, get_session_factory()).run())
    sampled = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=30)
    try:
        await asyncio.sleep(0.3)  # let the subscription settle
        await asyncio.to_thread(
            _publish_tls,
            port,
            pki,
            f"vpp/site/battery/{battery.id}/state",
            {"soc": 66.0, "power": -2.25, "timestamp": int(sampled.timestamp())},
        )
        for _ in range(100):
            rows = await BatteryStateRepository.get_latest(db_session, battery.id)
            if rows:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("telemetry published over TLS was not stored")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await adapter.disconnect()

    assert rows[0].timestamp.replace(tzinfo=timezone.utc) == sampled
    async with get_session_factory()() as session:
        resource = await session.get(ResourceModel, battery.id)
        assert resource is not None
        assert resource.current_power == pytest.approx(-2.25)


@pytest.mark.asyncio
async def test_tls_rejects_untrusted_broker(tls_broker, tmp_path_factory):
    """Server verification is on by default: a CA that did not sign the
    broker's certificate fails the handshake."""
    port, _pki = tls_broker
    other = tmp_path_factory.mktemp("other-pki")
    _write_pki(other)
    adapter = MQTTAdapter()
    adapter.configure(
        broker_host="localhost",
        broker_port=port,
        use_tls=True,
        ca_certs=str(other / "ca.pem"),
        certfile=str(other / "client.pem"),
        keyfile=str(other / "client.key"),
    )
    with pytest.raises(ssl.SSLError):
        await adapter.connect()
    if adapter._client is not None:
        adapter._client.loop_stop()
