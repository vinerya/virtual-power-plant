"""OpenLEADR VTN that feeds scripted demand-response signals to the VPP's VEN.

OpenLEADR (https://github.com/openleadr/openleadr-python) is the OpenADR
Alliance's open-source reference implementation. This script plays the
utility: it runs a plain-HTTP VTN, accepts any VEN registration, and queues a
sequence of events for the VEN the moment it registers. The VPP's OpenADR
adapter polls them, answers optIn/optOut, and (with
VPP_DR_AUTO_RESPONSE_ENABLED=true) dispatches the fleet.

Scenarios (``--scenario``):

* ``sequence`` (default) -- back-to-back events the DR orchestrator can
  translate (docs/protocols.md, "Translation rules"):
    1. SIMPLE level 1          -> 50 % of fleet export capability
    2. SIMPLE level 2          -> 75 %
    3. LOAD_DISPATCH delta     -> export ``--dispatch-kw`` kW
* ``simple`` -- a single SIMPLE event at ``--level``.

``--cancel-first-after N`` cancels the first event N seconds after the VEN
registers, to exercise the VEN's cancellation path.

Point the VPP at it with::

    VPP_OPENADR_ENABLED=true
    VPP_OPENADR_VTN_URL=http://localhost:8080/OpenADR2/Simple/2.0b   # host
    VPP_OPENADR_VTN_URL=http://host.docker.internal:8080/OpenADR2/Simple/2.0b   # Docker

Events are queued at registration, so restart this script to replay them.
Not collected by pytest (no ``test_`` prefix). Requires ``pip install openleadr``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from openleadr import OpenADRServer, enable_default_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
enable_default_logging(level=logging.INFO)
log = logging.getLogger("vtn")

VEN_ID = "ven_vpp"
_background: set[asyncio.Task] = set()
REGISTRATION_ID = "reg_vpp"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--host", default="127.0.0.1", help="bind address (0.0.0.0 to reach from other hosts)"
    )
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--scenario", choices=("sequence", "simple"), default="sequence")
    p.add_argument(
        "--lead", type=float, default=60, help="seconds from registration to the first event"
    )
    p.add_argument("--duration", type=float, default=300, help="seconds per event")
    p.add_argument("--gap", type=float, default=30, help="seconds between events in the sequence")
    p.add_argument("--level", type=int, default=1, help="SIMPLE level for --scenario simple (0-3)")
    p.add_argument(
        "--dispatch-kw", type=float, default=30, help="LOAD_DISPATCH delta (kW) in the sequence"
    )
    p.add_argument(
        "--cancel-first-after",
        type=float,
        default=0,
        help="cancel the first event this many seconds after registration (0 = never)",
    )
    p.add_argument(
        "--poll-freq", type=float, default=5, help="poll interval requested from the VEN (s)"
    )
    return p.parse_args()


ARGS = parse_args()
server = OpenADRServer(
    vtn_id="openleadr-vtn",
    http_host=ARGS.host,
    http_port=ARGS.port,
    requested_poll_freq=timedelta(seconds=ARGS.poll_freq),
)


async def on_event_response(ven_id: str, event_id: str, opt_type: str) -> None:
    log.info("VEN %s answered event %s: %s", ven_id, event_id, opt_type)


def _add(ven_id: str, signal_name: str, signal_type: str, start: datetime, payload: float) -> str:
    event_id = server.add_event(
        ven_id=ven_id,
        signal_name=signal_name,
        signal_type=signal_type,
        intervals=[
            {
                "dtstart": start,
                "duration": timedelta(seconds=ARGS.duration),
                "signal_payload": payload,
            }
        ],
        callback=on_event_response,
        response_required="always",
    )
    log.info(
        "Queued %s/%s=%s id=%s start=%s dur=%ss",
        signal_name,
        signal_type,
        payload,
        event_id,
        start.strftime("%H:%M:%S"),
        int(ARGS.duration),
    )
    return event_id


def queue_scenario(ven_id: str) -> list[str]:
    start = datetime.now(timezone.utc) + timedelta(seconds=ARGS.lead)
    step = timedelta(seconds=ARGS.duration + ARGS.gap)
    if ARGS.scenario == "simple":
        return [_add(ven_id, "SIMPLE", "level", start, ARGS.level)]
    return [
        _add(ven_id, "SIMPLE", "level", start, 1),
        _add(ven_id, "SIMPLE", "level", start + step, 2),
        _add(ven_id, "LOAD_DISPATCH", "delta", start + 2 * step, ARGS.dispatch_kw),
    ]


async def cancel_later(ven_id: str, event_id: str, delay: float) -> None:
    await asyncio.sleep(delay)
    server.cancel_event(ven_id, event_id)
    log.info("Cancelled event %s", event_id)


async def on_create_party_registration(registration_info: dict):
    log.info("VEN registered: %s", registration_info.get("ven_name"))
    if server.events.get(VEN_ID):
        log.info("Events already queued for %s; not re-queuing", VEN_ID)
    else:
        event_ids = queue_scenario(VEN_ID)
        if ARGS.cancel_first_after > 0:
            task = asyncio.create_task(cancel_later(VEN_ID, event_ids[0], ARGS.cancel_first_after))
            _background.add(task)
            task.add_done_callback(_background.discard)
    return VEN_ID, REGISTRATION_ID


async def main() -> None:
    server.add_handler("on_create_party_registration", on_create_party_registration)
    await server.run()
    log.info(
        "VTN listening on http://%s:%d/OpenADR2/Simple/2.0b, scenario=%s (Ctrl-C to stop)",
        ARGS.host,
        ARGS.port,
        ARGS.scenario,
    )
    try:
        await asyncio.Event().wait()
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
