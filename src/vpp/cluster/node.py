"""Identity of this API process within a (possibly multi-process) deployment."""

from __future__ import annotations

import os
import socket
import uuid
from datetime import datetime, timezone

_NODE_ID: str | None = None


def _hostname() -> str:
    """Hostname plus the PID-namespace id: pids are only comparable within one namespace.

    Two containers can share a hostname (and even a pid) while seeing
    different processes; the namespace inode tells them apart.
    """
    try:
        pidns = str(os.stat("/proc/self/ns/pid").st_ino)
    except OSError:
        pidns = "0"
    return f"{(socket.gethostname() or 'localhost')[:64]}~{pidns}"


def node_id() -> str:
    """``<hostname>~<pid namespace>:<pid>:<random>`` -- unique per process incarnation.

    The random suffix distinguishes a restarted process that reuses a pid
    (PID 1 in a restarted container) from its dead predecessor.
    """
    global _NODE_ID
    if _NODE_ID is None:
        _NODE_ID = f"{_hostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    return _NODE_ID


def holder_is_dead_local_process(holder: str) -> bool:
    """True when *holder* names a process on this host that no longer exists.

    Lets a restarted single-process deployment take over its crashed
    predecessor's leases at once instead of waiting for them to expire.
    Holders on other hosts (or in another PID namespace, i.e. another
    container) are never judged dead: their liveness is only visible
    through lease expiry.
    """
    if holder == node_id():
        return False
    try:
        host, pid_s, _suffix = holder.rsplit(":", 2)
        pid = int(pid_s)
    except ValueError:
        return False
    if host != _hostname():
        return False
    if pid == os.getpid():
        return True  # an earlier incarnation of this pid (e.g. PID 1 after a restart)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:  # EPERM: alive, owned by someone else
        return False
    return False


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes; everything here is stored in UTC."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
