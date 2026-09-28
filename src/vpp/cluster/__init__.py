"""Running the API as several worker processes.

* :mod:`vpp.cluster.lease` -- DB-backed leadership leases; singleton
  background work runs only in the process holding its lease.
* :mod:`vpp.cluster.rpc` -- calls forwarded through the database to the
  process holding a lease (the simulated trading venue, alert evaluation).
* :mod:`vpp.cluster.topology` -- lease names, startup validation and the
  topology log line.
"""

from vpp.cluster.lease import (
    CallbackRole,
    LeaderElector,
    TaskRole,
    get_elector,
    is_local,
    leadership,
    release,
    try_acquire,
)
from vpp.cluster.node import node_id
from vpp.cluster.rpc import (
    CallOutcomeUnknownError,
    ClusterCallError,
    LeaderUnavailableError,
    call,
    register_handler,
    submit,
)

__all__ = [
    "CallOutcomeUnknownError",
    "CallbackRole",
    "ClusterCallError",
    "LeaderElector",
    "LeaderUnavailableError",
    "TaskRole",
    "call",
    "get_elector",
    "is_local",
    "leadership",
    "node_id",
    "register_handler",
    "release",
    "submit",
    "try_acquire",
]
