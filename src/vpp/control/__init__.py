"""Device control: turn dispatch allocations into device setpoints."""

from vpp.control.actuator import (
    SetpointActuator,
    get_setpoint_actuator,
    set_setpoint_actuator,
)

__all__ = ["SetpointActuator", "get_setpoint_actuator", "set_setpoint_actuator"]
