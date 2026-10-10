# Protocol adapters

The base class and registry that every protocol adapter (OCPP, OpenADR,
IEEE 2030.5, Modbus, MQTT) implements. What each protocol supports and how
to configure it is in [Protocols, V2G and DR](../../protocols.md).

```python
from vpp.protocols import ProtocolAdapter, ProtocolRegistry, ProtocolStatus
```

::: vpp.protocols.base
