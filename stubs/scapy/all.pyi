# Minimal type stubs for scapy.all
from typing import Any

class Packet:
    """Base packet class."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __truediv__(self, other: Any) -> Packet: ...

class IP(Packet):
    """IP layer."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class UDP(Packet):
    """UDP layer."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Ether(Packet):
    """Ethernet layer."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Raw(Packet):
    """Raw data layer."""
    def __init__(self, load: bytes = b"", *args: Any, **kwargs: Any) -> None: ...

def send(pkt: Packet, *args: Any, **kwargs: Any) -> None:
    """Send packets."""

def sendp(pkt: Packet, *args: Any, **kwargs: Any) -> None:
    """Send packets at layer 2."""
