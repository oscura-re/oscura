# Minimal type stubs for scapy.all
from typing import Any

class Packet:
    """Base packet class."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class IP(Packet):
    """IP layer."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class UDP(Packet):
    """UDP layer."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Ether(Packet):
    """Ethernet layer."""
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
