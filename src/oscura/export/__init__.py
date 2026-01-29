"""Export module for Oscura protocol definitions.

This module provides functionality to export Oscura protocol definitions
to various formats for integration with other tools and workflows.

Supported export formats:
- Wireshark Lua dissectors
- Kaitai Struct definitions (.ksy)
- Scapy packet layers
- (Future) C/C++ parser code

Example:
    >>> from oscura.export.wireshark_dissector import WiresharkDissectorGenerator
    >>> from oscura.export.kaitai_struct import KaitaiStructGenerator
    >>> from oscura.export.scapy_layer import ScapyLayerGenerator
    >>> from oscura.workflows.reverse_engineering import ProtocolSpec
    >>> # Generate Wireshark dissector
    >>> wireshark_gen = WiresharkDissectorGenerator(config)
    >>> wireshark_gen.generate(spec, Path("myproto.lua"))
    >>> # Generate Kaitai Struct definition
    >>> kaitai_gen = KaitaiStructGenerator(config)
    >>> kaitai_gen.generate(spec, Path("myproto.ksy"))
    >>> # Generate Scapy layer
    >>> scapy_gen = ScapyLayerGenerator(config)
    >>> scapy_gen.generate(spec, messages, Path("proto_layer.py"))
"""

# Import main exports
from . import kaitai_struct, scapy_layer, wireshark

__all__ = [
    "kaitai_struct",
    "scapy_layer",
    "wireshark",
]
