"""J1939 heavy-duty vehicle protocol support.

This module provides J1939 protocol decoding for heavy-duty vehicles
(trucks, buses, agriculture, marine).
"""

__all__ = ["J1939Decoder", "J1939Message", "extract_pgn"]

try:
    from oscura.automotive.j1939.decoder import J1939Decoder, J1939Message, extract_pgn
except ImportError:
    pass
