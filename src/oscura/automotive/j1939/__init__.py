"""J1939 (SAE J1939) protocol support for heavy-duty vehicles.

This module provides comprehensive J1939 protocol analysis including:
- Message decoding and PGN extraction
- Transport protocol (TP.CM, TP.DT, BAM) multi-packet reassembly
- Suspect Parameter Number (SPN) decoding
- Standard SPN definitions

Example:
    >>> from oscura.automotive.j1939 import J1939Analyzer, J1939SPN
    >>> analyzer = J1939Analyzer()
    >>> msg = analyzer.parse_message(0x0CF00400, b'\\xff' * 8)
    >>> print(msg.identifier.pgn)
    61444
"""

from oscura.automotive.j1939.analyzer import (
    J1939SPN,
    J1939Analyzer,
    J1939Identifier,
    J1939Message,
)
from oscura.automotive.j1939.decoder import (
    J1939Decoder,
    extract_pgn,
)
from oscura.automotive.j1939.decoder import (
    J1939Message as DecoderMessage,
)
from oscura.automotive.j1939.spns import STANDARD_SPNS, get_standard_spns
from oscura.automotive.j1939.transport import TransportProtocol, TransportSession

__all__ = [
    "J1939SPN",
    # SPN definitions
    "STANDARD_SPNS",
    "DecoderMessage",
    # Analyzer (new comprehensive analyzer)
    "J1939Analyzer",
    # Decoder (legacy decoder)
    "J1939Decoder",
    "J1939Identifier",
    "J1939Message",
    # Transport protocol
    "TransportProtocol",
    "TransportSession",
    "extract_pgn",
    "get_standard_spns",
]
