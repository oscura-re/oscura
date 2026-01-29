"""UDS (Unified Diagnostic Services) protocol support per ISO 14229.

This module provides decoding and analysis of UDS diagnostic messages used
in modern automotive ECUs for diagnostics, programming, and security access.

Supported Services (ISO 14229-1):
- 0x10: Diagnostic Session Control
- 0x11: ECU Reset
- 0x14: Clear Diagnostic Information
- 0x19: Read DTC Information
- 0x22: Read Data By Identifier
- 0x23: Read Memory By Address
- 0x27: Security Access
- 0x28: Communication Control
- 0x2E: Write Data By Identifier
- 0x2F: Input Output Control By Identifier
- 0x31: Routine Control
- 0x34: Request Download
- 0x35: Request Upload
- 0x36: Transfer Data
- 0x37: Request Transfer Exit
- 0x3E: Tester Present
- 0x85: Control DTC Setting

Example:
    >>> from oscura.automotive.uds import UDSDecoder
    >>> from oscura.automotive.can.models import CANMessage
    >>> # Create a UDS request message (Diagnostic Session Control)
    >>> msg = CANMessage(
    ...     arbitration_id=0x7DF,
    ...     timestamp=1.0,
    ...     data=bytes([0x02, 0x10, 0x01])  # Length=2, SID=0x10, sub=0x01
    ... )
    >>> service = UDSDecoder.decode_service(msg)
    >>> print(service)
    UDSService(0x10 DiagnosticSessionControl [Request], sub=0x01)
"""

__all__ = [
    "UDSECU",
    "UDSAnalyzer",
    "UDSDecoder",
    "UDSMessage",
    "UDSNegativeResponse",
    "UDSService",
]

from oscura.automotive.uds.analyzer import UDSECU, UDSAnalyzer, UDSMessage
from oscura.automotive.uds.decoder import UDSDecoder
from oscura.automotive.uds.models import UDSNegativeResponse, UDSService
