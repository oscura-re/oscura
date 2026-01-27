"""CoAP (Constrained Application Protocol) analyzer.

This module provides comprehensive CoAP (RFC 7252) protocol analysis including
message parsing, request/response matching, blockwise transfer support, and
observe extension support.

Example:
    >>> from oscura.iot.coap import CoAPAnalyzer, CoAPMessage
    >>> analyzer = CoAPAnalyzer()
    >>> data = bytes([0x40, 0x01, 0x00, 0x01])  # CON GET message
    >>> message = analyzer.parse_message(data, timestamp=0.0)
    >>> print(message.msg_type, message.code)
    CON GET

References:
    RFC 7252: CoAP (Constrained Application Protocol)
    RFC 7959: Blockwise Transfer in CoAP
    RFC 7641: Observing Resources in CoAP
"""

from __future__ import annotations

from oscura.iot.coap.analyzer import CoAPAnalyzer, CoAPExchange, CoAPMessage
from oscura.iot.coap.options import CONTENT_FORMATS, OPTIONS

__all__ = [
    "CONTENT_FORMATS",
    "OPTIONS",
    "CoAPAnalyzer",
    "CoAPExchange",
    "CoAPMessage",
]
