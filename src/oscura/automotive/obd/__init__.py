"""OBD-II diagnostic protocol support.

This module provides OBD-II (On-Board Diagnostics) protocol decoding
for standard vehicle diagnostics.
"""

from __future__ import annotations

__all__ = ["PID", "OBD2Decoder", "OBD2Response"]

try:
    from oscura.automotive.obd.decoder import PID, OBD2Decoder, OBD2Response
except ImportError:
    pass
