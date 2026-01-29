"""UDS (Unified Diagnostic Services) data models per ISO 14229.

This module defines the core data structures for UDS protocol analysis.
"""

from dataclasses import dataclass

__all__ = [
    "UDSNegativeResponse",
    "UDSService",
]


@dataclass
class UDSService:
    """A decoded UDS service.

    Attributes:
        sid: Service ID (0x10-0xFF). For responses, this is the response SID (e.g., 0x50),
             not the request SID (0x10).
        name: Human-readable service name.
        request: True if request message, False if positive response.
        sub_function: Sub-function byte (if applicable).
        data: Service data payload (excluding SID and sub-function).
    """

    sid: int
    name: str
    request: bool
    sub_function: int | None = None
    data: bytes = b""

    @property
    def is_response(self) -> bool:
        """True if this is a response message."""
        return not self.request

    def __repr__(self) -> str:
        """Human-readable representation."""
        msg_type = "Request" if self.request else "Response"
        subfunc = f", sub=0x{self.sub_function:02X}" if self.sub_function is not None else ""
        data_str = self.data.hex().upper() if self.data else ""
        data_part = f", data={data_str}" if data_str else ""
        return f"UDSService(0x{self.sid:02X} {self.name} [{msg_type}]{subfunc}{data_part})"


@dataclass
class UDSNegativeResponse:
    """A UDS negative response (NRC).

    Negative responses follow format: [0x7F, requested_SID, NRC]

    Attributes:
        requested_sid: The Service ID that was requested.
        nrc: Negative Response Code (0x10-0xFF).
        nrc_name: Human-readable description of the NRC.
    """

    requested_sid: int
    nrc: int
    nrc_name: str

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"UDSNegativeResponse(requested_SID=0x{self.requested_sid:02X}, "
            f"NRC=0x{self.nrc:02X} [{self.nrc_name}])"
        )
