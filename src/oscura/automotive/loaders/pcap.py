"""PCAP file loader for CAN bus data.

This module provides loading of PCAP files containing SocketCAN frames.
PCAP is a common packet capture format that can contain CAN frames from
network interfaces or recorded with tools like Wireshark or tcpdump.

Supported formats:
    - SocketCAN frames (Linux can0, can1, etc.)
    - CAN frames from pcap-ng format

Requirements:
    - scapy library (install with: uv pip install scapy)
"""

from __future__ import annotations

from pathlib import Path

from oscura.automotive.can.models import CANMessage, CANMessageList

__all__ = ["load_pcap"]


def load_pcap(file_path: Path | str) -> CANMessageList:
    """Load CAN messages from a PCAP file.

    This function reads PCAP files containing SocketCAN frames and converts
    them to Oscura's CANMessage format. It uses scapy to parse the PCAP
    file and extract CAN frames.

    Args:
        file_path: Path to the PCAP or PCAPNG file.

    Returns:
        CANMessageList containing all parsed CAN messages.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ImportError: If scapy is not installed.
        ValueError: If file cannot be parsed or contains no CAN frames.

    Example:
        >>> messages = load_pcap("capture.pcap")
        >>> print(f"Loaded {len(messages)} messages")

    Note:
        Requires scapy to be installed:
            uv pip install oscura[automotive]

        Or manually:
            uv pip install scapy
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP file not found: {path}")

    rdpcap, CAN = _import_scapy_modules()
    messages = CANMessageList()

    try:
        packets = rdpcap(str(path))
        _extract_can_messages(packets, CAN, messages)

    except Exception as e:
        raise ValueError(f"Failed to parse PCAP file {path}: {e}") from e

    if len(messages) == 0:
        raise ValueError(
            f"No CAN frames found in PCAP file {path}. "
            "Ensure the capture contains SocketCAN or CAN frames."
        )

    return messages


def _import_scapy_modules() -> tuple[type, type]:
    """Import and return required scapy modules.

    Returns:
        Tuple of (rdpcap function, CAN class).

    Raises:
        ImportError: If scapy is not installed.
    """
    try:
        from scapy.all import rdpcap
        from scapy.layers.can import CAN

        return rdpcap, CAN  # type: ignore[return-value]
    except ImportError as e:
        msg = "scapy library is required for PCAP loading. Install with: uv pip install scapy"
        raise ImportError(msg) from e


def _extract_can_messages(
    packets: list[object], CAN: type, messages: CANMessageList
) -> float | None:
    """Extract CAN messages from PCAP packets.

    Args:
        packets: List of scapy packets.
        CAN: CAN layer class from scapy.
        messages: CANMessageList to populate.

    Returns:
        First timestamp value for reference.
    """
    first_timestamp: float | None = None

    for packet in packets:
        if CAN in packet:  # type: ignore[operator]
            can_frame = packet[CAN]  # type: ignore[index]
            timestamp = _extract_timestamp(packet, first_timestamp)

            if first_timestamp is None and hasattr(packet, "time"):
                first_timestamp = float(packet.time)

            can_msg = _create_can_message(can_frame, timestamp)
            messages.append(can_msg)

    return first_timestamp


def _extract_timestamp(packet: object, first_timestamp: float | None) -> float:
    """Extract and normalize timestamp from packet.

    Args:
        packet: Scapy packet with time attribute.
        first_timestamp: Reference timestamp for normalization.

    Returns:
        Normalized timestamp in seconds.
    """
    if hasattr(packet, "time"):
        if first_timestamp is None:
            return 0.0
        return float(packet.time) - first_timestamp
    return 0.0


def _create_can_message(can_frame: object, timestamp: float) -> CANMessage:
    """Create CANMessage from scapy CAN frame.

    Args:
        can_frame: Scapy CAN frame object.
        timestamp: Message timestamp.

    Returns:
        CANMessage object.
    """
    arb_id: int = int(can_frame.identifier)  # type: ignore[attr-defined]
    # Check if data attribute exists before accessing
    if hasattr(can_frame, "data"):
        data: bytes = bytes(can_frame.data)
    else:
        data = b""

    # Determine if extended ID (bit 31 indicates extended format)
    is_extended = bool(arb_id & 0x80000000)
    if is_extended:
        arb_id = arb_id & 0x1FFFFFFF  # Mask to get 29-bit ID

    # Determine if CAN-FD
    if hasattr(can_frame, "flags"):
        is_fd: bool = bool(can_frame.flags & 0x01)
    else:
        is_fd = False

    # Extract channel if available
    if hasattr(can_frame, "channel"):
        channel: int = int(can_frame.channel)
    else:
        channel = 0

    return CANMessage(
        arbitration_id=arb_id,
        timestamp=timestamp,
        data=data,
        is_extended=is_extended,
        is_fd=is_fd,
        channel=channel,
    )
