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
from typing import TYPE_CHECKING

from oscura.automotive.can.models import CANMessage, CANMessageList

if TYPE_CHECKING:
    from scapy.packet import Packet  # type: ignore[import-not-found]

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

    try:
        from scapy.all import rdpcap  # type: ignore[import-not-found]
        from scapy.layers.can import CAN  # type: ignore[import-not-found]
    except ImportError as e:
        msg = "scapy library is required for PCAP loading. Install with: uv pip install scapy"
        raise ImportError(msg) from e

    messages = CANMessageList()

    try:
        # Read PCAP file
        packets = rdpcap(str(path))

        # Extract CAN frames
        first_timestamp: float | None = None
        for packet in packets:
            # Check if packet contains CAN layer
            if CAN in packet:
                can_frame: Packet = packet[CAN]

                # Get timestamp
                if hasattr(packet, "time"):
                    if first_timestamp is None:
                        first_timestamp = float(packet.time)
                    timestamp = float(packet.time) - first_timestamp
                else:
                    timestamp = 0.0

                # Extract CAN ID and data
                arb_id = int(can_frame.identifier)

                # Get data bytes (scapy stores CAN data as bytes)
                if hasattr(can_frame, "data"):
                    data = bytes(can_frame.data)
                else:
                    data = b""

                # Determine if extended ID (bit 31 indicates extended format)
                # SocketCAN uses bit 31 for extended frame flag
                is_extended = bool(arb_id & 0x80000000)
                if is_extended:
                    arb_id = arb_id & 0x1FFFFFFF  # Mask to get 29-bit ID

                # Determine if CAN-FD (scapy may have an FD flag)
                is_fd = hasattr(can_frame, "flags") and (can_frame.flags & 0x01)

                # Extract channel if available
                channel = 0
                if hasattr(can_frame, "channel"):
                    channel = int(can_frame.channel)

                # Create CANMessage
                can_msg = CANMessage(
                    arbitration_id=arb_id,
                    timestamp=timestamp,
                    data=data,
                    is_extended=is_extended,
                    is_fd=is_fd,
                    channel=channel,
                )
                messages.append(can_msg)

    except Exception as e:
        raise ValueError(f"Failed to parse PCAP file {path}: {e}") from e

    if len(messages) == 0:
        raise ValueError(
            f"No CAN frames found in PCAP file {path}. "
            "Ensure the capture contains SocketCAN or CAN frames."
        )

    return messages
