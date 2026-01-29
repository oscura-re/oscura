"""ASAM MDF/MF4 (Measurement Data Format) file loader.

This module provides loading of MDF4/MF4 files using the asammdf library.
MDF is an industry-standard format for storing measurement data,
commonly used with CAN bus loggers like CANedge.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from oscura.automotive.can.models import CANMessage, CANMessageList

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["load_mdf"]


def load_mdf(file_path: Path | str) -> CANMessageList:
    """Load CAN messages from an MDF/MF4 file.

    This loader supports both MDF3 and MDF4 (MF4) formats. It handles various
    CAN logging structures commonly found in MDF files from tools like CANedge,
    Vector CANalyzer, ETAS INCA, and others.

    The loader attempts multiple extraction strategies:
    1. Bus logging format (CAN frames as structured records)
    2. Signal-based format (CAN ID, data, timestamps as separate channels)
    3. Raw frame format (binary CAN frames)

    Args:
        file_path: Path to the MDF/MF4 file.

    Returns:
        CANMessageList containing all parsed CAN messages.

    Raises:
        ImportError: If asammdf is not installed.
        FileNotFoundError: If file doesn't exist.
        ValueError: If file cannot be parsed or contains no CAN data.

    Example:
        >>> messages = load_mdf("capture.mf4")
        >>> print(f"Loaded {len(messages)} messages")
        >>> print(f"Unique IDs: {len(messages.unique_ids())}")
    """
    try:
        from asammdf import MDF
    except ImportError as e:
        raise ImportError(
            "asammdf is required for MDF/MF4 file support. "
            "Install with: pip install 'oscura[automotive]'"
        ) from e

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"MDF file not found: {path}")

    messages = CANMessageList()

    try:
        # Open MDF file (supports both MDF3 and MDF4)
        with MDF(str(path)) as mdf:
            # Strategy 1: Try to extract CAN bus logging data
            # Look for channels with CAN-related names
            _extract_can_bus_logging(mdf, messages)

            # Strategy 2: If no messages found, try signal-based extraction
            # Some MDF files store CAN ID, data, timestamp as separate signals
            if not messages.messages:
                _extract_can_signals(mdf, messages)

            # Strategy 3: Try iterating through all channels looking for CAN data patterns
            if not messages.messages:
                _extract_can_channels(mdf, messages)

    except Exception as e:
        raise ValueError(f"Failed to parse MDF file {path}: {e}") from e

    if not messages.messages:
        raise ValueError(
            f"No CAN messages found in MDF file {path}. "
            "File may not contain CAN bus data or uses unsupported format."
        )

    return messages


def _extract_can_bus_logging(mdf: Any, messages: CANMessageList) -> None:
    """Extract CAN messages from bus logging format.

    This handles MDF files that store CAN frames as structured bus logging data,
    common in Vector tools and CANedge loggers.

    Args:
        mdf: Opened MDF file object.
        messages: CANMessageList to append extracted messages to.
    """
    # Common CAN bus logging channel patterns
    can_bus_patterns = [
        "CAN_DataFrame",
        "CANBus",
        "CAN_Message",
        "CAN.DataFrames",
        "CAN Bus",
    ]

    for channel_name in mdf.channels_db:
        # Check if channel name matches CAN bus logging pattern
        if any(pattern in channel_name for pattern in can_bus_patterns):
            try:
                signal = mdf.get(channel_name)

                # Extract timestamps
                timestamps = signal.timestamps

                # CAN data could be in samples (structured) or as raw bytes
                samples = signal.samples

                # Handle different sample structures
                if hasattr(samples, "dtype") and samples.dtype.names:
                    # Structured array with fields
                    _extract_structured_can_frames(samples, timestamps, messages)
                else:
                    # Try to interpret as raw CAN frames
                    _extract_raw_can_frames(samples, timestamps, messages)

            except Exception:
                # Skip channels that fail to parse
                continue


def _extract_structured_can_frames(
    samples: npt.NDArray[Any], timestamps: npt.NDArray[Any], messages: CANMessageList
) -> None:
    """Extract CAN messages from structured array format.

    Args:
        samples: Structured numpy array with CAN frame fields.
        timestamps: Array of timestamps.
        messages: CANMessageList to append to.
    """
    field_names = samples.dtype.names
    if not field_names:
        return

    field_map = _find_can_field_mapping(field_names)
    if not field_map["id_field"]:
        return

    for i in range(len(samples)):
        try:
            can_msg = _extract_can_message_from_sample(samples, timestamps, i, field_map)
            messages.append(can_msg)
        except (IndexError, ValueError, TypeError):
            continue


def _find_can_field_mapping(field_names: tuple[str, ...]) -> dict[str, str | None]:
    """Find mapping of CAN field names in structured array.

    Args:
        field_names: Field names from structured array.

    Returns:
        Dictionary mapping field types to actual field names.
    """
    id_fields = ["ID", "id", "BusID", "Identifier", "ArbitrationID"]
    data_fields = ["Data", "data", "DataBytes", "Payload"]
    dlc_fields = ["DLC", "dlc", "DataLength"]

    return {
        "id_field": next((f for f in id_fields if f in field_names), None),
        "data_field": next((f for f in data_fields if f in field_names), None),
        "dlc_field": next((f for f in dlc_fields if f in field_names), None),
    }


def _extract_can_message_from_sample(
    samples: npt.NDArray[Any],
    timestamps: npt.NDArray[Any],
    index: int,
    field_map: dict[str, str | None],
) -> CANMessage:
    """Extract single CAN message from sample.

    Args:
        samples: Structured array.
        timestamps: Timestamp array.
        index: Sample index.
        field_map: Field name mapping.

    Returns:
        CANMessage instance.
    """
    arb_id = int(samples[field_map["id_field"]][index])
    data = _extract_can_data_field(samples, index, field_map)
    is_extended = arb_id > 0x7FF

    return CANMessage(
        arbitration_id=arb_id,
        timestamp=float(timestamps[index]),
        data=data,
        is_extended=is_extended,
        is_fd=False,
        channel=0,
    )


def _extract_can_data_field(
    samples: npt.NDArray[Any], index: int, field_map: dict[str, str | None]
) -> bytes:
    """Extract data bytes from sample.

    Args:
        samples: Structured array.
        index: Sample index.
        field_map: Field name mapping.

    Returns:
        Data bytes.
    """
    data_field = field_map["data_field"]
    if not data_field:
        return b""

    data_bytes = samples[data_field][index]
    if isinstance(data_bytes, bytes):
        data = data_bytes
    else:
        data = bytes(data_bytes)

    dlc_field = field_map["dlc_field"]
    if dlc_field:
        dlc = int(samples[dlc_field][index])
        data = data[:dlc]

    return data


def _extract_raw_can_frames(
    samples: npt.NDArray[Any], timestamps: npt.NDArray[Any], messages: CANMessageList
) -> None:
    """Extract CAN messages from raw byte format.

    Args:
        samples: Raw bytes array.
        timestamps: Array of timestamps.
        messages: CANMessageList to append to.
    """
    # Attempt to parse raw CAN frames
    # Typical raw CAN frame: [ID (4 bytes)] [DLC (1 byte)] [Data (up to 8 bytes)]
    for i in range(len(samples)):
        try:
            frame = samples[i]
            if isinstance(frame, bytes) and len(frame) >= 5:
                # Parse standard raw CAN frame structure
                arb_id = int.from_bytes(frame[0:4], byteorder="little")
                dlc = min(frame[4], 8)
                data = frame[5 : 5 + dlc]

                # Determine if extended ID
                is_extended = arb_id > 0x7FF

                can_msg = CANMessage(
                    arbitration_id=arb_id,
                    timestamp=float(timestamps[i]),
                    data=data,
                    is_extended=is_extended,
                    is_fd=False,
                    channel=0,
                )
                messages.append(can_msg)

        except (IndexError, ValueError, TypeError):
            continue


def _extract_can_signals(mdf: Any, messages: CANMessageList) -> None:
    """Extract CAN messages from separate signal channels.

    This handles MDF files where CAN ID, data, and timestamps are stored
    as separate channels/signals.

    Args:
        mdf: Opened MDF file object.
        messages: CANMessageList to append to.
    """
    # Look for separate CAN ID and data channels
    id_patterns = ["CAN_ID", "CANID", "CAN.ID", "Identifier"]
    data_patterns = ["CAN_Data", "CANData", "CAN.Data", "Payload"]

    id_channels = []
    data_channels = []

    for channel_name in mdf.channels_db:
        if any(pattern in channel_name for pattern in id_patterns):
            id_channels.append(channel_name)
        elif any(pattern in channel_name for pattern in data_patterns):
            data_channels.append(channel_name)

    # Try to match ID and data channels
    for id_channel in id_channels:
        for data_channel in data_channels:
            try:
                id_signal = mdf.get(id_channel)
                data_signal = mdf.get(data_channel)

                # Ensure same length
                if len(id_signal.timestamps) != len(data_signal.timestamps):
                    continue

                # Extract messages
                for i in range(len(id_signal.timestamps)):
                    arb_id = int(id_signal.samples[i])
                    data = bytes(data_signal.samples[i])
                    timestamp = float(id_signal.timestamps[i])

                    is_extended = arb_id > 0x7FF

                    can_msg = CANMessage(
                        arbitration_id=arb_id,
                        timestamp=timestamp,
                        data=data,
                        is_extended=is_extended,
                        is_fd=False,
                        channel=0,
                    )
                    messages.append(can_msg)

            except Exception:
                continue


def _extract_can_channels(mdf: Any, messages: CANMessageList) -> None:
    """Extract CAN messages by searching all channels for CAN patterns.

    This is a fallback strategy that looks for any channel containing
    'CAN' in the name and attempts to extract CAN data.

    Args:
        mdf: Opened MDF file object.
        messages: CANMessageList to append to.
    """
    can_keywords = ["CAN", "can"]

    for channel_name in mdf.channels_db:
        # Only process channels with CAN in the name
        if not any(keyword in channel_name for keyword in can_keywords):
            continue

        try:
            signal = mdf.get(channel_name)
            timestamps = signal.timestamps
            samples = signal.samples

            # Try different interpretations
            if hasattr(samples, "dtype"):
                if samples.dtype.names:
                    # Structured
                    _extract_structured_can_frames(samples, timestamps, messages)
                elif samples.dtype == "uint32" or samples.dtype == "int32":
                    # Could be CAN IDs
                    pass
                elif len(samples.shape) == 2 and samples.shape[1] >= 8:
                    # Could be CAN data bytes
                    pass

        except Exception:
            continue
