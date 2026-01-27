"""High-level reverse engineering workflow for unknown signals.

This module provides a complete workflow for reverse engineering unknown
digital signals from initial capture to protocol understanding.

Example:
    >>> import oscura as osc
    >>> trace = osc.load("unknown_capture.wfm")
    >>> result = osc.workflows.reverse_engineer_signal(trace)
    >>> print(result.protocol_spec)
    >>> print(f"Detected baud rate: {result.baud_rate}")
    >>> print(f"Frames decoded: {len(result.frames)}")

The workflow includes:
1. Signal characterization (voltage levels, signal type)
2. Clock recovery / baud rate detection
3. Bit stream extraction
4. Frame boundary detection
5. Sync pattern identification
6. Field structure inference
7. Checksum analysis
8. Protocol specification generation

References:
    - sigrok Protocol Analysis
    - UART: TIA-232-F
    - I2C: NXP UM10204
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


@dataclass
class InferredFrame:
    """An inferred protocol frame.

    Attributes:
        start_bit: Starting bit index.
        end_bit: Ending bit index.
        raw_bits: Raw bit string.
        raw_bytes: Raw bytes.
        fields: Identified field mapping.
        checksum_valid: Whether checksum validated (None if unknown).
    """

    start_bit: int
    end_bit: int
    raw_bits: str
    raw_bytes: bytes
    fields: dict[str, bytes] = field(default_factory=dict)
    checksum_valid: bool | None = None


@dataclass
class FieldSpec:
    """Specification for an inferred field.

    Attributes:
        name: Field name.
        offset: Byte offset in frame.
        size: Size in bytes (or expression for variable).
        field_type: Data type (uint8, bytes, checksum, etc.).
        value: Example or constant value.
    """

    name: str
    offset: int
    size: int | str
    field_type: str
    value: Any = None


@dataclass
class ProtocolSpec:
    """Inferred protocol specification.

    Attributes:
        name: Protocol name.
        baud_rate: Detected baud rate.
        frame_format: Frame format (e.g., "8N1").
        sync_pattern: Detected sync pattern (hex string).
        frame_length: Frame length in bytes (or None if variable).
        fields: List of field specifications.
        checksum_type: Detected checksum type (or None).
        checksum_position: Position of checksum in frame.
        confidence: Overall confidence score (0-1).
    """

    name: str
    baud_rate: float
    frame_format: str
    sync_pattern: str
    frame_length: int | None
    fields: list[FieldSpec]
    checksum_type: str | None
    checksum_position: int | None
    confidence: float


@dataclass
class ReverseEngineeringResult:
    """Complete results from reverse engineering workflow.

    Attributes:
        protocol_spec: Inferred protocol specification.
        frames: List of decoded frames.
        baud_rate: Detected baud rate.
        bit_stream: Extracted bit stream.
        byte_stream: Extracted byte stream.
        sync_positions: Positions where sync patterns found.
        characterization: Signal characterization results.
        confidence: Overall analysis confidence (0-1).
        warnings: List of analysis warnings.
    """

    protocol_spec: ProtocolSpec
    frames: list[InferredFrame]
    baud_rate: float
    bit_stream: str
    byte_stream: bytes
    sync_positions: list[int]
    characterization: dict[str, Any]
    confidence: float
    warnings: list[str]


def reverse_engineer_signal(
    trace: WaveformTrace,
    *,
    expected_baud_rates: list[int] | None = None,
    min_frames: int = 3,
    max_frame_length: int = 256,
    checksum_types: list[str] | None = None,
) -> ReverseEngineeringResult:
    """Complete reverse engineering workflow for unknown signals.

    Analyzes an unknown digital signal to infer protocol parameters,
    frame structure, and decode messages.

    Args:
        trace: Input waveform trace.
        expected_baud_rates: List of expected baud rates to try.
            Default: [9600, 19200, 38400, 57600, 115200].
        min_frames: Minimum frames required for analysis (default 3).
        max_frame_length: Maximum expected frame length in bytes.
        checksum_types: Checksum types to try.
            Default: ["xor", "sum8", "crc8", "crc16"].

    Returns:
        ReverseEngineeringResult with protocol specification and decoded frames.

    Example:
        >>> trace = osc.load("unknown_capture.wfm")
        >>> result = osc.workflows.reverse_engineer_signal(trace)
        >>> print(f"Baud rate: {result.baud_rate}")
        >>> print(f"Sync pattern: {result.protocol_spec.sync_pattern}")
        >>> print(f"Frame length: {result.protocol_spec.frame_length} bytes")
        >>> for frame in result.frames[:5]:
        ...     print(f"  {frame.raw_bytes.hex()}")
    """
    expected_baud_rates = expected_baud_rates or [
        9600,
        19200,
        38400,
        57600,
        115200,
        230400,
        460800,
    ]
    checksum_types = checksum_types or ["xor", "sum8", "crc8", "crc16"]

    warnings: list[str] = []
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    # ========== Step 1-3: Signal Analysis ==========
    characterization = _characterize_signal(data, sample_rate)
    baud_rate, baud_confidence = _detect_baud_rate(
        data, sample_rate, expected_baud_rates, characterization["threshold"]
    )
    _validate_baud_confidence(baud_confidence, warnings)

    bit_stream = _extract_bit_stream(data, sample_rate, baud_rate, characterization["threshold"])
    _validate_bit_stream(bit_stream, warnings)

    # ========== Step 4-5: Byte and Sync Analysis ==========
    byte_positions, byte_stream = _extract_bytes(bit_stream)
    _validate_byte_stream(byte_stream, warnings)

    sync_pattern, sync_positions, sync_confidence = _detect_sync_pattern(
        byte_stream, max_frame_length
    )

    # ========== Step 6-8: Frame and Protocol Analysis ==========
    frames = _extract_frames(bit_stream, byte_stream, byte_positions, sync_positions, sync_pattern)
    _validate_frame_count(frames, min_frames, warnings)

    field_specs = _infer_fields(frames, sync_pattern)
    checksum_type, checksum_pos, checksum_confidence = _detect_checksum(frames, checksum_types)
    _validate_frame_checksums(frames, checksum_type, checksum_pos)

    # ========== Build Results ==========
    protocol_spec = _build_protocol_spec(
        baud_rate,
        sync_pattern,
        frames,
        field_specs,
        checksum_type,
        checksum_pos,
        baud_confidence,
        sync_confidence,
        checksum_confidence,
    )

    return ReverseEngineeringResult(
        protocol_spec=protocol_spec,
        frames=frames,
        baud_rate=baud_rate,
        bit_stream=bit_stream,
        byte_stream=byte_stream,
        sync_positions=sync_positions,
        characterization=characterization,
        confidence=protocol_spec.confidence,
        warnings=warnings,
    )


def _validate_baud_confidence(baud_confidence: float, warnings: list[str]) -> None:
    """Validate baud rate confidence and add warning if low.

    Args:
        baud_confidence: Confidence score for baud rate detection.
        warnings: List to append warnings to.
    """
    if baud_confidence < 0.7:
        warnings.append(f"Low baud rate confidence: {baud_confidence:.2f}")


def _validate_bit_stream(bit_stream: str, warnings: list[str]) -> None:
    """Validate bit stream length and add warning if short.

    Args:
        bit_stream: Extracted bit stream.
        warnings: List to append warnings to.
    """
    if len(bit_stream) < 100:
        warnings.append("Short bit stream extracted")


def _validate_byte_stream(byte_stream: bytes, warnings: list[str]) -> None:
    """Validate byte stream length and add warning if short.

    Args:
        byte_stream: Extracted byte stream.
        warnings: List to append warnings to.
    """
    if len(byte_stream) < 10:
        warnings.append("Few bytes extracted")


def _validate_frame_count(
    frames: list[InferredFrame], min_frames: int, warnings: list[str]
) -> None:
    """Validate frame count and add warning if insufficient.

    Args:
        frames: List of extracted frames.
        min_frames: Minimum required frames.
        warnings: List to append warnings to.
    """
    if len(frames) < min_frames:
        warnings.append(f"Only {len(frames)} frames found (minimum {min_frames})")


def _validate_frame_checksums(
    frames: list[InferredFrame], checksum_type: str | None, checksum_pos: int | None
) -> None:
    """Validate checksums for all frames.

    Args:
        frames: List of frames to validate.
        checksum_type: Detected checksum type.
        checksum_pos: Position of checksum in frame.
    """
    if checksum_type:
        for frame in frames:
            frame.checksum_valid = _verify_checksum(frame.raw_bytes, checksum_type, checksum_pos)


def _build_protocol_spec(
    baud_rate: float,
    sync_pattern: bytes,
    frames: list[InferredFrame],
    field_specs: list[FieldSpec],
    checksum_type: str | None,
    checksum_pos: int | None,
    baud_confidence: float,
    sync_confidence: float,
    checksum_confidence: float,
) -> ProtocolSpec:
    """Build protocol specification from analysis results.

    Args:
        baud_rate: Detected baud rate.
        sync_pattern: Detected sync pattern.
        frames: List of extracted frames.
        field_specs: List of field specifications.
        checksum_type: Detected checksum type.
        checksum_pos: Position of checksum.
        baud_confidence: Baud rate detection confidence.
        sync_confidence: Sync pattern detection confidence.
        checksum_confidence: Checksum detection confidence.

    Returns:
        Complete protocol specification.
    """
    frame_lengths = [len(f.raw_bytes) for f in frames]
    frame_length = int(np.median(frame_lengths)) if len(set(frame_lengths)) == 1 else None

    overall_confidence = (
        baud_confidence * 0.3
        + sync_confidence * 0.3
        + checksum_confidence * 0.2
        + min(len(frames) / 10, 1.0) * 0.2
    )

    return ProtocolSpec(
        name="Unknown Protocol (Inferred)",
        baud_rate=baud_rate,
        frame_format="8N1",
        sync_pattern=sync_pattern.hex() if sync_pattern else "",
        frame_length=frame_length,
        fields=field_specs,
        checksum_type=checksum_type,
        checksum_position=checksum_pos,
        confidence=overall_confidence,
    )


def _characterize_signal(
    data: np.ndarray[Any, np.dtype[np.float64]], sample_rate: float
) -> dict[str, Any]:
    """Characterize signal voltage levels and type."""
    high_level = float(np.percentile(data, 95))
    low_level = float(np.percentile(data, 5))
    threshold = (high_level + low_level) / 2
    swing = high_level - low_level

    # Detect if inverted (idle low vs idle high)
    is_inverted = np.mean(data) > threshold

    # Detect signal type
    signal_type = "digital"  # Default assumption
    if swing < 0.5:
        signal_type = "low_swing"
    elif swing > 10:
        signal_type = "high_swing"

    return {
        "high_level": high_level,
        "low_level": low_level,
        "threshold": threshold,
        "swing": swing,
        "is_inverted": is_inverted,
        "signal_type": signal_type,
        "sample_rate": sample_rate,
    }


def _detect_baud_rate(
    data: np.ndarray[Any, np.dtype[np.float64]],
    sample_rate: float,
    expected_rates: list[int],
    threshold: float,
) -> tuple[float, float]:
    """Detect baud rate from edge timing."""
    # Convert to digital
    digital = data > threshold
    edges = np.where(np.diff(digital.astype(int)) != 0)[0]

    if len(edges) < 20:
        # Default to most common baud rate if not enough edges
        return 115200, 0.3

    # Measure edge-to-edge intervals
    intervals = np.diff(edges)
    intervals = intervals[intervals > 5]  # Filter very short glitches

    if len(intervals) < 10:
        return 115200, 0.3

    # Find minimum interval (single bit period)
    min_interval = float(np.percentile(intervals, 5))

    # Estimate baud rate
    estimated_baud = sample_rate / min_interval

    # Find closest standard baud rate
    closest_baud = min(expected_rates, key=lambda x: abs(x - estimated_baud))

    # Calculate confidence based on how close we are
    error_percent = abs(estimated_baud - closest_baud) / closest_baud
    confidence = max(0.0, 1.0 - error_percent * 5)

    return float(closest_baud), confidence


def _extract_bit_stream(
    data: np.ndarray[Any, np.dtype[np.float64]],
    sample_rate: float,
    baud_rate: float,
    threshold: float,
) -> str:
    """Extract bit stream by sampling at bit centers."""
    samples_per_bit = int(sample_rate / baud_rate)
    n_bits = len(data) // samples_per_bit

    bits: list[str] = []
    for i in range(n_bits):
        sample_idx = i * samples_per_bit + samples_per_bit // 2
        if sample_idx < len(data):
            bit = "1" if data[sample_idx] > threshold else "0"
            bits.append(bit)

    return "".join(bits)


def _extract_bytes(bit_stream: str) -> tuple[list[int], bytes]:
    """Extract bytes from bit stream (8N1 format)."""
    byte_positions: list[int] = []
    byte_values: list[int] = []

    bit_pos = 0
    while bit_pos < len(bit_stream) - 10:
        # Look for start bit (0)
        if bit_stream[bit_pos] == "0":
            # Extract 8 data bits (LSB first for UART)
            byte_bits = bit_stream[bit_pos + 1 : bit_pos + 9]
            if len(byte_bits) == 8:
                byte_val = sum(int(byte_bits[i]) << i for i in range(8))
                byte_positions.append(bit_pos)
                byte_values.append(byte_val)
            bit_pos += 10  # Skip to next potential start
        else:
            bit_pos += 1

    return byte_positions, bytes(byte_values)


def _detect_sync_pattern(
    byte_stream: bytes,
    max_frame_length: int,
) -> tuple[bytes, list[int], float]:
    """Detect sync pattern by finding repeated byte sequences.

    Args:
        byte_stream: Stream of bytes to analyze.
        max_frame_length: Maximum expected frame length.

    Returns:
        Tuple of (pattern, positions, confidence).
    """
    if len(byte_stream) < 20:
        return b"", [], 0.0

    # Try common sync patterns first
    best_pattern, best_positions, best_confidence = _find_common_sync_patterns(byte_stream)

    # If no common pattern found, try to find repeating sequences
    if best_confidence < 0.5:
        pattern, positions, confidence = _find_repeating_patterns(byte_stream)
        if confidence > best_confidence:
            best_pattern, best_positions, best_confidence = pattern, positions, confidence

    return best_pattern, best_positions, best_confidence


def _find_common_sync_patterns(byte_stream: bytes) -> tuple[bytes, list[int], float]:
    """Search for common sync patterns.

    Args:
        byte_stream: Stream of bytes to analyze.

    Returns:
        Tuple of (best_pattern, positions, confidence).
    """
    common_patterns = [
        bytes([0xAA, 0x55]),
        bytes([0x55, 0xAA]),
        bytes([0x7E]),  # HDLC
        bytes([0xA5]),
        bytes([0x5A]),
        bytes([0xFF, 0x00]),
    ]

    best_pattern = b""
    best_positions: list[int] = []
    best_confidence = 0.0

    for pattern in common_patterns:
        positions = _find_pattern_positions(byte_stream, pattern)

        if len(positions) >= 3:
            confidence = _calculate_pattern_confidence(positions)

            if confidence > best_confidence:
                best_pattern = pattern
                best_positions = positions
                best_confidence = confidence

    return best_pattern, best_positions, best_confidence


def _find_repeating_patterns(byte_stream: bytes) -> tuple[bytes, list[int], float]:
    """Search for repeating byte sequences.

    Args:
        byte_stream: Stream of bytes to analyze.

    Returns:
        Tuple of (best_pattern, positions, confidence).
    """
    best_pattern = b""
    best_positions: list[int] = []
    best_confidence = 0.0

    for pattern_len in range(1, 4):
        for start in range(min(50, len(byte_stream) - pattern_len)):
            pattern = byte_stream[start : start + pattern_len]
            positions = _find_pattern_positions(byte_stream, pattern)

            if len(positions) >= 5:
                spacings = np.diff(positions)
                if len(spacings) > 0 and np.std(spacings) / (np.median(spacings) + 1) < 0.2:
                    confidence = min(len(positions) / 10, 1.0)
                    if confidence > best_confidence:
                        best_pattern = pattern
                        best_positions = positions
                        best_confidence = confidence

    return best_pattern, best_positions, best_confidence


def _find_pattern_positions(byte_stream: bytes, pattern: bytes) -> list[int]:
    """Find all positions where pattern occurs in byte stream.

    Args:
        byte_stream: Stream to search.
        pattern: Pattern to find.

    Returns:
        List of byte positions where pattern occurs.
    """
    positions = []
    for i in range(len(byte_stream) - len(pattern)):
        if byte_stream[i : i + len(pattern)] == pattern:
            positions.append(i)
    return positions


def _calculate_pattern_confidence(positions: list[int]) -> float:
    """Calculate confidence score for pattern regularity.

    Args:
        positions: List of pattern positions.

    Returns:
        Confidence score (0.0 to 1.0).
    """
    spacings = np.diff(positions)
    if len(spacings) == 0:
        return 0.0

    median_spacing = float(np.median(spacings))
    regularity = 1.0 - float(np.std(spacings)) / (median_spacing + 1)

    return min(len(positions) / 10, 1.0) * max(regularity, 0)


def _extract_frames(
    bit_stream: str,
    byte_stream: bytes,
    byte_positions: list[int],
    sync_positions: list[int],
    sync_pattern: bytes,
) -> list[InferredFrame]:
    """Extract frames based on sync positions."""
    frames: list[InferredFrame] = []

    if not sync_positions or len(sync_positions) < 2:
        return frames

    # Calculate frame length from sync spacing
    spacings = np.diff(sync_positions)
    frame_length = int(np.median(spacings))

    for _i, sync_pos in enumerate(sync_positions[:-1]):
        end_pos = min(sync_pos + frame_length, len(byte_stream))
        frame_bytes = byte_stream[sync_pos:end_pos]

        if len(frame_bytes) < 3:
            continue

        # Get bit positions
        start_bit = byte_positions[sync_pos] if sync_pos < len(byte_positions) else sync_pos * 10
        end_bit = byte_positions[end_pos - 1] if end_pos - 1 < len(byte_positions) else end_pos * 10

        frames.append(
            InferredFrame(
                start_bit=start_bit,
                end_bit=end_bit,
                raw_bits=bit_stream[start_bit:end_bit] if end_bit <= len(bit_stream) else "",
                raw_bytes=frame_bytes,
            )
        )

    return frames


def _infer_fields(frames: list[InferredFrame], sync_pattern: bytes) -> list[FieldSpec]:
    """Infer field structure from frames."""
    if not frames:
        return []

    fields: list[FieldSpec] = []
    sync_len = len(sync_pattern)

    _add_sync_field(fields, sync_pattern)
    length_offset = _try_add_length_field(fields, frames, sync_len)
    _add_data_field(fields, frames, length_offset, sync_len)
    _add_checksum_field(fields)

    return fields


def _add_sync_field(fields: list[FieldSpec], sync_pattern: bytes) -> None:
    """Add sync field if pattern exists.

    Args:
        fields: Field list to append to.
        sync_pattern: Sync pattern bytes.
    """
    if len(sync_pattern) > 0:
        fields.append(
            FieldSpec(
                name="sync",
                offset=0,
                size=len(sync_pattern),
                field_type="constant",
                value=sync_pattern.hex(),
            )
        )


def _try_add_length_field(
    fields: list[FieldSpec], frames: list[InferredFrame], sync_len: int
) -> int:
    """Try to detect and add length field.

    Args:
        fields: Field list to append to.
        frames: Frame list to analyze.
        sync_len: Length of sync pattern.

    Returns:
        Offset after length field (or sync_len if no length detected).
    """
    frame_lengths = [len(f.raw_bytes) for f in frames]
    min_len = min(frame_lengths)
    length_offset = sync_len

    if min_len <= length_offset:
        return length_offset

    length_values = [f.raw_bytes[length_offset] for f in frames if len(f.raw_bytes) > length_offset]

    if len(length_values) <= 2:
        return length_offset

    # Check correlation
    if len(set(length_values)) > 1:
        correlation = np.corrcoef(length_values, frame_lengths)[0, 1]
    else:
        correlation = 0

    if correlation > 0.8 or (len(set(length_values)) == 1 and length_values[0] == frame_lengths[0]):
        fields.append(FieldSpec(name="length", offset=length_offset, size=1, field_type="uint8"))
        return length_offset + 1

    return length_offset


def _add_data_field(
    fields: list[FieldSpec],
    frames: list[InferredFrame],
    length_offset: int,
    sync_len: int,
) -> None:
    """Add data field for payload.

    Args:
        fields: Field list to append to.
        frames: Frame list.
        length_offset: Offset after length field.
        sync_len: Length of sync pattern.
    """
    min_len = min(len(f.raw_bytes) for f in frames)

    if min_len > length_offset + 1:
        fields.append(
            FieldSpec(
                name="data",
                offset=length_offset,
                size="length - sync_len - 2",
                field_type="bytes",
            )
        )


def _add_checksum_field(fields: list[FieldSpec]) -> None:
    """Add checksum field at end.

    Args:
        fields: Field list to append to.
    """
    fields.append(FieldSpec(name="checksum", offset=-1, size=1, field_type="checksum"))


def _detect_checksum(
    frames: list[InferredFrame],
    checksum_types: list[str],
) -> tuple[str | None, int | None, float]:
    """Detect checksum type by trying different algorithms."""
    if len(frames) < 3:
        return None, None, 0.0

    best_type: str | None = None
    best_pos: int | None = None
    best_matches = 0

    for chk_type in checksum_types:
        # Assume checksum is last byte
        pos = -1
        matches = 0

        for frame in frames:
            if len(frame.raw_bytes) < 3:
                continue

            data = frame.raw_bytes[:-1]  # All but last byte
            expected = frame.raw_bytes[-1]

            calculated = _calculate_checksum(data, chk_type)
            if calculated == expected:
                matches += 1

        if matches > best_matches:
            best_matches = matches
            best_type = chk_type
            best_pos = pos

    confidence = best_matches / len(frames) if frames else 0
    if confidence < 0.5:
        return None, None, confidence

    return best_type, best_pos, confidence


def _calculate_checksum(data: bytes, checksum_type: str) -> int:
    """Calculate checksum using specified algorithm."""
    checksum_funcs = {
        "xor": _checksum_xor,
        "sum8": _checksum_sum8,
        "crc8": _checksum_crc8,
        "crc16": _checksum_crc16,
    }

    func = checksum_funcs.get(checksum_type)
    return func(data) if func else 0


def _checksum_xor(data: bytes) -> int:
    """Calculate XOR checksum."""
    result = 0
    for b in data:
        result ^= b
    return result


def _checksum_sum8(data: bytes) -> int:
    """Calculate 8-bit sum checksum."""
    return sum(data) & 0xFF


def _checksum_crc8(data: bytes) -> int:
    """Calculate simple CRC-8."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _checksum_crc16(data: bytes) -> int:
    """Calculate CRC-16-CCITT (return low byte only)."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFF


def _verify_checksum(data: bytes, checksum_type: str, checksum_pos: int | None) -> bool:
    """Verify checksum in frame."""
    if checksum_pos == -1 or checksum_pos is None:
        # Checksum is last byte
        frame_data = data[:-1]
        expected = data[-1]
    else:
        frame_data = data[:checksum_pos] + data[checksum_pos + 1 :]
        expected = data[checksum_pos]

    calculated = _calculate_checksum(frame_data, checksum_type)
    return calculated == expected
