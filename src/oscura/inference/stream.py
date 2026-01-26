"""Stream reassembly and message framing for network protocols.

    - RE-STR-001: UDP Stream Reconstruction
    - RE-STR-002: TCP Stream Reassembly
    - RE-STR-003: Message Framing and Segmentation

This module provides tools for reconstructing application-layer data from
transport-layer segments, handling out-of-order delivery, gaps, and
message boundaries.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class StreamSegment:
    """A segment of stream data.

    Implements RE-STR-001, RE-STR-002: Stream segment.

    Attributes:
        sequence_number: Sequence number (TCP) or packet number (UDP).
        data: Segment payload.
        timestamp: Capture timestamp.
        src: Source address.
        dst: Destination address.
        flags: Protocol flags.
        is_retransmit: Whether this is a retransmission.
    """

    sequence_number: int
    data: bytes
    timestamp: float = 0.0
    src: str = ""
    dst: str = ""
    flags: int = 0
    is_retransmit: bool = False


@dataclass
class ReassembledStream:
    """A fully reassembled stream.

    Implements RE-STR-001, RE-STR-002: Reassembled stream.

    Attributes:
        data: Complete reassembled data.
        src: Source address.
        dst: Destination address.
        start_time: Stream start time.
        end_time: Stream end time.
        segments: Number of segments.
        gaps: List of (start, end) gap ranges.
        retransmits: Number of retransmissions detected.
        out_of_order: Number of out-of-order segments.
    """

    data: bytes
    src: str
    dst: str
    start_time: float
    end_time: float
    segments: int
    gaps: list[tuple[int, int]] = field(default_factory=list)
    retransmits: int = 0
    out_of_order: int = 0


@dataclass
class MessageFrame:
    """A framed message from stream data.

    Implements RE-STR-003: Message frame.

    Attributes:
        data: Message data.
        offset: Offset in stream.
        length: Message length.
        frame_type: Detected frame type.
        is_complete: Whether message is complete.
        sequence: Message sequence number if detected.
    """

    data: bytes
    offset: int
    length: int
    frame_type: str = "unknown"
    is_complete: bool = True
    sequence: int | None = None


@dataclass
class FramingResult:
    """Result of message framing.

    Implements RE-STR-003: Framing result.

    Attributes:
        messages: List of extracted messages.
        framing_type: Detected framing type.
        delimiter: Detected delimiter if applicable.
        length_field_offset: Length field offset if applicable.
        length_field_size: Length field size if applicable.
        remaining: Unframed bytes at end.
    """

    messages: list[MessageFrame]
    framing_type: str
    delimiter: bytes | None = None
    length_field_offset: int | None = None
    length_field_size: int | None = None
    remaining: bytes = b""


class UDPStreamReassembler:
    """Reassemble UDP datagram streams.

    Implements RE-STR-001: UDP Stream Reconstruction.

    UDP doesn't guarantee order, so this reassembler orders datagrams
    by sequence number or timestamp and handles gaps.

    Example:
        >>> reassembler = UDPStreamReassembler()
        >>> for packet in packets:
        ...     reassembler.add_segment(packet)
        >>> stream = reassembler.get_stream()
    """

    def __init__(
        self,
        sequence_key: Callable[[Any], int] | None = None,
        max_gap: int = 1000,
    ) -> None:
        """Initialize UDP reassembler.

        Args:
            sequence_key: Function to extract sequence number from packet.
            max_gap: Maximum sequence gap before treating as new stream.
        """
        self.sequence_key = sequence_key
        self.max_gap = max_gap
        self._segments: dict[str, list[StreamSegment]] = defaultdict(list)

    def add_segment(
        self,
        packet: dict[str, Any] | bytes,
        flow_key: str | None = None,
    ) -> None:
        """Add a UDP datagram to the reassembler.

        Args:
            packet: Packet data or dict with metadata.
            flow_key: Optional flow identifier.
        """
        if isinstance(packet, bytes):
            segment = StreamSegment(
                sequence_number=len(self._segments.get(flow_key or "default", [])),
                data=packet,
            )
        else:
            seq = 0
            if self.sequence_key is not None:
                try:
                    seq = self.sequence_key(packet)
                except (KeyError, TypeError):
                    pass

            segment = StreamSegment(
                sequence_number=seq,
                data=packet.get("data", packet.get("payload", b"")),
                timestamp=packet.get("timestamp", 0.0),
                src=packet.get("src", packet.get("src_ip", "")),
                dst=packet.get("dst", packet.get("dst_ip", "")),
            )

        key = flow_key or f"{segment.src}-{segment.dst}"
        self._segments[key].append(segment)

    def _get_empty_stream(self) -> ReassembledStream:
        """Create empty stream result.

        Returns:
            Empty ReassembledStream.
        """
        return ReassembledStream(
            data=b"",
            src="",
            dst="",
            start_time=0.0,
            end_time=0.0,
            segments=0,
        )

    def _resolve_flow_key(self, flow_key: str | None) -> str | None:
        """Resolve flow key to actual key or None if empty.

        Args:
            flow_key: Requested flow key.

        Returns:
            Resolved flow key or None.
        """
        if flow_key is None:
            if not self._segments:
                return None
            return next(iter(self._segments.keys()))
        return flow_key

    def _count_out_of_order(self, segments: list[StreamSegment]) -> int:
        """Count out-of-order segments.

        Args:
            segments: List of segments (unsorted).

        Returns:
            Number of out-of-order segments.
        """
        out_of_order = 0
        max_seq_seen = -1
        for segment in segments:
            if segment.sequence_number < max_seq_seen:
                out_of_order += 1
            max_seq_seen = max(max_seq_seen, segment.sequence_number)
        return out_of_order

    def _detect_gaps(self, sorted_segments: list[StreamSegment]) -> list[tuple[int, int]]:
        """Detect gaps in sequence numbers.

        Args:
            sorted_segments: Segments sorted by sequence number.

        Returns:
            List of (expected, actual) gap tuples.
        """
        gaps: list[tuple[int, int]] = []
        for i in range(1, len(sorted_segments)):
            expected = sorted_segments[i - 1].sequence_number + len(sorted_segments[i - 1].data)
            actual = sorted_segments[i].sequence_number
            if actual > expected:
                gaps.append((expected, actual))
        return gaps

    def _get_time_range(self, sorted_segments: list[StreamSegment]) -> tuple[float, float]:
        """Get start and end times from segments.

        Args:
            sorted_segments: List of segments.

        Returns:
            Tuple of (start_time, end_time).
        """
        timestamps = [s.timestamp for s in sorted_segments if s.timestamp > 0]
        if timestamps:
            return min(timestamps), max(timestamps)
        return 0.0, 0.0

    def _extract_addresses(self, sorted_segments: list[StreamSegment]) -> tuple[str, str]:
        """Extract source and destination addresses from segments.

        Args:
            sorted_segments: List of segments.

        Returns:
            Tuple of (src, dst).
        """
        if sorted_segments:
            return sorted_segments[0].src, sorted_segments[0].dst
        return "", ""

    def get_stream(self, flow_key: str | None = None) -> ReassembledStream:
        """Get reassembled stream for a flow.

        Implements RE-STR-001: UDP stream reconstruction.

        Args:
            flow_key: Flow identifier.

        Returns:
            ReassembledStream with ordered data.
        """
        # Resolve flow key
        resolved_key = self._resolve_flow_key(flow_key)
        if resolved_key is None:
            return self._get_empty_stream()

        # Get segments
        segments = self._segments.get(resolved_key, [])
        if not segments:
            return self._get_empty_stream()

        # Sort by sequence number
        sorted_segments = sorted(segments, key=lambda s: s.sequence_number)

        # Build stream components
        data = b"".join(s.data for s in sorted_segments)
        out_of_order = self._count_out_of_order(segments)
        gaps = self._detect_gaps(sorted_segments)
        start_time, end_time = self._get_time_range(sorted_segments)
        src, dst = self._extract_addresses(sorted_segments)

        return ReassembledStream(
            data=data,
            src=src,
            dst=dst,
            start_time=start_time,
            end_time=end_time,
            segments=len(sorted_segments),
            gaps=gaps,
            retransmits=0,
            out_of_order=out_of_order,
        )

    def get_all_streams(self) -> dict[str, ReassembledStream]:
        """Get all reassembled streams.

        Returns:
            Dictionary mapping flow keys to streams.
        """
        return {key: self.get_stream(key) for key in self._segments}

    def clear(self) -> None:
        """Clear all segments."""
        self._segments.clear()


class TCPStreamReassembler:
    """Reassemble TCP byte streams.

    Implements RE-STR-002: TCP Stream Reassembly.

    Handles TCP sequence numbers, retransmissions, and ordering
    to reconstruct the original byte stream.

    Example:
        >>> reassembler = TCPStreamReassembler()
        >>> for segment in tcp_segments:
        ...     reassembler.add_segment(segment)
        >>> stream = reassembler.get_stream()
    """

    def __init__(
        self,
        initial_sequence: int | None = None,
        max_buffer_size: int = 10 * 1024 * 1024,
    ) -> None:
        """Initialize TCP reassembler.

        Args:
            initial_sequence: Initial sequence number (auto-detect if None).
            max_buffer_size: Maximum buffer size in bytes.
        """
        self.initial_sequence = initial_sequence
        self.max_buffer_size = max_buffer_size

        self._segments: dict[str, list[StreamSegment]] = defaultdict(list)
        self._isn: dict[str, int | None] = {}  # Initial sequence numbers
        self._seen_seqs: dict[str, set[int]] = defaultdict(set)  # Track seen sequence numbers

    def add_segment(
        self,
        segment: dict[str, Any] | StreamSegment,
        flow_key: str | None = None,
    ) -> None:
        """Add a TCP segment to the reassembler.

        Args:
            segment: TCP segment data or StreamSegment.
            flow_key: Optional flow identifier.
        """
        if isinstance(segment, dict):
            seq_num = segment.get("seq") or segment.get("sequence_number") or 0
            seg_data = segment.get("data") or segment.get("payload") or b""
            seg = StreamSegment(
                sequence_number=seq_num,
                data=seg_data if isinstance(seg_data, bytes) else b"",
                timestamp=segment.get("timestamp", 0.0),
                src=segment.get("src", ""),
                dst=segment.get("dst", ""),
                flags=segment.get("flags", 0),
            )
        else:
            seg = segment

        key = flow_key or f"{seg.src}-{seg.dst}"

        # Detect initial sequence number (SYN)
        if key not in self._isn or self._isn[key] is None:
            if seg.flags & 0x02:  # SYN flag
                # SYN consumes one sequence number, so ISN+1 is first data byte
                self._isn[key] = seg.sequence_number + 1
                return  # Don't store SYN itself

            if self.initial_sequence is not None:
                self._isn[key] = self.initial_sequence
            else:
                # Use first data segment's sequence as initial
                self._isn[key] = seg.sequence_number

        # Check for retransmit: same sequence number seen before WITH data
        # Empty segments (ACK-only) shouldn't cause data segments to be marked as retransmits
        if seg.sequence_number in self._seen_seqs[key] and seg.data:
            # Check if there's already a segment with data at this sequence
            has_data_at_seq = any(
                s.sequence_number == seg.sequence_number and s.data for s in self._segments[key]
            )
            if has_data_at_seq:
                seg.is_retransmit = True

        if seg.data:  # Only track sequences with data
            self._seen_seqs[key].add(seg.sequence_number)

        self._segments[key].append(seg)

    def _extract_addresses(self, sorted_segments: list[StreamSegment]) -> tuple[str, str]:
        """Extract source and destination addresses from segments.

        Args:
            sorted_segments: Sorted list of stream segments.

        Returns:
            Tuple of (source, destination) addresses.
        """
        if not sorted_segments:
            return ("", "")
        return (sorted_segments[0].src, sorted_segments[0].dst)

    def _count_anomalies(self, segments: list[StreamSegment]) -> tuple[int, int]:
        """Count retransmits and out-of-order segments.

        Args:
            segments: List of segments in arrival order.

        Returns:
            Tuple of (retransmits, out_of_order) counts.
        """
        # Count retransmits
        retransmits = sum(1 for seg in segments if seg.is_retransmit)

        # Count out-of-order segments
        out_of_order = 0
        for i, seg in enumerate(segments):
            for j in range(i):
                if segments[j].sequence_number > seg.sequence_number:
                    out_of_order += 1
                    break

        return (retransmits, out_of_order)

    def _detect_isn(self, flow_key: str, segments: list[StreamSegment]) -> int:
        """Detect initial sequence number for flow.

        Args:
            flow_key: Flow identifier.
            segments: List of segments.

        Returns:
            Initial sequence number.
        """
        isn = self._isn.get(flow_key, 0) or 0

        # If ISN wasn't detected via SYN, use minimum sequence number
        if isn == 0 or isn > min(s.sequence_number for s in segments):
            isn = min(s.sequence_number for s in segments)

        return isn

    def _build_data_buffer(
        self, sorted_segments: list[StreamSegment], isn: int
    ) -> tuple[bytes, list[tuple[int, int]]]:
        """Build data buffer from sorted segments, handling gaps and overlaps.

        Args:
            sorted_segments: Segments sorted by sequence number.
            isn: Initial sequence number.

        Returns:
            Tuple of (reassembled_data, gaps).
        """
        data_buffer = bytearray()
        current_offset = 0
        gaps = []

        for seg in sorted_segments:
            if seg.is_retransmit:
                continue

            rel_seq = (seg.sequence_number - isn) % (2**32)

            if rel_seq > current_offset:
                # Gap detected
                gaps.append((current_offset, rel_seq))
                data_buffer.extend(b"\x00" * (rel_seq - current_offset))
                current_offset = rel_seq

            if rel_seq < current_offset:
                # Overlap - use only non-overlapping part
                overlap = current_offset - rel_seq
                if overlap < len(seg.data):
                    data_buffer.extend(seg.data[overlap:])
                    current_offset += len(seg.data) - overlap
            else:
                data_buffer.extend(seg.data)
                current_offset += len(seg.data)

        return (bytes(data_buffer), gaps)

    def get_stream(self, flow_key: str | None = None) -> ReassembledStream:
        """Get reassembled TCP stream.

        Implements RE-STR-002: TCP stream reassembly.

        Args:
            flow_key: Flow identifier.

        Returns:
            ReassembledStream with complete data.
        """
        # Handle empty or default flow key
        if flow_key is None:
            if not self._segments:
                return ReassembledStream(
                    data=b"",
                    src="",
                    dst="",
                    start_time=0.0,
                    end_time=0.0,
                    segments=0,
                )
            flow_key = next(iter(self._segments.keys()))

        segments = self._segments.get(flow_key, [])
        if not segments:
            return ReassembledStream(
                data=b"",
                src="",
                dst="",
                start_time=0.0,
                end_time=0.0,
                segments=0,
            )

        # Detect ISN and count anomalies
        isn = self._detect_isn(flow_key, segments)
        retransmits, out_of_order = self._count_anomalies(segments)

        # Sort and reassemble
        sorted_segments = sorted(segments, key=lambda s: (s.sequence_number - isn) % (2**32))
        data, gaps = self._build_data_buffer(sorted_segments, isn)

        # Extract metadata
        src, dst = self._extract_addresses(sorted_segments)
        timestamps = [s.timestamp for s in sorted_segments if s.timestamp > 0]

        return ReassembledStream(
            data=data,
            src=src,
            dst=dst,
            start_time=min(timestamps) if timestamps else 0.0,
            end_time=max(timestamps) if timestamps else 0.0,
            segments=len(sorted_segments),
            gaps=gaps,
            retransmits=retransmits,
            out_of_order=out_of_order,
        )

    def get_all_streams(self) -> dict[str, ReassembledStream]:
        """Get all reassembled TCP streams."""
        return {key: self.get_stream(key) for key in self._segments}

    def clear(self) -> None:
        """Clear all data."""
        self._segments.clear()
        self._isn.clear()
        self._seen_seqs.clear()


class MessageFramer:
    """Extract framed messages from stream data.

    Implements RE-STR-003: Message Framing and Segmentation.

    Supports multiple framing methods: delimiter-based, length-prefixed,
    and fixed-size.

    Example:
        >>> framer = MessageFramer(framing_type='delimiter', delimiter=b'\\r\\n')
        >>> result = framer.frame(stream_data)
        >>> for msg in result.messages:
        ...     print(msg.data)
    """

    def __init__(
        self,
        framing_type: Literal["delimiter", "length_prefix", "fixed", "auto"] = "auto",
        delimiter: bytes | None = None,
        length_field_offset: int = 0,
        length_field_size: int = 2,
        length_field_endian: Literal["big", "little"] = "big",
        length_includes_header: bool = False,
        fixed_size: int = 0,
    ) -> None:
        """Initialize message framer.

        Args:
            framing_type: Type of framing to use.
            delimiter: Delimiter bytes for delimiter-based framing.
            length_field_offset: Offset of length field.
            length_field_size: Size of length field in bytes.
            length_field_endian: Endianness of length field.
            length_includes_header: Whether length includes header.
            fixed_size: Fixed message size.
        """
        self.framing_type = framing_type
        self.delimiter = delimiter
        self.length_field_offset = length_field_offset
        self.length_field_size = length_field_size
        self.length_field_endian = length_field_endian
        self.length_includes_header = length_includes_header
        self.fixed_size = fixed_size

    def frame(self, data: bytes) -> FramingResult:
        """Extract framed messages from data.

        Implements RE-STR-003: Message framing workflow.

        Args:
            data: Stream data to frame.

        Returns:
            FramingResult with extracted messages.

        Example:
            >>> result = framer.frame(stream_data)
            >>> print(f"Found {len(result.messages)} messages")
        """
        if self.framing_type == "auto":
            return self._auto_frame(data)
        elif self.framing_type == "delimiter":
            return self._frame_by_delimiter(data)
        elif self.framing_type == "length_prefix":
            return self._frame_by_length(data)
        else:  # framing_type == "fixed"
            return self._frame_fixed(data)

    def detect_framing(self, data: bytes) -> str:
        """Detect framing type from data.

        Implements RE-STR-003: Framing detection.

        Args:
            data: Sample data.

        Returns:
            Detected framing type string.
        """
        # Check for delimiter-based framing
        if self._is_delimiter_framed(data):
            return "delimiter"

        # Check for length-prefixed framing
        if self._is_length_prefixed(data):
            return "length_prefix"

        # Check for fixed-size framing
        if self._is_fixed_size(data):
            return "fixed"

        return "unknown"

    def _is_delimiter_framed(self, data: bytes) -> bool:
        """Check if data uses delimiter-based framing.

        Args:
            data: Sample data.

        Returns:
            True if delimiter framing detected.
        """
        common_delimiters = [b"\r\n", b"\n", b"\x00", b"\r"]
        for delim in common_delimiters:
            count = data.count(delim)
            if count >= 3:
                # Check for regular spacing
                parts = data.split(delim)
                if parts and len({len(p) for p in parts if p}) <= 3:
                    return True
        return False

    def _is_length_prefixed(self, data: bytes) -> bool:
        """Check if data uses length-prefixed framing.

        Args:
            data: Sample data.

        Returns:
            True if length-prefixed framing detected.
        """
        if len(data) < 4:
            return False

        # Try big-endian 2-byte length
        for offset in range(min(8, len(data) - 2)):
            length = int.from_bytes(data[offset : offset + 2], "big")
            if 4 < length < len(data) and length < 65536:
                # Check if data continues with similar pattern
                next_offset = offset + length
                if next_offset + 2 < len(data):
                    next_length = int.from_bytes(data[next_offset : next_offset + 2], "big")
                    if 4 < next_length < len(data):
                        return True
        return False

    def _is_fixed_size(self, data: bytes) -> bool:
        """Check if data uses fixed-size framing.

        Args:
            data: Sample data.

        Returns:
            True if fixed-size framing detected.
        """
        if len(data) < 32:
            return False

        # Look for repeating pattern
        for size in range(4, 128):
            if len(data) % size == 0:
                chunks = [data[i : i + size] for i in range(0, len(data), size)]
                if len(chunks) >= 3:
                    # Check structural similarity
                    first = chunks[0][:4] if len(chunks[0]) >= 4 else chunks[0]
                    matches = sum(1 for c in chunks[1:] if c[: len(first)] == first)
                    if matches >= len(chunks) * 0.5:
                        return True
        return False

    def _auto_frame(self, data: bytes) -> FramingResult:
        """Automatically detect and apply framing.

        Args:
            data: Stream data.

        Returns:
            FramingResult with detected framing.
        """
        framing_type = self.detect_framing(data)

        if framing_type == "delimiter":
            # Find the delimiter
            for delim in [b"\r\n", b"\n", b"\x00", b"\r"]:
                if data.count(delim) >= 3:
                    self.delimiter = delim
                    break
            return self._frame_by_delimiter(data)

        elif framing_type == "length_prefix":
            return self._frame_by_length(data)

        elif framing_type == "fixed":
            # Try to detect fixed size
            for size in range(4, 128):
                if len(data) % size == 0 and len(data) // size >= 3:
                    self.fixed_size = size
                    break
            return self._frame_fixed(data)

        else:
            # Return as single message
            return FramingResult(
                messages=[
                    MessageFrame(
                        data=data,
                        offset=0,
                        length=len(data),
                        frame_type="unknown",
                    )
                ],
                framing_type="unknown",
            )

    def _frame_by_delimiter(self, data: bytes) -> FramingResult:
        """Frame by delimiter.

        Args:
            data: Stream data.

        Returns:
            FramingResult.
        """
        if self.delimiter is None:
            return FramingResult(messages=[], framing_type="delimiter")

        messages = []
        offset = 0
        parts = data.split(self.delimiter)

        for i, part in enumerate(parts):
            if part:  # Skip empty parts
                messages.append(
                    MessageFrame(
                        data=part,
                        offset=offset,
                        length=len(part),
                        frame_type="delimited",
                        sequence=i,
                    )
                )
            offset += len(part) + len(self.delimiter)

        # Check for remaining bytes
        remaining = b""
        if parts and not parts[-1]:
            # Ends with delimiter, no remaining
            pass
        elif parts:
            remaining = parts[-1] if not data.endswith(self.delimiter) else b""

        return FramingResult(
            messages=messages,
            framing_type="delimiter",
            delimiter=self.delimiter,
            remaining=remaining,
        )

    def _frame_by_length(self, data: bytes) -> FramingResult:
        """Frame by length prefix.

        Args:
            data: Stream data.

        Returns:
            FramingResult.
        """
        messages = []
        offset = 0
        sequence = 0

        while offset + self.length_field_offset + self.length_field_size <= len(data):
            # Read length field
            length_start = offset + self.length_field_offset
            length_bytes = data[length_start : length_start + self.length_field_size]

            if self.length_field_endian == "big":
                length = int.from_bytes(length_bytes, "big")
            else:
                length = int.from_bytes(length_bytes, "little")

            # Calculate total message size
            if self.length_includes_header:
                msg_size = length
                header_size = self.length_field_offset + self.length_field_size
            else:
                header_size = self.length_field_offset + self.length_field_size
                msg_size = header_size + length

            # Check if complete message available
            if offset + msg_size > len(data):
                break

            messages.append(
                MessageFrame(
                    data=data[offset : offset + msg_size],
                    offset=offset,
                    length=msg_size,
                    frame_type="length_prefixed",
                    sequence=sequence,
                )
            )

            offset += msg_size
            sequence += 1

        remaining = data[offset:] if offset < len(data) else b""

        return FramingResult(
            messages=messages,
            framing_type="length_prefix",
            length_field_offset=self.length_field_offset,
            length_field_size=self.length_field_size,
            remaining=remaining,
        )

    def _frame_fixed(self, data: bytes) -> FramingResult:
        """Frame by fixed size.

        Args:
            data: Stream data.

        Returns:
            FramingResult.
        """
        if self.fixed_size <= 0:
            return FramingResult(messages=[], framing_type="fixed")

        messages = []
        offset = 0
        sequence = 0

        while offset + self.fixed_size <= len(data):
            messages.append(
                MessageFrame(
                    data=data[offset : offset + self.fixed_size],
                    offset=offset,
                    length=self.fixed_size,
                    frame_type="fixed",
                    sequence=sequence,
                )
            )
            offset += self.fixed_size
            sequence += 1

        remaining = data[offset:] if offset < len(data) else b""

        return FramingResult(
            messages=messages,
            framing_type="fixed",
            remaining=remaining,
        )


# =============================================================================
# Convenience functions
# =============================================================================


def reassemble_udp_stream(
    packets: Sequence[dict[str, Any] | bytes],
    sequence_key: Callable[[Any], int] | None = None,
) -> ReassembledStream:
    """Reassemble UDP datagram stream.

    Implements RE-STR-001: UDP Stream Reconstruction.

    Args:
        packets: List of UDP packets.
        sequence_key: Function to extract sequence number.

    Returns:
        ReassembledStream with ordered data.

    Example:
        >>> stream = reassemble_udp_stream(udp_packets)
        >>> print(f"Reassembled {len(stream.data)} bytes")
    """
    reassembler = UDPStreamReassembler(sequence_key=sequence_key)
    for packet in packets:
        reassembler.add_segment(packet)
    return reassembler.get_stream()


def reassemble_tcp_stream(
    segments: Sequence[dict[str, Any]],
    flow_key: str | None = None,
) -> ReassembledStream:
    """Reassemble TCP byte stream.

    Implements RE-STR-002: TCP Stream Reassembly.

    Args:
        segments: List of TCP segments.
        flow_key: Optional flow identifier.

    Returns:
        ReassembledStream with complete data.

    Example:
        >>> stream = reassemble_tcp_stream(tcp_segments)
        >>> print(f"Reassembled {len(stream.data)} bytes with {stream.gaps} gaps")
    """
    reassembler = TCPStreamReassembler()
    for segment in segments:
        reassembler.add_segment(segment, flow_key)
    return reassembler.get_stream(flow_key)


def extract_messages(
    data: bytes,
    framing_type: Literal["auto", "delimiter", "length_prefix", "fixed"] = "auto",
    delimiter: bytes | None = None,
    length_field_offset: int = 0,
    length_field_size: int = 2,
    fixed_size: int = 0,
) -> FramingResult:
    """Extract framed messages from stream data.

    Implements RE-STR-003: Message Framing and Segmentation.

    Args:
        data: Stream data.
        framing_type: Type of framing.
        delimiter: Delimiter for delimiter-based framing.
        length_field_offset: Length field offset.
        length_field_size: Length field size.
        fixed_size: Fixed message size.

    Returns:
        FramingResult with extracted messages.

    Example:
        >>> result = extract_messages(data, framing_type='delimiter', delimiter=b'\\r\\n')
        >>> for msg in result.messages:
        ...     print(msg.data)
    """
    framer = MessageFramer(
        framing_type=framing_type,
        delimiter=delimiter,
        length_field_offset=length_field_offset,
        length_field_size=length_field_size,
        fixed_size=fixed_size,
    )
    return framer.frame(data)


def detect_message_framing(data: bytes) -> dict[str, Any]:
    """Detect message framing type in data.

    Implements RE-STR-003: Framing detection.

    Args:
        data: Stream data sample.

    Returns:
        Dictionary with detected framing parameters.

    Example:
        >>> framing = detect_message_framing(stream_data)
        >>> print(f"Detected: {framing['type']}")
    """
    framer = MessageFramer()
    framing_type = framer.detect_framing(data)

    result: dict[str, Any] = {"type": framing_type}

    if framing_type == "delimiter":
        # Find the delimiter
        for delim in [b"\r\n", b"\n", b"\x00", b"\r"]:
            if data.count(delim) >= 3:
                result["delimiter"] = delim
                result["message_count"] = data.count(delim)
                break

    elif framing_type == "length_prefix":
        result["length_field_offset"] = 0
        result["length_field_size"] = 2

    elif framing_type == "fixed":
        # Try to detect fixed size
        for size in range(4, 128):
            if len(data) % size == 0 and len(data) // size >= 3:
                result["fixed_size"] = size
                result["message_count"] = len(data) // size
                break

    return result


__all__ = [
    "FramingResult",
    "MessageFrame",
    "MessageFramer",
    "ReassembledStream",
    # Data classes
    "StreamSegment",
    "TCPStreamReassembler",
    # Classes
    "UDPStreamReassembler",
    "detect_message_framing",
    "extract_messages",
    "reassemble_tcp_stream",
    # Functions
    "reassemble_udp_stream",
]
