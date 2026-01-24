"""Stream processing utilities for packet analysis.

This module provides generator-based stream processing for
O(1) memory usage when analyzing large packet captures.


Example:
    >>> from oscura.analyzers.packet.stream import stream_packets
    >>> for packet in stream_packets(file_path, format="pcap"):
    ...     process(packet)  # Process one at a time

References:
    Python generators and itertools patterns
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, TypeVar

from oscura.analyzers.packet.parser import BinaryParser

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass
class StreamPacket:
    """Packet from a stream.

    Attributes:
        timestamp: Packet timestamp in seconds.
        data: Raw packet data.
        metadata: Additional packet metadata.
    """

    timestamp: float
    data: bytes
    metadata: dict[str, Any]


def stream_file(
    file_path: str | Path,
    chunk_size: int = 65536,
) -> Iterator[bytes]:
    """Stream file in chunks.

    Args:
        file_path: Path to file.
        chunk_size: Bytes per chunk (default 64KB).

    Yields:
        Byte chunks from file.

    Example:
        >>> for chunk in stream_file("large_capture.bin"):
        ...     process_chunk(chunk)
    """
    path = Path(file_path)

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def stream_records(
    file_or_buffer: str | Path | BinaryIO | bytes,
    record_size: int,
) -> Iterator[bytes]:
    """Stream fixed-size records.

    Args:
        file_or_buffer: Source file path, file object, or bytes.
        record_size: Size of each record in bytes.

    Yields:
        Records as bytes objects.

    Example:
        >>> for record in stream_records("data.bin", record_size=128):
        ...     parse_record(record)
    """
    if isinstance(file_or_buffer, bytes):
        buffer: BinaryIO = io.BytesIO(file_or_buffer)
        should_close = True
    elif isinstance(file_or_buffer, str | Path):
        buffer = open(file_or_buffer, "rb")  # noqa: SIM115
        should_close = True
    else:
        buffer = file_or_buffer
        should_close = False

    try:
        while True:
            record = buffer.read(record_size)
            if len(record) < record_size:
                break
            yield record
    finally:
        if should_close:
            buffer.close()


def stream_packets(
    file_or_buffer: str | Path | BinaryIO | bytes,
    *,
    header_parser: BinaryParser | None = None,
    length_field: int = 1,
    header_included: bool = False,
) -> Iterator[StreamPacket]:
    """Stream variable-length packets.

    Parses packets with length-prefixed format.

    Args:
        file_or_buffer: Source.
        header_parser: Parser for packet header.
        length_field: Index of length field in header (default 1).
        header_included: True if length includes header.

    Yields:
        StreamPacket objects.

    Example:
        >>> header = BinaryParser(">HH")  # sync, length
        >>> for pkt in stream_packets("capture.bin", header_parser=header):
        ...     print(f"Packet: {len(pkt.data)} bytes")
    """
    if header_parser is None:
        # Default: 2-byte big-endian length prefix
        header_parser = BinaryParser(">H")
        length_field = 0

    header_size = header_parser.size

    if isinstance(file_or_buffer, bytes):
        buffer: BinaryIO = io.BytesIO(file_or_buffer)
        should_close = True
    elif isinstance(file_or_buffer, str | Path):
        buffer = open(file_or_buffer, "rb")  # noqa: SIM115
        should_close = True
    else:
        buffer = file_or_buffer
        should_close = False

    try:
        packet_num = 0

        while True:
            # Read header
            header_bytes = buffer.read(header_size)
            if len(header_bytes) < header_size:
                break

            header = header_parser.unpack(header_bytes)
            length = header[length_field]

            # Compute payload size
            payload_size = length - header_size if header_included else length

            if payload_size < 0:
                break

            # Read payload
            payload = buffer.read(payload_size)
            if len(payload) < payload_size:
                break

            packet_num += 1

            yield StreamPacket(
                timestamp=packet_num,  # Placeholder
                data=header_bytes + payload,
                metadata={"header": header, "packet_num": packet_num},
            )

    finally:
        if should_close:
            buffer.close()


def stream_delimited(
    file_or_buffer: str | Path | BinaryIO | bytes,
    delimiter: bytes = b"\n",
    *,
    max_record_size: int = 1048576,
) -> Iterator[bytes]:
    """Stream delimiter-separated records.

    Args:
        file_or_buffer: Source.
        delimiter: Record delimiter (default newline).
        max_record_size: Maximum record size (default 1MB).

    Yields:
        Records as bytes (without delimiter).

    Example:
        >>> for line in stream_delimited("log.txt", b"\\n"):
        ...     process_line(line)
    """
    if isinstance(file_or_buffer, bytes):
        buffer: BinaryIO = io.BytesIO(file_or_buffer)
        should_close = True
    elif isinstance(file_or_buffer, str | Path):
        buffer = open(file_or_buffer, "rb")  # noqa: SIM115
        should_close = True
    else:
        buffer = file_or_buffer
        should_close = False

    try:
        partial = b""

        while True:
            chunk = buffer.read(65536)
            if not chunk:
                if partial:
                    yield partial
                break

            data = partial + chunk
            parts = data.split(delimiter)

            # Yield complete records
            for part in parts[:-1]:
                if len(part) <= max_record_size:
                    yield part

            # Keep partial record for next iteration
            partial = parts[-1]

            # Guard against memory issues
            if len(partial) > max_record_size:
                yield partial[:max_record_size]
                partial = b""

    finally:
        if should_close:
            buffer.close()


def pipeline(
    source: Iterator[T],
    *transforms: Callable[[Iterator[T]], Iterator[T]],
) -> Iterator[T]:
    """Chain processing transforms.

    Args:
        source: Source iterator.
        *transforms: Transform functions.

    Yields:
        Transformed items.

    Example:
        >>> def filter_large(packets):
        ...     for pkt in packets:
        ...         if len(pkt.data) > 100:
        ...             yield pkt
        ...
        >>> def decode(packets):
        ...     for pkt in packets:
        ...         pkt.metadata["decoded"] = decode_packet(pkt.data)
        ...         yield pkt
        ...
        >>> for pkt in pipeline(stream_packets(f), filter_large, decode):
        ...     print(pkt)
    """
    result: Iterator = source  # type: ignore[type-arg]

    for transform in transforms:
        result = transform(result)

    yield from result


def batch(
    source: Iterator[T],
    size: int,
) -> Iterator[list[T]]:
    """Batch items from iterator.

    Args:
        source: Source iterator.
        size: Batch size.

    Yields:
        Lists of items.

    Example:
        >>> for batch_items in batch(stream_packets(f), size=100):
        ...     process_batch(batch_items)
    """
    current_batch: list[T] = []

    for item in source:
        current_batch.append(item)
        if len(current_batch) >= size:
            yield current_batch
            current_batch = []

    if current_batch:
        yield current_batch


def take(source: Iterator[T], n: int) -> Iterator[T]:
    """Take first n items.

    Args:
        source: Source iterator.
        n: Number of items to take.

    Yields:
        First n items.
    """
    count = 0
    for item in source:
        if count >= n:
            break
        yield item
        count += 1  # noqa: SIM113


def skip(source: Iterator[T], n: int) -> Iterator[T]:
    """Skip first n items.

    Args:
        source: Source iterator.
        n: Number of items to skip.

    Yields:
        Items after first n.
    """
    count = 0
    for item in source:
        if count >= n:
            yield item
        count += 1  # noqa: SIM113


__all__ = [
    "StreamPacket",
    "batch",
    "pipeline",
    "skip",
    "stream_delimited",
    "stream_file",
    "stream_packets",
    "stream_records",
    "take",
]
