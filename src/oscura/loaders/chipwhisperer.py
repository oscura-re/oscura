"""ChipWhisperer trace loader.

This module loads power/EM traces from ChipWhisperer capture files (.npy, .trs).

ChipWhisperer is a widely-used open-source platform for side-channel analysis
and hardware security testing.

Example:
    >>> from oscura.loaders.chipwhisperer import load_chipwhisperer
    >>> traces, metadata = load_chipwhisperer("capture_data.npy")
    >>> print(f"Loaded {len(traces)} traces")

References:
    ChipWhisperer Project: https://github.com/newaetech/chipwhisperer
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import NDArray

__all__ = [
    "ChipWhispererTraceSet",
    "load_chipwhisperer",
    "load_chipwhisperer_npy",
    "load_chipwhisperer_trs",
]


@dataclass
class ChipWhispererTraceSet:
    """ChipWhisperer trace set container.

    Attributes:
        traces: Power/EM traces (n_traces, n_samples).
        plaintexts: Input plaintexts (n_traces, plaintext_size).
        ciphertexts: Output ciphertexts (n_traces, ciphertext_size).
        keys: Encryption keys if known (n_traces, key_size).
        sample_rate: Sample rate in Hz.
        metadata: Additional metadata.
    """

    traces: NDArray[np.floating[Any]]
    plaintexts: NDArray[np.integer[Any]] | None = None
    ciphertexts: NDArray[np.integer[Any]] | None = None
    keys: NDArray[np.integer[Any]] | None = None
    sample_rate: float = 1e6
    metadata: dict[str, object] | None = None

    @property
    def n_traces(self) -> int:
        """Number of traces."""
        return int(self.traces.shape[0])

    @property
    def n_samples(self) -> int:
        """Number of samples per trace."""
        return int(self.traces.shape[1])


def load_chipwhisperer(
    path: str | PathLike[str],
    *,
    sample_rate: float | None = None,
) -> ChipWhispererTraceSet:
    """Load ChipWhisperer traces from file.

    Auto-detects file format (.npy, .trs) and delegates to appropriate loader.

    Args:
        path: Path to ChipWhisperer trace file.
        sample_rate: Override sample rate (if not in file).

    Returns:
        ChipWhispererTraceSet with traces and metadata.

    Raises:
        LoaderError: If file cannot be loaded.
        FormatError: If file format invalid.

    Example:
        >>> traceset = load_chipwhisperer("traces.npy")
        >>> print(f"Loaded {traceset.n_traces} traces")
        >>> print(f"Samples per trace: {traceset.n_samples}")
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError("File not found", file_path=str(path))

    ext = path.suffix.lower()

    if ext == ".npy":
        return load_chipwhisperer_npy(path, sample_rate=sample_rate)
    elif ext == ".trs":
        return load_chipwhisperer_trs(path, sample_rate=sample_rate)
    else:
        raise FormatError(
            f"Unsupported ChipWhisperer format: {ext}",
            file_path=str(path),
            expected=".npy or .trs",
            got=ext,
        )


def load_chipwhisperer_npy(
    path: str | PathLike[str],
    *,
    sample_rate: float | None = None,
) -> ChipWhispererTraceSet:
    """Load ChipWhisperer traces from .npy file.

    ChipWhisperer often saves trace data as numpy .npy files with
    associated metadata in .npy files (textin.npy, textout.npy, etc.).

    Args:
        path: Path to traces .npy file.
        sample_rate: Override sample rate.

    Returns:
        ChipWhispererTraceSet with traces and metadata.

    Raises:
        LoaderError: If file cannot be loaded.

    Example:
        >>> traceset = load_chipwhisperer_npy("traces.npy")
        >>> # Look for associated files
        >>> if traceset.plaintexts is not None:
        ...     print("Plaintexts available")
    """
    path = Path(path)
    traces = _load_chipwhisperer_traces(path)

    # Load associated metadata files
    base_path = path.parent
    base_name = path.stem
    plaintexts = _load_optional_npy(base_path, base_name, "textin")
    ciphertexts = _load_optional_npy(base_path, base_name, "textout")
    keys = _load_optional_npy(base_path, base_name, "keys")

    sample_rate = sample_rate or 1e6  # Default 1 MS/s

    return ChipWhispererTraceSet(
        traces=traces.astype(np.float64),
        plaintexts=plaintexts.astype(np.uint8) if plaintexts is not None else None,
        ciphertexts=ciphertexts.astype(np.uint8) if ciphertexts is not None else None,
        keys=keys.astype(np.uint8) if keys is not None else None,
        sample_rate=sample_rate,
        metadata={"source_file": str(path), "format": "chipwhisperer_npy"},
    )


def _load_chipwhisperer_traces(path: Path) -> NDArray[np.float64]:
    """Load and validate ChipWhisperer trace array."""
    try:
        loaded_data: NDArray[Any] = np.load(path)

        result: NDArray[np.float64]
        if loaded_data.ndim == 1:
            result = loaded_data.reshape(1, -1).astype(np.float64)
        elif loaded_data.ndim == 2:
            result = loaded_data.astype(np.float64)
        else:
            raise FormatError(
                f"Expected 1D or 2D trace array, got {loaded_data.ndim}D",
                file_path=str(path),
            )
        return result

    except (OSError, ValueError) as e:
        raise LoaderError(
            "Failed to load trace file",
            file_path=str(path),
            details=str(e),
        ) from e


def _load_optional_npy(base_path: Path, base_name: str, suffix: str) -> NDArray[np.float64] | None:
    """Load optional ChipWhisperer metadata file."""
    # Try with base name prefix first
    filepath = base_path / f"{base_name}_{suffix}.npy"
    if not filepath.exists():
        filepath = base_path / f"{suffix}.npy"

    if filepath.exists():
        try:
            loaded: NDArray[Any] = np.load(filepath)
            result: NDArray[np.float64] = loaded.astype(np.float64)
            return result
        except Exception:
            pass  # Optional file, silently ignore if corrupt

    return None


def load_chipwhisperer_trs(
    path: str | PathLike[str],
    *,
    sample_rate: float | None = None,
) -> ChipWhispererTraceSet:
    """Load ChipWhisperer traces from Inspector .trs file.

    The .trs format is used by Riscure Inspector and supported by ChipWhisperer.

    TRS file structure:
    - Header with metadata
    - Trace data (interleaved with trace-specific data)

    Args:
        path: Path to .trs file.
        sample_rate: Override sample rate.

    Returns:
        ChipWhispererTraceSet with traces and metadata.

    Raises:
        LoaderError: If file cannot be loaded.
        FormatError: If TRS format invalid.

    Example:
        >>> traceset = load_chipwhisperer_trs("capture.trs")
        >>> print(f"Loaded {traceset.n_traces} traces")

    References:
        Inspector Trace Set (.trs) file format specification
    """
    path = Path(path)

    try:
        with open(path, "rb") as f:
            tags = _read_trs_header(f)
            n_traces, n_samples, sample_coding, data_length = _parse_trs_tags(tags, str(path))
            dtype = _get_trs_dtype(sample_coding, str(path))
            traces, plaintexts = _read_trs_traces(f, n_traces, n_samples, data_length, dtype)

    except OSError as e:
        raise LoaderError("Failed to read TRS file", file_path=str(path), details=str(e)) from e
    except Exception as e:
        if isinstance(e, (LoaderError, FormatError)):
            raise
        raise LoaderError("Failed to parse TRS file", file_path=str(path), details=str(e)) from e

    sample_rate = sample_rate or 1e6

    return ChipWhispererTraceSet(
        traces=traces,
        plaintexts=plaintexts,
        ciphertexts=None,
        keys=None,
        sample_rate=sample_rate,
        metadata={
            "source_file": str(path),
            "format": "chipwhisperer_trs",
            "n_traces": n_traces,
            "n_samples": n_samples,
            "sample_coding": sample_coding,
        },
    )


def _read_trs_header(f: Any) -> dict[int, bytes]:
    """Read TRS header tags in Tag-Length-Value format.

    Args:
        f: File object to read from.

    Returns:
        Dictionary mapping tag IDs to values.

    Raises:
        FormatError: If tag length exceeds MAX_TAG_LENGTH (10MB) or
                     tag count exceeds MAX_TAG_COUNT (1000).
    """
    MAX_TAG_LENGTH = 10 * 1024 * 1024  # 10MB per tag
    MAX_TAG_COUNT = 1000  # Maximum number of tags

    tags: dict[int, bytes] = {}
    while True:
        # Safety check: prevent excessive tag counts
        if len(tags) >= MAX_TAG_COUNT:
            raise FormatError(
                f"TRS header tag count exceeded {MAX_TAG_COUNT} limit. "
                "This may indicate malformed or malicious TRS file.",
                file_path="<unknown>",
            )

        tag_byte = f.read(1)
        if not tag_byte or tag_byte == b"\x5f":
            break

        tag = tag_byte[0]
        length = int.from_bytes(f.read(1), byteorder="little")

        if length == 0xFF:
            length = int.from_bytes(f.read(4), byteorder="little")

        # Safety check: prevent excessive tag lengths
        if length > MAX_TAG_LENGTH:
            raise FormatError(
                f"TRS tag length {length} bytes exceeded {MAX_TAG_LENGTH // (1024 * 1024)}MB limit. "
                "This may indicate malformed or malicious TRS file.",
                file_path="<unknown>",
            )

        value = f.read(length)
        tags[tag] = value

    return tags


def _parse_trs_tags(tags: dict[int, bytes], file_path: str) -> tuple[int, int, int, int]:
    """Parse critical TRS tags and validate."""
    n_traces = int.from_bytes(tags.get(0x41, b"\x00\x00"), byteorder="little")
    n_samples = int.from_bytes(tags.get(0x42, b"\x00\x00"), byteorder="little")
    sample_coding = tags.get(0x43, b"\x01")[0]
    data_length = int.from_bytes(tags.get(0x44, b"\x00\x00"), byteorder="little")

    if n_traces == 0 or n_samples == 0:
        raise FormatError("Invalid TRS file: zero traces or samples", file_path=file_path)

    return n_traces, n_samples, sample_coding, data_length


def _get_trs_dtype(
    sample_coding: int, file_path: str
) -> type[np.int8] | type[np.int16] | type[np.float32]:
    """Map TRS sample coding to numpy dtype."""
    dtype_map: dict[int, type[np.int8] | type[np.int16] | type[np.float32]] = {
        1: np.int8,
        2: np.int16,
        4: np.float32,
    }

    if sample_coding not in dtype_map:
        raise FormatError(f"Unsupported sample coding: {sample_coding}", file_path=file_path)

    result = dtype_map[sample_coding]
    return result


def _read_trs_traces(
    f: Any,
    n_traces: int,
    n_samples: int,
    data_length: int,
    dtype: type[np.int8] | type[np.int16] | type[np.float32],
) -> tuple[NDArray[np.float64], NDArray[np.uint8] | None]:
    """Read trace data from TRS file."""
    traces = np.zeros((n_traces, n_samples), dtype=np.float64)
    plaintexts = np.zeros((n_traces, data_length), dtype=np.uint8) if data_length > 0 else None

    for trace_idx in range(n_traces):
        if data_length > 0:
            trace_data = np.frombuffer(f.read(data_length), dtype=np.uint8)
            if plaintexts is not None:
                plaintexts[trace_idx] = trace_data

        trace_samples = np.frombuffer(f.read(n_samples * dtype(0).itemsize), dtype=dtype)
        traces[trace_idx] = trace_samples.astype(np.float64)

    return traces, plaintexts


def to_waveform_trace(
    traceset: ChipWhispererTraceSet,
    trace_index: int = 0,
) -> WaveformTrace:
    """Convert ChipWhisperer trace to WaveformTrace.

    Args:
        traceset: ChipWhisperer trace set.
        trace_index: Index of trace to convert.

    Returns:
        WaveformTrace for single trace.

    Raises:
        IndexError: If trace_index out of range.

    Example:
        >>> traceset = load_chipwhisperer("traces.npy")
        >>> trace = to_waveform_trace(traceset, trace_index=0)
        >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")
    """
    if not 0 <= trace_index < traceset.n_traces:
        raise IndexError(f"trace_index {trace_index} out of range [0, {traceset.n_traces})")

    metadata = TraceMetadata(
        sample_rate=traceset.sample_rate,
        source_file=str(traceset.metadata.get("source_file", "")) if traceset.metadata else "",
        channel_name=f"trace_{trace_index}",
    )

    return WaveformTrace(
        data=traceset.traces[trace_index],
        metadata=metadata,
    )
