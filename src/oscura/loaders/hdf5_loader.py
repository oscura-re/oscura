"""HDF5 file loader for waveform data.

This module provides loading of waveform data from HDF5 (.h5) files
with automatic dataset discovery and attribute-based metadata extraction.


Example:
    >>> from oscura.loaders.hdf5_loader import load_hdf5
    >>> trace = load_hdf5("data.h5")
    >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from os import PathLike

# Try to import h5py
try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


# Common dataset names for waveform data
DATASET_NAMES = [
    "data",
    "waveform",
    "signal",
    "samples",
    "voltage",
    "trace",
    "ch1",
    "ch2",
    "channel1",
    "channel2",
    "analog",
]

# Common attribute names for sample rate
SAMPLE_RATE_ATTRS = [
    "sample_rate",
    "samplerate",
    "sampling_rate",
    "fs",
    "rate",
    "sample_interval",
    "dt",
]


class HDF5MmapTrace:
    """Memory-mapped waveform trace backed by HDF5 dataset.

    Provides lazy access to HDF5 dataset without loading into memory.
    Useful for huge files that don't fit in RAM.

    Attributes:
        file_path: Path to the HDF5 file.
        dataset_path: Path to dataset within HDF5 file.
        sample_rate: Sample rate in Hz.
        length: Number of samples in the trace.
        metadata: TraceMetadata object.

    Example:
        >>> trace = HDF5MmapTrace("huge.h5", "/data", metadata)
        >>> # Access subset without loading entire file
        >>> subset = trace[1000:2000]
    """

    def __init__(
        self,
        file_path: str | Path,
        dataset_path: str,
        metadata: TraceMetadata,
    ) -> None:
        """Initialize HDF5 memory-mapped trace.

        Args:
            file_path: Path to HDF5 file.
            dataset_path: Path to dataset within file (e.g., "/data").
            metadata: TraceMetadata with sample rate and other info.

        Raises:
            LoaderError: If file not found or invalid.
        """
        self._file_path = Path(file_path)
        self._dataset_path = dataset_path
        self._metadata = metadata
        self._h5_file: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None

        if not self._file_path.exists():
            raise LoaderError(f"File not found: {self._file_path}")

    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz."""
        return self._metadata.sample_rate

    @property
    def length(self) -> int:
        """Number of samples."""
        self._ensure_open()
        assert self._dataset is not None
        return len(self._dataset)

    @property
    def metadata(self) -> TraceMetadata:
        """Trace metadata."""
        return self._metadata

    def _ensure_open(self) -> None:
        """Ensure HDF5 file is open."""
        if self._h5_file is None or self._dataset is None:
            self._h5_file = h5py.File(self._file_path, "r")
            self._dataset = self._h5_file[self._dataset_path]

    def __getitem__(self, key: int | slice) -> np.ndarray[Any, Any]:
        """Access data by index or slice.

        Args:
            key: Index or slice.

        Returns:
            Numpy array of data.
        """
        self._ensure_open()
        assert self._dataset is not None
        data = self._dataset[key]
        return np.asarray(data, dtype=np.float64)

    def __len__(self) -> int:
        """Return number of samples."""
        return self.length

    def close(self) -> None:
        """Close HDF5 file handle."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._dataset = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()

    def __enter__(self) -> HDF5MmapTrace:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HDF5MmapTrace("
            f"file={self._file_path.name}, "
            f"dataset={self._dataset_path}, "
            f"sample_rate={self.sample_rate:.2e} Hz, "
            f"length={self.length:,} samples)"
        )


def load_hdf5(
    path: str | PathLike[str],
    *,
    dataset: str | None = None,
    channel: str | int | None = None,
    sample_rate: float | None = None,
    mmap: bool = False,
) -> WaveformTrace | HDF5MmapTrace:
    """Load waveform data from an HDF5 file.

    Loads waveform data and metadata from HDF5 files. Automatically
    discovers datasets and extracts sample rate from attributes.

    Args:
        path: Path to the HDF5 file.
        dataset: Specific dataset path to load. If None, auto-detects.
        channel: Alias for dataset (for API consistency with other loaders).
        sample_rate: Override sample rate (if not found in attributes).
        mmap: If True, return memory-mapped trace for large files.

    Returns:
        WaveformTrace containing the waveform data and metadata.
        If mmap=True, returns HDF5MmapTrace instead.

    Raises:
        LoaderError: If the file cannot be loaded.
        FormatError: If no valid waveform data is found.

    Example:
        >>> trace = load_hdf5("data.h5")
        >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")

        >>> # Load specific dataset
        >>> trace = load_hdf5("multi.h5", dataset="/measurements/ch1")

        >>> # Load as memory-mapped for large files
        >>> trace = load_hdf5("huge_data.h5", mmap=True)
    """
    _validate_hdf5_availability()
    file_path = _validate_file_path(path)
    dataset_path = _resolve_dataset_name(dataset, channel)

    try:
        with h5py.File(file_path, "r") as f:
            ds = _locate_dataset(f, dataset_path, file_path)
            _validate_dataset(ds, file_path)
            data = _extract_data_array(ds)
            metadata = _build_metadata(f, ds, sample_rate, file_path)

            if mmap:
                return _create_mmap_trace(file_path, ds.name, metadata)

            return WaveformTrace(data=data, metadata=metadata)

    except OSError as e:
        raise LoaderError(
            "Failed to read HDF5 file",
            file_path=str(file_path),
            details=str(e),
        ) from e
    except Exception as e:
        if isinstance(e, LoaderError | FormatError):
            raise
        raise LoaderError(
            "Failed to load HDF5 file",
            file_path=str(file_path),
            details=str(e),
        ) from e


def _validate_hdf5_availability() -> None:
    """Validate that h5py package is available.

    Raises:
        LoaderError: If h5py is not installed.
    """
    if not H5PY_AVAILABLE:
        raise LoaderError(
            "HDF5 support not available",
            details="h5py package is required for HDF5 loading",
            fix_hint="Install h5py: pip install h5py",
        )


def _validate_file_path(path: str | PathLike[str]) -> Path:
    """Validate that the file path exists.

    Args:
        path: File path to validate.

    Returns:
        Validated Path object.

    Raises:
        LoaderError: If file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise LoaderError("File not found", file_path=str(file_path))
    return file_path


def _resolve_dataset_name(dataset: str | None, channel: str | int | None) -> str | None:
    """Resolve dataset name from dataset or channel parameter.

    Args:
        dataset: Explicit dataset path.
        channel: Channel name/number (alternative to dataset).

    Returns:
        Resolved dataset name, or None for auto-detection.

    Example:
        >>> _resolve_dataset_name(None, 1)
        '1'
        >>> _resolve_dataset_name("/data", None)
        '/data'
    """
    if dataset is None and channel is not None:
        return str(channel)
    return dataset


def _locate_dataset(f: h5py.File, dataset_path: str | None, file_path: Path) -> h5py.Dataset:
    """Locate dataset in HDF5 file by path or auto-detection.

    Args:
        f: Open HDF5 file handle.
        dataset_path: Specific dataset path, or None for auto-detect.
        file_path: Path to HDF5 file (for error messages).

    Returns:
        Located HDF5 dataset.

    Raises:
        FormatError: If dataset not found.

    Example:
        >>> with h5py.File("data.h5") as f:
        ...     ds = _locate_dataset(f, "/waveform", Path("data.h5"))
    """
    if dataset_path is not None:
        if dataset_path in f:
            return f[dataset_path]

        # Try fuzzy name matching
        ds = _find_dataset_by_name(f, dataset_path)
        if ds is None:
            available = list_datasets(file_path)
            raise FormatError(
                f"Dataset not found: {dataset_path}",
                file_path=str(file_path),
                expected=dataset_path,
                got=f"Available: {', '.join(available)}",
            )
        return ds

    # Auto-detect waveform dataset
    ds = _find_waveform_dataset(f)
    if ds is None:
        available = list_datasets(file_path)
        raise FormatError(
            "No waveform data found in HDF5 file",
            file_path=str(file_path),
            expected=f"Dataset named: {', '.join(DATASET_NAMES)}",
            got=f"Datasets: {', '.join(available)}",
        )
    return ds


def _validate_dataset(ds: Any, file_path: Path) -> None:
    """Validate that object is a valid HDF5 dataset.

    Args:
        ds: Object to validate.
        file_path: Path to HDF5 file (for error messages).

    Raises:
        FormatError: If not a dataset.
    """
    if not isinstance(ds, h5py.Dataset):
        raise FormatError(
            "Selected path is not a dataset",
            file_path=str(file_path),
            got=type(ds).__name__,
        )


def _extract_data_array(ds: h5py.Dataset) -> np.ndarray[Any, Any]:
    """Extract and flatten data array from dataset.

    Args:
        ds: HDF5 dataset.

    Returns:
        1D numpy array of float64 data.

    Example:
        >>> data = _extract_data_array(dataset)
        >>> data.shape
        (10000,)
    """
    data = np.asarray(ds, dtype=np.float64)
    if data.ndim > 1:
        data = data.ravel()
    return data


def _build_metadata(
    f: h5py.File, ds: h5py.Dataset, sample_rate: float | None, file_path: Path
) -> TraceMetadata:
    """Build trace metadata from HDF5 attributes.

    Args:
        f: Open HDF5 file handle.
        ds: HDF5 dataset.
        sample_rate: User-provided sample rate override.
        file_path: Path to HDF5 file.

    Returns:
        TraceMetadata with extracted attributes.

    Example:
        >>> metadata = _build_metadata(f, ds, None, Path("data.h5"))
        >>> metadata.sample_rate
        1000000.0
    """
    detected_sample_rate = sample_rate if sample_rate is not None else _find_sample_rate(f, ds)
    if detected_sample_rate is None:
        detected_sample_rate = 1e6  # Default 1 MHz

    vertical_scale = _get_attr(ds, ["vertical_scale", "v_scale", "scale"])
    vertical_offset = _get_attr(ds, ["vertical_offset", "v_offset", "offset"])
    channel_name = _get_channel_name(ds)

    return TraceMetadata(
        sample_rate=float(detected_sample_rate),
        vertical_scale=float(vertical_scale) if vertical_scale else None,
        vertical_offset=float(vertical_offset) if vertical_offset else None,
        source_file=str(file_path),
        channel_name=str(channel_name),
    )


def _get_channel_name(ds: h5py.Dataset) -> str:
    """Get channel name from dataset attributes or path.

    Args:
        ds: HDF5 dataset.

    Returns:
        Channel name string.

    Example:
        >>> _get_channel_name(dataset)
        'CH1'
    """
    channel_name = _get_attr(ds, ["channel_name", "name", "channel"])
    if channel_name is None:
        channel_name = ds.name.split("/")[-1] if ds.name else "CH1"
    return channel_name


def _create_mmap_trace(
    file_path: Path, dataset_path: str, metadata: TraceMetadata
) -> HDF5MmapTrace:
    """Create memory-mapped HDF5 trace.

    Args:
        file_path: Path to HDF5 file.
        dataset_path: Path to dataset within file.
        metadata: Trace metadata.

    Returns:
        Memory-mapped trace object.

    Example:
        >>> trace = _create_mmap_trace(Path("data.h5"), "/waveform", metadata)
        >>> trace[1000:2000]  # Access subset without loading entire file
    """
    return HDF5MmapTrace(
        file_path=file_path,
        dataset_path=dataset_path,
        metadata=metadata,
    )


def _find_waveform_dataset(f: h5py.File) -> h5py.Dataset | None:
    """Find a waveform dataset in the HDF5 file."""
    result: h5py.Dataset | None = None

    def visitor(name: str, obj: Any) -> None:
        """Visit HDF5 items to find waveform dataset.

        Args:
            name: Dataset path within HDF5 file.
            obj: HDF5 object (dataset or group).
        """
        nonlocal result
        if result is not None:
            return
        if isinstance(obj, h5py.Dataset):
            name_lower = name.lower().split("/")[-1]
            # Check for common names
            for ds_name in DATASET_NAMES:
                if ds_name in name_lower:
                    result = obj
                    return
            # Check if it's a 1D numeric array
            if obj.ndim == 1 and obj.size > 10 and np.issubdtype(obj.dtype, np.number):
                if result is None:
                    result = obj

    f.visititems(visitor)
    return result


def _find_dataset_by_name(f: h5py.File, name: str) -> h5py.Dataset | None:
    """Find a dataset by name (case-insensitive partial match)."""
    name_lower = name.lower()
    result: h5py.Dataset | None = None

    def visitor(path: str, obj: Any) -> None:
        """Visit HDF5 items to find dataset by name.

        Args:
            path: Dataset path within HDF5 file.
            obj: HDF5 object (dataset or group).
        """
        nonlocal result
        if result is not None:
            return
        if isinstance(obj, h5py.Dataset):
            path_lower = path.lower()
            if name_lower in path_lower:
                result = obj

    f.visititems(visitor)
    return result


def _find_sample_rate(f: h5py.File, ds: h5py.Dataset) -> float | None:
    """Find sample rate from HDF5 attributes.

    Args:
        f: HDF5 file handle.
        ds: Dataset to search for sample rate.

    Returns:
        Sample rate in Hz, or None if not found.
    """
    # Check dataset, then parent, then root, then metadata group
    search_locations = [ds, ds.parent, f, f.get("metadata")]

    for location in search_locations:
        if location is None:
            continue

        rate = _extract_sample_rate_from_attrs(location)
        if rate is not None:
            return rate

    return None


def _extract_sample_rate_from_attrs(obj: h5py.Dataset | h5py.Group | h5py.File) -> float | None:
    """Extract sample rate from HDF5 object attributes.

    Args:
        obj: HDF5 object with attributes.

    Returns:
        Sample rate in Hz, or None if not found.
    """
    for attr_name in SAMPLE_RATE_ATTRS:
        if attr_name in obj.attrs:
            value = obj.attrs[attr_name]
            if attr_name in ("sample_interval", "dt") and value > 0:
                return 1.0 / float(value)
            return float(value)

    return None


def _get_attr(obj: h5py.Dataset | h5py.Group, names: list[str]) -> Any | None:
    """Get attribute value by trying multiple names."""
    for name in names:
        if name in obj.attrs:
            value = obj.attrs[name]
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return value
    return None


def list_datasets(path: str | PathLike[str]) -> list[str]:
    """List all datasets in an HDF5 file.

    Args:
        path: Path to the HDF5 file.

    Returns:
        List of dataset paths.

    Raises:
        LoaderError: If h5py is not available or file not found.

    Example:
        >>> datasets = list_datasets("data.h5")
        >>> print(datasets)
        ['/measurements/ch1', '/measurements/ch2', '/time']
    """
    if not H5PY_AVAILABLE:
        raise LoaderError(
            "HDF5 support not available",
            details="h5py package is required",
        )

    path = Path(path)
    if not path.exists():
        raise LoaderError("File not found", file_path=str(path))

    datasets: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        """Visit HDF5 items to collect dataset paths.

        Args:
            name: Dataset path within HDF5 file.
            obj: HDF5 object (dataset or group).
        """
        if isinstance(obj, h5py.Dataset):
            datasets.append("/" + name)

    try:
        with h5py.File(path, "r") as f:
            f.visititems(visitor)
    except Exception as e:
        raise LoaderError(
            "Failed to read HDF5 file",
            file_path=str(path),
            details=str(e),
        ) from e

    return datasets


def get_attributes(
    path: str | PathLike[str],
    dataset: str | None = None,
) -> dict[str, Any]:
    """Get attributes from an HDF5 file or dataset.

    Args:
        path: Path to the HDF5 file.
        dataset: Dataset path. If None, returns root attributes.

    Returns:
        Dictionary of attributes.

    Raises:
        LoaderError: If h5py is not available or file not found.
    """
    if not H5PY_AVAILABLE:
        raise LoaderError("HDF5 support not available")

    path = Path(path)
    if not path.exists():
        raise LoaderError("File not found", file_path=str(path))

    try:
        with h5py.File(path, "r") as f:
            obj = f[dataset] if dataset is not None else f

            attrs = {}
            for key, value in obj.attrs.items():
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                elif isinstance(value, np.ndarray):
                    value = value.tolist()
                attrs[key] = value

            return attrs

    except Exception as e:
        raise LoaderError(
            "Failed to read HDF5 attributes",
            file_path=str(path),
            details=str(e),
        ) from e


__all__ = ["get_attributes", "list_datasets", "load_hdf5"]
