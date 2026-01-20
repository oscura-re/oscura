"""HDF5 export functionality.

This module provides trace export to HDF5 format with metadata attributes.


Example:
    >>> from oscura.exporters.hdf5 import export_hdf5
    >>> export_hdf5(trace, "output.h5")

References:
    HDF5 specification (https://www.hdfgroup.org/)
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from oscura.core.types import DigitalTrace, WaveformTrace


def export_hdf5(
    data: WaveformTrace | DigitalTrace | dict[str, WaveformTrace | DigitalTrace],
    path: str | Path,
    *,
    compression: str | None = "gzip",
    compression_opts: int = 4,
    include_metadata: bool = True,
) -> None:
    """Export data to HDF5 format.

    Args:
        data: Data to export. Can be:
            - Single WaveformTrace or DigitalTrace
            - Dictionary mapping names to traces
        path: Output file path.
        compression: Compression algorithm ("gzip", "lzf", None).
        compression_opts: Compression level (1-9 for gzip).
        include_metadata: Include trace metadata as attributes.

    Raises:
        ImportError: If h5py is not installed.

    Example:
        >>> export_hdf5(trace, "waveform.h5")
        >>> export_hdf5({"ch1": ch1, "ch2": ch2}, "channels.h5")
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for HDF5 export. Install with: pip install h5py")

    path = Path(path)

    if isinstance(data, WaveformTrace | DigitalTrace):
        data = {"trace": data}

    with h5py.File(path, "w") as f:
        # Add file-level metadata
        f.attrs["created"] = datetime.now().isoformat()
        f.attrs["oscura_version"] = "1.0"
        f.attrs["format"] = "oscura_hdf5"

        for name, trace in data.items():
            _write_trace_dataset(
                f,
                name,
                trace,
                compression,
                compression_opts,
                include_metadata,
            )


def _write_trace_dataset(
    f: "h5py.File",
    name: str,
    trace: WaveformTrace | DigitalTrace,
    compression: str | None,
    compression_opts: int,
    include_metadata: bool,
) -> None:
    """Write trace to HDF5 dataset.

    Args:
        f: HDF5 file object.
        name: Dataset name.
        trace: Trace to write.
        compression: Compression algorithm.
        compression_opts: Compression level.
        include_metadata: Include metadata as attributes.
    """
    # Create dataset
    dtype = np.float64 if isinstance(trace, WaveformTrace) else np.bool_

    kwargs = {}
    if compression:
        kwargs["compression"] = compression
        if compression == "gzip":
            kwargs["compression_opts"] = compression_opts  # type: ignore[assignment]

    ds = f.create_dataset(name, data=trace.data.astype(dtype), **kwargs)

    # Add metadata attributes
    if include_metadata:
        meta = trace.metadata

        ds.attrs["sample_rate"] = meta.sample_rate
        ds.attrs["time_base"] = meta.time_base

        if meta.vertical_scale is not None:
            ds.attrs["vertical_scale"] = meta.vertical_scale

        if meta.vertical_offset is not None:
            ds.attrs["vertical_offset"] = meta.vertical_offset

        if meta.acquisition_time is not None:
            ds.attrs["acquisition_time"] = meta.acquisition_time.isoformat()

        if meta.source_file is not None:
            ds.attrs["source_file"] = str(meta.source_file)

        if meta.channel_name is not None:
            ds.attrs["channel_name"] = meta.channel_name

        if meta.trigger_info:
            for key, value in meta.trigger_info.items():
                ds.attrs[f"trigger_{key}"] = value

        # Type indicator
        ds.attrs["trace_type"] = "waveform" if isinstance(trace, WaveformTrace) else "digital"


def export_measurement_results(
    results: dict[str, Any],
    path: str | Path,
    *,
    group_name: str = "measurements",
) -> None:
    """Export measurement results to HDF5.

    Args:
        results: Dictionary of measurement results.
        path: Output file path.
        group_name: HDF5 group name for results.

    Raises:
        ImportError: If h5py is not installed.

    Example:
        >>> results = measure(trace)
        >>> export_measurement_results(results, "measurements.h5")
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for HDF5 export")

    path = Path(path)

    with h5py.File(path, "a") as f:
        grp = f.require_group(group_name)

        for name, value in results.items():
            if isinstance(value, dict):
                # Nested dict (value/unit pairs)
                sub_grp = grp.require_group(name)
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        sub_grp.create_dataset(k, data=v)
                    else:
                        sub_grp.attrs[k] = v
            elif isinstance(value, np.ndarray):
                grp.create_dataset(name, data=value)
            else:
                grp.attrs[name] = value


def append_trace(
    path: str | Path,
    name: str,
    trace: WaveformTrace | DigitalTrace,
    *,
    compression: str | None = "gzip",
) -> None:
    """Append trace to existing HDF5 file.

    Args:
        path: HDF5 file path.
        name: Dataset name for new trace.
        trace: Trace to append.
        compression: Compression algorithm.

    Raises:
        ImportError: If h5py is not installed.

    Example:
        >>> append_trace("data.h5", "ch3", channel3_trace)
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for HDF5 export")

    path = Path(path)

    with h5py.File(path, "a") as f:
        _write_trace_dataset(f, name, trace, compression, 4, True)


__all__ = [
    "append_trace",
    "export_hdf5",
    "export_measurement_results",
]
