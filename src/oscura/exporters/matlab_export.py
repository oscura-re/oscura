"""MATLAB export functionality.

This module provides trace export to MATLAB .mat format with metadata.


Example:
    >>> from oscura.exporters.matlab_export import export_mat
    >>> export_mat(trace, "waveform.mat")
    >>> export_mat({"ch1": ch1, "ch2": ch2}, "channels.mat")

References:
    MATLAB MAT-File Format (https://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import scipy.io as sio

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from oscura.core.types import DigitalTrace, WaveformTrace


def export_mat(
    data: WaveformTrace | DigitalTrace | dict[str, Any],
    path: str | Path,
    *,
    version: str = "5",
    compression: bool = True,
    include_metadata: bool = True,
) -> None:
    """Export data to MATLAB .mat format.

    Args:
        data: Data to export. Can be:
            - Single WaveformTrace or DigitalTrace
            - Dictionary mapping names to traces or data
        path: Output file path.
        version: MAT-file version ("5", "7.3"). Version 5 is more compatible.
            Version 7.3 requires h5py and uses HDF5 backend for large files.
        compression: Enable compression.
        include_metadata: Include trace metadata in output.

    Raises:
        ImportError: If scipy is not installed, or h5py for version 7.3.

    Raises:
        TypeError: If data type is not supported.

    Example:
        >>> export_mat(trace, "waveform.mat")
        >>> export_mat({"ch1": ch1, "ch2": ch2}, "channels.mat")
        >>> export_mat(measurements, "results.mat", version="5")

    Note:
        Version 5 is the default and most compatible format, readable by
        scipy.io.loadmat and MATLAB.

        Version 7.3 uses HDF5 backend and supports:
        - Files > 2 GB
        - Compression
        - But requires h5py and cannot be read by scipy.io.loadmat

    References:
        EXP-008
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for MATLAB export. Install with: pip install scipy")

    path = Path(path)

    # Prepare data dictionary for MATLAB
    mat_dict: dict[str, Any] = {}

    if isinstance(data, WaveformTrace | DigitalTrace):
        # Single trace - use standard variable names
        _add_trace_to_dict(mat_dict, "trace", data, include_metadata)
    elif isinstance(data, dict):
        for name, value in data.items():
            if isinstance(value, WaveformTrace | DigitalTrace):
                _add_trace_to_dict(mat_dict, name, value, include_metadata)
            else:
                # Convert numpy arrays and other types
                mat_dict[_sanitize_varname(name)] = _convert_value(value)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Add export metadata
    # Note: MATLAB field names cannot start with underscore
    if include_metadata:
        mat_dict["oscura_export"] = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "format": "oscura_matlab",
        }

    # Save to .mat file
    if version == "7.3":
        # Use HDF5 backend (requires h5py)
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for MATLAB v7.3 export. "
                "Install with: pip install h5py, or use version='5'"
            )
        _save_hdf5_mat(path, mat_dict, compression)
    else:
        # Version 5 format (default, most compatible)
        sio.savemat(
            str(path),
            mat_dict,
            do_compression=compression,
            oned_as="column",
        )


def _save_hdf5_mat(path: Path, mat_dict: dict[str, Any], compression: bool) -> None:
    """Save MATLAB 7.3 format file using h5py (HDF5 backend).

    Args:
        path: Output file path.
        mat_dict: Dictionary of MATLAB variables.
        compression: Enable compression.
    """
    compression_opts = "gzip" if compression else None

    with h5py.File(path, "w") as f:
        # Set MATLAB 7.3 header attributes
        f.attrs["MATLAB_class"] = np.bytes_("struct")

        for key, value in mat_dict.items():
            _write_hdf5_value(f, key, value, compression_opts)


def _write_hdf5_value(
    parent: h5py.File | h5py.Group, key: str, value: Any, compression: str | None
) -> None:
    """Write a value to HDF5 file in MATLAB 7.3 compatible format.

    Args:
        parent: HDF5 file or group object.
        key: Variable name.
        value: Value to write.
        compression: Compression algorithm.
    """
    if isinstance(value, np.ndarray):
        # Create dataset for arrays
        if compression and value.size > 100:
            parent.create_dataset(key, data=value, compression=compression)
        else:
            parent.create_dataset(key, data=value)
        # Set MATLAB class attribute
        parent[key].attrs["MATLAB_class"] = np.bytes_("double")
    elif isinstance(value, dict):
        # Create group for structs/dicts
        grp = parent.create_group(key)
        grp.attrs["MATLAB_class"] = np.bytes_("struct")
        for k, v in value.items():
            _write_hdf5_value(grp, k, v, compression)
    elif isinstance(value, str):
        # String as uint16 array (MATLAB format)
        dt = h5py.string_dtype(encoding="utf-8")
        parent.create_dataset(key, data=value, dtype=dt)
        parent[key].attrs["MATLAB_class"] = np.bytes_("char")
    elif isinstance(value, int | float):
        # Scalar as 1x1 array
        parent.create_dataset(key, data=np.array([[value]]))
        parent[key].attrs["MATLAB_class"] = np.bytes_("double")
    elif isinstance(value, bool):
        parent.create_dataset(key, data=np.array([[value]], dtype=np.uint8))
        parent[key].attrs["MATLAB_class"] = np.bytes_("logical")
    elif isinstance(value, list):
        # Convert list to array
        arr = np.array(value)
        if compression and arr.size > 100:
            parent.create_dataset(key, data=arr, compression=compression)
        else:
            parent.create_dataset(key, data=arr)
        parent[key].attrs["MATLAB_class"] = np.bytes_("double")


def _add_trace_to_dict(
    mat_dict: dict[str, Any],
    name: str,
    trace: WaveformTrace | DigitalTrace,
    include_metadata: bool,
) -> None:
    """Add trace to MATLAB dictionary with metadata.

    Args:
        mat_dict: MATLAB variable dictionary.
        name: Variable name for trace.
        trace: Trace to add.
        include_metadata: Include metadata fields.
    """
    name = _sanitize_varname(name)

    # Add waveform data
    if isinstance(trace, WaveformTrace):
        mat_dict[f"{name}_data"] = trace.data
        mat_dict[f"{name}_time"] = trace.time_vector
    else:  # DigitalTrace
        mat_dict[f"{name}_data"] = trace.data.astype(np.uint8)
        mat_dict[f"{name}_time"] = trace.time_vector

    # Add metadata as struct
    if include_metadata:
        meta = trace.metadata
        metadata_struct: dict[str, Any] = {
            "sample_rate": meta.sample_rate,
            "time_base": meta.time_base,
            "num_samples": len(trace.data),
            "duration": trace.duration,
        }

        if meta.vertical_scale is not None:
            metadata_struct["vertical_scale"] = meta.vertical_scale
        if meta.vertical_offset is not None:
            metadata_struct["vertical_offset"] = meta.vertical_offset
        if meta.acquisition_time is not None:
            metadata_struct["acquisition_time"] = meta.acquisition_time.isoformat()
        if meta.source_file is not None:
            metadata_struct["source_file"] = str(meta.source_file)
        if meta.channel_name is not None:
            metadata_struct["channel_name"] = meta.channel_name
        if meta.trigger_info:
            metadata_struct["trigger_info"] = meta.trigger_info

        metadata_struct["trace_type"] = (
            "waveform" if isinstance(trace, WaveformTrace) else "digital"
        )

        mat_dict[f"{name}_metadata"] = metadata_struct


def _sanitize_varname(name: str) -> str:
    """Sanitize variable name for MATLAB compatibility.

    Args:
        name: Variable name to sanitize.

    Returns:
        Sanitized variable name compatible with MATLAB.

    Note:
        MATLAB variable names must:
        - Start with a letter
        - Contain only letters, digits, and underscores
        - Be <= 63 characters
    """
    import re

    # Replace invalid characters with underscores
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # Ensure starts with letter
    if name and not name[0].isalpha():
        name = "var_" + name

    # Truncate to 63 characters
    if len(name) > 63:
        name = name[:63]

    return name if name else "var"


def _convert_value(value: Any) -> Any:
    """Convert Python value to MATLAB-compatible format.

    Args:
        value: Python value to convert.

    Returns:
        MATLAB-compatible representation of the value.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.array(value)
    if isinstance(value, dict):
        # Convert nested dict
        return {_sanitize_varname(k): _convert_value(v) for k, v in value.items()}
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, complex):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    # For other types, try converting to string
    return str(value)


def export_multi_trace_mat(
    traces: list[WaveformTrace | DigitalTrace],
    path: str | Path,
    *,
    names: list[str] | None = None,
    version: str = "5",
    include_metadata: bool = True,
) -> None:
    """Export multiple traces to single MATLAB file.

    Args:
        traces: List of traces to export.
        path: Output file path.
        names: Variable names for each trace. If not provided, uses trace_1, trace_2, etc.
        version: MAT-file version ("5", "7.3").
        include_metadata: Include trace metadata.

    Raises:
        ValueError: If number of names does not match number of traces.

    Example:
        >>> export_multi_trace_mat(
        ...     [ch1, ch2, ch3],
        ...     "channels.mat",
        ...     names=["ch1", "ch2", "ch3"]
        ... )

    References:
        EXP-008
    """
    if names is None:
        names = [f"trace_{i + 1}" for i in range(len(traces))]

    if len(names) != len(traces):
        raise ValueError("Number of names must match number of traces")

    # Create dictionary mapping names to traces
    trace_dict = dict(zip(names, traces, strict=True))

    # Export using main function
    export_mat(trace_dict, path, version=version, include_metadata=include_metadata)


__all__ = [
    "export_mat",
    "export_multi_trace_mat",
]
