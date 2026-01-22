"""JSON export functionality.

This module provides measurement results export to JSON format.


Example:
    >>> from oscura.exporters.json_export import export_json
    >>> export_json(measurements, "results.json")

References:
    RFC 8259 (JSON format)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace


class OscuraJSONEncoder(json.JSONEncoder):
    """JSON encoder with numpy, datetime, and Oscura object support."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, WaveformTrace):
            return {
                "_type": "WaveformTrace",
                "data": obj.data.tolist(),
                "metadata": self.default(obj.metadata),
            }
        if isinstance(obj, DigitalTrace):
            return {
                "_type": "DigitalTrace",
                "data": obj.data.tolist(),
                "metadata": self.default(obj.metadata),
                "edges": obj.edges,
            }
        if isinstance(obj, TraceMetadata):
            return {
                "_type": "TraceMetadata",
                "sample_rate": obj.sample_rate,
                "time_base": obj.time_base,
                "vertical_scale": obj.vertical_scale,
                "vertical_offset": obj.vertical_offset,
                "acquisition_time": obj.acquisition_time.isoformat()
                if obj.acquisition_time
                else None,
                "trigger_info": obj.trigger_info,
                "source_file": obj.source_file,
                "channel_name": obj.channel_name,
            }
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer | np.floating):
            val = float(obj)
            # Handle Infinity and NaN - convert to null for JSON compliance (RFC 8259)
            if math.isinf(val) or math.isnan(val):
                return None
            return val
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, float):
            # Also handle Python float inf/nan
            if math.isinf(obj) or math.isnan(obj):
                return None
            return obj
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, complex):
            # Handle complex with inf/nan components
            if (
                math.isinf(obj.real)
                or math.isnan(obj.real)
                or math.isinf(obj.imag)
                or math.isnan(obj.imag)
            ):
                return None
            return {"real": obj.real, "imag": obj.imag}
        if isinstance(obj, bytes):
            return obj.hex()
        if is_dataclass(obj):
            # Convert dataclasses to dict, then recursively encode
            return asdict(obj)  # type: ignore[arg-type]
        return super().default(obj)


def export_json(
    data: WaveformTrace | DigitalTrace | dict[str, Any] | list[Any],
    path: str | Path,
    *,
    pretty: bool = True,
    include_metadata: bool = True,
    compress: bool = False,
) -> None:
    """Export data to JSON format.

    Args:
        data: Data to export. Can be:
            - WaveformTrace or DigitalTrace (full trace with metadata)
            - Dictionary of measurements or data
            - List of data
        path: Output file path.
        pretty: Use pretty printing with indentation.
        include_metadata: Include export metadata.
        compress: Compress output (save as .json.gz).

    Example:
        >>> results = measure(trace)
        >>> export_json(results, "measurements.json")
        >>> export_json(trace, "waveform.json", pretty=True)
        >>> export_json(trace, "waveform.json.gz", compress=True)

    References:
        EXP-003
    """
    path = Path(path)

    output: dict[str, Any] = {}

    if include_metadata:
        output["_metadata"] = {
            "format": "oscura_json",
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
        }

    output["data"] = data

    # Sanitize to handle inf/nan in nested dictionaries (Python float inf/nan)
    # are handled directly by json encoder before calling default()
    from oscura.reporting.output import _sanitize_for_serialization

    output = _sanitize_for_serialization(output)

    # Serialize to JSON string
    if pretty:
        json_str = json.dumps(output, cls=OscuraJSONEncoder, indent=2)
    else:
        json_str = json.dumps(output, cls=OscuraJSONEncoder)

    # Write to file (with optional compression)
    if compress:
        import gzip

        # Ensure .gz extension
        if not str(path).endswith(".gz"):
            path = path.with_suffix(path.suffix + ".gz")

        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json_str)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)


def export_measurements(
    measurements: dict[str, Any],
    path: str | Path,
    *,
    trace_info: dict[str, Any] | None = None,
    pretty: bool = True,
) -> None:
    """Export measurement results to JSON.

    Specialized function for measurement export with trace info.

    Args:
        measurements: Dictionary of measurements.
        path: Output file path.
        trace_info: Optional trace metadata.
        pretty: Use pretty printing.

    Example:
        >>> measurements = measure(trace)
        >>> trace_info = {
        ...     "source_file": "scope_capture.wfm",
        ...     "sample_rate": 1e9,
        ...     "duration": 0.001
        ... }
        >>> export_measurements(measurements, "results.json", trace_info=trace_info)
    """
    path = Path(path)

    output = {
        "_metadata": {
            "format": "oscura_measurements",
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
        },
        "measurements": measurements,
    }

    if trace_info:
        output["trace_info"] = trace_info

    # Sanitize to ensure inf/nan handling
    from oscura.reporting.output import _sanitize_for_serialization

    output = _sanitize_for_serialization(output)

    with open(path, "w") as f:
        if pretty:
            json.dump(output, f, cls=OscuraJSONEncoder, indent=2)
        else:
            json.dump(output, f, cls=OscuraJSONEncoder)


def export_protocol_decode(
    packets: list[dict[str, Any]],
    path: str | Path,
    *,
    protocol: str = "unknown",
    trace_info: dict[str, Any] | None = None,
    pretty: bool = True,
) -> None:
    """Export protocol decode results to JSON.

    Args:
        packets: List of decoded packets.
        path: Output file path.
        protocol: Protocol name.
        trace_info: Optional trace metadata.
        pretty: Use pretty printing.

    Example:
        >>> packets = [{"timestamp": 0.001, "data": "0x48"}]
        >>> export_protocol_decode(packets, "uart_decode.json", protocol="uart")
    """
    path = Path(path)

    output = {
        "_metadata": {
            "format": "oscura_protocol",
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "protocol": protocol,
        },
        "packets": packets,
        "summary": {
            "total_packets": len(packets),
        },
    }

    if trace_info:
        output["trace_info"] = trace_info

    # Sanitize to ensure inf/nan handling
    from oscura.reporting.output import _sanitize_for_serialization

    output = _sanitize_for_serialization(output)

    with open(path, "w") as f:
        if pretty:
            json.dump(output, f, cls=OscuraJSONEncoder, indent=2)
        else:
            json.dump(output, f, cls=OscuraJSONEncoder)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON data file.

    Args:
        path: Input file path.

    Returns:
        Loaded data dictionary.

    Example:
        >>> data = load_json("results.json")
        >>> measurements = data.get("measurements", data.get("data", {}))
    """
    path = Path(path)

    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


__all__ = [
    "OscuraJSONEncoder",
    "export_json",
    "export_measurements",
    "export_protocol_decode",
    "load_json",
]
