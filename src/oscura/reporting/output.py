"""Output management for comprehensive analysis reports.

This module provides directory structure and file management for analysis
report outputs, including plots, JSON/YAML data exports, and logs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from oscura.reporting.config import AnalysisDomain


def _sanitize_for_serialization(obj: Any, max_depth: int = 10) -> Any:
    """Convert non-serializable objects for JSON/YAML output.

    Handles generators, numpy arrays, and other problematic types
    that can appear in analysis results.

    Args:
        obj: Object to sanitize.
        max_depth: Maximum recursion depth to prevent infinite loops.

    Returns:
        Serialization-safe version of the object.
    """
    import types

    from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace

    if max_depth <= 0:
        return "<max depth exceeded>"

    try:
        # Don't sanitize Oscura types - let the JSONEncoder handle them
        if isinstance(obj, WaveformTrace | DigitalTrace | TraceMetadata):
            return obj
        if isinstance(obj, dict):
            # Sanitize both keys and values, convert non-string keys to strings
            sanitized = {}
            for k, v in obj.items():
                # Convert bytes keys to hex strings
                if isinstance(k, bytes):
                    k = f"0x{k.hex()}"
                # Convert other non-string keys to strings
                elif not isinstance(k, str | int | float | bool | type(None)):
                    k = str(k)
                sanitized[k] = _sanitize_for_serialization(v, max_depth - 1)
            return sanitized
        elif isinstance(obj, list | tuple):
            return [_sanitize_for_serialization(item, max_depth - 1) for item in obj]
        elif isinstance(obj, types.GeneratorType):
            # Convert generators to lists, but catch errors
            try:
                items = list(obj)
                return [_sanitize_for_serialization(item, max_depth - 1) for item in items]
            except Exception:
                # Return None for incompatible generators (cleaner than error string)
                return None
        elif isinstance(obj, np.ndarray):
            # Limit large arrays
            if obj.size > 10000:
                return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
            return obj.tolist()
        elif isinstance(obj, np.generic):
            # Catch all numpy scalar types (int, float, complex, bool, str, etc.)
            # This includes np.integer, np.floating, np.bool_, np.complexfloating, etc.
            return obj.item()
        elif isinstance(obj, np.integer | np.floating):
            # Redundant but kept for clarity
            return obj.item()
        elif isinstance(obj, np.bool_):
            # Redundant but kept for clarity
            return bool(obj)
        elif isinstance(obj, float):
            # Handle Python float inf/nan (not caught by JSONEncoder.default)
            import math

            if math.isinf(obj) or math.isnan(obj):
                return None
            return obj
        elif isinstance(obj, complex):
            # Handle complex numbers with inf/nan components
            import math

            if (
                math.isinf(obj.real)
                or math.isnan(obj.real)
                or math.isinf(obj.imag)
                or math.isnan(obj.imag)
            ):
                return None
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, bytes):
            # Limit large byte sequences
            if len(obj) > 1000:
                return f"<bytes len={len(obj)}>"
            return obj.hex()
        elif hasattr(obj, "__dict__") and not isinstance(obj, type):
            # Convert dataclasses and objects to dicts
            try:
                return {
                    k: _sanitize_for_serialization(v, max_depth - 1)
                    for k, v in obj.__dict__.items()
                }
            except Exception:
                return str(obj)
        elif callable(obj):
            return f"<callable: {getattr(obj, '__name__', str(obj))}>"
        else:
            # Try to convert to string as last resort
            try:
                return obj
            except Exception:
                return str(obj)
    except Exception as e:
        return f"<error: {type(e).__name__}: {str(e)[:50]}>"


class OutputManager:
    """Manages output directory structure and file operations for analysis reports.

    Creates timestamped output directories with organized subdirectories for
    different types of analysis outputs (plots, data files, logs, errors).

    Attributes:
        root: Root directory path for this analysis output.
        timestamp: Timestamp for this output session.
        timestamp_str: Formatted timestamp string.

    Requirements:
    """

    def __init__(
        self,
        base_dir: Path,
        input_name: str,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialize output manager.

        Args:
            base_dir: Base directory for all outputs.
            input_name: Name of the input file/dataset being analyzed.
            timestamp: Timestamp for this session (defaults to now).

        Examples:
            >>> manager = OutputManager(Path("/output"), "signal_data")
            >>> manager.root.name
            '20260101_120000_signal_data_analysis'
        """
        self._timestamp = timestamp or datetime.now()
        self._timestamp_str = self._timestamp.strftime("%Y%m%d_%H%M%S")

        # Create timestamped directory name
        dirname = f"{self._timestamp_str}_{input_name}_analysis"
        self._root = base_dir / dirname

    @property
    def root(self) -> Path:
        """Root directory path for this analysis output."""
        return self._root

    @property
    def timestamp(self) -> datetime:
        """Timestamp for this output session."""
        return self._timestamp

    @property
    def timestamp_str(self) -> str:
        """Formatted timestamp string (YYYYMMDD_HHMMSS)."""
        return self._timestamp_str

    def create(self) -> Path:
        """Create output directory structure.

        Creates the root directory and standard subdirectories:
        - plots/: Visualization outputs
        - errors/: Error logs and diagnostics
        - logs/: Analysis logs
        - input/: Input file copies/metadata

        Returns:
            Path to the created root directory.

        Note:
            This method is idempotent - calling it multiple times is safe.

        Requirements:

        Examples:
            >>> manager = OutputManager(Path("/tmp/output"), "test")
            >>> root = manager.create()
            >>> (root / "plots").exists()
            True
        """
        self._root.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        subdirs = ["plots", "errors", "logs", "input"]
        for subdir in subdirs:
            (self._root / subdir).mkdir(exist_ok=True)

        return self._root

    def create_domain_dir(self, domain: AnalysisDomain) -> Path:
        """Create and return domain-specific subdirectory.

        Creates a subdirectory for organizing outputs from a specific
        analysis domain (e.g., spectral/, digital/, jitter/).

        Args:
            domain: Analysis domain.

        Returns:
            Path to the created domain directory.

        Requirements:

        Examples:
            >>> manager = OutputManager(Path("/tmp/output"), "test")
            >>> manager.create()
            >>> domain_dir = manager.create_domain_dir(AnalysisDomain.SPECTRAL)
            >>> domain_dir.name
            'spectral'
        """
        domain_dir = self._root / domain.value
        domain_dir.mkdir(parents=True, exist_ok=True)
        return domain_dir

    def save_json(
        self,
        name: str,
        data: dict[str, Any],
        subdir: str | None = None,
    ) -> Path:
        """Save data as JSON file with pretty formatting.

        Args:
            name: Filename (without .json extension).
            data: Dictionary to serialize.
            subdir: Optional subdirectory within root.

        Returns:
            Path to the saved JSON file.

        Requirements:

        Examples:
            >>> manager = OutputManager(Path("/tmp/output"), "test")
            >>> manager.create()
            >>> path = manager.save_json("metrics", {"snr": 42.5})
            >>> path.name
            'metrics.json'
        """
        target_dir = self._root / subdir if subdir else self._root
        target_dir.mkdir(parents=True, exist_ok=True)

        filepath = target_dir / f"{name}.json"
        with filepath.open("w") as f:
            json.dump(data, f, indent=2, default=str)

        return filepath

    def save_yaml(
        self,
        name: str,
        data: dict[str, Any],
        subdir: str | None = None,
    ) -> Path:
        """Save data as YAML file.

        Args:
            name: Filename (without .yaml extension).
            data: Dictionary to serialize.
            subdir: Optional subdirectory within root.

        Returns:
            Path to the saved YAML file.

        Requirements:

        Examples:
            >>> manager = OutputManager(Path("/tmp/output"), "test")
            >>> manager.create()
            >>> path = manager.save_yaml("config", {"enabled": True})
            >>> path.name
            'config.yaml'
        """
        target_dir = self._root / subdir if subdir else self._root
        target_dir.mkdir(parents=True, exist_ok=True)

        filepath = target_dir / f"{name}.yaml"
        # Sanitize data to handle generators, numpy arrays, etc.
        sanitized_data = _sanitize_for_serialization(data)
        with filepath.open("w") as f:
            yaml.dump(sanitized_data, f, default_flow_style=False, sort_keys=False)

        return filepath

    def save_plot(
        self,
        domain: AnalysisDomain,
        name: str,
        fig: Any,
        format: str = "png",
        dpi: int = 150,
    ) -> Path:
        """Save matplotlib figure to plots directory.

        Saves plot with domain-prefixed filename in the plots/ subdirectory.

        Args:
            domain: Analysis domain for this plot.
            name: Plot name (without extension).
            fig: Matplotlib figure object.
            format: Image format (png, pdf, svg, etc.).
            dpi: Resolution in dots per inch.

        Returns:
            Path to the saved plot file.

        Requirements:

        Examples:
            >>> import matplotlib.pyplot as plt
            >>> manager = OutputManager(Path("/tmp/output"), "test")
            >>> manager.create()
            >>> fig, ax = plt.subplots()
            >>> path = manager.save_plot(AnalysisDomain.SPECTRAL, "fft", fig)
            >>> path.name
            'spectral_fft.png'
        """
        plots_dir = self._root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{domain.value}_{name}.{format}"
        filepath = plots_dir / filename

        fig.savefig(filepath, format=format, dpi=dpi, bbox_inches="tight")

        return filepath

    def save_text(
        self,
        name: str,
        content: str,
        subdir: str | None = None,
    ) -> Path:
        """Save text content to file.

        Args:
            name: Filename (with extension).
            content: Text content to write.
            subdir: Optional subdirectory within root.

        Returns:
            Path to the saved text file.

        Examples:
            >>> manager = OutputManager(Path("/tmp/output"), "test")
            >>> manager.create()
            >>> path = manager.save_text("summary.txt", "Analysis complete")
            >>> path.name
            'summary.txt'
        """
        target_dir = self._root / subdir if subdir else self._root
        target_dir.mkdir(parents=True, exist_ok=True)

        filepath = target_dir / name
        filepath.write_text(content)

        return filepath
