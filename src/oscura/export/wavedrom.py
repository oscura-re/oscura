"""WaveDrom timing diagram generation.

Creates WaveDrom JSON format timing diagrams from digital signals.
WaveDrom format can be rendered as SVG/PNG using wavedrom-cli or online tools.

Example:
    >>> from oscura.export.wavedrom import export_wavedrom, WaveDromBuilder
    >>> builder = WaveDromBuilder()
    >>> builder.add_clock("CLK", period=100e-9)
    >>> builder.add_signal("DATA", edges=[10e-9, 50e-9, 150e-9])
    >>> builder.add_arrow(10e-9, 40e-9, "t_su = 30ns")
    >>> json_output = builder.to_json()
    >>> export_wavedrom(json_output, "timing_diagram.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from oscura.core.types import DigitalTrace, WaveformTrace


@dataclass
class WaveDromSignal:
    """A WaveDrom signal definition.

    Attributes:
        name: Signal name.
        wave: Wave string (WaveDrom format).
        data: Optional data labels.
        node: Optional node markers for arrows.
    """

    name: str
    wave: str
    data: list[str] | None = None
    node: str | None = None


@dataclass
class WaveDromEdge:
    """A WaveDrom edge/arrow annotation.

    Attributes:
        from_node: Source node name.
        to_node: Destination node name.
        label: Arrow label text.
        style: Arrow style.
    """

    from_node: str
    to_node: str
    label: str
    style: str = ""


class WaveDromBuilder:
    """Builder for WaveDrom timing diagrams.

    Example:
        >>> builder = WaveDromBuilder(title="74LS74 Setup Time")
        >>> builder.add_clock("CLK", period=100e-9, start_time=0)
        >>> builder.add_signal("D", edges=[10e-9, 50e-9])
        >>> builder.add_arrow(10e-9, 40e-9, "t_su = 30ns")
        >>> json_str = builder.to_json()
    """

    def __init__(
        self,
        *,
        title: str | None = None,
        time_scale: float = 1e-9,  # Default to nanoseconds
    ):
        """Initialize WaveDrom builder.

        Args:
            title: Optional diagram title.
            time_scale: Time scale for signal conversion (default 1ns).
        """
        self.title = title
        self.time_scale = time_scale
        self.signals: list[WaveDromSignal] = []
        self.edges: list[WaveDromEdge] = []
        self.config: dict[str, Any] = {"hscale": 2, "skin": "narrow"}
        self._time_offset = 0.0
        self._time_end = 0.0

    def add_clock(
        self,
        name: str,
        *,
        period: float,
        start_time: float = 0.0,
        duty_cycle: float = 0.5,
        initial_state: Literal["high", "low"] = "low",
        duration: float | None = None,
    ) -> None:
        """Add a clock signal.

        Args:
            name: Signal name.
            period: Clock period in seconds.
            start_time: Start time in seconds.
            duty_cycle: Duty cycle (0.0-1.0).
            initial_state: Initial clock state.
            duration: Optional duration (defaults to auto-calculated).
        """
        if duration is None:
            duration = max(self._time_end - start_time, period * 10)

        # Convert to time steps
        time_steps = int((start_time - self._time_offset) / self.time_scale)
        num_periods = int(duration / period)

        # Build wave string
        wave = "." * time_steps  # Leading dots

        if initial_state == "low":
            wave += "p" * num_periods  # pulsing clock
        else:
            wave += "n" * num_periods  # inverted pulsing clock

        self.signals.append(WaveDromSignal(name=name, wave=wave))
        self._time_end = max(self._time_end, start_time + duration)

    def add_signal(
        self,
        name: str,
        *,
        edges: list[float] | None = None,
        wave_string: str | None = None,
        data: list[str] | None = None,
        nodes: str | None = None,
    ) -> None:
        """Add a digital signal.

        Args:
            name: Signal name.
            edges: List of edge timestamps (rising/falling alternating).
            wave_string: Direct WaveDrom wave string (overrides edges).
            data: Optional data labels.
            nodes: Optional node markers (e.g., "A.B.C").
        """
        if wave_string is not None:
            # Use direct wave string
            self.signals.append(WaveDromSignal(name=name, wave=wave_string, data=data, node=nodes))
            return

        if edges is None:
            raise ValueError("Must provide either edges or wave_string")

        # Convert edges to wave string
        wave = self._edges_to_wave(edges)
        self.signals.append(WaveDromSignal(name=name, wave=wave, data=data, node=nodes))

    def add_data_bus(
        self,
        name: str,
        *,
        transitions: list[tuple[float, str]],
        initial_value: str = "x",
    ) -> None:
        """Add a data bus with labeled values.

        Args:
            name: Bus name.
            transitions: List of (timestamp, value) tuples.
            initial_value: Initial bus value.
        """
        if not transitions:
            raise ValueError("Must provide at least one transition")

        # Sort by timestamp
        transitions_sorted = sorted(transitions, key=lambda x: x[0])

        # Build wave string and data labels
        wave = ""
        data: list[str] = []
        current_time = self._time_offset

        for timestamp, value in transitions_sorted:
            # Add stable periods
            steps = int((timestamp - current_time) / self.time_scale)
            if steps > 0:
                wave += "." * (steps - 1)

            # Add transition
            wave += "="
            data.append(value)
            current_time = timestamp

        self.signals.append(WaveDromSignal(name=name, wave=wave, data=data))

    def add_arrow(
        self,
        from_time: float,
        to_time: float,
        label: str,
        *,
        from_signal_idx: int = 0,
        to_signal_idx: int = 0,
    ) -> None:
        """Add an arrow annotation between two time points.

        Args:
            from_time: Start time in seconds.
            to_time: End time in seconds.
            label: Arrow label text.
            from_signal_idx: Source signal index.
            to_signal_idx: Destination signal index.
        """
        # Create node markers
        from_node = f"A{len(self.edges)}"
        to_node = f"B{len(self.edges)}"

        # Add nodes to signals (simplified - would need to track positions)
        self.edges.append(WaveDromEdge(from_node=from_node, to_node=to_node, label=label))

    def add_group(self, name: str, signals: list[WaveDromSignal]) -> None:
        """Add a group of signals.

        Args:
            name: Group name.
            signals: List of signals in group.
        """
        # WaveDrom groups are represented as nested lists
        # This is a simplified implementation

    def _edges_to_wave(self, edges: list[float]) -> str:
        """Convert edge timestamps to WaveDrom wave string.

        Args:
            edges: List of edge timestamps.

        Returns:
            WaveDrom wave string.
        """
        if not edges:
            return "0"

        wave = ""
        current_time = self._time_offset
        state = 0  # Start low

        for edge_time in sorted(edges):
            # Calculate steps to this edge
            steps = int((edge_time - current_time) / self.time_scale)

            # Add stable period
            if steps > 0:
                wave += "." * (steps - 1)

            # Add transition
            if state == 0:
                wave += "1"  # Rising edge
                state = 1
            else:
                wave += "0"  # Falling edge
                state = 0

            current_time = edge_time

        return wave if wave else "0"

    def to_dict(self) -> dict[str, Any]:
        """Export to WaveDrom dictionary.

        Returns:
            Dictionary in WaveDrom JSON format.
        """
        result: dict[str, Any] = {}

        if self.title:
            result["head"] = {"text": self.title}

        # Build signal list
        signal_list: list[dict[str, Any]] = []
        for sig in self.signals:
            sig_dict: dict[str, Any] = {"name": sig.name, "wave": sig.wave}
            if sig.data:
                sig_dict["data"] = sig.data
            if sig.node:
                sig_dict["node"] = sig.node
            signal_list.append(sig_dict)

        result["signal"] = signal_list

        # Add edges if any
        if self.edges:
            edge_list = [f"{e.from_node}{e.style}>{e.to_node} {e.label}" for e in self.edges]
            result["edge"] = edge_list

        # Add configuration
        if self.config:
            result["config"] = self.config

        return result

    def to_json(self, *, indent: int = 2) -> str:
        """Export to WaveDrom JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str | Path) -> None:
        """Save to file.

        Args:
            filepath: Output file path.
        """
        filepath = Path(filepath)
        with filepath.open("w") as f:
            f.write(self.to_json())


def from_digital_trace(
    trace: DigitalTrace,
    *,
    name: str = "signal",
    start_time: float = 0.0,
    duration: float | None = None,
) -> WaveDromSignal:
    """Create WaveDrom signal from DigitalTrace.

    Args:
        trace: Input digital trace.
        name: Signal name.
        start_time: Start time offset.
        duration: Optional duration limit.

    Returns:
        WaveDromSignal object.
    """
    # Extract edges from digital trace
    data = trace.data
    transitions = np.diff(data.astype(np.int8))

    edges: list[float] = []
    time_base = trace.metadata.time_base

    for i, trans in enumerate(transitions):
        if trans != 0:
            edges.append((i + 1) * time_base + start_time)

    # Limit duration if specified
    if duration is not None:
        edges = [e for e in edges if e < start_time + duration]

    # Build wave string
    builder = WaveDromBuilder(time_scale=time_base)
    builder.add_signal(name, edges=edges)

    return builder.signals[0]


def export_wavedrom(
    signals: dict[str, WaveformTrace | DigitalTrace],
    filepath: str | Path,
    *,
    title: str | None = None,
    time_scale: float = 1e-9,
    annotations: list[tuple[float, float, str]] | None = None,
) -> None:
    """Export signals to WaveDrom JSON file.

    Args:
        signals: Dictionary mapping signal names to traces.
        filepath: Output file path.
        title: Optional diagram title.
        time_scale: Time scale for conversion (default 1ns).
        annotations: Optional list of (from_time, to_time, label) tuples.

    Example:
        >>> signals = {
        ...     "CLK": clock_trace,
        ...     "DATA": data_trace,
        ...     "CS": cs_trace,
        ... }
        >>> export_wavedrom(signals, "timing.json", title="SPI Transaction")
    """
    builder = WaveDromBuilder(title=title, time_scale=time_scale)

    # Add signals
    for name, trace in signals.items():
        # Detect if clock-like
        from oscura.analyzers.waveform.measurements import frequency
        from oscura.core.types import WaveformTrace

        # frequency() only works with WaveformTrace, skip for DigitalTrace
        if isinstance(trace, WaveformTrace):
            freq = frequency(trace)
        else:
            freq = np.nan
        if not np.isnan(freq) and freq > 0:
            # Looks like a clock
            period = float(1.0 / freq)
            builder.add_clock(name, period=period)
        else:
            # Regular signal - extract edges
            from oscura.analyzers.digital import detect_edges

            edges = detect_edges(trace)
            builder.add_signal(name, edges=list(edges))

    # Add annotations
    if annotations:
        for from_t, to_t, label in annotations:
            builder.add_arrow(from_t, to_t, label)

    builder.save(filepath)


__all__ = [
    "WaveDromBuilder",
    "WaveDromEdge",
    "WaveDromSignal",
    "export_wavedrom",
    "from_digital_trace",
]
