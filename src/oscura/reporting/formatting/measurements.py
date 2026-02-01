"""Measurement formatting with intelligent SI prefix scaling and unit display.

This module provides automatic formatting for measurement dictionaries,
converting raw numeric values into human-readable strings with proper
SI prefixes and units.

Example:
    >>> from oscura.reporting.formatting.measurements import format_measurement
    >>> measurement = {"value": 2.3e-9, "unit": "s"}
    >>> format_measurement(measurement)
    '2.30 ns'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oscura.reporting.formatting.numbers import NumberFormatter


@dataclass
class MeasurementFormatter:
    """Format measurement dictionaries with intelligent number formatting and units.

    This class provides centralized, professional formatting for measurement data,
    automatically applying SI prefixes and units for optimal readability.

    Attributes:
        number_formatter: NumberFormatter instance for value formatting.
        default_sig_figs: Default significant figures for display.
        show_units: Whether to append units to formatted values.
        show_specs: Whether to include specification comparison in output.
    """

    number_formatter: NumberFormatter | None = None
    default_sig_figs: int = 4
    show_units: bool = True
    show_specs: bool = False

    def __post_init__(self) -> None:
        """Initialize number formatter if not provided."""
        if self.number_formatter is None:
            self.number_formatter = NumberFormatter(sig_figs=self.default_sig_figs)

    def format_single(self, value: float, unit: str = "") -> str:
        """Format single measurement value with SI prefix and unit.

        Args:
            value: Numeric value to format.
            unit: Unit string (e.g., "s", "Hz", "V", "ratio", "%").

        Returns:
            Formatted string with SI prefix and unit (e.g., "2.30 ns").

        Example:
            >>> formatter = MeasurementFormatter()
            >>> formatter.format_single(2.3e-9, "s")
            '2.30 ns'
            >>> formatter.format_single(440.0, "Hz")
            '440.0 Hz'
            >>> formatter.format_single(0.5, "ratio")
            '50.00 %'
        """
        assert self.number_formatter is not None

        # Handle ratio values (duty_cycle, overshoot, undershoot)
        # These are stored as fractions (0.5) but should display as percentages (50%)
        if unit == "ratio":
            percentage_value = value * 100
            formatted = self.number_formatter.format(percentage_value, "")
            return f"{formatted} %" if self.show_units else formatted

        # Handle percentages (THD, etc.) that are already in percentage units
        if unit == "%":
            formatted = self.number_formatter.format(value, "")
            return f"{formatted} %" if self.show_units else formatted

        # Handle dimensionless values
        if unit == "":
            formatted = self.number_formatter.format(value, "")
            return formatted

        # Use NumberFormatter's built-in SI prefix support for other units
        formatted = self.number_formatter.format(value, unit)
        return formatted if self.show_units else formatted.replace(unit, "").strip()

    def format_measurement(self, measurement: dict[str, Any]) -> str:
        """Format complete measurement dict with value, unit, spec, status.

        Supports both legacy format and MeasurementResult format.

        Args:
            measurement: Dictionary containing:
                Legacy format:
                  - value: numeric measurement value
                  - unit: unit string (optional)
                  - spec: specification limit (optional)
                  - spec_type: "max", "min", or "exact" (optional)
                  - passed: boolean pass/fail status (optional)
                MeasurementResult format:
                  - value: float | None
                  - unit: str
                  - applicable: bool
                  - reason: str | None
                  - display: str (pre-formatted)

        Returns:
            Formatted string representation of measurement.

        Example:
            >>> formatter = MeasurementFormatter(show_specs=True)
            >>> measurement = {
            ...     "value": 2.3e-9,
            ...     "unit": "s",
            ...     "spec": 10e-9,
            ...     "spec_type": "max",
            ...     "passed": True
            ... }
            >>> formatter.format_measurement(measurement)
            '2.30 ns (spec: < 10.0 ns) ✓'

            >>> # MeasurementResult format with inapplicable measurement
            >>> inapplicable = {
            ...     "value": None,
            ...     "unit": "Hz",
            ...     "applicable": False,
            ...     "reason": "Aperiodic signal",
            ...     "display": "N/A"
            ... }
            >>> formatter.format_measurement(inapplicable)
            'N/A (Aperiodic signal)'
        """
        # Check if this is MeasurementResult format
        if "applicable" in measurement:
            if not measurement["applicable"]:
                # Inapplicable measurement - show N/A with reason
                reason = measurement.get("reason")
                if reason:
                    return f"N/A ({reason})"
                return "N/A"

            # Applicable measurement - use pre-formatted display if available
            if "display" in measurement:
                formatted = str(measurement["display"])
            else:
                # Fallback to manual formatting
                value = measurement.get("value")
                unit = measurement.get("unit", "")
                if value is None:
                    return "N/A"
                formatted = self.format_single(value, unit)
        else:
            # Legacy format
            value = measurement.get("value")
            unit = measurement.get("unit", "")

            if value is None:
                return "N/A"

            if not isinstance(value, (int, float)):
                return str(value)

            # Format the value
            formatted = self.format_single(value, unit)

        # Add spec comparison if requested
        if self.show_specs and "spec" in measurement:
            spec = measurement["spec"]
            spec_type = measurement.get("spec_type", "exact")
            spec_formatted = self.format_single(spec, measurement.get("unit", ""))

            if spec_type == "max":
                formatted += f" (spec: < {spec_formatted})"
            elif spec_type == "min":
                formatted += f" (spec: > {spec_formatted})"
            else:  # exact
                formatted += f" (spec: {spec_formatted})"

            # Add pass/fail indicator
            if "passed" in measurement:
                formatted += " ✓" if measurement["passed"] else " ✗"

        return formatted

    def format_measurement_dict(
        self, measurements: dict[str, dict[str, Any]], html: bool = True
    ) -> str:
        """Format dictionary of measurements as formatted text or HTML.

        Args:
            measurements: Dictionary mapping parameter names to measurement dicts.
            html: If True, return HTML list; if False, return plain text.

        Returns:
            HTML unordered list or multi-line string with formatted measurements.

        Example:
            >>> formatter = MeasurementFormatter()
            >>> measurements = {
            ...     "rise_time": {"value": 2.3e-9, "unit": "s"},
            ...     "frequency": {"value": 440.0, "unit": "Hz"}
            ... }
            >>> print(formatter.format_measurement_dict(measurements, html=False))
            Rise Time: 2.30 ns
            Frequency: 440.0 Hz
        """
        lines = []
        for key, measurement in measurements.items():
            # Convert snake_case to Title Case
            display_name = key.replace("_", " ").title()
            formatted_value = self.format_measurement(measurement)

            if html:
                lines.append(f"<li><strong>{display_name}:</strong> {formatted_value}</li>")
            else:
                lines.append(f"{display_name}: {formatted_value}")

        if html:
            return f"<ul>\n{''.join(lines)}\n</ul>"
        else:
            return "\n".join(lines)

    def to_display_dict(self, measurements: dict[str, dict[str, Any]]) -> dict[str, str]:
        """Convert measurements to display-ready string dictionary.

        Args:
            measurements: Dictionary mapping parameter names to measurement dicts.

        Returns:
            Dictionary mapping parameter names to formatted value strings.

        Example:
            >>> formatter = MeasurementFormatter()
            >>> measurements = {"rise_time": {"value": 2.3e-9, "unit": "s"}}
            >>> formatter.to_display_dict(measurements)
            {'rise_time': '2.30 ns'}
        """
        return {key: self.format_measurement(meas) for key, meas in measurements.items()}


def format_measurement(measurement: dict[str, Any], sig_figs: int = 4) -> str:
    """Quick format single measurement dict.

    Convenience function for formatting a single measurement without
    creating a MeasurementFormatter instance.

    Args:
        measurement: Measurement dictionary with value and optional unit.
        sig_figs: Number of significant figures.

    Returns:
        Formatted measurement string.

    Example:
        >>> format_measurement({"value": 2.3e-9, "unit": "s"})
        '2.300 ns'
    """
    formatter = MeasurementFormatter(number_formatter=NumberFormatter(sig_figs=sig_figs))
    return formatter.format_measurement(measurement)


def format_measurement_dict(
    measurements: dict[str, dict[str, Any]], sig_figs: int = 4, html: bool = True
) -> str:
    """Quick format measurement dictionary.

    Convenience function for formatting multiple measurements without
    creating a MeasurementFormatter instance.

    Args:
        measurements: Dictionary mapping names to measurement dicts.
        sig_figs: Number of significant figures.
        html: If True, return HTML list; if False, return plain text.

    Returns:
        HTML unordered list or multi-line formatted string.

    Example:
        >>> measurements = {
        ...     "rise_time": {"value": 2.3e-9, "unit": "s"},
        ...     "frequency": {"value": 440.0, "unit": "Hz"}
        ... }
        >>> print(format_measurement_dict(measurements, html=False))
        Rise Time: 2.300 ns
        Frequency: 440.0 Hz
    """
    formatter = MeasurementFormatter(number_formatter=NumberFormatter(sig_figs=sig_figs))
    return formatter.format_measurement_dict(measurements, html=html)


def convert_to_measurement_dict(
    raw_measurements: dict[str, float], unit_map: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Convert raw measurement dictionary to structured measurement format.

    Helper function to convert simple {name: value} dictionaries into
    the full measurement format with units.

    Args:
        raw_measurements: Dictionary mapping parameter names to numeric values.
        unit_map: Dictionary mapping parameter names to unit strings.

    Returns:
        Dictionary in measurement format with value and unit fields.

    Example:
        >>> raw = {"rise_time": 2.3e-9, "frequency": 440.0}
        >>> units = {"rise_time": "s", "frequency": "Hz"}
        >>> convert_to_measurement_dict(raw, units)
        {
            'rise_time': {'value': 2.3e-09, 'unit': 's'},
            'frequency': {'value': 440.0, 'unit': 'Hz'}
        }
    """
    return {
        key: {"value": value, "unit": unit_map.get(key, "")}
        for key, value in raw_measurements.items()
        if isinstance(value, (int, float))
    }
