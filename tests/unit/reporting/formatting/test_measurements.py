"""Comprehensive unit tests for oscura.reporting.formatting.measurements module.

Tests all public functions and classes in the measurements formatting module,
with focus on edge cases, error handling, and measurement dictionary formatting.
"""

from __future__ import annotations

from typing import Any

import pytest

from oscura.reporting.formatting.measurements import (
    MeasurementFormatter,
    convert_to_measurement_dict,
    format_measurement,
    format_measurement_dict,
)
from oscura.reporting.formatting.numbers import NumberFormatter

pytestmark = pytest.mark.unit


class TestMeasurementFormatterInit:
    """Test MeasurementFormatter initialization and configuration."""

    def test_default_initialization(self):
        """Test MeasurementFormatter with default values."""
        fmt = MeasurementFormatter()
        assert fmt.default_sig_figs == 4
        assert fmt.show_units is True
        assert fmt.show_specs is False
        assert fmt.number_formatter is not None

    def test_custom_sig_figs(self):
        """Test MeasurementFormatter with custom significant figures."""
        fmt = MeasurementFormatter(default_sig_figs=5)
        assert fmt.default_sig_figs == 5

    def test_custom_number_formatter(self):
        """Test MeasurementFormatter with custom NumberFormatter."""
        custom_fmt = NumberFormatter(sig_figs=6)
        fmt = MeasurementFormatter(number_formatter=custom_fmt)
        assert fmt.number_formatter is custom_fmt
        assert fmt.number_formatter.sig_figs == 6

    def test_show_units_disabled(self):
        """Test MeasurementFormatter with units display disabled."""
        fmt = MeasurementFormatter(show_units=False)
        assert fmt.show_units is False

    def test_show_specs_enabled(self):
        """Test MeasurementFormatter with specs display enabled."""
        fmt = MeasurementFormatter(show_specs=True)
        assert fmt.show_specs is True

    def test_post_init_creates_formatter(self):
        """Test that __post_init__ creates NumberFormatter if not provided."""
        fmt = MeasurementFormatter(default_sig_figs=5)
        assert isinstance(fmt.number_formatter, NumberFormatter)
        assert fmt.number_formatter.sig_figs == 5


class TestMeasurementFormatterFormatSingle:
    """Test MeasurementFormatter.format_single() method."""

    def test_format_single_basic(self):
        """Test basic single value formatting."""
        fmt = MeasurementFormatter()
        result = fmt.format_single(2.3e-9, "s")
        assert "2.300" in result
        assert "ns" in result or "n" in result

    def test_format_single_ratio(self):
        """Test formatting ratio values (converts to percentage)."""
        fmt = MeasurementFormatter()
        result = fmt.format_single(0.5, "ratio")
        assert "50.00" in result
        assert "%" in result

    def test_format_single_ratio_without_units(self):
        """Test formatting ratio without showing units."""
        fmt = MeasurementFormatter(show_units=False)
        result = fmt.format_single(0.5, "ratio")
        assert "50.00" in result
        # Should not have % when units are disabled
        assert "%" not in result

    def test_format_single_percentage(self):
        """Test formatting percentage values."""
        fmt = MeasurementFormatter()
        result = fmt.format_single(75.5, "%")
        assert "75.50" in result
        assert "%" in result

    def test_format_single_percentage_without_units(self):
        """Test formatting percentage without showing units."""
        fmt = MeasurementFormatter(show_units=False)
        result = fmt.format_single(75.5, "%")
        assert "75.50" in result
        assert "%" not in result

    def test_format_single_dimensionless(self):
        """Test formatting dimensionless values (empty unit)."""
        fmt = MeasurementFormatter()
        result = fmt.format_single(42.0, "")
        assert "42.00" in result
        # Should not have trailing space
        assert not result.endswith(" ")

    def test_format_single_with_unit(self):
        """Test formatting with standard unit."""
        fmt = MeasurementFormatter()
        result = fmt.format_single(440.0, "Hz")
        assert "440.0" in result
        assert "Hz" in result

    def test_format_single_without_units_display(self):
        """Test formatting with show_units=False strips unit from result."""
        fmt = MeasurementFormatter(show_units=False)
        result = fmt.format_single(2.3e-6, "s")
        # Should have the value but not the unit
        assert "2.300" in result
        # Unit should be removed from the formatted string
        assert not result.endswith("s")

    def test_format_single_microseconds(self):
        """Test formatting microsecond values."""
        fmt = MeasurementFormatter()
        result = fmt.format_single(2.3e-6, "s")
        assert "2.300" in result
        assert "\u03bc" in result or "u" in result  # micro prefix

    def test_format_single_milliseconds(self):
        """Test formatting millisecond values."""
        fmt = MeasurementFormatter()
        result = fmt.format_single(0.0023, "s")
        assert "2.300" in result
        assert "m" in result

    def test_format_single_zero(self):
        """Test formatting zero value."""
        fmt = MeasurementFormatter()
        result = fmt.format_single(0.0, "V")
        assert "0" in result
        assert "V" in result

    def test_format_single_negative(self):
        """Test formatting negative value."""
        fmt = MeasurementFormatter()
        result = fmt.format_single(-2.3e-6, "A")
        assert "-" in result
        assert "2.300" in result


class TestMeasurementFormatterFormatMeasurement:
    """Test MeasurementFormatter.format_measurement() method."""

    def test_format_measurement_basic(self):
        """Test basic measurement dictionary formatting."""
        fmt = MeasurementFormatter()
        measurement = {"value": 2.3e-9, "unit": "s"}
        result = fmt.format_measurement(measurement)
        assert "2.300" in result
        assert "ns" in result or "n" in result

    def test_format_measurement_none_value(self):
        """Test formatting measurement with None value returns N/A."""
        fmt = MeasurementFormatter()
        measurement = {"value": None, "unit": "s"}
        result = fmt.format_measurement(measurement)
        assert result == "N/A"

    def test_format_measurement_missing_value(self):
        """Test formatting measurement without value key returns N/A."""
        fmt = MeasurementFormatter()
        measurement = {"unit": "s"}
        result = fmt.format_measurement(measurement)
        assert result == "N/A"

    def test_format_measurement_non_numeric_value(self):
        """Test formatting measurement with non-numeric value returns string."""
        fmt = MeasurementFormatter()
        measurement = {"value": "INVALID", "unit": "s"}
        result = fmt.format_measurement(measurement)
        assert result == "INVALID"

    def test_format_measurement_missing_unit(self):
        """Test formatting measurement without unit (defaults to empty string)."""
        fmt = MeasurementFormatter()
        measurement = {"value": 42.0}
        result = fmt.format_measurement(measurement)
        assert "42.00" in result

    def test_format_measurement_with_spec_max(self):
        """Test formatting measurement with max specification."""
        fmt = MeasurementFormatter(show_specs=True)
        measurement = {
            "value": 2.3e-9,
            "unit": "s",
            "spec": 10e-9,
            "spec_type": "max",
        }
        result = fmt.format_measurement(measurement)
        assert "2.300" in result
        assert "spec:" in result
        assert "<" in result  # max spec indicator
        assert "10.0" in result

    def test_format_measurement_with_spec_min(self):
        """Test formatting measurement with min specification."""
        fmt = MeasurementFormatter(show_specs=True)
        measurement = {
            "value": 12e-9,
            "unit": "s",
            "spec": 10e-9,
            "spec_type": "min",
        }
        result = fmt.format_measurement(measurement)
        assert "12.0" in result
        assert "spec:" in result
        assert ">" in result  # min spec indicator
        assert "10.0" in result

    def test_format_measurement_with_spec_exact(self):
        """Test formatting measurement with exact specification."""
        fmt = MeasurementFormatter(show_specs=True)
        measurement = {
            "value": 10e-9,
            "unit": "s",
            "spec": 10e-9,
            "spec_type": "exact",
        }
        result = fmt.format_measurement(measurement)
        assert "10.0" in result
        assert "spec:" in result
        # No comparison symbol for exact
        assert "<" not in result
        assert ">" not in result

    def test_format_measurement_with_spec_default_type(self):
        """Test formatting measurement with spec but no spec_type (defaults to exact)."""
        fmt = MeasurementFormatter(show_specs=True)
        measurement = {
            "value": 10e-9,
            "unit": "s",
            "spec": 10e-9,
        }
        result = fmt.format_measurement(measurement)
        assert "10.0" in result
        assert "spec:" in result

    def test_format_measurement_with_pass_status(self):
        """Test formatting measurement with pass status (checkmark)."""
        fmt = MeasurementFormatter(show_specs=True)
        measurement = {
            "value": 2.3e-9,
            "unit": "s",
            "spec": 10e-9,
            "spec_type": "max",
            "passed": True,
        }
        result = fmt.format_measurement(measurement)
        assert "\u2713" in result  # checkmark

    def test_format_measurement_with_fail_status(self):
        """Test formatting measurement with fail status (X mark)."""
        fmt = MeasurementFormatter(show_specs=True)
        measurement = {
            "value": 12e-9,
            "unit": "s",
            "spec": 10e-9,
            "spec_type": "max",
            "passed": False,
        }
        result = fmt.format_measurement(measurement)
        assert "\u2717" in result  # X mark

    def test_format_measurement_spec_without_show_specs(self):
        """Test that specs are not shown when show_specs=False."""
        fmt = MeasurementFormatter(show_specs=False)
        measurement = {
            "value": 2.3e-9,
            "unit": "s",
            "spec": 10e-9,
            "spec_type": "max",
            "passed": True,
        }
        result = fmt.format_measurement(measurement)
        # Should not have spec or pass/fail indicators
        assert "spec:" not in result
        assert "\u2713" not in result
        assert "\u2717" not in result

    def test_format_measurement_passed_without_spec(self):
        """Test that passed status is ignored without spec."""
        fmt = MeasurementFormatter(show_specs=True)
        measurement = {
            "value": 2.3e-9,
            "unit": "s",
            "passed": True,
        }
        result = fmt.format_measurement(measurement)
        # Should not show pass/fail without spec
        assert "\u2713" not in result


class TestMeasurementFormatterFormatMeasurementDict:
    """Test MeasurementFormatter.format_measurement_dict() method."""

    def test_format_measurement_dict_html(self):
        """Test formatting measurement dict as HTML."""
        fmt = MeasurementFormatter()
        measurements: dict[str, dict[str, Any]] = {
            "rise_time": {"value": 2.3e-9, "unit": "s"},
            "frequency": {"value": 440.0, "unit": "Hz"},
        }
        result = fmt.format_measurement_dict(measurements, html=True)
        assert "<ul>" in result
        assert "</ul>" in result
        assert "<li>" in result
        assert "</li>" in result
        assert "<strong>Rise Time:</strong>" in result
        assert "<strong>Frequency:</strong>" in result
        assert "2.300" in result
        assert "440.0" in result

    def test_format_measurement_dict_plain_text(self):
        """Test formatting measurement dict as plain text."""
        fmt = MeasurementFormatter()
        measurements = {
            "rise_time": {"value": 2.3e-9, "unit": "s"},
            "frequency": {"value": 440.0, "unit": "Hz"},
        }
        result = fmt.format_measurement_dict(measurements, html=False)
        assert "<ul>" not in result
        assert "Rise Time:" in result
        assert "Frequency:" in result
        assert "\n" in result
        lines = result.split("\n")
        assert len(lines) == 2

    def test_format_measurement_dict_snake_case_to_title(self):
        """Test that snake_case keys are converted to Title Case."""
        fmt = MeasurementFormatter()
        measurements = {
            "total_harmonic_distortion": {"value": 0.5, "unit": "%"},
        }
        result = fmt.format_measurement_dict(measurements, html=False)
        assert "Total Harmonic Distortion:" in result

    def test_format_measurement_dict_empty(self):
        """Test formatting empty measurement dictionary."""
        fmt = MeasurementFormatter()
        measurements: dict[str, dict[str, Any]] = {}
        result = fmt.format_measurement_dict(measurements, html=True)
        assert "<ul>" in result
        assert "</ul>" in result
        # Should be empty list
        assert result.strip() == "<ul>\n\n</ul>"

    def test_format_measurement_dict_single_entry(self):
        """Test formatting measurement dict with single entry."""
        fmt = MeasurementFormatter()
        measurements = {"voltage": {"value": 3.3, "unit": "V"}}
        result = fmt.format_measurement_dict(measurements, html=False)
        assert "Voltage:" in result
        assert "3.300" in result


class TestMeasurementFormatterToDisplayDict:
    """Test MeasurementFormatter.to_display_dict() method."""

    def test_to_display_dict_basic(self):
        """Test converting measurements to display dictionary."""
        fmt = MeasurementFormatter()
        measurements = {
            "rise_time": {"value": 2.3e-9, "unit": "s"},
            "frequency": {"value": 440.0, "unit": "Hz"},
        }
        result = fmt.to_display_dict(measurements)
        assert isinstance(result, dict)
        assert "rise_time" in result
        assert "frequency" in result
        assert "2.300" in result["rise_time"]
        assert "440.0" in result["frequency"]

    def test_to_display_dict_preserves_keys(self):
        """Test that to_display_dict preserves original keys."""
        fmt = MeasurementFormatter()
        measurements = {
            "snake_case_key": {"value": 1.0, "unit": "V"},
        }
        result = fmt.to_display_dict(measurements)
        assert "snake_case_key" in result
        # Keys should not be converted to title case
        assert "Snake Case Key" not in result

    def test_to_display_dict_empty(self):
        """Test converting empty measurement dict."""
        fmt = MeasurementFormatter()
        measurements: dict[str, dict[str, Any]] = {}
        result = fmt.to_display_dict(measurements)
        assert result == {}

    def test_to_display_dict_with_none_values(self):
        """Test that None values are handled as N/A."""
        fmt = MeasurementFormatter()
        measurements: dict[str, dict[str, Any]] = {
            "valid": {"value": 1.0, "unit": "V"},
            "invalid": {"value": None, "unit": "V"},
        }
        result = fmt.to_display_dict(measurements)
        assert result["valid"] != "N/A"
        assert result["invalid"] == "N/A"


class TestFormatMeasurementFunction:
    """Test format_measurement() convenience function."""

    def test_format_measurement_basic(self):
        """Test basic format_measurement function."""
        measurement = {"value": 2.3e-9, "unit": "s"}
        result = format_measurement(measurement)
        assert "2.300" in result
        assert "ns" in result or "n" in result

    def test_format_measurement_custom_sig_figs(self):
        """Test format_measurement with custom significant figures."""
        measurement = {"value": 2.3e-9, "unit": "s"}
        result = format_measurement(measurement, sig_figs=3)
        assert "2.30" in result

    def test_format_measurement_default_sig_figs(self):
        """Test format_measurement uses default 4 sig figs."""
        measurement = {"value": 1.23456789, "unit": "V"}
        result = format_measurement(measurement)
        # Should have 4 significant figures (1.2346 rounded)
        assert "1.234" in result or "1.235" in result

    def test_format_measurement_with_ratio(self):
        """Test format_measurement with ratio unit."""
        measurement = {"value": 0.75, "unit": "ratio"}
        result = format_measurement(measurement)
        assert "75.00" in result
        assert "%" in result


class TestFormatMeasurementDictFunction:
    """Test format_measurement_dict() convenience function."""

    def test_format_measurement_dict_html_default(self):
        """Test format_measurement_dict returns HTML by default."""
        measurements = {
            "rise_time": {"value": 2.3e-9, "unit": "s"},
        }
        result = format_measurement_dict(measurements)
        assert "<ul>" in result
        assert "</ul>" in result

    def test_format_measurement_dict_plain_text(self):
        """Test format_measurement_dict with html=False."""
        measurements = {
            "rise_time": {"value": 2.3e-9, "unit": "s"},
        }
        result = format_measurement_dict(measurements, html=False)
        assert "<ul>" not in result
        assert "Rise Time:" in result

    def test_format_measurement_dict_custom_sig_figs(self):
        """Test format_measurement_dict with custom significant figures."""
        measurements = {
            "value": {"value": 1.23456789, "unit": "V"},
        }
        result = format_measurement_dict(measurements, sig_figs=3)
        assert "1.23" in result

    def test_format_measurement_dict_empty(self):
        """Test format_measurement_dict with empty dict."""
        measurements: dict[str, dict[str, Any]] = {}
        result = format_measurement_dict(measurements)
        assert "<ul>" in result
        assert "</ul>" in result


class TestConvertToMeasurementDict:
    """Test convert_to_measurement_dict() helper function."""

    def test_convert_to_measurement_dict_basic(self):
        """Test basic conversion from raw values to measurement dict."""
        raw = {"rise_time": 2.3e-9, "frequency": 440.0}
        units = {"rise_time": "s", "frequency": "Hz"}
        result = convert_to_measurement_dict(raw, units)
        assert "rise_time" in result
        assert "frequency" in result
        assert result["rise_time"]["value"] == 2.3e-9
        assert result["rise_time"]["unit"] == "s"
        assert result["frequency"]["value"] == 440.0
        assert result["frequency"]["unit"] == "Hz"

    def test_convert_to_measurement_dict_missing_unit(self):
        """Test conversion when unit is not in unit_map (defaults to empty string)."""
        raw: dict[str, float] = {"unknown": 42.0}
        units: dict[str, str] = {}
        result = convert_to_measurement_dict(raw, units)
        assert result["unknown"]["unit"] == ""

    def test_convert_to_measurement_dict_filters_non_numeric(self):
        """Test that non-numeric values are filtered out."""
        raw: dict[str, Any] = {
            "valid": 42.0,
            "invalid_str": "not a number",
            "invalid_list": [1, 2, 3],
        }
        units: dict[str, str] = {"valid": "V", "invalid_str": "V", "invalid_list": "V"}
        result = convert_to_measurement_dict(raw, units)
        assert "valid" in result
        assert "invalid_str" not in result
        assert "invalid_list" not in result

    def test_convert_to_measurement_dict_int_values(self):
        """Test that integer values are preserved."""
        raw: dict[str, float] = {"count": 100}
        units: dict[str, str] = {"count": ""}
        result = convert_to_measurement_dict(raw, units)
        assert result["count"]["value"] == 100

    def test_convert_to_measurement_dict_float_values(self):
        """Test that float values are preserved."""
        raw: dict[str, float] = {"voltage": 3.3}
        units: dict[str, str] = {"voltage": "V"}
        result = convert_to_measurement_dict(raw, units)
        assert result["voltage"]["value"] == 3.3

    def test_convert_to_measurement_dict_zero_value(self):
        """Test that zero values are included."""
        raw: dict[str, float] = {"zero": 0.0, "positive": 1.0}
        units: dict[str, str] = {"zero": "V", "positive": "V"}
        result = convert_to_measurement_dict(raw, units)
        assert "zero" in result
        assert result["zero"]["value"] == 0.0

    def test_convert_to_measurement_dict_negative_value(self):
        """Test that negative values are preserved."""
        raw: dict[str, float] = {"offset": -2.5}
        units: dict[str, str] = {"offset": "V"}
        result = convert_to_measurement_dict(raw, units)
        assert result["offset"]["value"] == -2.5

    def test_convert_to_measurement_dict_empty_inputs(self):
        """Test conversion with empty inputs."""
        raw: dict[str, float] = {}
        units: dict[str, str] = {}
        result = convert_to_measurement_dict(raw, units)
        assert result == {}


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_format_single_very_small_ratio(self):
        """Test formatting very small ratio values with SI prefix scaling."""
        fmt = MeasurementFormatter()
        result = fmt.format_single(0.001, "ratio")
        # 0.001 * 100 = 0.1%, which gets SI-scaled to 100.0000 m%
        assert "100.0000" in result
        assert "m" in result  # milli prefix
        assert "%" in result

    def test_format_single_very_large_ratio(self):
        """Test formatting ratio > 1 (edge case)."""
        fmt = MeasurementFormatter()
        result = fmt.format_single(1.5, "ratio")
        assert "150" in result
        assert "%" in result

    def test_format_measurement_with_all_fields(self):
        """Test formatting measurement with all possible fields."""
        fmt = MeasurementFormatter(show_specs=True)
        measurement = {
            "value": 2.3e-9,
            "unit": "s",
            "spec": 10e-9,
            "spec_type": "max",
            "passed": True,
        }
        result = fmt.format_measurement(measurement)
        # Should contain value, spec, and pass indicator
        assert "2.300" in result
        assert "spec:" in result
        assert "\u2713" in result

    def test_format_measurement_integer_value(self):
        """Test formatting measurement with integer value."""
        fmt = MeasurementFormatter()
        measurement = {"value": 100, "unit": ""}
        result = fmt.format_measurement(measurement)
        assert "100" in result

    def test_formatter_consistency(self):
        """Test that formatter produces consistent results."""
        fmt = MeasurementFormatter()
        measurement = {"value": 2.3e-9, "unit": "s"}
        r1 = fmt.format_measurement(measurement)
        r2 = fmt.format_measurement(measurement)
        assert r1 == r2

    def test_format_single_with_unicode_disabled(self):
        """Test formatting with unicode prefixes disabled."""
        custom_fmt = NumberFormatter(unicode_prefixes=False)
        fmt = MeasurementFormatter(number_formatter=custom_fmt)
        result = fmt.format_single(2.3e-6, "s")
        # Should use 'u' instead of Unicode mu
        assert "u" in result
        assert "\u03bc" not in result


class TestIntegrationScenarios:
    """Integration tests combining multiple features."""

    def test_complete_measurement_workflow(self):
        """Test complete workflow from raw values to HTML output."""
        # Raw measurement data
        raw: dict[str, float] = {
            "rise_time": 2.3e-9,
            "fall_time": 1.8e-9,
            "frequency": 1e6,
            "duty_cycle": 0.5,
        }
        units: dict[str, str] = {
            "rise_time": "s",
            "fall_time": "s",
            "frequency": "Hz",
            "duty_cycle": "ratio",
        }

        # Convert to measurement dict
        measurements = convert_to_measurement_dict(raw, units)

        # Format to HTML
        result = format_measurement_dict(measurements, html=True)

        # Verify all measurements are present
        assert "Rise Time" in result
        assert "Fall Time" in result
        assert "Frequency" in result
        assert "Duty Cycle" in result
        assert "<ul>" in result
        assert "</ul>" in result

    def test_measurement_with_specifications(self):
        """Test measurement formatting with specifications."""
        fmt = MeasurementFormatter(show_specs=True)
        measurements: dict[str, dict[str, Any]] = {
            "rise_time": {
                "value": 2.3e-9,
                "unit": "s",
                "spec": 5e-9,
                "spec_type": "max",
                "passed": True,
            },
            "frequency": {
                "value": 1.1e6,
                "unit": "Hz",
                "spec": 1e6,
                "spec_type": "min",
                "passed": True,
            },
        }

        display_dict = fmt.to_display_dict(measurements)

        # Both should have spec indicators
        assert "spec:" in display_dict["rise_time"]
        assert "spec:" in display_dict["frequency"]
        # Both should show pass
        assert "\u2713" in display_dict["rise_time"]
        assert "\u2713" in display_dict["frequency"]

    def test_mixed_valid_invalid_measurements(self):
        """Test handling mix of valid and invalid measurements."""
        fmt = MeasurementFormatter()
        measurements: dict[str, dict[str, Any]] = {
            "valid": {"value": 1.0, "unit": "V"},
            "none_value": {"value": None, "unit": "V"},
            "string_value": {"value": "ERROR", "unit": "V"},
            "missing_value": {"unit": "V"},
        }

        display_dict = fmt.to_display_dict(measurements)

        assert "1.000" in display_dict["valid"]
        assert display_dict["none_value"] == "N/A"
        assert display_dict["string_value"] == "ERROR"
        assert display_dict["missing_value"] == "N/A"
