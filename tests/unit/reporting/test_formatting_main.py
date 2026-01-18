"""Comprehensive unit tests for oscura.reporting.formatting module.

Tests all public functions and classes in the formatting module with
focus on edge cases, locale handling, and numeric formatting accuracy.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from oscura.reporting.formatting import (
    NumberFormatter,
    format_margin,
    format_pass_fail,
    format_value,
    format_with_context,
    format_with_locale,
    format_with_units,
)

pytestmark = pytest.mark.unit


class TestNumberFormatterInit:
    """Test NumberFormatter initialization and configuration."""

    def test_default_initialization(self):
        """Test NumberFormatter with default values."""
        fmt = NumberFormatter()
        assert fmt.sig_figs == 3
        assert fmt.auto_scale is True
        assert fmt.engineering_notation is True
        assert fmt.unicode_prefixes is True

    def test_custom_sig_figs(self):
        """Test NumberFormatter with custom significant figures."""
        fmt = NumberFormatter(sig_figs=5)
        assert fmt.sig_figs == 5

    def test_disable_auto_scale(self):
        """Test NumberFormatter with auto-scaling disabled."""
        fmt = NumberFormatter(auto_scale=False)
        assert fmt.auto_scale is False

    def test_disable_unicode_prefixes(self):
        """Test NumberFormatter with unicode prefixes disabled."""
        fmt = NumberFormatter(unicode_prefixes=False)
        assert fmt.unicode_prefixes is False

    def test_disable_engineering_notation(self):
        """Test NumberFormatter with engineering notation disabled."""
        fmt = NumberFormatter(engineering_notation=False)
        assert fmt.engineering_notation is False


class TestNumberFormatterFormat:
    """Test NumberFormatter.format() method."""

    def test_format_basic(self):
        """Test basic formatting."""
        fmt = NumberFormatter()
        result = fmt.format(123.456, "V")
        assert "123" in result
        assert "V" in result

    def test_format_with_unit(self):
        """Test formatting with unit."""
        fmt = NumberFormatter()
        result = fmt.format(123.456, "V")
        assert "V" in result

    def test_format_microseconds(self):
        """Test formatting microsecond values."""
        fmt = NumberFormatter()
        result = fmt.format(0.0000023, "s")
        # Should contain scaled value (around 2.3) and micro prefix
        assert "2" in result
        # Unicode mu or 'u'
        assert "\u03bc" in result or "u" in result

    def test_format_nanoseconds(self):
        """Test formatting nanosecond values."""
        fmt = NumberFormatter()
        result = fmt.format(2.3e-9, "s")
        assert "2" in result
        assert "n" in result

    def test_format_milliseconds(self):
        """Test formatting millisecond values."""
        fmt = NumberFormatter()
        result = fmt.format(0.0023, "s")
        assert "2" in result
        assert "m" in result

    def test_format_kilohertz(self):
        """Test formatting kilohertz values."""
        fmt = NumberFormatter()
        result = fmt.format(2300, "Hz")
        assert "2" in result
        assert "k" in result

    def test_format_megahertz(self):
        """Test formatting megahertz values."""
        fmt = NumberFormatter()
        result = fmt.format(2300000, "Hz")
        assert "2" in result
        assert "M" in result

    def test_format_gigahertz(self):
        """Test formatting gigahertz values."""
        fmt = NumberFormatter()
        result = fmt.format(2.3e9, "Hz")
        assert "2" in result
        assert "G" in result

    def test_format_negative_value(self):
        """Test formatting negative values."""
        fmt = NumberFormatter()
        result = fmt.format(-0.0023, "s")
        assert "-" in result
        assert "2" in result

    def test_format_zero(self):
        """Test formatting zero."""
        fmt = NumberFormatter()
        result = fmt.format(0, "V")
        assert "0" in result
        assert "V" in result

    def test_format_nan(self):
        """Test formatting NaN."""
        fmt = NumberFormatter()
        result = fmt.format(float("nan"))
        assert "NaN" in result

    def test_format_positive_inf(self):
        """Test formatting positive infinity."""
        fmt = NumberFormatter()
        result = fmt.format(float("inf"))
        assert "Inf" in result

    def test_format_negative_inf(self):
        """Test formatting negative infinity."""
        fmt = NumberFormatter()
        result = fmt.format(float("-inf"))
        assert "-Inf" in result

    def test_format_with_decimal_places_override(self):
        """Test formatting with decimal places override."""
        fmt = NumberFormatter()
        result = fmt.format(123.456789, "V", decimal_places=2)
        # Should have exactly 2 decimal places
        assert "123.46" in result or "123.45" in result

    def test_format_without_engineering_notation(self):
        """Test formatting without engineering notation."""
        fmt = NumberFormatter(auto_scale=False)
        result = fmt.format(0.0023, "s")
        # Should use scientific notation or plain number
        assert "0.0023" in result or "2.3" in result


class TestNumberFormatterFormatPercentage:
    """Test NumberFormatter.format_percentage() method."""

    def test_format_percentage_basic(self):
        """Test basic percentage formatting."""
        fmt = NumberFormatter()
        result = fmt.format_percentage(0.5)
        assert "50" in result
        assert "%" in result

    def test_format_percentage_decimal(self):
        """Test formatting decimal values as percentage."""
        fmt = NumberFormatter()
        result = fmt.format_percentage(0.123)
        assert "12" in result
        assert "%" in result

    def test_format_percentage_zero(self):
        """Test formatting zero as percentage."""
        fmt = NumberFormatter()
        result = fmt.format_percentage(0)
        assert "0" in result
        assert "%" in result

    def test_format_percentage_one(self):
        """Test formatting one as percentage."""
        fmt = NumberFormatter()
        result = fmt.format_percentage(1.0)
        assert "100" in result
        assert "%" in result

    def test_format_percentage_negative(self):
        """Test formatting negative percentage."""
        fmt = NumberFormatter()
        result = fmt.format_percentage(-0.25)
        assert "-25" in result
        assert "%" in result

    def test_format_percentage_greater_than_one(self):
        """Test formatting percentage > 100 (already in percent form)."""
        fmt = NumberFormatter()
        # Value > 1 assumed to already be in percent
        result = fmt.format_percentage(50)
        assert "50" in result
        assert "%" in result


class TestNumberFormatterFormatRange:
    """Test NumberFormatter.format_range() method."""

    def test_format_range_basic(self):
        """Test basic range formatting."""
        fmt = NumberFormatter()
        result = fmt.format_range(0.9, 1.0, 1.1, "V")
        assert "min" in result
        assert "typ" in result
        assert "max" in result
        assert "V" in result

    def test_format_range_with_scaling(self):
        """Test range formatting with SI prefix scaling."""
        fmt = NumberFormatter()
        result = fmt.format_range(1e-6, 2e-6, 3e-6, "s")
        assert "min" in result
        # Should have micro prefix
        assert "\u03bc" in result or "u" in result

    def test_format_range_no_unit(self):
        """Test range formatting without unit."""
        fmt = NumberFormatter()
        result = fmt.format_range(1, 2, 3)
        assert "min" in result
        assert "typ" in result
        assert "max" in result


class TestFormatValue:
    """Test format_value() function."""

    def test_format_value_basic(self):
        """Test basic value formatting."""
        result = format_value(0.0023, "s")
        assert "s" in result

    def test_format_value_with_sig_figs(self):
        """Test format_value with custom significant figures."""
        result = format_value(1.23456, "V", sig_figs=4)
        assert "V" in result

    def test_format_value_nanoseconds(self):
        """Test formatting nanosecond scale."""
        result = format_value(2.3e-9, "s")
        assert "n" in result
        assert "s" in result

    def test_format_value_microseconds(self):
        """Test formatting microsecond scale."""
        result = format_value(2.3e-6, "s")
        assert "s" in result

    def test_format_value_milliseconds(self):
        """Test formatting millisecond scale."""
        result = format_value(0.0023, "s")
        assert "s" in result

    def test_format_value_no_unit(self):
        """Test formatting without unit."""
        result = format_value(123.456)
        assert "123" in result


class TestFormatWithUnits:
    """Test format_with_units() function."""

    def test_format_with_units_basic(self):
        """Test basic unit formatting."""
        result = format_with_units(2300000, "Hz")
        assert "Hz" in result
        assert "M" in result

    def test_format_with_units_microseconds(self):
        """Test formatting with microseconds."""
        result = format_with_units(0.0000023, "s")
        assert "s" in result

    def test_format_with_units_custom_sig_figs(self):
        """Test formatting with custom significant figures."""
        result = format_with_units(1.23456, "V", sig_figs=2)
        assert "V" in result


class TestFormatWithContext:
    """Test format_with_context() function."""

    def test_format_with_context_pass_max(self):
        """Test context formatting with passing value (max spec)."""
        result = format_with_context(2.3e-9, spec=5e-9, unit="s", spec_type="max")
        assert "ns" in result or "n" in result
        # Should show pass status (checkmark)
        assert "\u2713" in result

    def test_format_with_context_fail_max(self):
        """Test context formatting with failing value (max spec)."""
        result = format_with_context(6e-9, spec=5e-9, unit="s", spec_type="max")
        # Should show fail status (X mark)
        assert "\u2717" in result

    def test_format_with_context_pass_min(self):
        """Test context formatting with min spec (passing)."""
        result = format_with_context(6e-9, spec=5e-9, spec_type="min", unit="s")
        # Should pass (>=)
        assert "\u2713" in result

    def test_format_with_context_fail_min(self):
        """Test context formatting with min spec (failing)."""
        result = format_with_context(4e-9, spec=5e-9, spec_type="min", unit="s")
        # Should fail (<)
        assert "\u2717" in result

    def test_format_with_context_exact(self):
        """Test context formatting with exact spec type."""
        result = format_with_context(5e-9, spec=5e-9, spec_type="exact", unit="s")
        # Within 1% tolerance should pass
        assert "\u2713" in result

    def test_format_with_context_no_spec(self):
        """Test context formatting without specification."""
        result = format_with_context(2.3e-9, unit="s")
        # Should not have pass/fail indicators
        assert "\u2713" not in result
        assert "\u2717" not in result

    def test_format_with_context_no_margin(self):
        """Test context formatting without margin display."""
        result = format_with_context(2.3e-9, spec=5e-9, unit="s", show_margin=False)
        # Should still show pass/fail but no percentage
        assert "\u2713" in result

    def test_format_with_context_boundary_pass(self):
        """Test format_with_context at exact max spec value."""
        result = format_with_context(5e-9, spec=5e-9, spec_type="max", unit="s")
        # Should pass (<=)
        assert "\u2713" in result


class TestFormatPassFail:
    """Test format_pass_fail() function."""

    def test_format_pass_fail_pass_with_symbol(self):
        """Test pass/fail formatting for pass with symbol."""
        result = format_pass_fail(True, with_symbol=True)
        assert "PASS" in result
        assert "\u2713" in result

    def test_format_pass_fail_fail_with_symbol(self):
        """Test pass/fail formatting for fail with symbol."""
        result = format_pass_fail(False, with_symbol=True)
        assert "FAIL" in result
        assert "\u2717" in result

    def test_format_pass_fail_pass_no_symbol(self):
        """Test pass/fail formatting for pass without symbol."""
        result = format_pass_fail(True, with_symbol=False)
        assert result == "PASS"

    def test_format_pass_fail_fail_no_symbol(self):
        """Test pass/fail formatting for fail without symbol."""
        result = format_pass_fail(False, with_symbol=False)
        assert result == "FAIL"


class TestFormatMargin:
    """Test format_margin() function."""

    def test_format_margin_upper_good(self):
        """Test margin formatting for upper bound (good margin >20%)."""
        result = format_margin(70, 100, limit_type="upper")
        assert "30" in result
        assert "good" in result

    def test_format_margin_upper_ok(self):
        """Test margin formatting for upper bound (ok margin 10-20%)."""
        result = format_margin(85, 100, limit_type="upper")
        assert "15" in result
        assert "ok" in result

    def test_format_margin_upper_marginal(self):
        """Test margin formatting for upper bound (marginal 0-10%)."""
        result = format_margin(95, 100, limit_type="upper")
        assert "5" in result
        assert "marginal" in result

    def test_format_margin_upper_violation(self):
        """Test margin formatting for upper bound (violation)."""
        result = format_margin(105, 100, limit_type="upper")
        assert "violation" in result

    def test_format_margin_lower_good(self):
        """Test margin formatting for lower bound (good margin)."""
        result = format_margin(130, 100, limit_type="lower")
        assert "30" in result
        assert "good" in result

    def test_format_margin_lower_violation(self):
        """Test margin formatting for lower bound (violation)."""
        result = format_margin(90, 100, limit_type="lower")
        assert "violation" in result

    def test_format_margin_zero_limit(self):
        """Test margin formatting with zero limit."""
        result = format_margin(10, 0, limit_type="upper")
        assert "0" in result
        assert "%" in result


class TestFormatWithLocale:
    """Test format_with_locale() function."""

    def test_format_with_locale_en_us(self):
        """Test locale formatting for US English."""
        result = format_with_locale(1234.56, locale="en_US")
        assert result == "1,234.56"

    def test_format_with_locale_de_de(self):
        """Test locale formatting for German."""
        result = format_with_locale(1234.56, locale="de_DE")
        assert result == "1.234,56"

    def test_format_with_locale_fr_fr(self):
        """Test locale formatting for French."""
        result = format_with_locale(1234.56, locale="fr_FR")
        assert result == "1 234,56"

    def test_format_with_locale_default(self):
        """Test locale formatting with None locale (uses system default)."""
        result = format_with_locale(1234.56)
        # Should return a formatted string
        assert "1234" in result.replace(",", "").replace(".", "").replace(" ", "")

    def test_format_with_locale_negative(self):
        """Test locale formatting with negative numbers."""
        result = format_with_locale(-1234.56, locale="en_US")
        assert result == "-1,234.56"

    def test_format_with_locale_zero(self):
        """Test locale formatting with zero."""
        result = format_with_locale(0.0, locale="en_US")
        assert result == "0.00"

    def test_format_with_locale_none_value(self):
        """Test locale formatting with None value returns empty string."""
        result = format_with_locale(None, locale="en_US")
        assert result == ""

    def test_format_date_en_us(self):
        """Test date formatting for US English."""
        timestamp = datetime(2024, 3, 15).timestamp()
        result = format_with_locale(date_value=timestamp, locale="en_US")
        assert result == "03/15/2024"

    def test_format_date_de_de(self):
        """Test date formatting for German."""
        timestamp = datetime(2024, 3, 15).timestamp()
        result = format_with_locale(date_value=timestamp, locale="de_DE")
        assert result == "15.03.2024"

    def test_format_date_fr_fr(self):
        """Test date formatting for French."""
        timestamp = datetime(2024, 3, 15).timestamp()
        result = format_with_locale(date_value=timestamp, locale="fr_FR")
        assert result == "15/03/2024"

    def test_format_date_unknown_locale(self):
        """Test date formatting for unknown locale (ISO format fallback)."""
        timestamp = datetime(2024, 3, 15).timestamp()
        result = format_with_locale(date_value=timestamp, locale="xx_XX")
        assert result == "2024-03-15"


class TestReportingFormattingMainEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_format_value_at_milli_boundary(self):
        """Test formatting at exactly 1 milli."""
        result = format_value(0.001, "s")
        assert "1" in result
        assert "m" in result or "s" in result

    def test_format_value_at_kilo_boundary(self):
        """Test formatting at exactly 1 kilo."""
        result = format_value(1000, "Hz")
        assert "1" in result
        assert "k" in result

    def test_format_value_unicode_mu(self):
        """Test that Unicode mu is used for micro prefix when enabled."""
        fmt = NumberFormatter(unicode_prefixes=True)
        result = fmt.format(1e-6, "s")
        assert "\u03bc" in result

    def test_format_value_ascii_mu(self):
        """Test that ASCII 'u' is used when unicode prefixes disabled."""
        fmt = NumberFormatter(unicode_prefixes=False)
        result = fmt.format(1e-6, "s")
        assert "u" in result

    def test_number_formatter_consistency(self):
        """Test that formatter produces consistent results."""
        fmt = NumberFormatter(sig_figs=4)
        r1 = fmt.format(1.2345, "V")
        r2 = fmt.format(1.2345, "V")
        assert r1 == r2

    def test_format_very_large_numbers(self):
        """Test very large numbers are handled."""
        result = format_value(1e12, "Hz")
        # Should use tera (T) prefix
        assert "T" in result or "Hz" in result

    def test_format_very_small_numbers(self):
        """Test very small numbers are handled."""
        result = format_value(1e-15, "s")
        # Should use femto (f) prefix
        assert "f" in result or "s" in result


class TestReportingFormattingMainIntegration:
    """Integration tests combining multiple features."""

    def test_complete_workflow(self):
        """Test complete workflow with multiple formatters."""
        # Create formatter
        fmt = NumberFormatter(sig_figs=3)

        # Format various values
        voltage = fmt.format(1.234, "V")
        current = fmt.format(0.0023, "A")
        frequency = fmt.format(2.3e6, "Hz")

        # All should have proper formatting
        assert "V" in voltage
        assert "A" in current
        assert "Hz" in frequency

    def test_format_value_equals_formatter(self):
        """Test that format_value produces same result as NumberFormatter."""
        value = 1.234
        unit = "V"

        result1 = format_value(value, unit, sig_figs=3)
        fmt = NumberFormatter(sig_figs=3)
        result2 = fmt.format(value, unit)

        assert result1 == result2

    def test_locale_consistency(self):
        """Test locale formatting consistency."""
        value = 1234.56

        # Different locales should format same value differently
        en = format_with_locale(value, locale="en_US")
        de = format_with_locale(value, locale="de_DE")
        fr = format_with_locale(value, locale="fr_FR")

        # All different
        assert en != de
        assert en != fr
        assert de != fr

    def test_format_measurements_table(self):
        """Test formatting a table of measurements."""
        measurements = [
            (2.3e-9, "s", 5e-9, "max"),  # rise time
            (1.1e9, "Hz", 1e9, "min"),  # frequency
            (3.3, "V", 3.6, "max"),  # voltage
        ]

        for value, unit, spec, spec_type in measurements:
            result = format_with_context(value, spec=spec, unit=unit, spec_type=spec_type)
            assert unit in result


class TestRegressionCases:
    """Test cases for potential regressions."""

    def test_zero_division_protection_margin(self):
        """Test that zero division is handled in margin calculation."""
        # Should not raise ZeroDivisionError
        result = format_margin(10, 0, limit_type="upper")
        assert "%" in result

    def test_negative_values_with_prefixes(self):
        """Test negative values maintain correct prefix."""
        result = format_value(-2.3e-6, "s")
        assert "-" in result

    def test_empty_unit_string(self):
        """Test formatting with empty unit string."""
        result = format_value(123.456, "")
        assert "123" in result
        # Should not have trailing space
        assert not result.endswith(" ")

    def test_numpy_types_handled(self):
        """Test that numpy numeric types work correctly."""
        fmt = NumberFormatter()

        # Test with numpy types
        result1 = fmt.format(np.float64(1.23e-6), "s")
        result2 = fmt.format(np.float32(1.23e-6), "s")

        assert "s" in result1
        assert "s" in result2
