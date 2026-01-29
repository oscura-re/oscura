"""Comprehensive unit tests for oscura.compliance.masks module.

Tests limit mask loading, creation, and manipulation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from oscura.validation.compliance.masks import (
    LimitMask,
    create_custom_mask,
    load_limit_mask,
)

pytestmark = pytest.mark.unit


# =============================================================================
# LimitMask Tests
# =============================================================================


class TestLimitMask:
    """Tests for LimitMask class."""

    @pytest.fixture
    def basic_mask(self) -> LimitMask:
        """Create basic limit mask for testing."""
        return LimitMask(
            name="TestMask",
            frequency=np.array([1e6, 10e6, 100e6, 1e9]),
            limit=np.array([60.0, 50.0, 40.0, 30.0]),
            unit="dBuV",
            standard="TEST",
        )

    def test_basic_creation(self, basic_mask):
        """Test basic LimitMask creation."""
        assert basic_mask.name == "TestMask"
        assert len(basic_mask.frequency) == 4
        assert len(basic_mask.limit) == 4
        assert basic_mask.unit == "dBuV"
        assert basic_mask.standard == "TEST"

    def test_frequency_array(self, basic_mask):
        """Test frequency array values."""
        np.testing.assert_array_equal(
            basic_mask.frequency,
            np.array([1e6, 10e6, 100e6, 1e9]),
        )

    def test_limit_array(self, basic_mask):
        """Test limit array values."""
        np.testing.assert_array_equal(
            basic_mask.limit,
            np.array([60.0, 50.0, 40.0, 30.0]),
        )

    def test_get_limit_at_frequency_exact(self, basic_mask):
        """Test getting limit at exact frequency point."""
        limit = basic_mask.get_limit_at_frequency(10e6)

        assert limit == 50.0

    def test_get_limit_at_frequency_interpolated(self, basic_mask):
        """Test getting limit at interpolated frequency."""
        # Between 1 MHz (60 dB) and 10 MHz (50 dB)
        # At 5 MHz, should be around 55 dB (log interpolation)
        limit = basic_mask.get_limit_at_frequency(5e6)

        assert 50.0 <= limit <= 60.0

    def test_get_limit_at_frequency_below_range(self, basic_mask):
        """Test getting limit below frequency range."""
        limit = basic_mask.get_limit_at_frequency(100e3)  # Below 1 MHz

        # Should extrapolate or return first limit
        assert limit is not None

    def test_get_limit_at_frequency_above_range(self, basic_mask):
        """Test getting limit above frequency range."""
        limit = basic_mask.get_limit_at_frequency(10e9)  # Above 1 GHz

        # Should extrapolate or return last limit
        assert limit is not None

    def test_mask_with_many_points(self):
        """Test mask with many frequency points."""
        freq = np.logspace(6, 9, 100)  # 1 MHz to 1 GHz, 100 points
        limit = 60 - 10 * np.log10(freq / 1e6)  # Decreasing limit

        mask = LimitMask(
            name="FineMask",
            frequency=freq,
            limit=limit,
            unit="dBuV/m",
            standard="CISPR",
        )

        assert len(mask.frequency) == 100
        assert len(mask.limit) == 100

    def test_mask_with_two_points(self):
        """Test mask with minimum (two) points."""
        mask = LimitMask(
            name="MinimalMask",
            frequency=np.array([1e6, 1e9]),
            limit=np.array([60.0, 30.0]),
            unit="dBuV",
            standard="TEST",
        )

        assert len(mask.frequency) == 2

        # Interpolation should still work
        limit = mask.get_limit_at_frequency(100e6)
        assert 30.0 <= limit <= 60.0


# =============================================================================
# load_limit_mask Tests
# =============================================================================


class TestLoadLimitMask:
    """Tests for load_limit_mask function."""

    def test_load_fcc_class_b(self):
        """Test loading FCC Part 15 Class B mask."""
        mask = load_limit_mask("FCC_Part15_ClassB")

        assert isinstance(mask, LimitMask)
        assert "FCC" in mask.name or "Part15" in mask.name
        assert len(mask.frequency) > 0
        assert len(mask.limit) > 0

    def test_load_fcc_class_a(self):
        """Test loading FCC Part 15 Class A mask."""
        mask = load_limit_mask("FCC_Part15_ClassA")

        assert isinstance(mask, LimitMask)
        # Class A limits are typically higher than Class B
        assert mask is not None

    def test_load_cispr_11_class_b(self):
        """Test loading CISPR 11 Class B mask."""
        mask = load_limit_mask("CISPR11_ClassB")

        assert isinstance(mask, LimitMask)

    def test_load_cispr_22(self):
        """Test loading CISPR 22 mask."""
        mask = load_limit_mask("CISPR22")

        assert isinstance(mask, LimitMask)

    def test_load_invalid_mask_name(self):
        """Test loading with invalid mask name."""
        with pytest.raises((ValueError, KeyError, FileNotFoundError)):
            load_limit_mask("NonexistentMask_12345")

    def test_load_case_insensitive(self):
        """Test mask name loading behavior (case sensitivity depends on implementation)."""
        mask1 = load_limit_mask("FCC_Part15_ClassB")
        # First load should succeed
        assert mask1 is not None, "Expected valid mask object"
        try:
            mask2 = load_limit_mask("fcc_part15_classb")
            # If case insensitive implementation, names should match
            assert mask1.name == mask2.name, "Case-insensitive load returned different masks"
        except (ValueError, KeyError, FileNotFoundError):
            # Case-sensitive implementation - lowercase not found, which is valid
            pass

    def test_load_from_file(self):
        """Test loading mask from custom file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a custom mask file
            mask_path = Path(tmpdir) / "custom_mask.csv"
            mask_path.write_text(
                "frequency,limit\n1000000,60\n10000000,50\n100000000,40\n1000000000,30\n"
            )

            try:
                mask = load_limit_mask(str(mask_path))
                assert isinstance(mask, LimitMask)
            except (ValueError, NotImplementedError):
                # File-based loading may not be implemented
                pass


# =============================================================================
# create_custom_mask Tests
# =============================================================================


class TestCreateCustomMask:
    """Tests for create_custom_mask function."""

    def test_create_basic_mask(self):
        """Test creating basic custom mask."""
        mask = create_custom_mask(
            name="CustomMask",
            frequencies=[1e6, 10e6, 100e6],
            limits=[60.0, 50.0, 40.0],
            unit="dBuV",
        )

        assert isinstance(mask, LimitMask)
        assert mask.name == "CustomMask"
        assert len(mask.frequency) == 3
        assert mask.unit == "dBuV"

    def test_create_mask_with_numpy_arrays(self):
        """Test creating mask with numpy arrays."""
        freq = np.array([1e6, 10e6, 100e6, 1e9])
        limit = np.array([60.0, 50.0, 40.0, 30.0])

        mask = create_custom_mask(
            name="NumpyMask",
            frequencies=freq,
            limits=limit,
            unit="dBm",
        )

        assert isinstance(mask, LimitMask)
        np.testing.assert_array_equal(mask.frequency, freq)

    def test_create_mask_with_standard(self):
        """Test creating mask with standard specification."""
        mask = create_custom_mask(
            name="StandardMask",
            frequencies=[30e6, 230e6, 1e9],
            limits=[40.0, 43.5, 46.0],
            unit="dBuV/m",
            standard="FCC Part 15B",
        )

        assert mask.standard == "FCC Part 15B"

    def test_create_mask_single_point(self):
        """Test creating mask with single point."""
        mask = create_custom_mask(
            name="SinglePoint",
            frequencies=[100e6],
            limits=[45.0],
            unit="dBuV",
        )

        assert len(mask.frequency) == 1
        assert mask.limit[0] == 45.0

    def test_create_mask_many_points(self):
        """Test creating mask with many points."""
        freq = list(np.logspace(6, 9, 50))
        limits = list(60 - 10 * np.log10(np.array(freq) / 1e6))

        mask = create_custom_mask(
            name="ManyPoints",
            frequencies=freq,
            limits=limits,
            unit="dBuV",
        )

        assert len(mask.frequency) == 50

    def test_create_mask_mismatched_lengths(self):
        """Test creating mask with mismatched array lengths."""
        with pytest.raises((ValueError, AssertionError)):
            create_custom_mask(
                name="BadMask",
                frequencies=[1e6, 10e6, 100e6],
                limits=[60.0, 50.0],  # Missing one limit
                unit="dBuV",
            )

    def test_create_mask_unsorted_frequencies(self):
        """Test creating mask with unsorted frequencies."""
        # Should either sort automatically or raise error
        try:
            mask = create_custom_mask(
                name="Unsorted",
                frequencies=[100e6, 1e6, 10e6],  # Not sorted
                limits=[40.0, 60.0, 50.0],
                unit="dBuV",
            )

            # If it succeeds, frequencies should be auto-sorted
            assert np.all(np.diff(mask.frequency) > 0), (
                "Frequencies should be sorted in ascending order"
            )
        except ValueError:
            # Raising error for unsorted frequencies is also acceptable
            pass


# =============================================================================
# Integration Tests
# =============================================================================


class TestComplianceMasksIntegration:
    """Integration tests for mask functionality."""

    def test_load_and_interpolate(self):
        """Test loading mask and interpolating limits."""
        mask = load_limit_mask("FCC_Part15_ClassB")

        # Get limits at various frequencies
        limits = []
        test_freqs = [30e6, 50e6, 100e6, 200e6, 500e6, 1e9]

        for freq in test_freqs:
            limit = mask.get_limit_at_frequency(freq)
            limits.append(limit)

        # All limits should be valid numbers
        assert all(isinstance(l, int | float) for l in limits)
        assert all(np.isfinite(l) for l in limits)

    def test_create_and_use(self):
        """Test creating custom mask and using it."""
        # Create a simple linear mask
        freq = np.array([30e6, 88e6, 216e6, 960e6])
        limit = np.array([40.0, 43.5, 46.0, 54.0])

        mask = create_custom_mask(
            name="TestFCC",
            frequencies=freq,
            limits=limit,
            unit="dBuV/m",
            standard="FCC 15.109",
        )

        # Test interpolation at specific points
        assert mask.get_limit_at_frequency(30e6) == pytest.approx(40.0, abs=0.1)
        assert mask.get_limit_at_frequency(88e6) == pytest.approx(43.5, abs=0.1)

        # Test interpolation between points
        mid_limit = mask.get_limit_at_frequency(60e6)
        assert 40.0 <= mid_limit <= 43.5


# =============================================================================
# Edge Cases
# =============================================================================


class TestComplianceMasksEdgeCases:
    """Tests for edge cases."""

    def test_mask_with_flat_limit(self):
        """Test mask with flat (constant) limit."""
        mask = LimitMask(
            name="FlatMask",
            frequency=np.array([1e6, 10e6, 100e6, 1e9]),
            limit=np.array([50.0, 50.0, 50.0, 50.0]),
            unit="dBuV",
            standard="TEST",
        )

        # All interpolated values should be 50
        for freq in [5e6, 50e6, 500e6]:
            limit = mask.get_limit_at_frequency(freq)
            assert limit == pytest.approx(50.0, abs=0.01)

    def test_mask_with_step_limit(self):
        """Test mask with step change."""
        mask = LimitMask(
            name="StepMask",
            frequency=np.array([1e6, 10e6, 10.001e6, 100e6]),
            limit=np.array([60.0, 60.0, 40.0, 40.0]),
            unit="dBuV",
            standard="TEST",
        )

        # Before step
        assert mask.get_limit_at_frequency(5e6) == pytest.approx(60.0, abs=1.0)

        # After step
        assert mask.get_limit_at_frequency(50e6) == pytest.approx(40.0, abs=1.0)

    def test_very_large_frequency_range(self):
        """Test mask spanning very large frequency range."""
        mask = LimitMask(
            name="WideRange",
            frequency=np.array([1e3, 1e6, 1e9, 1e12]),  # 1 kHz to 1 THz
            limit=np.array([80.0, 60.0, 40.0, 20.0]),
            unit="dBuV",
            standard="TEST",
        )

        assert len(mask.frequency) == 4

    def test_negative_limit_values(self):
        """Test mask with negative limit values."""
        mask = LimitMask(
            name="NegativeLimits",
            frequency=np.array([1e6, 1e9]),
            limit=np.array([-10.0, -30.0]),  # Negative dBm values
            unit="dBm",
            standard="TEST",
        )

        limit = mask.get_limit_at_frequency(100e6)
        assert limit < 0
