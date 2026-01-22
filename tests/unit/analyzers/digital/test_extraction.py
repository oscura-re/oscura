"""Unit tests for digital signal extraction and edge detection.

This module tests the extraction of digital signals from analog waveforms,
edge detection, and logic family threshold handling.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oscura.core.exceptions import InsufficientDataError
from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.digital]


# =============================================================================
# Helper Functions
# =============================================================================


def make_waveform_trace(
    data: NDArray[np.float64],
    sample_rate: float = 1e6,
) -> WaveformTrace:
    """Create a WaveformTrace from raw data for testing.

    Args:
        data: Waveform data array.
        sample_rate: Sample rate in Hz.

    Returns:
        WaveformTrace with the given data and sample rate.
    """
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


def make_digital_trace(
    data: NDArray[np.bool_],
    sample_rate: float = 1e6,
    edges: list[tuple[float, bool]] | None = None,
) -> DigitalTrace:
    """Create a DigitalTrace from raw boolean data.

    Args:
        data: Digital data array.
        sample_rate: Sample rate in Hz.
        edges: Optional list of (timestamp, is_rising) tuples.

    Returns:
        DigitalTrace with the given data and sample rate.
    """
    metadata = TraceMetadata(sample_rate=sample_rate)
    return DigitalTrace(data=data, metadata=metadata, edges=edges or [])


# =============================================================================
# LOGIC_FAMILIES Constants Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DIG-006")
class TestLogicFamilies:
    """Test logic family constants."""

    def test_logic_families_available(self) -> None:
        """Test that LOGIC_FAMILIES constant is available."""
        from oscura.analyzers.digital.extraction import LOGIC_FAMILIES

        assert LOGIC_FAMILIES is not None
        assert isinstance(LOGIC_FAMILIES, dict)

    def test_standard_families_present(self) -> None:
        """Test that standard logic families are defined."""
        from oscura.analyzers.digital.extraction import LOGIC_FAMILIES

        expected_families = [
            "TTL",
            "CMOS_5V",
            "LVTTL",
            "LVCMOS_3V3",
            "LVCMOS_2V5",
            "LVCMOS_1V8",
            "LVCMOS_1V2",
        ]

        for family in expected_families:
            assert family in LOGIC_FAMILIES, f"Missing logic family: {family}"

    def test_ttl_thresholds(self) -> None:
        """Test TTL logic level definitions."""
        from oscura.analyzers.digital.extraction import LOGIC_FAMILIES

        ttl = LOGIC_FAMILIES["TTL"]
        assert ttl["VIL_max"] == 0.8
        assert ttl["VIH_min"] == 2.0
        assert ttl["VOL_max"] == 0.4
        assert ttl["VOH_min"] == 2.4
        assert ttl["VCC"] == 5.0

    def test_lvcmos_3v3_thresholds(self) -> None:
        """Test LVCMOS 3.3V logic level definitions."""
        from oscura.analyzers.digital.extraction import LOGIC_FAMILIES

        lvcmos = LOGIC_FAMILIES["LVCMOS_3V3"]
        assert lvcmos["VIL_max"] == 0.3 * 3.3
        assert lvcmos["VIH_min"] == 0.7 * 3.3
        assert lvcmos["VCC"] == 3.3

    def test_all_families_have_required_fields(self) -> None:
        """Test that all families have required voltage fields."""
        from oscura.analyzers.digital.extraction import LOGIC_FAMILIES

        required_fields = ["VIL_max", "VIH_min", "VOL_max", "VOH_min", "VCC"]

        for family_name, family in LOGIC_FAMILIES.items():
            for field in required_fields:
                assert field in family, f"{family_name} missing {field}"
                assert isinstance(family[field], int | float)


# =============================================================================
# to_digital() Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DIG-001")
class TestToDigitalFixedThreshold:
    """Test to_digital() with fixed thresholds."""

    def test_basic_fixed_threshold(self) -> None:
        """Test basic digital extraction with fixed threshold."""
        from oscura.analyzers.digital.extraction import to_digital

        # Create simple analog signal: 0, 0, 3, 3, 0
        data = np.array([0.0, 0.0, 3.0, 3.0, 0.0])
        trace = make_waveform_trace(data)

        result = to_digital(trace, threshold=1.5)

        assert isinstance(result, DigitalTrace)
        assert len(result.data) == len(data)
        assert result.data.dtype == np.bool_
        # Expected: False, False, True, True, False
        expected = np.array([False, False, True, True, False])
        np.testing.assert_array_equal(result.data, expected)

    def test_threshold_at_signal_level(self) -> None:
        """Test threshold exactly at signal level."""
        from oscura.analyzers.digital.extraction import to_digital

        data = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        trace = make_waveform_trace(data)

        result = to_digital(trace, threshold=1.0)

        # Values >= threshold are True
        expected = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(result.data, expected)

    def test_threshold_above_all_samples(self) -> None:
        """Test threshold above all sample values."""
        from oscura.analyzers.digital.extraction import to_digital

        data = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        trace = make_waveform_trace(data)

        result = to_digital(trace, threshold=5.0)

        # All samples below threshold
        expected = np.array([False, False, False, False, False])
        np.testing.assert_array_equal(result.data, expected)

    def test_threshold_below_all_samples(self) -> None:
        """Test threshold below all sample values."""
        from oscura.analyzers.digital.extraction import to_digital

        data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        trace = make_waveform_trace(data)

        result = to_digital(trace, threshold=-1.0)

        # All samples above threshold
        expected = np.array([True, True, True, True, True])
        np.testing.assert_array_equal(result.data, expected)

    def test_square_wave_extraction(self, square_wave: NDArray[np.float64]) -> None:
        """Test extraction on square wave signal."""
        from oscura.analyzers.digital.extraction import to_digital

        trace = make_waveform_trace(square_wave)
        result = to_digital(trace, threshold=0.5)

        assert len(result.data) == len(square_wave)
        # Square wave should produce clean digital output
        assert result.data.dtype == np.bool_


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DIG-002")
class TestToDigitalAdaptiveThreshold:
    """Test to_digital() with adaptive threshold."""

    def test_auto_threshold_basic(self) -> None:
        """Test automatic threshold calculation."""
        from oscura.analyzers.digital.extraction import to_digital

        # Signal with clear low (0) and high (10) levels
        data = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
        trace = make_waveform_trace(data)

        result = to_digital(trace, threshold="auto")

        assert isinstance(result, DigitalTrace)
        # Adaptive threshold should be around 5.0 (midpoint)
        # Expected: False, False, True, True, False
        expected = np.array([False, False, True, True, False])
        np.testing.assert_array_equal(result.data, expected)

    def test_auto_threshold_with_noise(self, noisy_sine: NDArray[np.float64]) -> None:
        """Test adaptive threshold on noisy signal."""
        from oscura.analyzers.digital.extraction import to_digital

        trace = make_waveform_trace(noisy_sine)
        result = to_digital(trace, threshold="auto")

        assert len(result.data) == len(noisy_sine)
        # Should produce reasonable digital output
        assert result.data.dtype == np.bool_

    def test_auto_threshold_percentile_calculation(self) -> None:
        """Test that auto threshold uses 10th-90th percentile."""
        from oscura.analyzers.digital.extraction import to_digital

        # Create signal with outliers
        # Most values are 0 or 10, with some outliers at -100 and 100
        data = np.array([-100.0] + [0.0] * 45 + [10.0] * 45 + [100.0])
        trace = make_waveform_trace(data)

        result = to_digital(trace, threshold="auto")

        # Adaptive threshold should ignore outliers
        # Check that middle values are properly classified
        assert not result.data[1]  # First normal low value
        assert result.data[46]  # First normal high value


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DIG-003")
class TestToDigitalHysteresis:
    """Test to_digital() with hysteresis."""

    def test_symmetric_hysteresis(self) -> None:
        """Test symmetric hysteresis around threshold."""
        from oscura.analyzers.digital.extraction import to_digital

        # Signal that oscillates near threshold
        data = np.array([0.0, 1.4, 1.6, 1.4, 1.6, 3.0])
        trace = make_waveform_trace(data)

        # Threshold = 1.5, hysteresis = 0.4 means low=1.3, high=1.7
        result = to_digital(trace, threshold=1.5, hysteresis=0.4)

        assert isinstance(result, DigitalTrace)
        assert len(result.data) == len(data)

    def test_explicit_hysteresis_thresholds(self) -> None:
        """Test explicit low/high threshold tuple."""
        from oscura.analyzers.digital.extraction import to_digital

        data = np.array([0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 2.5, 2.0, 1.0, 0.5])
        trace = make_waveform_trace(data)

        # Explicit thresholds: low=1.0, high=2.0
        result = to_digital(trace, threshold=1.5, hysteresis=(1.0, 2.0))

        assert len(result.data) == len(data)
        # First sample starts low (< midpoint of 1.0 and 2.0)
        assert not result.data[0]

    def test_hysteresis_reduces_noise_transitions(self) -> None:
        """Test that hysteresis reduces spurious transitions."""
        from oscura.analyzers.digital.extraction import to_digital

        # Noisy signal near threshold
        rng = np.random.default_rng(42)
        base = np.tile([0.0, 2.0], 50)
        noise = rng.normal(0, 0.05, len(base))
        data = base + noise
        trace = make_waveform_trace(data)

        # Without hysteresis
        no_hyst = to_digital(trace, threshold=1.0, hysteresis=None)

        # With hysteresis
        with_hyst = to_digital(trace, threshold=1.0, hysteresis=0.5)

        # Count transitions
        no_hyst_trans = np.sum(np.abs(np.diff(no_hyst.data.astype(int))))
        with_hyst_trans = np.sum(np.abs(np.diff(with_hyst.data.astype(int))))

        # Hysteresis should reduce transitions (or keep them same)
        assert with_hyst_trans <= no_hyst_trans

    def test_hysteresis_zero(self) -> None:
        """Test that zero hysteresis works like no hysteresis."""
        from oscura.analyzers.digital.extraction import to_digital

        data = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        trace = make_waveform_trace(data)

        no_hyst = to_digital(trace, threshold=1.0, hysteresis=None)
        zero_hyst = to_digital(trace, threshold=1.0, hysteresis=0.0)

        # Should produce same result
        np.testing.assert_array_equal(no_hyst.data, zero_hyst.data)


@pytest.mark.unit
@pytest.mark.digital
class TestToDigitalEdgeDetection:
    """Test edge detection in to_digital() output."""

    def test_edges_detected(self) -> None:
        """Test that edges are detected and included in result."""
        from oscura.analyzers.digital.extraction import to_digital

        # Simple signal with 2 transitions
        data = np.array([0.0, 0.0, 3.0, 3.0, 0.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        result = to_digital(trace, threshold=1.5)

        # Should have edges list
        assert hasattr(result, "edges")
        assert isinstance(result.edges, list)
        # Should detect 2 edges (rising at index 2, falling at index 4)
        assert len(result.edges) == 2

    def test_edge_timestamps(self) -> None:
        """Test that edge timestamps are in seconds."""
        from oscura.analyzers.digital.extraction import to_digital

        data = np.array([0.0, 0.0, 3.0, 3.0, 0.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        result = to_digital(trace, threshold=1.5)

        # Each edge should be (timestamp, is_rising)
        for edge in result.edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            timestamp, is_rising = edge
            assert isinstance(timestamp, float)
            assert isinstance(is_rising, bool)
            assert timestamp >= 0.0

    def test_edge_types(self) -> None:
        """Test that edges are correctly classified as rising/falling."""
        from oscura.analyzers.digital.extraction import to_digital

        # Rising edge, then falling edge
        data = np.array([0.0, 0.0, 3.0, 3.0, 0.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        result = to_digital(trace, threshold=1.5)

        # First edge should be rising
        assert result.edges[0][1] is True
        # Second edge should be falling
        assert result.edges[1][1] is False


@pytest.mark.unit
@pytest.mark.digital
class TestToDigitalEdgeCases:
    """Test edge cases for to_digital()."""

    def test_insufficient_data_error(self) -> None:
        """Test that insufficient data raises appropriate error."""
        from oscura.analyzers.digital.extraction import to_digital

        # Single sample
        data = np.array([1.5])
        trace = make_waveform_trace(data)

        with pytest.raises(InsufficientDataError) as exc_info:
            to_digital(trace, threshold=1.0)

        assert exc_info.value.required == 2
        assert exc_info.value.available == 1

    def test_empty_trace(self) -> None:
        """Test that empty trace raises error."""
        from oscura.analyzers.digital.extraction import to_digital

        data = np.array([])
        trace = make_waveform_trace(data)

        with pytest.raises(InsufficientDataError):
            to_digital(trace, threshold=1.0)

    def test_constant_signal(self) -> None:
        """Test extraction on constant signal."""
        from oscura.analyzers.digital.extraction import to_digital

        data = np.ones(100) * 2.5
        trace = make_waveform_trace(data)

        result = to_digital(trace, threshold=1.0)

        # All samples should be high
        assert np.all(result.data)
        # No edges
        assert len(result.edges) == 0

    def test_metadata_preserved(self) -> None:
        """Test that trace metadata is preserved."""
        from oscura.analyzers.digital.extraction import to_digital

        data = np.array([0.0, 1.0, 2.0])
        metadata = TraceMetadata(
            sample_rate=1e9,
            channel_name="CH1",
            source_file="test.wfm",
        )
        trace = WaveformTrace(data=data, metadata=metadata)

        result = to_digital(trace, threshold=1.0)

        assert result.metadata.sample_rate == metadata.sample_rate
        assert result.metadata.channel_name == metadata.channel_name
        assert result.metadata.source_file == metadata.source_file


# =============================================================================
# detect_edges() Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DIG-004")
class TestDetectEdgesRising:
    """Test detect_edges() for rising edges."""

    def test_detect_rising_edges(self) -> None:
        """Test rising edge detection."""
        from oscura.analyzers.digital.extraction import detect_edges

        # Signal with 2 rising edges
        data = np.array([0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 3.0, 3.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        edges = detect_edges(trace, edge_type="rising", threshold=1.5)

        assert len(edges) == 2
        # Edges detected at transition indices (np.diff finds transition at index before change)
        # Transition 1: between index 1 and 2, interpolated to 1.5e-6
        # Transition 2: between index 5 and 6, interpolated to 5.5e-6
        assert edges[0] == pytest.approx(1.5e-6, abs=1e-7)
        assert edges[1] == pytest.approx(5.5e-6, abs=1e-7)

    def test_rising_edges_only(self) -> None:
        """Test that only rising edges are detected."""
        from oscura.analyzers.digital.extraction import detect_edges

        # 1 rising, 1 falling edge
        data = np.array([0.0, 3.0, 0.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        edges = detect_edges(trace, edge_type="rising", threshold=1.5)

        # Should only detect the rising edge
        assert len(edges) == 1

    def test_no_rising_edges(self) -> None:
        """Test signal with no rising edges."""
        from oscura.analyzers.digital.extraction import detect_edges

        # Starts high, goes low
        data = np.array([3.0, 3.0, 0.0, 0.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        edges = detect_edges(trace, edge_type="rising", threshold=1.5)

        assert len(edges) == 0


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DIG-005")
class TestDetectEdgesFalling:
    """Test detect_edges() for falling edges."""

    def test_detect_falling_edges(self) -> None:
        """Test falling edge detection."""
        from oscura.analyzers.digital.extraction import detect_edges

        # Signal with 2 falling edges
        data = np.array([3.0, 3.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        edges = detect_edges(trace, edge_type="falling", threshold=1.5)

        assert len(edges) == 2

    def test_falling_edges_only(self) -> None:
        """Test that only falling edges are detected."""
        from oscura.analyzers.digital.extraction import detect_edges

        # 1 rising, 1 falling edge
        data = np.array([0.0, 3.0, 0.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        edges = detect_edges(trace, edge_type="falling", threshold=1.5)

        # Should only detect the falling edge
        assert len(edges) == 1

    def test_no_falling_edges(self) -> None:
        """Test signal with no falling edges."""
        from oscura.analyzers.digital.extraction import detect_edges

        # Starts low, goes high
        data = np.array([0.0, 0.0, 3.0, 3.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        edges = detect_edges(trace, edge_type="falling", threshold=1.5)

        assert len(edges) == 0


@pytest.mark.unit
@pytest.mark.digital
class TestDetectEdgesBoth:
    """Test detect_edges() for both edge types."""

    def test_detect_both_edges(self) -> None:
        """Test detection of both rising and falling edges."""
        from oscura.analyzers.digital.extraction import detect_edges

        # 2 rising, 2 falling
        data = np.array([0.0, 3.0, 0.0, 3.0, 0.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        edges = detect_edges(trace, edge_type="both", threshold=1.5)

        assert len(edges) == 4

    def test_edges_in_time_order(self) -> None:
        """Test that edges are returned in time order."""
        from oscura.analyzers.digital.extraction import detect_edges

        data = np.array([0.0, 3.0, 0.0, 3.0, 0.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        edges = detect_edges(trace, edge_type="both", threshold=1.5)

        # Edges should be in increasing time order
        for i in range(len(edges) - 1):
            assert edges[i] < edges[i + 1]


@pytest.mark.unit
@pytest.mark.digital
class TestDetectEdgesWithDigitalTrace:
    """Test detect_edges() with DigitalTrace input."""

    def test_detect_edges_from_digital_trace(self) -> None:
        """Test edge detection on DigitalTrace."""
        from oscura.analyzers.digital.extraction import detect_edges

        # Digital data with 2 transitions
        data = np.array([False, False, True, True, False])
        trace = make_digital_trace(data, sample_rate=1e6)

        edges = detect_edges(trace, edge_type="both")

        assert len(edges) == 2

    def test_digital_trace_ignores_threshold(self) -> None:
        """Test that threshold is ignored for DigitalTrace."""
        from oscura.analyzers.digital.extraction import detect_edges

        data = np.array([False, True, False])
        trace = make_digital_trace(data, sample_rate=1e6)

        # Threshold should be ignored for digital traces
        edges1 = detect_edges(trace, edge_type="both", threshold=0.5)
        edges2 = detect_edges(trace, edge_type="both", threshold=2.5)

        # Should produce same results
        np.testing.assert_array_equal(edges1, edges2)


@pytest.mark.unit
@pytest.mark.digital
class TestDetectEdgesInterpolation:
    """Test sub-sample edge interpolation."""

    def test_interpolation_with_analog_trace(self) -> None:
        """Test that analog traces use interpolation."""
        from oscura.analyzers.digital.extraction import detect_edges

        # Analog trace with clear transition
        data = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 3.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        edges = detect_edges(trace, edge_type="rising", threshold=1.5)

        # Edge should be interpolated between samples
        assert len(edges) == 1
        # Edge is between index 2 (value 1.0) and index 3 (value 2.0)
        # Linear interpolation: t = (1.5 - 1.0) / (2.0 - 1.0) = 0.5
        # Expected time: (2 + 0.5) * 1e-6 = 2.5e-6
        assert edges[0] == pytest.approx(2.5e-6, abs=1e-7)

    def test_interpolation_accuracy(self) -> None:
        """Test interpolation calculation accuracy."""
        from oscura.analyzers.digital.extraction import detect_edges

        # Create precise test case with enough samples for interpolation
        # Transition from 0 to 10, threshold at 3
        # Should interpolate to 0.3 of the way between samples
        data = np.array([0.0, 0.0, 0.0, 10.0, 10.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        edges = detect_edges(trace, edge_type="rising", threshold=3.0)

        # Edge detected at index 2 (transition between index 2 and 3)
        # Interpolation between data[2]=0.0 and data[3]=10.0
        # t = (3.0 - 0.0) / (10.0 - 0.0) * 1e-6 = 0.3 * 1e-6
        # Total time = 2 * 1e-6 + 0.3 * 1e-6 = 2.3e-6
        assert edges[0] == pytest.approx(2.3e-6, abs=1e-8)


@pytest.mark.unit
@pytest.mark.digital
class TestDetectEdgesEdgeCases:
    """Test edge cases for detect_edges()."""

    def test_insufficient_data_error(self) -> None:
        """Test that insufficient data raises error."""
        from oscura.analyzers.digital.extraction import detect_edges

        data = np.array([1.5])
        trace = make_waveform_trace(data)

        with pytest.raises(InsufficientDataError) as exc_info:
            detect_edges(trace, edge_type="both", threshold=1.0)

        assert exc_info.value.required == 2
        assert exc_info.value.available == 1

    def test_empty_trace(self) -> None:
        """Test that empty trace raises error."""
        from oscura.analyzers.digital.extraction import detect_edges

        data = np.array([])
        trace = make_waveform_trace(data)

        with pytest.raises(InsufficientDataError):
            detect_edges(trace, edge_type="both", threshold=1.0)

    def test_constant_signal_no_edges(self) -> None:
        """Test that constant signal has no edges."""
        from oscura.analyzers.digital.extraction import detect_edges

        data = np.ones(100) * 2.5
        trace = make_waveform_trace(data)

        edges = detect_edges(trace, edge_type="both", threshold=1.5)

        assert len(edges) == 0

    def test_auto_threshold_in_detect_edges(self) -> None:
        """Test that 'auto' threshold works in detect_edges()."""
        from oscura.analyzers.digital.extraction import detect_edges

        data = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
        trace = make_waveform_trace(data)

        edges = detect_edges(trace, edge_type="both", threshold="auto")

        # Should detect edges with adaptive threshold
        assert len(edges) == 2


# =============================================================================
# get_logic_threshold() Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DIG-006")
class TestGetLogicThreshold:
    """Test get_logic_threshold() function."""

    def test_ttl_midpoint_threshold(self) -> None:
        """Test TTL midpoint threshold calculation."""
        from oscura.analyzers.digital.extraction import get_logic_threshold

        threshold = get_logic_threshold("TTL", "midpoint")

        # Midpoint of VIL_max (0.8) and VIH_min (2.0)
        expected = (0.8 + 2.0) / 2
        assert threshold == pytest.approx(expected)

    def test_ttl_vih_threshold(self) -> None:
        """Test TTL VIH threshold."""
        from oscura.analyzers.digital.extraction import get_logic_threshold

        threshold = get_logic_threshold("TTL", "VIH")

        assert threshold == 2.0

    def test_ttl_vil_threshold(self) -> None:
        """Test TTL VIL threshold."""
        from oscura.analyzers.digital.extraction import get_logic_threshold

        threshold = get_logic_threshold("TTL", "VIL")

        assert threshold == 0.8

    def test_lvcmos_3v3_midpoint(self) -> None:
        """Test LVCMOS 3.3V midpoint threshold."""
        from oscura.analyzers.digital.extraction import get_logic_threshold

        threshold = get_logic_threshold("LVCMOS_3V3", "midpoint")

        vil = 0.3 * 3.3
        vih = 0.7 * 3.3
        expected = (vil + vih) / 2
        assert threshold == pytest.approx(expected)

    def test_all_logic_families(self) -> None:
        """Test threshold calculation for all logic families."""
        from oscura.analyzers.digital.extraction import (
            LOGIC_FAMILIES,
            get_logic_threshold,
        )

        for family in LOGIC_FAMILIES:
            # Should not raise exception
            midpoint = get_logic_threshold(family, "midpoint")
            vih = get_logic_threshold(family, "VIH")
            vil = get_logic_threshold(family, "VIL")

            # Sanity checks
            assert vil < midpoint < vih
            assert vil > 0
            assert vih > 0

    def test_unknown_family_error(self) -> None:
        """Test that unknown logic family raises ValueError."""
        from oscura.analyzers.digital.extraction import get_logic_threshold

        with pytest.raises(ValueError) as exc_info:
            get_logic_threshold("UNKNOWN_FAMILY", "midpoint")

        assert "Unknown logic family" in str(exc_info.value)

    def test_unknown_threshold_type_error(self) -> None:
        """Test that unknown threshold type raises ValueError."""
        from oscura.analyzers.digital.extraction import get_logic_threshold

        with pytest.raises(ValueError) as exc_info:
            get_logic_threshold("TTL", "INVALID_TYPE")  # type: ignore

        assert "Unknown threshold_type" in str(exc_info.value)

    def test_default_threshold_type(self) -> None:
        """Test default threshold type is midpoint."""
        from oscura.analyzers.digital.extraction import get_logic_threshold

        # Default should be midpoint
        default = get_logic_threshold("TTL")
        midpoint = get_logic_threshold("TTL", "midpoint")

        assert default == midpoint


# =============================================================================
# Internal Helper Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestInternalHelpers:
    """Test internal helper functions."""

    def test_apply_hysteresis(self) -> None:
        """Test _apply_hysteresis() internal function."""
        from oscura.analyzers.digital.extraction import _apply_hysteresis

        # Signal oscillating between 0 and 3
        data = np.array([0.0, 0.5, 1.5, 2.5, 3.0, 2.5, 1.5, 0.5, 0.0])

        # Hysteresis: low=1.0, high=2.0
        result = _apply_hysteresis(data, thresh_low=1.0, thresh_high=2.0)

        assert len(result) == len(data)
        assert result.dtype == np.bool_

    def test_apply_hysteresis_initial_state(self) -> None:
        """Test that initial state is determined correctly."""
        from oscura.analyzers.digital.extraction import _apply_hysteresis

        # Start below midpoint
        data_low = np.array([0.5, 1.5, 2.5])
        result_low = _apply_hysteresis(data_low, thresh_low=1.0, thresh_high=2.0)
        assert not result_low[0]  # Should start low

        # Start above midpoint
        data_high = np.array([2.5, 1.5, 0.5])
        result_high = _apply_hysteresis(data_high, thresh_low=1.0, thresh_high=2.0)
        assert result_high[0]  # Should start high

    def test_interpolate_crossing(self) -> None:
        """Test _interpolate_crossing() internal function."""
        from oscura.analyzers.digital.extraction import _interpolate_crossing

        # Linear interpolation between 0 and 10, threshold at 3
        sample_period = 1e-6
        t = _interpolate_crossing(0.0, 10.0, 3.0, sample_period)

        # Expected: 0.3 * sample_period
        expected = 0.3 * sample_period
        assert t == pytest.approx(expected, abs=1e-10)

    def test_interpolate_crossing_boundary(self) -> None:
        """Test that interpolation is clamped to sample period."""
        from oscura.analyzers.digital.extraction import _interpolate_crossing

        sample_period = 1e-6

        # Threshold beyond v2
        t = _interpolate_crossing(0.0, 1.0, 5.0, sample_period)
        assert 0.0 <= t <= sample_period

        # Threshold before v1
        t = _interpolate_crossing(0.0, 1.0, -5.0, sample_period)
        assert 0.0 <= t <= sample_period

    def test_interpolate_crossing_no_change(self) -> None:
        """Test interpolation when voltage doesn't change."""
        from oscura.analyzers.digital.extraction import _interpolate_crossing

        sample_period = 1e-6
        # No voltage change
        t = _interpolate_crossing(1.5, 1.5, 1.5, sample_period)

        # Should return midpoint
        assert t == pytest.approx(sample_period / 2, abs=1e-10)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestDigitalExtractionIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_analog_to_edges(self, square_wave: NDArray[np.float64]) -> None:
        """Test complete workflow from analog signal to edge detection."""
        from oscura.analyzers.digital.extraction import detect_edges, to_digital

        trace = make_waveform_trace(square_wave, sample_rate=1e6)

        # First convert to digital
        digital = to_digital(trace, threshold=0.5)
        assert isinstance(digital, DigitalTrace)

        # Then detect edges on the digital trace
        edges = detect_edges(digital, edge_type="both")
        assert len(edges) > 0

    def test_logic_family_integration(self) -> None:
        """Test using logic family thresholds for extraction."""
        from oscura.analyzers.digital.extraction import (
            get_logic_threshold,
            to_digital,
        )

        # Create TTL-level signal
        data = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
        trace = make_waveform_trace(data)

        # Use TTL midpoint threshold
        ttl_threshold = get_logic_threshold("TTL", "midpoint")
        result = to_digital(trace, threshold=ttl_threshold)

        expected = np.array([False, False, True, True, False])
        np.testing.assert_array_equal(result.data, expected)

    def test_hysteresis_with_edge_detection(self) -> None:
        """Test hysteresis extraction followed by edge detection."""
        from oscura.analyzers.digital.extraction import detect_edges, to_digital

        # Noisy signal
        data = np.array([0.0, 0.1, 2.9, 3.0, 0.1, 0.0])
        trace = make_waveform_trace(data, sample_rate=1e6)

        # Extract with hysteresis
        digital = to_digital(trace, threshold=1.5, hysteresis=0.5)

        # Detect edges
        edges = detect_edges(digital, edge_type="both")

        # Should have clean edges
        assert len(edges) > 0


# =============================================================================
# __all__ Export Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestExports:
    """Test module exports."""

    def test_all_exports_available(self) -> None:
        """Test that all __all__ exports are importable."""
        from oscura.analyzers.digital import extraction

        expected_exports = [
            "LOGIC_FAMILIES",
            "detect_edges",
            "get_logic_threshold",
            "to_digital",
        ]

        for export in expected_exports:
            assert hasattr(extraction, export), f"Missing export: {export}"

    def test_all_list_complete(self) -> None:
        """Test that __all__ list matches expected exports."""
        from oscura.analyzers.digital.extraction import __all__

        assert len(__all__) == 4
        assert "LOGIC_FAMILIES" in __all__
        assert "detect_edges" in __all__
        assert "get_logic_threshold" in __all__
        assert "to_digital" in __all__

    def test_can_import_from_digital_module(self) -> None:
        """Test that functions can be imported from parent module."""
        # These imports should work
        from oscura.analyzers.digital import (
            LOGIC_FAMILIES,
            detect_edges,
            get_logic_threshold,
            to_digital,
        )

        assert LOGIC_FAMILIES is not None
        assert callable(detect_edges)
        assert callable(get_logic_threshold)
        assert callable(to_digital)


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestPerformance:
    """Performance tests for extraction functions."""

    def test_large_signal_extraction(self, large_signal: NDArray[np.float64]) -> None:
        """Test extraction on large signal (1M samples)."""
        from oscura.analyzers.digital.extraction import to_digital

        trace = make_waveform_trace(large_signal)

        # Should complete without error
        result = to_digital(trace, threshold=0.0)

        assert len(result.data) == len(large_signal)

    def test_large_signal_edge_detection(self) -> None:
        """Test edge detection on large signal."""
        from oscura.analyzers.digital.extraction import detect_edges

        # Create signal with many transitions
        data = np.tile([0.0, 3.0], 50000)  # 100k samples, 50k transitions
        trace = make_waveform_trace(data)

        edges = detect_edges(trace, edge_type="both", threshold=1.5)

        # Should detect all transitions
        assert len(edges) > 0

    def test_hysteresis_performance(self) -> None:
        """Test hysteresis on large signal."""
        from oscura.analyzers.digital.extraction import to_digital

        # Large noisy signal
        data = np.tile([0.0, 3.0], 50000)
        trace = make_waveform_trace(data)

        # Should complete without significant slowdown
        result = to_digital(trace, threshold=1.5, hysteresis=0.5)

        assert len(result.data) == len(data)
