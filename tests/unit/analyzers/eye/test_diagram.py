"""Comprehensive unit tests for eye diagram generation and analysis.

Tests for src/oscura/analyzers/eye/diagram.py

This test suite provides comprehensive coverage of the eye diagram module,
including:
- Eye diagram generation from waveforms
- Eye diagram generation from clock edges
- Auto-centering functionality
- EyeDiagram dataclass
- Edge cases and error conditions
- Input validation
- Histogram generation
- Numerical stability

Coverage targets:
- All public functions: generate_eye, generate_eye_from_edges, auto_center_eye_diagram
- All public classes: EyeDiagram
- Edge detection and triggering
- Sample rate handling
- Error conditions
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from oscura.analyzers.eye.diagram import (
    EyeDiagram,
    auto_center_eye_diagram,
    generate_eye,
    generate_eye_from_edges,
)
from oscura.core.exceptions import AnalysisError, InsufficientDataError
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.eye]


# =============================================================================
# Test Data Generators
# =============================================================================


def create_serial_waveform(
    n_bits: int,
    bit_rate: float = 1e9,
    sample_rate: float = 10e9,
    amplitude: float = 1.0,
    bit_pattern: str | None = None,
    transition_time: float = 0.1,
    noise_level: float = 0.0,
    seed: int = 42,
) -> WaveformTrace:
    """Generate a synthetic serial data waveform.

    Args:
        n_bits: Number of bits to generate.
        bit_rate: Bit rate in bits/second.
        sample_rate: Sample rate in Hz.
        amplitude: Signal amplitude in volts.
        bit_pattern: Bit pattern ('prbs', 'alternating', 'all_ones', 'all_zeros').
        transition_time: Rise/fall time as fraction of UI.
        noise_level: RMS noise level as fraction of amplitude.
        seed: Random seed for reproducibility.

    Returns:
        WaveformTrace containing serial data signal.
    """
    rng = np.random.default_rng(seed)
    unit_interval = 1.0 / bit_rate
    samples_per_bit = int(sample_rate * unit_interval)
    n_samples = n_bits * samples_per_bit

    # Generate bit pattern
    if bit_pattern == "alternating":
        bits = np.tile([0, 1], (n_bits + 1) // 2)[:n_bits]
    elif bit_pattern == "all_ones":
        bits = np.ones(n_bits, dtype=int)
    elif bit_pattern == "all_zeros":
        bits = np.zeros(n_bits, dtype=int)
    else:  # prbs or default
        bits = rng.integers(0, 2, n_bits)

    # Generate waveform with transitions
    data = np.zeros(n_samples)
    transition_samples = int(samples_per_bit * transition_time)

    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        target = amplitude if bit else 0.0

        if transition_samples > 0 and i > 0:
            # Add transition from previous bit
            prev_target = amplitude if bits[i - 1] else 0.0
            if prev_target != target:
                transition_end = min(start + transition_samples, end)
                transition_len = transition_end - start
                data[start:transition_end] = np.linspace(prev_target, target, transition_len)
                data[transition_end:end] = target
            else:
                data[start:end] = target
        else:
            data[start:end] = target

    # Add noise if requested
    if noise_level > 0:
        noise = rng.normal(0, noise_level * amplitude, n_samples)
        data += noise

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


def create_clock_edges(
    n_edges: int,
    period: float = 1e-9,
    jitter_rms: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic clock edge timestamps.

    Args:
        n_edges: Number of edges to generate.
        period: Clock period in seconds.
        jitter_rms: RMS jitter in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Array of edge timestamps in seconds.
    """
    rng = np.random.default_rng(seed)
    edges = np.arange(n_edges) * period

    if jitter_rms > 0:
        jitter = rng.normal(0, jitter_rms, n_edges)
        edges += jitter

    return edges


# =============================================================================
# Tests for EyeDiagram Dataclass
# =============================================================================


@pytest.mark.unit
@pytest.mark.eye
class TestEyeDiagram:
    """Test EyeDiagram dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating an EyeDiagram instance."""
        data = np.random.randn(100, 200)
        time_axis = np.linspace(0, 2, 200)

        eye = EyeDiagram(
            data=data,
            time_axis=time_axis,
            unit_interval=1e-9,
            samples_per_ui=100,
            n_traces=100,
            sample_rate=100e9,
        )

        assert eye.data.shape == (100, 200)
        assert len(eye.time_axis) == 200
        assert eye.unit_interval == 1e-9
        assert eye.samples_per_ui == 100
        assert eye.n_traces == 100
        assert eye.sample_rate == 100e9
        assert eye.histogram is None
        assert eye.voltage_bins is None
        assert eye.time_bins is None

    def test_dataclass_with_histogram(self) -> None:
        """Test EyeDiagram with histogram data."""
        data = np.random.randn(50, 100)
        time_axis = np.linspace(0, 1, 100)
        histogram = np.random.rand(100, 50)
        voltage_bins = np.linspace(-3, 3, 101)
        time_bins = np.linspace(0, 1, 51)

        eye = EyeDiagram(
            data=data,
            time_axis=time_axis,
            unit_interval=1e-9,
            samples_per_ui=100,
            n_traces=50,
            sample_rate=100e9,
            histogram=histogram,
            voltage_bins=voltage_bins,
            time_bins=time_bins,
        )

        assert eye.histogram is not None
        assert eye.histogram.shape == (100, 50)
        assert eye.voltage_bins is not None
        assert len(eye.voltage_bins) == 101
        assert eye.time_bins is not None
        assert len(eye.time_bins) == 51

    def test_dataclass_attributes_immutable(self) -> None:
        """Test that dataclass attributes can be assigned."""
        data = np.random.randn(10, 20)
        time_axis = np.linspace(0, 2, 20)

        eye = EyeDiagram(
            data=data,
            time_axis=time_axis,
            unit_interval=1e-9,
            samples_per_ui=10,
            n_traces=10,
            sample_rate=10e9,
        )

        # Should be able to access all attributes
        assert isinstance(eye.data, np.ndarray)
        assert isinstance(eye.time_axis, np.ndarray)
        assert isinstance(eye.unit_interval, float)
        assert isinstance(eye.samples_per_ui, int)
        assert isinstance(eye.n_traces, int)
        assert isinstance(eye.sample_rate, float)


# =============================================================================
# Tests for generate_eye Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.eye
class TestGenerateEye:
    """Test generate_eye function for basic eye diagram generation."""

    def test_generate_eye_basic(self) -> None:
        """Test basic eye diagram generation with alternating pattern."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
            bit_pattern="alternating",
        )

        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        assert isinstance(eye, EyeDiagram)
        assert eye.n_traces > 0
        assert eye.samples_per_ui > 0
        assert eye.unit_interval == 1e-9
        assert eye.sample_rate == 10e9
        assert len(eye.time_axis) == eye.data.shape[1]

    def test_generate_eye_prbs_pattern(self) -> None:
        """Test eye diagram generation with PRBS pattern."""
        trace = create_serial_waveform(
            n_bits=200,
            bit_rate=1e9,
            sample_rate=10e9,
            bit_pattern="prbs",
        )

        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        assert eye.n_traces > 0
        assert eye.data.shape == (eye.n_traces, len(eye.time_axis))

    def test_generate_eye_single_ui(self) -> None:
        """Test eye diagram generation with single UI display."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        eye = generate_eye(trace, unit_interval=1e-9, n_ui=1)

        assert eye.n_traces > 0
        assert np.max(eye.time_axis) <= 1.0  # Single UI

    def test_generate_eye_two_ui(self) -> None:
        """Test eye diagram generation with two UI display."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        assert eye.n_traces > 0
        assert np.max(eye.time_axis) <= 2.0  # Two UI

    def test_generate_eye_rising_edge_trigger(self) -> None:
        """Test eye diagram with rising edge trigger."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
            bit_pattern="alternating",
        )

        eye = generate_eye(
            trace,
            unit_interval=1e-9,
            n_ui=2,
            trigger_edge="rising",
            trigger_level=0.5,
        )

        assert eye.n_traces > 0

    def test_generate_eye_falling_edge_trigger(self) -> None:
        """Test eye diagram with falling edge trigger."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
            bit_pattern="alternating",
        )

        eye = generate_eye(
            trace,
            unit_interval=1e-9,
            n_ui=2,
            trigger_edge="falling",
            trigger_level=0.5,
        )

        assert eye.n_traces > 0

    def test_generate_eye_custom_trigger_level(self) -> None:
        """Test eye diagram with custom trigger level."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        eye_low = generate_eye(trace, unit_interval=1e-9, trigger_level=0.2)
        eye_mid = generate_eye(trace, unit_interval=1e-9, trigger_level=0.5)
        eye_high = generate_eye(trace, unit_interval=1e-9, trigger_level=0.8)

        # All should produce valid eye diagrams
        assert eye_low.n_traces > 0
        assert eye_mid.n_traces > 0
        assert eye_high.n_traces > 0

    def test_generate_eye_max_traces_limit(self) -> None:
        """Test limiting maximum number of traces."""
        trace = create_serial_waveform(
            n_bits=200,
            bit_rate=1e9,
            sample_rate=10e9,
            bit_pattern="alternating",
        )

        eye = generate_eye(trace, unit_interval=1e-9, max_traces=10)

        assert eye.n_traces <= 10

    def test_generate_eye_with_histogram(self) -> None:
        """Test eye diagram generation with histogram."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        eye = generate_eye(
            trace,
            unit_interval=1e-9,
            generate_histogram=True,
            histogram_bins=(50, 50),
        )

        assert eye.histogram is not None
        assert eye.histogram.shape == (50, 50)
        assert eye.voltage_bins is not None
        assert len(eye.voltage_bins) == 51  # N bins + 1 edges
        assert eye.time_bins is not None
        assert len(eye.time_bins) == 51

    def test_generate_eye_without_histogram(self) -> None:
        """Test eye diagram generation without histogram."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        eye = generate_eye(trace, unit_interval=1e-9, generate_histogram=False)

        assert eye.histogram is None
        assert eye.voltage_bins is None
        assert eye.time_bins is None

    def test_generate_eye_custom_histogram_bins(self) -> None:
        """Test eye diagram with custom histogram bin counts."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        eye = generate_eye(
            trace,
            unit_interval=1e-9,
            generate_histogram=True,
            histogram_bins=(80, 120),
        )

        assert eye.histogram is not None
        assert eye.histogram.shape == (80, 120)

    def test_generate_eye_time_axis_range(self) -> None:
        """Test that time axis has correct range."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        assert eye.time_axis[0] == pytest.approx(0.0, abs=1e-10)
        assert eye.time_axis[-1] < 2.0  # Less than 2 UI (endpoint=False)

    def test_generate_eye_data_shape_consistency(self) -> None:
        """Test that eye data shape is consistent with metadata."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        assert eye.data.shape[0] == eye.n_traces
        assert eye.data.shape[1] == len(eye.time_axis)


# =============================================================================
# Tests for Error Conditions in generate_eye
# =============================================================================


@pytest.mark.unit
@pytest.mark.eye
class TestGenerateEyeErrors:
    """Test error conditions in generate_eye function."""

    def test_generate_eye_unit_interval_too_short(self) -> None:
        """Test that too short unit interval raises AnalysisError."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        # UI so short that samples_per_ui < 4
        with pytest.raises(AnalysisError) as exc_info:
            generate_eye(trace, unit_interval=1e-12)

        assert "Unit interval too short" in str(exc_info.value)
        assert "samples/UI" in str(exc_info.value)

    def test_generate_eye_insufficient_data(self) -> None:
        """Test that insufficient data raises InsufficientDataError."""
        # Create trace with very few samples
        metadata = TraceMetadata(sample_rate=10e9)
        trace = WaveformTrace(data=np.zeros(10), metadata=metadata)

        with pytest.raises(InsufficientDataError) as exc_info:
            generate_eye(trace, unit_interval=1e-9)

        assert exc_info.value.analysis_type == "eye_diagram_generation"

    def test_generate_eye_no_trigger_events(self) -> None:
        """Test that signal with no trigger events raises error."""
        # All zeros - no trigger crossings
        metadata = TraceMetadata(sample_rate=10e9)
        trace = WaveformTrace(data=np.zeros(1000), metadata=metadata)

        with pytest.raises(InsufficientDataError) as exc_info:
            generate_eye(trace, unit_interval=1e-9)

        assert "Not enough trigger events" in str(exc_info.value)

    def test_generate_eye_insufficient_trigger_events(self) -> None:
        """Test that single trigger event raises error."""
        # Create signal with single transition
        metadata = TraceMetadata(sample_rate=10e9)
        data = np.concatenate([np.zeros(500), np.ones(500)])
        trace = WaveformTrace(data=data, metadata=metadata)

        with pytest.raises(InsufficientDataError) as exc_info:
            generate_eye(trace, unit_interval=1e-9, trigger_edge="rising")

        assert "Not enough trigger events" in str(exc_info.value)

    def test_generate_eye_no_complete_traces(self) -> None:
        """Test that signal where no complete traces can be extracted raises error."""
        # Very short signal with trigger at end
        metadata = TraceMetadata(sample_rate=10e9)
        data = np.concatenate([np.zeros(50), [0.5], np.ones(10)])
        trace = WaveformTrace(data=data, metadata=metadata)

        with pytest.raises(InsufficientDataError) as exc_info:
            generate_eye(trace, unit_interval=1e-9)

        # The actual error could be about trigger events or complete traces
        error_msg = str(exc_info.value)
        assert (
            "Not enough trigger events" in error_msg
            or "Could not extract any complete eye traces" in error_msg
        )


# =============================================================================
# Tests for generate_eye_from_edges Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.eye
class TestGenerateEyeFromEdges:
    """Test generate_eye_from_edges function."""

    def test_generate_eye_from_edges_basic(self) -> None:
        """Test basic eye generation from edge timestamps."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
            bit_pattern="alternating",
        )
        edges = create_clock_edges(n_edges=50, period=1e-9)

        eye = generate_eye_from_edges(
            trace,
            edges,
            n_ui=2,
            samples_per_ui=100,
        )

        assert isinstance(eye, EyeDiagram)
        assert eye.n_traces > 0
        assert eye.samples_per_ui == 100
        assert eye.unit_interval == pytest.approx(1e-9, rel=0.1)

    def test_generate_eye_from_edges_with_jitter(self) -> None:
        """Test eye generation from edges with jitter."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )
        edges = create_clock_edges(n_edges=50, period=1e-9, jitter_rms=10e-12)

        eye = generate_eye_from_edges(trace, edges, n_ui=2, samples_per_ui=100)

        assert eye.n_traces > 0
        # Unit interval should be calculated from median of edge differences
        assert eye.unit_interval == pytest.approx(1e-9, rel=0.1)

    def test_generate_eye_from_edges_single_ui(self) -> None:
        """Test eye generation from edges with single UI."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
        )
        edges = create_clock_edges(n_edges=30, period=1e-9)

        eye = generate_eye_from_edges(trace, edges, n_ui=1, samples_per_ui=50)

        assert eye.n_traces > 0
        assert np.max(eye.time_axis) <= 1.1  # Single UI (with small tolerance)

    def test_generate_eye_from_edges_max_traces(self) -> None:
        """Test limiting maximum traces from edges."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )
        edges = create_clock_edges(n_edges=100, period=1e-9)

        eye = generate_eye_from_edges(trace, edges, n_ui=2, samples_per_ui=100, max_traces=10)

        assert eye.n_traces <= 10

    def test_generate_eye_from_edges_custom_samples_per_ui(self) -> None:
        """Test eye generation with custom samples per UI."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
        )
        edges = create_clock_edges(n_edges=30, period=1e-9)

        eye = generate_eye_from_edges(trace, edges, n_ui=2, samples_per_ui=200)

        assert eye.samples_per_ui == 200
        assert eye.data.shape[1] == 400  # 2 UI * 200 samples/UI

    def test_generate_eye_from_edges_no_histogram(self) -> None:
        """Test that edges-based eye doesn't generate histogram."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
        )
        edges = create_clock_edges(n_edges=30, period=1e-9)

        eye = generate_eye_from_edges(trace, edges, n_ui=2, samples_per_ui=100)

        # Histogram not generated in generate_eye_from_edges
        assert eye.histogram is None
        assert eye.voltage_bins is None
        assert eye.time_bins is None


# =============================================================================
# Tests for Error Conditions in generate_eye_from_edges
# =============================================================================


@pytest.mark.unit
@pytest.mark.eye
class TestGenerateEyeFromEdgesErrors:
    """Test error conditions in generate_eye_from_edges."""

    def test_generate_eye_from_edges_insufficient_edges(self) -> None:
        """Test that fewer than 3 edges raises error."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
        )
        edges = np.array([0.0, 1e-9])  # Only 2 edges

        with pytest.raises(InsufficientDataError) as exc_info:
            generate_eye_from_edges(trace, edges, n_ui=2, samples_per_ui=100)

        assert exc_info.value.required == 3
        assert exc_info.value.available == 2

    def test_generate_eye_from_edges_no_valid_windows(self) -> None:
        """Test that edges outside trace range raises error."""
        metadata = TraceMetadata(sample_rate=10e9)
        trace = WaveformTrace(data=np.zeros(1000), metadata=metadata)
        # Edges far beyond trace duration
        edges = np.array([1.0, 2.0, 3.0])  # 1-3 seconds, trace is ~100ns

        with pytest.raises(InsufficientDataError) as exc_info:
            generate_eye_from_edges(trace, edges, n_ui=2, samples_per_ui=100)

        assert "Could not extract any eye traces" in str(exc_info.value)

    def test_generate_eye_from_edges_empty_array(self) -> None:
        """Test that empty edge array raises error."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
        )
        edges = np.array([])

        with pytest.raises(InsufficientDataError):
            generate_eye_from_edges(trace, edges, n_ui=2, samples_per_ui=100)


# =============================================================================
# Tests for auto_center_eye_diagram Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.eye
class TestAutoCenterEyeDiagram:
    """Test auto_center_eye_diagram function."""

    def test_auto_center_basic(self) -> None:
        """Test basic auto-centering of eye diagram."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
            bit_pattern="alternating",
        )
        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        centered = auto_center_eye_diagram(eye)

        assert isinstance(centered, EyeDiagram)
        assert centered.data.shape == eye.data.shape
        assert centered.n_traces == eye.n_traces
        assert centered.samples_per_ui == eye.samples_per_ui

    def test_auto_center_trigger_fraction(self) -> None:
        """Test auto-centering with different trigger fractions."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
            bit_pattern="alternating",
        )
        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        centered_20 = auto_center_eye_diagram(eye, trigger_fraction=0.2)
        centered_50 = auto_center_eye_diagram(eye, trigger_fraction=0.5)
        centered_80 = auto_center_eye_diagram(eye, trigger_fraction=0.8)

        # All should produce valid centered diagrams
        assert centered_20.n_traces == eye.n_traces
        assert centered_50.n_traces == eye.n_traces
        assert centered_80.n_traces == eye.n_traces

    def test_auto_center_symmetric_range(self) -> None:
        """Test auto-centering with symmetric amplitude range."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )
        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        centered = auto_center_eye_diagram(eye, symmetric_range=True)

        # Mean should be close to zero after centering
        assert np.abs(np.mean(centered.data)) < np.std(centered.data) * 0.1

    def test_auto_center_no_symmetric_range(self) -> None:
        """Test auto-centering without symmetric range."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )
        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        centered = auto_center_eye_diagram(eye, symmetric_range=False)

        assert centered.n_traces == eye.n_traces

    def test_auto_center_histogram_invalidated(self) -> None:
        """Test that histogram is invalidated after centering."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )
        eye = generate_eye(trace, unit_interval=1e-9, generate_histogram=True)

        assert eye.histogram is not None

        centered = auto_center_eye_diagram(eye)

        # Histogram should be invalidated
        assert centered.histogram is None
        assert centered.voltage_bins is None
        assert centered.time_bins is None

    def test_auto_center_no_crossings_warning(self) -> None:
        """Test that no crossings produces warning and returns original."""
        # Create eye with all high values (no crossings)
        metadata = TraceMetadata(sample_rate=10e9)
        trace = WaveformTrace(data=np.ones(1000), metadata=metadata)

        # Manually create eye diagram (skip normal generation)
        eye = EyeDiagram(
            data=np.ones((10, 100)),
            time_axis=np.linspace(0, 2, 100),
            unit_interval=1e-9,
            samples_per_ui=50,
            n_traces=10,
            sample_rate=10e9,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            centered = auto_center_eye_diagram(eye)

            # Should have warning
            assert len(w) == 1
            assert "cannot auto-center" in str(w[0].message).lower()

        # Should return original eye
        assert np.array_equal(centered.data, eye.data)

    def test_auto_center_invalid_trigger_fraction(self) -> None:
        """Test that invalid trigger fraction raises ValueError."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
        )
        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        with pytest.raises(ValueError) as exc_info:
            auto_center_eye_diagram(eye, trigger_fraction=1.5)

        assert "trigger_fraction must be in [0, 1]" in str(exc_info.value)

        with pytest.raises(ValueError):
            auto_center_eye_diagram(eye, trigger_fraction=-0.1)

    def test_auto_center_preserves_metadata(self) -> None:
        """Test that centering preserves eye metadata."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )
        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        centered = auto_center_eye_diagram(eye)

        assert centered.unit_interval == eye.unit_interval
        assert centered.samples_per_ui == eye.samples_per_ui
        assert centered.n_traces == eye.n_traces
        assert centered.sample_rate == eye.sample_rate
        assert np.array_equal(centered.time_axis, eye.time_axis)


# =============================================================================
# Edge Cases and Numerical Stability
# =============================================================================


@pytest.mark.unit
@pytest.mark.eye
class TestEyeDiagramEdgeCases:
    """Test edge cases and numerical stability."""

    def test_generate_eye_very_high_sample_rate(self) -> None:
        """Test with very high sample rate (many samples per UI)."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=100e9,  # 100 samples per UI
        )

        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        assert eye.n_traces > 0
        assert eye.samples_per_ui >= 100

    def test_generate_eye_low_sample_rate(self) -> None:
        """Test with low sample rate (few samples per UI)."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=100e6,
            sample_rate=500e6,  # 5 samples per UI
        )

        eye = generate_eye(trace, unit_interval=10e-9, n_ui=2)

        assert eye.n_traces > 0
        assert eye.samples_per_ui >= 4  # Minimum requirement

    def test_generate_eye_with_noise(self) -> None:
        """Test eye diagram generation with noisy signal."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
            noise_level=0.1,
        )

        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        assert eye.n_traces > 0

    def test_generate_eye_slow_transitions(self) -> None:
        """Test with slow rise/fall times."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
            transition_time=0.4,  # 40% of UI
        )

        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        assert eye.n_traces > 0

    def test_generate_eye_fast_transitions(self) -> None:
        """Test with fast rise/fall times."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=1e9,
            sample_rate=10e9,
            transition_time=0.05,  # 5% of UI
        )

        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        assert eye.n_traces > 0

    def test_generate_eye_different_amplitudes(self) -> None:
        """Test with different signal amplitudes."""
        for amplitude in [0.5, 1.0, 3.3, 5.0]:
            trace = create_serial_waveform(
                n_bits=50,
                bit_rate=1e9,
                sample_rate=10e9,
                amplitude=amplitude,
            )

            eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

            assert eye.n_traces > 0

    def test_generate_eye_from_edges_non_uniform_periods(self) -> None:
        """Test edge-based generation with non-uniform periods."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        # Create edges with varying periods
        edges = np.cumsum(np.array([1e-9, 1.01e-9, 0.99e-9, 1e-9, 1.02e-9] * 10))

        eye = generate_eye_from_edges(trace, edges, n_ui=2, samples_per_ui=100)

        assert eye.n_traces > 0
        # Unit interval should be median period
        assert eye.unit_interval == pytest.approx(1e-9, rel=0.05)

    def test_auto_center_with_offset_signal(self) -> None:
        """Test auto-centering with DC offset signal."""
        # Create signal with DC offset
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )
        trace.data += 2.0  # Add DC offset

        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)
        centered = auto_center_eye_diagram(eye, symmetric_range=True)

        # Mean should be near zero after centering
        assert np.abs(np.mean(centered.data)) < 0.5

    def test_generate_eye_minimal_valid_signal(self) -> None:
        """Test with minimal valid signal (just enough data)."""
        trace = create_serial_waveform(
            n_bits=5,
            bit_rate=1e9,
            sample_rate=10e9,
            bit_pattern="alternating",
        )

        # Should work with minimal data
        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        assert eye.n_traces >= 1


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.eye
class TestEyeDiagramIntegration:
    """Integration tests combining multiple functions."""

    def test_generate_and_center_workflow(self) -> None:
        """Test complete workflow: generate eye then auto-center."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
            bit_pattern="prbs",
        )

        # Generate eye
        eye = generate_eye(
            trace,
            unit_interval=1e-9,
            n_ui=2,
            generate_histogram=True,
        )

        assert eye.n_traces > 0
        assert eye.histogram is not None

        # Center eye
        centered = auto_center_eye_diagram(eye, trigger_fraction=0.5)

        assert centered.n_traces == eye.n_traces
        assert centered.histogram is None  # Invalidated

    def test_edges_to_eye_workflow(self) -> None:
        """Test workflow from edge recovery to eye diagram."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        # Simulate recovered edges
        edges = create_clock_edges(n_edges=50, period=1e-9, jitter_rms=5e-12)

        # Generate eye from edges
        eye = generate_eye_from_edges(trace, edges, n_ui=2, samples_per_ui=100)

        assert eye.n_traces > 0

        # Center the eye
        centered = auto_center_eye_diagram(eye)

        assert centered.n_traces == eye.n_traces

    def test_comparison_threshold_vs_edges(self) -> None:
        """Test that threshold and edge-based methods produce similar results."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
            bit_pattern="alternating",
        )

        # Threshold-based
        eye_threshold = generate_eye(trace, unit_interval=1e-9, n_ui=2)

        # Edge-based with known edges
        edges = create_clock_edges(n_edges=50, period=1e-9)
        eye_edges = generate_eye_from_edges(
            trace, edges, n_ui=2, samples_per_ui=eye_threshold.samples_per_ui
        )

        # Both should produce similar trace counts (within reason)
        assert abs(eye_threshold.n_traces - eye_edges.n_traces) < 20

    def test_multiple_trigger_levels(self) -> None:
        """Test eye generation with multiple trigger levels."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        results = []
        for level in [0.2, 0.5, 0.8]:
            eye = generate_eye(trace, unit_interval=1e-9, trigger_level=level)
            results.append(eye)

        # All should produce valid eyes
        for eye in results:
            assert eye.n_traces > 0


# =============================================================================
# Performance and Stress Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.eye
class TestEyeDiagramPerformance:
    """Performance and stress tests."""

    def test_large_number_of_bits(self) -> None:
        """Test with large number of bits."""
        trace = create_serial_waveform(
            n_bits=1000,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        eye = generate_eye(trace, unit_interval=1e-9, n_ui=2, max_traces=100)

        assert eye.n_traces <= 100

    def test_high_oversampling(self) -> None:
        """Test with high oversampling ratio."""
        trace = create_serial_waveform(
            n_bits=50,
            bit_rate=100e6,
            sample_rate=10e9,  # 100x oversampling
        )

        eye = generate_eye(trace, unit_interval=10e-9, n_ui=2)

        assert eye.n_traces > 0
        assert eye.samples_per_ui >= 100

    def test_many_edges_from_edges(self) -> None:
        """Test edge-based generation with many edges."""
        trace = create_serial_waveform(
            n_bits=500,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        edges = create_clock_edges(n_edges=200, period=1e-9)

        eye = generate_eye_from_edges(trace, edges, n_ui=2, samples_per_ui=100, max_traces=50)

        assert eye.n_traces <= 50


# =============================================================================
# Histogram Validation Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.eye
class TestEyeHistogram:
    """Test histogram generation and properties."""

    def test_histogram_shape(self) -> None:
        """Test histogram has correct shape."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        eye = generate_eye(
            trace,
            unit_interval=1e-9,
            generate_histogram=True,
            histogram_bins=(60, 80),
        )

        assert eye.histogram is not None
        assert eye.histogram.shape == (60, 80)

    def test_histogram_bins_are_edges(self) -> None:
        """Test that bin arrays have correct length (N+1 for N bins)."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        eye = generate_eye(
            trace,
            unit_interval=1e-9,
            generate_histogram=True,
            histogram_bins=(50, 50),
        )

        assert len(eye.voltage_bins) == 51
        assert len(eye.time_bins) == 51

    def test_histogram_coverage(self) -> None:
        """Test that histogram covers data range."""
        trace = create_serial_waveform(
            n_bits=100,
            bit_rate=1e9,
            sample_rate=10e9,
        )

        eye = generate_eye(
            trace,
            unit_interval=1e-9,
            generate_histogram=True,
        )

        data_min = np.min(eye.data)
        data_max = np.max(eye.data)

        assert eye.voltage_bins[0] <= data_min
        assert eye.voltage_bins[-1] >= data_max
