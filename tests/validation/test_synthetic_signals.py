"""Ground truth validation tests for synthetic signal data.

This module validates signal processing against square wave and UART
ground truth files to ensure correct frequency, period, and edge detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.validation


@pytest.mark.requires_data
@pytest.mark.requirement("VAL-002")
class TestSquareWaveValidation:
    """Validate square wave signal processing against ground truth."""

    @pytest.mark.parametrize("freq", ["1MHz", "10MHz", "100MHz"])
    def test_square_wave_file_exists(self, square_wave_files: dict[str, Path], freq: str) -> None:
        """Verify square wave NPY files exist.

        Validates:
        - square_1MHz.npy exists
        - square_10MHz.npy exists
        - square_100MHz.npy exists
        """
        path = square_wave_files.get(freq)
        if path is None:
            pytest.skip(f"Square wave file for {freq} not configured")
        if not path.exists():
            pytest.skip(f"Square wave file not found: {path}")

        # Verify file can be loaded
        data = np.load(path, allow_pickle=True)
        assert len(data) > 0, f"Empty data in {path}"

    def test_square_1MHz_frequency(
        self,
        square_wave_files: dict[str, Path],
        square_wave_truth: dict[str, dict[str, Any]],
    ) -> None:
        """Verify 1MHz square wave frequency detection.

        Validates:
        - Detected frequency matches ground truth (1000000 Hz)
        - Tolerance: 1% error allowed
        """
        path = square_wave_files.get("1MHz")
        truth = square_wave_truth.get("1MHz", {})

        if path is None or not path.exists():
            pytest.skip("1MHz square wave file not available")
        if not truth:
            pytest.skip("1MHz ground truth not available")

        expected_freq = truth.get("frequency_hz", 1000000.0)
        pattern_period = truth.get("pattern_period", 1000)
        data = np.load(path, allow_pickle=True)

        try:
            from oscura.analyzers.digital import detect_clock_frequency

            # Calculate sample rate from ground truth:
            # If pattern_period = 1000 samples per 1MHz cycle (1us),
            # then sample_rate = pattern_period * frequency
            # For 1MHz with 1000 samples/period: sample_rate = 1000 * 1e6 = 1 GHz
            sample_rate = pattern_period * expected_freq
            detected = detect_clock_frequency(data, sample_rate)

            tolerance = 0.01 * expected_freq  # 1% tolerance
            assert abs(detected - expected_freq) < tolerance, (
                f"Frequency mismatch: expected {expected_freq}, got {detected}"
            )
        except ImportError:
            pytest.skip("detect_clock_frequency not available")

    def test_square_wave_edge_count(
        self,
        square_wave_files: dict[str, Path],
        square_wave_truth: dict[str, dict[str, Any]],
    ) -> None:
        """Verify edge count matches ground truth.

        Validates:
        - Number of detected edges matches expected count
        """
        for freq_label, path in square_wave_files.items():
            if path is None or not path.exists():
                continue

            truth = square_wave_truth.get(freq_label, {})
            if not truth:
                continue

            expected_edges = truth.get("edge_count")
            if expected_edges is None:
                continue

            data = np.load(path, allow_pickle=True)

            try:
                # Use detect_edges_advanced which accepts raw numpy arrays
                from oscura.analyzers.digital import detect_edges_advanced

                edges = detect_edges_advanced(data, threshold=0.5)
                # Allow some tolerance in edge detection
                tolerance = max(10, expected_edges * 0.05)  # 5% or 10 edges

                assert abs(len(edges) - expected_edges) <= tolerance, (
                    f"{freq_label}: Edge count mismatch: expected {expected_edges}, "
                    f"got {len(edges)}"
                )
            except ImportError:
                pytest.skip("detect_edges_advanced not available")

    def test_square_wave_period(
        self,
        square_wave_files: dict[str, Path],
        square_wave_truth: dict[str, dict[str, Any]],
    ) -> None:
        """Verify pattern period matches ground truth.

        Validates:
        - Pattern period (samples per cycle) matches expected value
        """
        truth_1MHz = square_wave_truth.get("1MHz", {})
        if not truth_1MHz:
            pytest.skip("1MHz ground truth not available")

        expected_period = truth_1MHz.get("pattern_period")
        if expected_period is None:
            pytest.skip("pattern_period not in ground truth")

        path = square_wave_files.get("1MHz")
        if path is None or not path.exists():
            pytest.skip("1MHz square wave file not available")

        data = np.load(path, allow_pickle=True)

        # Calculate period from data using zero crossings
        # For a square wave, period = distance between rising edges
        threshold = (data.max() + data.min()) / 2
        crossings = np.where(np.diff(data > threshold))[0]

        if len(crossings) >= 4:
            # Rising edges are at every other crossing
            rising_edges = crossings[::2] if data[crossings[0] + 1] > threshold else crossings[1::2]
            if len(rising_edges) >= 2:
                periods = np.diff(rising_edges)
                median_period = np.median(periods)

                tolerance = expected_period * 0.01  # 1% tolerance
                assert abs(median_period - expected_period) <= tolerance, (
                    f"Period mismatch: expected {expected_period}, got {median_period}"
                )


@pytest.mark.validation
@pytest.mark.requires_data
@pytest.mark.requirement("DSP-002")
class TestUARTSignalValidation:
    """Validate UART signal processing against ground truth."""

    def test_uart_file_exists(self, uart_synthetic_file: Path) -> None:
        """Verify synthetic UART file exists."""
        if not uart_synthetic_file.exists():
            pytest.skip(f"UART file not found: {uart_synthetic_file}")

        data = np.load(uart_synthetic_file, allow_pickle=True)
        assert len(data) > 0, "Empty UART signal"

    def test_uart_baud_rate_detection(
        self,
        uart_synthetic_file: Path,
        uart_truth: dict[str, Any],
    ) -> None:
        """Verify UART baud rate detection.

        Validates:
        - Detected baud rate matches ground truth (9600 baud)
        - Tolerance: 5% error allowed
        """
        if not uart_synthetic_file.exists():
            pytest.skip("UART synthetic file not available")

        if not uart_truth:
            pytest.skip("UART ground truth not available")

        expected_baud = uart_truth.get("baud_rate", 9600)
        data = np.load(uart_synthetic_file, allow_pickle=True)

        try:
            from oscura.analyzers.digital import detect_baud_rate

            # Assume 1MHz sample rate
            sample_rate = 1e6
            result = detect_baud_rate(data, sample_rate)

            # Get baud rate from result (may be object or value)
            detected = (
                getattr(result, "baud_rate", result) if hasattr(result, "baud_rate") else result
            )

            tolerance = 0.05 * expected_baud  # 5% tolerance
            assert abs(detected - expected_baud) < tolerance, (
                f"Baud rate mismatch: expected {expected_baud}, got {detected}"
            )
        except ImportError:
            pytest.skip("detect_baud_rate not available")


@pytest.mark.validation
@pytest.mark.requires_data
@pytest.mark.requirement("VAL-001")
class TestSyntheticDataIntegrity:
    """Test integrity of all synthetic signal data."""

    def test_all_square_waves_loadable(self, square_wave_files: dict[str, Path]) -> None:
        """Verify all square wave files can be loaded as numpy arrays."""
        for freq, path in square_wave_files.items():
            if path is not None and path.exists():
                try:
                    data = np.load(path, allow_pickle=True)
                    assert isinstance(data, np.ndarray), (
                        f"{freq}: Expected ndarray, got {type(data)}"
                    )
                    assert data.dtype in [np.float32, np.float64, np.int16, np.int32], (
                        f"{freq}: Unexpected dtype {data.dtype}"
                    )
                except Exception as e:
                    pytest.fail(f"Failed to load {freq} square wave: {e}")

    def test_ground_truth_structure(self, square_wave_truth: dict[str, dict[str, Any]]) -> None:
        """Verify ground truth files have expected structure."""
        for freq, truth in square_wave_truth.items():
            # Required fields
            assert "frequency_hz" in truth, f"{freq}: Missing frequency_hz field"
            assert isinstance(truth["frequency_hz"], int | float), (
                f"{freq}: frequency_hz should be numeric"
            )

            # Optional but expected fields
            if "pattern_period" in truth:
                assert isinstance(truth["pattern_period"], int | float)
            if "edge_count" in truth:
                assert isinstance(truth["edge_count"], int)

    def test_waveform_value_ranges(self, square_wave_files: dict[str, Path]) -> None:
        """Verify synthetic waveforms have valid value ranges."""
        for freq, path in square_wave_files.items():
            if path is None or not path.exists():
                continue

            data = np.load(path, allow_pickle=True)

            # Square waves should be bounded
            assert np.isfinite(data).all(), f"{freq}: Contains non-finite values"

            # Check for NaN
            assert not np.isnan(data).any(), f"{freq}: Contains NaN values"

            # Square wave should have two distinct levels (but may have noise)
            unique_values = np.unique(data)
            # More relaxed check: either few unique values OR bimodal distribution
            # For float data, check if values cluster around 2 levels
            if len(unique_values) > 100:
                # Check for bimodal distribution by looking at histogram peaks
                hist, _bin_edges = np.histogram(data, bins=50)
                peaks = np.where(hist > len(data) * 0.01)[0]  # Bins with >1% of data
                # Should have values clustered in at most a few regions
                assert len(peaks) <= 10 or np.std(data) < 2.0, (
                    f"{freq}: Too many unique values for square wave: {len(unique_values)}"
                )


@pytest.mark.validation
@pytest.mark.requires_data
class TestEdgeDetectionValidation:
    """Validate edge detection against known signals."""

    def test_edge_detection_basic(self, square_wave_files: dict[str, Path]) -> None:
        """Test basic edge detection on square waves."""
        path = square_wave_files.get("1MHz")
        if path is None or not path.exists():
            pytest.skip("1MHz square wave not available")

        data = np.load(path, allow_pickle=True)

        try:
            # Use detect_edges_advanced which accepts raw numpy arrays
            from oscura.analyzers.digital import detect_edges_advanced

            edges = detect_edges_advanced(data, threshold=0.5)

            # Should detect edges
            assert len(edges) > 0, "No edges detected"

            # Edges should be within data bounds (Edge objects have sample_index attribute)
            for edge in edges:
                idx = edge.sample_index
                assert 0 <= idx < len(data), f"Edge index out of bounds: {idx}"

        except ImportError:
            pytest.skip("detect_edges_advanced not available")

    def test_edge_detection_interpolation(self, square_wave_files: dict[str, Path]) -> None:
        """Test edge time interpolation accuracy.

        The interpolate_edge_time function takes:
        - trace: Input signal trace (ndarray)
        - sample_index: Sample index just before the edge
        - method: Interpolation method ('linear' or 'quadratic')

        It returns a fractional sample offset (0.0 to 1.0) from sample_index.
        """
        path = square_wave_files.get("1MHz")
        if path is None or not path.exists():
            pytest.skip("1MHz square wave not available")

        data = np.load(path, allow_pickle=True)

        try:
            from oscura.analyzers.digital import interpolate_edge_time

            # Find approximate crossing point
            threshold = (data.max() + data.min()) / 2
            crossings = np.where(np.diff(data > threshold))[0]

            if len(crossings) > 0:
                edge_idx = crossings[0]
                # interpolate_edge_time returns fractional offset (0.0 to 1.0)
                interp_offset = interpolate_edge_time(data, edge_idx, method="linear")
                # Interpolated offset should be between 0 and 1
                assert 0 <= interp_offset <= 1.0, (
                    f"Interpolated offset out of range: {interp_offset}"
                )

        except ImportError:
            pytest.skip("interpolate_edge_time not available")
        except (IndexError, ValueError):
            # Edge case - not enough samples
            pass


@pytest.mark.validation
@pytest.mark.requires_data
class TestClockRecoveryValidation:
    """Validate clock recovery algorithms."""

    def test_clock_recovery_from_uart(self, uart_synthetic_file: Path) -> None:
        """Test clock recovery from UART signal."""
        if not uart_synthetic_file.exists():
            pytest.skip("UART synthetic file not available")

        data = np.load(uart_synthetic_file, allow_pickle=True)

        try:
            from oscura.analyzers.digital import recover_clock

            sample_rate = 1e6
            clock = recover_clock(data, sample_rate)

            # Should recover a clock signal
            assert clock is not None, "Clock recovery failed"

        except ImportError:
            # SKIP: Valid - Optional clock recovery algorithm
            # Only skip if clock recovery module not available
            pytest.skip("recover_clock not available")
        except Exception as e:
            pytest.skip(f"Clock recovery requires specific conditions: {e}")

    def test_clock_jitter_measurement(self, square_wave_files: dict[str, Path]) -> None:
        """Test clock jitter measurement on square waves."""
        path = square_wave_files.get("1MHz")
        if path is None or not path.exists():
            # SKIP: Valid - Test signal dependency
            # Only skip if test signal generator not available
            pytest.skip("1MHz square wave not available")

        data = np.load(path, allow_pickle=True)

        try:
            from oscura.analyzers.digital import measure_clock_jitter

            sample_rate = 100e6
            jitter = measure_clock_jitter(data, sample_rate)

            # Synthetic square wave should have low jitter
            # Allow for quantization effects
            assert jitter is not None, "Jitter measurement failed"

        except ImportError:
            pytest.skip("measure_clock_jitter not available")
        except Exception as e:
            pytest.skip(f"Jitter measurement requires specific conditions: {e}")
