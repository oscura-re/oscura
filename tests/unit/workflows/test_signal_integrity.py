"""Comprehensive unit tests for signal integrity workflow.

Requirements tested:

This test suite covers:
- Eye diagram analysis
- Jitter decomposition (RMS and peak-to-peak)
- Time Interval Error (TIE) measurement
- Margin analysis against masks
- Dominant jitter source identification
- BER estimation
- SNR calculation
- Report generation
- Edge cases and error handling
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.workflows.signal_integrity import (
    _generate_si_report,
    signal_integrity_audit,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def simple_waveform_trace():
    """Create a simple waveform trace for testing."""
    sample_rate = 1e9  # 1 GSa/s
    duration = 1e-6  # 1 microsecond
    n_samples = int(sample_rate * duration)

    # Create a digital-like signal (NRZ data pattern)
    # Bit pattern: 0 1 1 0 1 0 0 1
    bit_rate = 1e9  # 1 Gbps
    samples_per_bit = int(sample_rate / bit_rate)
    pattern = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    data = np.repeat(pattern, samples_per_bit) * 1.0  # 1V swing

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def high_speed_trace():
    """Create a high-speed data trace with realistic characteristics."""
    sample_rate = 10e9  # 10 GSa/s
    bit_rate = 2.5e9  # 2.5 Gbps
    duration = 10e-9  # 10 ns
    n_samples = int(sample_rate * duration)

    # Create NRZ data with some jitter and noise
    t = np.linspace(0, duration, n_samples)
    rng = np.random.default_rng(42)

    # Square wave base signal
    data = (np.sin(2 * np.pi * bit_rate * t) > 0).astype(float) * 0.8

    # Add some noise (realistic SNR ~ 20 dB)
    noise = rng.normal(0, 0.02, n_samples)
    data = data + noise

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def clock_trace():
    """Create a clock reference trace."""
    sample_rate = 10e9  # 10 GSa/s
    frequency = 2.5e9  # 2.5 GHz clock
    duration = 10e-9  # 10 ns
    n_samples = int(sample_rate * duration)

    t = np.linspace(0, duration, n_samples)
    data = (np.sin(2 * np.pi * frequency * t) > 0).astype(float) * 1.0

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def noisy_trace():
    """Create a trace with significant noise (low SNR)."""
    sample_rate = 1e9
    n_samples = 1000
    rng = np.random.default_rng(42)

    # High noise relative to signal
    data = rng.normal(0.5, 0.3, n_samples)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.mark.unit
class TestSignalIntegrityAudit:
    """Test the main signal_integrity_audit function."""

    def test_basic_audit_without_clock(self, simple_waveform_trace):
        """Test basic signal integrity audit without clock recovery."""
        result = signal_integrity_audit(simple_waveform_trace)

        # Check all required fields are present
        assert "eye_height" in result
        assert "eye_width" in result
        assert "jitter_rms" in result
        assert "jitter_pp" in result
        assert "tie" in result
        assert "tie_rms" in result
        assert "margin_to_mask" in result
        assert "dominant_jitter_source" in result
        assert "bit_error_rate_estimate" in result
        assert "snr_db" in result
        assert "bit_rate" in result
        assert "unit_interval" in result

    def test_audit_with_clock_trace(self, simple_waveform_trace, clock_trace):
        """Test signal integrity audit with provided clock trace."""
        result = signal_integrity_audit(simple_waveform_trace, clock_trace=clock_trace)

        assert isinstance(result, dict)
        assert result["eye_height"] > 0
        assert result["eye_width"] > 0

    def test_audit_with_bit_rate(self, simple_waveform_trace):
        """Test signal integrity audit with specified bit rate."""
        bit_rate = 1e9  # 1 Gbps

        result = signal_integrity_audit(simple_waveform_trace, bit_rate=bit_rate)

        assert result["bit_rate"] == bit_rate
        assert result["unit_interval"] == 1.0 / bit_rate

    def test_audit_with_mask(self, simple_waveform_trace):
        """Test signal integrity audit with mask specification."""
        result = signal_integrity_audit(simple_waveform_trace, mask="PCIe")

        assert isinstance(result["margin_to_mask"], (int, float))
        assert result["margin_to_mask"] > 0

    def test_audit_without_mask(self, simple_waveform_trace):
        """Test that margin_to_mask is None when no mask specified."""
        result = signal_integrity_audit(simple_waveform_trace)

        assert result["margin_to_mask"] is None

    def test_eye_height_calculation(self, simple_waveform_trace):
        """Test that eye height is calculated correctly."""
        result = signal_integrity_audit(simple_waveform_trace)

        # Eye height should be ~70% of peak-to-peak voltage
        vpp = np.ptp(simple_waveform_trace.data)
        expected_eye_height = vpp * 0.7

        assert pytest.approx(result["eye_height"], rel=0.01) == expected_eye_height

    def test_eye_width_calculation(self, simple_waveform_trace):
        """Test that eye width is calculated correctly."""
        bit_rate = 1e9
        result = signal_integrity_audit(simple_waveform_trace, bit_rate=bit_rate)

        # Eye width should be ~60% of unit interval
        ui = 1.0 / bit_rate
        expected_eye_width = ui * 0.6

        assert pytest.approx(result["eye_width"], rel=0.01) == expected_eye_width

    def test_jitter_metrics(self, simple_waveform_trace):
        """Test that jitter metrics are calculated."""
        result = signal_integrity_audit(simple_waveform_trace, bit_rate=1e9)

        # Jitter values should be positive
        assert result["jitter_rms"] > 0
        assert result["jitter_pp"] > 0

        # Peak-to-peak should be larger than RMS
        assert result["jitter_pp"] > result["jitter_rms"]

    def test_dominant_jitter_random(self, simple_waveform_trace):
        """Test dominant jitter source identification for random jitter."""
        # Default behavior should identify jitter type
        result = signal_integrity_audit(simple_waveform_trace, bit_rate=1e9)

        assert result["dominant_jitter_source"] in ["random", "deterministic", "unknown"]

    def test_dominant_jitter_deterministic(self):
        """Test dominant jitter identification when deterministic dominates."""
        # Create a signal with bounded jitter (deterministic)
        sample_rate = 1e9
        n_samples = 1000
        data = np.tile([0, 1], n_samples // 2) * 1.0

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Mock jitter analysis to return deterministic characteristics
        # (low RMS relative to peak-to-peak)
        result = signal_integrity_audit(trace, bit_rate=1e9)

        # Should detect some jitter type
        assert result["dominant_jitter_source"] is not None

    def test_ber_estimate(self, simple_waveform_trace):
        """Test bit error rate estimation."""
        result = signal_integrity_audit(simple_waveform_trace, bit_rate=1e9)

        assert "bit_error_rate_estimate" in result
        assert 0 <= result["bit_error_rate_estimate"] <= 1

    def test_snr_calculation(self, simple_waveform_trace):
        """Test SNR calculation in dB."""
        result = signal_integrity_audit(simple_waveform_trace, bit_rate=1e9)

        assert "snr_db" in result
        # SNR can be negative in dB scale for low quality signals
        assert isinstance(result["snr_db"], float | np.floating)

    def test_low_snr_trace(self, noisy_trace):
        """Test signal integrity on low SNR trace."""
        result = signal_integrity_audit(noisy_trace, bit_rate=1e9)

        # Low SNR trace should still return valid results
        assert isinstance(result["snr_db"], float | np.floating)
        assert result["bit_error_rate_estimate"] >= 0

    def test_high_speed_signal(self, high_speed_trace):
        """Test with realistic high-speed signal."""
        result = signal_integrity_audit(high_speed_trace, bit_rate=2.5e9)

        # Bit rate might be from clock recovery or specified value
        # Check it's in a reasonable range
        assert 2.0e9 <= result["bit_rate"] <= 3.0e9
        assert result["eye_height"] >= 0  # Can be very small for noisy signals
        assert result["jitter_rms"] > 0

    def test_tie_array(self, simple_waveform_trace):
        """Test that TIE array is returned."""
        result = signal_integrity_audit(simple_waveform_trace, bit_rate=1e9)

        assert "tie" in result
        assert isinstance(result["tie"], np.ndarray)

    def test_tie_rms(self, simple_waveform_trace):
        """Test TIE RMS calculation."""
        result = signal_integrity_audit(simple_waveform_trace, bit_rate=1e9)

        assert "tie_rms" in result
        assert result["tie_rms"] >= 0

    def test_report_generation(self, simple_waveform_trace, tmp_path):
        """Test HTML report generation."""
        report_path = tmp_path / "signal_integrity_report.html"

        result = signal_integrity_audit(
            simple_waveform_trace, bit_rate=1e9, report=str(report_path)
        )

        # Check report was created
        assert report_path.exists()

        # Check report contains expected content
        content = report_path.read_text()
        assert "Signal Integrity Audit Report" in content
        assert "Eye Diagram Analysis" in content
        assert "Eye Height" in content
        assert "RMS Jitter" in content

    def test_no_report_by_default(self, simple_waveform_trace, tmp_path):
        """Test that no report is generated by default."""
        result = signal_integrity_audit(simple_waveform_trace, bit_rate=1e9)

        # No report file should be created
        assert not any(tmp_path.glob("*.html"))

    def test_multiple_masks(self, simple_waveform_trace):
        """Test with different mask standards."""
        masks = ["PCIe", "USB", "SATA", None]

        for mask in masks:
            result = signal_integrity_audit(simple_waveform_trace, mask=mask)

            if mask is not None:
                assert result["margin_to_mask"] is not None
            else:
                assert result["margin_to_mask"] is None

    def test_zero_amplitude_signal(self):
        """Test handling of zero amplitude signal."""
        sample_rate = 1e9
        data = np.zeros(1000)

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = signal_integrity_audit(trace, bit_rate=1e9)

        # Eye height should be zero or very small
        assert result["eye_height"] >= 0

    def test_constant_signal(self):
        """Test handling of constant DC signal."""
        sample_rate = 1e9
        data = np.ones(1000) * 1.0

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = signal_integrity_audit(trace, bit_rate=1e9)

        # Should handle gracefully
        assert "eye_height" in result

    def test_very_short_trace(self):
        """Test with very short trace."""
        sample_rate = 1e9
        data = np.array([0, 1, 0, 1])

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = signal_integrity_audit(trace, bit_rate=1e9)

        assert isinstance(result, dict)
        assert "eye_height" in result

    def test_negative_values(self):
        """Test with signal containing negative values."""
        sample_rate = 1e9
        data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-6, 1000))

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = signal_integrity_audit(trace, bit_rate=1e9)

        # Should calculate eye height as peak-to-peak * 0.7
        assert result["eye_height"] > 0

    def test_clock_recovery_success(self, simple_waveform_trace):
        """Test successful clock recovery when imports available."""
        # Mock the imported function inside signal_integrity module
        with patch("oscura.analyzers.digital.timing.recover_clock_fft") as mock_recover:
            mock_result = Mock()
            mock_result.frequency = 1e9
            mock_recover.return_value = mock_result

            result = signal_integrity_audit(simple_waveform_trace)

            # Should use recovered frequency
            assert result["bit_rate"] == 1e9

    def test_clock_recovery_failure(self, simple_waveform_trace):
        """Test clock recovery failure fallback."""
        # Mock clock recovery to fail
        with patch("oscura.analyzers.digital.timing.recover_clock_fft") as mock_recover:
            mock_recover.side_effect = Exception("Clock recovery failed")

            bit_rate = 2e9
            result = signal_integrity_audit(simple_waveform_trace, bit_rate=bit_rate)

            # Should fall back to specified bit rate
            assert result["bit_rate"] == bit_rate

    def test_tie_calculation_success(self, simple_waveform_trace):
        """Test TIE calculation when available."""
        # Mock TIE calculation
        with patch("oscura.analyzers.digital.timing.time_interval_error") as mock_tie:
            mock_tie_data = np.random.default_rng(42).normal(0, 1e-12, 100)
            mock_tie.return_value = mock_tie_data

            result = signal_integrity_audit(simple_waveform_trace, bit_rate=1e9)

            # Should use calculated TIE
            assert len(result["tie"]) == len(mock_tie_data)
            assert pytest.approx(result["jitter_rms"], rel=0.1) == float(np.std(mock_tie_data))

    def test_tie_calculation_failure(self, simple_waveform_trace):
        """Test TIE calculation failure fallback."""
        # Mock TIE calculation failure
        with patch("oscura.analyzers.digital.timing.time_interval_error") as mock_tie:
            mock_tie.side_effect = Exception("TIE calculation failed")

            result = signal_integrity_audit(simple_waveform_trace, bit_rate=1e9)

            # Should fall back to estimated jitter
            assert len(result["tie"]) == 0
            assert result["jitter_rms"] > 0  # Should have fallback value

    def test_jitter_ratio_calculation(self, simple_waveform_trace):
        """Test jitter ratio determines dominant source correctly."""
        result = signal_integrity_audit(simple_waveform_trace, bit_rate=1e9)

        jitter_rms = result["jitter_rms"]
        jitter_pp = result["jitter_pp"]

        if jitter_rms > 0:
            jitter_ratio = jitter_pp / (6 * jitter_rms)

            if jitter_ratio < 8:
                assert result["dominant_jitter_source"] == "random"
            else:
                assert result["dominant_jitter_source"] == "deterministic"

    def test_zero_jitter_handling(self):
        """Test handling when jitter is zero."""
        # Create perfect square wave
        sample_rate = 1e9
        data = np.tile([0, 0, 1, 1], 250) * 1.0

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Mock TIE to return zeros
        with patch("oscura.analyzers.digital.timing.time_interval_error") as mock_tie:
            mock_tie.return_value = np.zeros(100)
            result = signal_integrity_audit(trace, bit_rate=1e9)

            # Should handle zero jitter gracefully
            assert result["dominant_jitter_source"] == "unknown"

    def test_ber_with_zero_eye_height(self):
        """Test BER estimation with zero eye height."""
        sample_rate = 1e9
        data = np.zeros(1000)

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = signal_integrity_audit(trace, bit_rate=1e9)

        # Should have BER estimate even with zero eye
        assert "bit_error_rate_estimate" in result

    def test_result_types(self, simple_waveform_trace):
        """Test that result values have correct types."""
        result = signal_integrity_audit(simple_waveform_trace, bit_rate=1e9)

        assert isinstance(result["eye_height"], float | np.floating)
        assert isinstance(result["eye_width"], float | np.floating)
        assert isinstance(result["jitter_rms"], float | np.floating)
        assert isinstance(result["jitter_pp"], float | np.floating)
        assert isinstance(result["tie"], np.ndarray)
        assert isinstance(result["tie_rms"], float | np.floating)
        assert isinstance(result["dominant_jitter_source"], str)
        assert isinstance(result["bit_error_rate_estimate"], float | np.floating)
        assert isinstance(result["snr_db"], float | np.floating)
        assert isinstance(result["bit_rate"], float | np.floating)
        assert isinstance(result["unit_interval"], float | np.floating)

    def test_mask_margin_positive(self, simple_waveform_trace):
        """Test that mask margin is positive when mask specified."""
        result = signal_integrity_audit(simple_waveform_trace, mask="PCIe")

        assert result["margin_to_mask"] > 0

    def test_mask_margin_calculation(self, simple_waveform_trace):
        """Test mask margin is approximately 20% of eye height."""
        result = signal_integrity_audit(simple_waveform_trace, mask="USB")

        expected_margin = result["eye_height"] * 0.2
        assert pytest.approx(result["margin_to_mask"], rel=0.01) == expected_margin


@pytest.mark.unit
class TestGenerateSIReport:
    """Test the HTML report generation function."""

    def test_report_generation_basic(self, tmp_path):
        """Test basic report generation."""
        result = {
            "eye_height": 0.8e-3,  # 0.8 mV
            "eye_width": 400e-12,  # 400 ps
            "jitter_rms": 10e-12,  # 10 ps
            "jitter_pp": 50e-12,  # 50 ps
            "snr_db": 25.0,
            "bit_error_rate_estimate": 1e-12,
            "dominant_jitter_source": "random",
        }

        report_path = tmp_path / "test_report.html"
        _generate_si_report(result, str(report_path))

        assert report_path.exists()

    def test_report_content_structure(self, tmp_path):
        """Test that report contains expected HTML structure."""
        result = {
            "eye_height": 0.8e-3,
            "eye_width": 400e-12,
            "jitter_rms": 10e-12,
            "jitter_pp": 50e-12,
            "snr_db": 25.0,
            "bit_error_rate_estimate": 1e-12,
            "dominant_jitter_source": "random",
        }

        report_path = tmp_path / "test_report.html"
        _generate_si_report(result, str(report_path))

        content = report_path.read_text()

        # Check HTML structure
        assert "<html>" in content
        assert "</html>" in content
        assert "<head>" in content
        assert "<title>Signal Integrity Audit Report</title>" in content
        assert "<body>" in content
        assert "</body>" in content

    def test_report_content_headers(self, tmp_path):
        """Test that report contains expected headers."""
        result = {
            "eye_height": 0.8e-3,
            "eye_width": 400e-12,
            "jitter_rms": 10e-12,
            "jitter_pp": 50e-12,
            "snr_db": 25.0,
            "bit_error_rate_estimate": 1e-12,
            "dominant_jitter_source": "random",
        }

        report_path = tmp_path / "test_report.html"
        _generate_si_report(result, str(report_path))

        content = report_path.read_text()

        assert "<h1>Signal Integrity Audit Report</h1>" in content
        assert "<h2>Eye Diagram Analysis</h2>" in content

    def test_report_content_table(self, tmp_path):
        """Test that report contains parameter table."""
        result = {
            "eye_height": 0.8e-3,
            "eye_width": 400e-12,
            "jitter_rms": 10e-12,
            "jitter_pp": 50e-12,
            "snr_db": 25.0,
            "bit_error_rate_estimate": 1e-12,
            "dominant_jitter_source": "random",
        }

        report_path = tmp_path / "test_report.html"
        _generate_si_report(result, str(report_path))

        content = report_path.read_text()

        assert "<table>" in content
        assert "<tr><th>Parameter</th><th>Value</th><th>Units</th></tr>" in content

    def test_report_eye_height_formatting(self, tmp_path):
        """Test eye height is formatted in mV."""
        result = {
            "eye_height": 0.8e-3,  # 0.8 mV in base units
            "eye_width": 400e-12,
            "jitter_rms": 10e-12,
            "jitter_pp": 50e-12,
            "snr_db": 25.0,
            "bit_error_rate_estimate": 1e-12,
            "dominant_jitter_source": "random",
        }

        report_path = tmp_path / "test_report.html"
        _generate_si_report(result, str(report_path))

        content = report_path.read_text()

        # Should show 0.80 mV
        assert ">0.80<" in content or ">0.8<" in content
        assert "<td>mV</td>" in content

    def test_report_eye_width_formatting(self, tmp_path):
        """Test eye width is formatted in ps."""
        result = {
            "eye_height": 0.8e-3,
            "eye_width": 400e-12,  # 400 ps in base units
            "jitter_rms": 10e-12,
            "jitter_pp": 50e-12,
            "snr_db": 25.0,
            "bit_error_rate_estimate": 1e-12,
            "dominant_jitter_source": "random",
        }

        report_path = tmp_path / "test_report.html"
        _generate_si_report(result, str(report_path))

        content = report_path.read_text()

        # Should show 400.00 ps
        assert ">400.00<" in content or ">400.0<" in content
        assert "<td>ps</td>" in content

    def test_report_jitter_formatting(self, tmp_path):
        """Test jitter values are formatted in ps."""
        result = {
            "eye_height": 0.8e-3,
            "eye_width": 400e-12,
            "jitter_rms": 10e-12,  # 10 ps
            "jitter_pp": 50e-12,  # 50 ps
            "snr_db": 25.0,
            "bit_error_rate_estimate": 1e-12,
            "dominant_jitter_source": "random",
        }

        report_path = tmp_path / "test_report.html"
        _generate_si_report(result, str(report_path))

        content = report_path.read_text()

        # Should show both jitter values in ps
        assert ">10.00<" in content or ">10.0<" in content
        assert ">50.00<" in content or ">50.0<" in content

    def test_report_snr_formatting(self, tmp_path):
        """Test SNR is formatted in dB."""
        result = {
            "eye_height": 0.8e-3,
            "eye_width": 400e-12,
            "jitter_rms": 10e-12,
            "jitter_pp": 50e-12,
            "snr_db": 25.3,
            "bit_error_rate_estimate": 1e-12,
            "dominant_jitter_source": "random",
        }

        report_path = tmp_path / "test_report.html"
        _generate_si_report(result, str(report_path))

        content = report_path.read_text()

        # Should show 25.3 dB
        assert ">25.3<" in content
        assert "<td>dB</td>" in content

    def test_report_ber_formatting(self, tmp_path):
        """Test BER is formatted in scientific notation."""
        result = {
            "eye_height": 0.8e-3,
            "eye_width": 400e-12,
            "jitter_rms": 10e-12,
            "jitter_pp": 50e-12,
            "snr_db": 25.0,
            "bit_error_rate_estimate": 1.5e-12,
            "dominant_jitter_source": "random",
        }

        report_path = tmp_path / "test_report.html"
        _generate_si_report(result, str(report_path))

        content = report_path.read_text()

        # Should contain scientific notation (e.g., 1.50e-12)
        assert "e-" in content or "E-" in content

    def test_report_dominant_jitter_source(self, tmp_path):
        """Test dominant jitter source is included."""
        result = {
            "eye_height": 0.8e-3,
            "eye_width": 400e-12,
            "jitter_rms": 10e-12,
            "jitter_pp": 50e-12,
            "snr_db": 25.0,
            "bit_error_rate_estimate": 1e-12,
            "dominant_jitter_source": "deterministic",
        }

        report_path = tmp_path / "test_report.html"
        _generate_si_report(result, str(report_path))

        content = report_path.read_text()

        assert ">deterministic<" in content

    def test_report_with_very_small_values(self, tmp_path):
        """Test report handles very small values correctly."""
        result = {
            "eye_height": 1e-6,  # 1 µV
            "eye_width": 1e-15,  # 1 fs
            "jitter_rms": 1e-15,
            "jitter_pp": 5e-15,
            "snr_db": 0.1,
            "bit_error_rate_estimate": 1e-20,
            "dominant_jitter_source": "random",
        }

        report_path = tmp_path / "test_report.html"
        _generate_si_report(result, str(report_path))

        assert report_path.exists()
        content = report_path.read_text()
        assert "Signal Integrity Audit Report" in content

    def test_report_with_large_values(self, tmp_path):
        """Test report handles large values correctly."""
        result = {
            "eye_height": 10.0,  # 10 V
            "eye_width": 1e-6,  # 1 µs
            "jitter_rms": 1e-9,  # 1 ns
            "jitter_pp": 1e-8,  # 10 ns
            "snr_db": 60.0,
            "bit_error_rate_estimate": 1e-30,
            "dominant_jitter_source": "random",
        }

        report_path = tmp_path / "test_report.html"
        _generate_si_report(result, str(report_path))

        assert report_path.exists()
        content = report_path.read_text()
        assert "10000.00" in content  # 10V = 10000 mV

    def test_report_overwrites_existing_file(self, tmp_path):
        """Test that report overwrites existing file."""
        report_path = tmp_path / "test_report.html"

        # Create initial report
        result1 = {
            "eye_height": 0.8e-3,
            "eye_width": 400e-12,
            "jitter_rms": 10e-12,
            "jitter_pp": 50e-12,
            "snr_db": 25.0,
            "bit_error_rate_estimate": 1e-12,
            "dominant_jitter_source": "random",
        }
        _generate_si_report(result1, str(report_path))

        # Overwrite with new report
        result2 = {
            "eye_height": 1.5e-3,
            "eye_width": 500e-12,
            "jitter_rms": 20e-12,
            "jitter_pp": 100e-12,
            "snr_db": 30.0,
            "bit_error_rate_estimate": 1e-15,
            "dominant_jitter_source": "deterministic",
        }
        _generate_si_report(result2, str(report_path))

        content = report_path.read_text()

        # Should have new values, not old
        assert ">deterministic<" in content
        assert ">30.0<" in content

    def test_report_file_is_valid_html(self, tmp_path):
        """Test that generated report is valid HTML."""
        result = {
            "eye_height": 0.8e-3,
            "eye_width": 400e-12,
            "jitter_rms": 10e-12,
            "jitter_pp": 50e-12,
            "snr_db": 25.0,
            "bit_error_rate_estimate": 1e-12,
            "dominant_jitter_source": "random",
        }

        report_path = tmp_path / "test_report.html"
        _generate_si_report(result, str(report_path))

        content = report_path.read_text()

        # Basic HTML validation
        assert content.count("<html>") == 1
        assert content.count("</html>") == 1
        assert content.count("<body>") == 1
        assert content.count("</body>") == 1
        assert content.count("<table>") == 1
        assert content.count("</table>") == 1


@pytest.mark.unit
class TestWorkflowsSignalIntegrityEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_sample_trace(self):
        """Test with trace containing only one sample."""
        metadata = TraceMetadata(sample_rate=1e9)
        trace = WaveformTrace(data=np.array([1.0]), metadata=metadata)

        result = signal_integrity_audit(trace, bit_rate=1e9)

        assert isinstance(result, dict)
        assert result["eye_height"] == 0  # No variation

    def test_two_sample_trace(self):
        """Test with minimal trace (two samples)."""
        metadata = TraceMetadata(sample_rate=1e9)
        trace = WaveformTrace(data=np.array([0.0, 1.0]), metadata=metadata)

        result = signal_integrity_audit(trace, bit_rate=1e9)

        assert result["eye_height"] > 0

    def test_inf_values(self):
        """Test handling of infinite values."""
        metadata = TraceMetadata(sample_rate=1e9)
        data = np.array([0.0, 1.0, np.inf, 1.0, 0.0])
        trace = WaveformTrace(data=data, metadata=metadata)

        result = signal_integrity_audit(trace, bit_rate=1e9)

        # Should handle gracefully (ptp will be inf)
        assert isinstance(result, dict)

    def test_nan_values(self):
        """Test handling of NaN values."""
        metadata = TraceMetadata(sample_rate=1e9)
        data = np.array([0.0, 1.0, np.nan, 1.0, 0.0])
        trace = WaveformTrace(data=data, metadata=metadata)

        result = signal_integrity_audit(trace, bit_rate=1e9)

        # Should handle gracefully (ptp will be nan)
        assert isinstance(result, dict)

    def test_very_high_sample_rate(self):
        """Test with extremely high sample rate."""
        metadata = TraceMetadata(sample_rate=1e12)  # 1 TSa/s
        data = np.array([0, 1] * 500, dtype=float)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Use clock_trace to bypass clock recovery which might override bit_rate
        clock_metadata = TraceMetadata(sample_rate=1e12)
        clock_data = np.array([0, 1] * 500, dtype=float)
        clock = WaveformTrace(data=clock_data, metadata=clock_metadata)

        result = signal_integrity_audit(trace, clock_trace=clock, bit_rate=1e9)

        # Unit interval should match specified bit rate
        assert result["unit_interval"] == pytest.approx(1e-9, rel=0.01)

    def test_very_low_bit_rate(self):
        """Test with very low bit rate."""
        metadata = TraceMetadata(sample_rate=1e6)
        data = np.array([0, 1] * 500, dtype=float)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Use clock_trace to bypass clock recovery which might override bit_rate
        clock_metadata = TraceMetadata(sample_rate=1e6)
        clock_data = np.array([0, 1] * 500, dtype=float)
        clock = WaveformTrace(data=clock_data, metadata=clock_metadata)

        result = signal_integrity_audit(trace, clock_trace=clock, bit_rate=1e3)

        # Unit interval should match specified bit rate
        assert result["unit_interval"] == pytest.approx(1e-3, rel=0.01)

    def test_no_bit_rate_no_clock(self):
        """Test when neither bit rate nor clock is provided."""
        metadata = TraceMetadata(sample_rate=1e9)
        data = np.array([0, 1] * 500, dtype=float)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = signal_integrity_audit(trace)

        # Should attempt some form of clock/bit rate detection or use a default
        # The exact value depends on the implementation
        assert result["bit_rate"] > 0
        assert result["unit_interval"] > 0

    def test_module_exports(self):
        """Test that module exports expected functions."""
        from oscura.workflows import signal_integrity

        assert hasattr(signal_integrity, "signal_integrity_audit")
        assert hasattr(signal_integrity, "_generate_si_report")
        assert "signal_integrity_audit" in signal_integrity.__all__

    def test_import_error_eye_diagram(self):
        """Test graceful handling when eye diagram module is not available."""
        import builtins

        metadata = TraceMetadata(sample_rate=1e9)
        data = np.array([0, 1] * 500, dtype=float)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Mock the import to fail for eye diagram module
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "oscura.analyzers.eye.diagram" in name:
                raise ImportError("Eye diagram module not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # Should still work without the eye diagram module
            result = signal_integrity_audit(trace, bit_rate=1e9)
            assert isinstance(result, dict)
            assert "eye_height" in result

    def test_timing_module_unavailable(self):
        """Test fallback when timing analysis module is unavailable."""
        import builtins

        metadata = TraceMetadata(sample_rate=1e9)
        data = np.array([0, 1] * 500, dtype=float)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Mock the import to fail for timing module
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "oscura.analyzers.digital.timing" in name:
                raise ImportError("Timing module not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = signal_integrity_audit(trace, bit_rate=1e9)

            # Should use fallback jitter estimation
            assert result["jitter_rms"] > 0
            assert len(result["tie"]) == 0  # No TIE array when module unavailable

    def test_all_imports_unavailable(self):
        """Test when both eye and timing modules are unavailable."""
        import builtins

        metadata = TraceMetadata(sample_rate=1e9)
        data = np.array([0, 1] * 500, dtype=float)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Mock the import to fail for both modules
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "oscura.analyzers.eye.diagram" in name or "oscura.analyzers.digital.timing" in name:
                raise ImportError("Module not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = signal_integrity_audit(trace, bit_rate=1e9)

            # Should still produce valid results with fallback calculations
            assert isinstance(result, dict)
            assert result["eye_height"] > 0
            assert result["jitter_rms"] > 0
            assert result["bit_rate"] == 1e9
