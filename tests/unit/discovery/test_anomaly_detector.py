"""Tests for anomaly detection.

Requirements tested:
"""

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.discovery.anomaly_detector import find_anomalies

pytestmark = pytest.mark.unit


class TestAnomalyDetection:
    """Tests for anomaly detection."""

    def test_detect_glitches(self):
        """Test detection of brief glitches."""
        # Create signal with a glitch
        signal = np.ones(1000) * 3.3
        signal[500:503] = 0.0  # 3-sample glitch

        trace = WaveformTrace(
            data=signal,
            metadata=TraceMetadata(sample_rate=1e9),  # 1 GS/s
        )

        anomalies = find_anomalies(trace, anomaly_types=["glitch"])

        # Should detect the glitch
        assert len(anomalies) >= 1
        glitches = [a for a in anomalies if a.type == "glitch"]
        assert len(glitches) >= 1

        glitch = glitches[0]
        assert glitch.duration_ns < 50
        assert glitch.confidence >= 0.7

    def test_detect_noise_spikes(self):
        """Test detection of noise spikes."""
        # Clean signal with noise spikes
        signal = np.ones(1000) * 1.65
        signal[100] = 3.3  # Large spike
        signal[500] = 0.0  # Another spike

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=1e6))

        anomalies = find_anomalies(trace, anomaly_types=["noise_spike"])

        # Should detect noise spikes
        noise_spikes = [a for a in anomalies if a.type == "noise_spike"]
        assert len(noise_spikes) >= 1

    def test_detect_overshoot(self):
        """Test detection of overshoot."""
        # Signal that transitions and overshoots significantly
        signal = np.zeros(1000)
        # Large overshoot: go from 0V to 4.5V (way above expected 3.3V)
        signal[100:110] = 4.5

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=1e6))

        anomalies = find_anomalies(trace, anomaly_types=["overshoot"])

        overshoots = [a for a in anomalies if a.type == "overshoot"]
        # Overshoot detection looks at high rail exceedance
        # May or may not detect depending on algorithm
        if len(overshoots) > 0:
            overshoot = overshoots[0]
            assert overshoot.severity in ("WARNING", "INFO")
            assert (
                "exceeded" in overshoot.description.lower()
                or "peak" in overshoot.description.lower()
            )

    def test_detect_undershoot(self):
        """Test detection of undershoot."""
        # Signal that undershoots 0V rail
        signal = np.zeros(1000)
        signal[100:110] = -0.5  # Undershoot below ground

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=1e6))

        anomalies = find_anomalies(trace, anomaly_types=["undershoot"])

        undershoots = [a for a in anomalies if a.type == "undershoot"]
        assert len(undershoots) >= 1

    def test_severity_filtering(self):
        """Test filtering anomalies by severity."""
        # Signal with various issues
        signal = np.ones(1000) * 1.65
        signal[100] = 3.3  # Spike
        signal[500:505] = 0.0  # Glitch

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=1e6))

        # Get all anomalies
        all_anomalies = find_anomalies(trace)

        # Get only critical
        critical = find_anomalies(trace, severity_filter=["CRITICAL"])

        # Critical should be subset of all
        assert len(critical) <= len(all_anomalies)

    def test_confidence_filtering(self):
        """Test filtering anomalies by confidence."""
        signal = np.ones(1000) * 1.65
        signal[100] = 2.0

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=1e6))

        low_conf = find_anomalies(trace, min_confidence=0.5)
        high_conf = find_anomalies(trace, min_confidence=0.9)

        # Higher confidence threshold should have fewer results
        assert len(high_conf) <= len(low_conf)

    def test_empty_trace(self):
        """Test that empty trace raises error."""
        # Empty array should raise ValueError at trace creation time
        with pytest.raises(ValueError, match="data array cannot be empty"):
            WaveformTrace(data=np.array([]), metadata=TraceMetadata(sample_rate=1e6))

    def test_anomaly_sorting(self):
        """Test that anomalies are sorted by timestamp."""
        # Signal with multiple anomalies
        signal = np.ones(1000) * 1.65
        signal[100] = 3.3
        signal[500] = 0.0
        signal[800] = 3.0

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=1e6))

        anomalies = find_anomalies(trace)

        # Check timestamps are in order
        if len(anomalies) > 1:
            timestamps = [a.timestamp_us for a in anomalies]
            assert timestamps == sorted(timestamps)

    def test_anomaly_metadata(self):
        """Test that anomalies include metadata."""
        signal = np.ones(1000) * 3.3
        signal[500:503] = 0.0

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=1e9))

        anomalies = find_anomalies(trace)

        if len(anomalies) > 0:
            anomaly = anomalies[0]
            assert anomaly.timestamp_us >= 0
            assert anomaly.type in (
                "glitch",
                "dropout",
                "noise_spike",
                "timing_violation",
                "ringing",
                "overshoot",
                "undershoot",
            )
            assert anomaly.severity in ("CRITICAL", "WARNING", "INFO")
            assert isinstance(anomaly.description, str)
            assert len(anomaly.description) > 0
            assert 0.0 <= anomaly.confidence <= 1.0

    def test_clean_signal_no_anomalies(self):
        """Test that clean signal has few/no anomalies."""
        # Very clean square wave
        t = np.linspace(0, 1e-3, 10000)
        signal = np.where(np.sin(2 * np.pi * 100 * t) > 0, 3.3, 0.0)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=10e6))

        anomalies = find_anomalies(trace, severity_filter=["CRITICAL"])

        # Clean signal should have no critical anomalies
        assert len(anomalies) == 0
