"""Tests for logic family detection and vintage logic support."""

import numpy as np
import pytest

from oscura.analyzers.digital.extraction import (
    LOGIC_FAMILIES,
    detect_logic_family,
    detect_open_collector,
    get_logic_threshold,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.digital]


class TestVintageLogicFamilies:
    """Test vintage logic family support."""

    def test_ecl_family_exists(self):
        """Test ECL logic family is defined."""
        assert "ECL" in LOGIC_FAMILIES
        ecl = LOGIC_FAMILIES["ECL"]
        assert ecl["VIL_max"] == -1.475
        assert ecl["VIH_min"] == -1.105
        assert ecl["differential"] is True

    def test_ecl_100k_family(self):
        """Test ECL 100K (faster variant)."""
        assert "ECL_100K" in LOGIC_FAMILIES
        ecl100k = LOGIC_FAMILIES["ECL_100K"]
        # 100K has tighter thresholds
        assert ecl100k["VIL_max"] == -1.810
        assert ecl100k["VIH_min"] == -1.620

    def test_rtl_family(self):
        """Test RTL (Resistor-Transistor Logic)."""
        assert "RTL" in LOGIC_FAMILIES
        rtl = LOGIC_FAMILIES["RTL"]
        assert rtl["VIL_max"] == 0.4
        assert rtl["VIH_min"] == 0.9
        assert rtl["VCC"] == 3.6

    def test_dtl_family(self):
        """Test DTL (Diode-Transistor Logic)."""
        assert "DTL" in LOGIC_FAMILIES
        dtl = LOGIC_FAMILIES["DTL"]
        assert dtl["VIL_max"] == 0.5
        assert dtl["VIH_min"] == 2.0

    def test_pmos_family(self):
        """Test PMOS (negative voltage)."""
        assert "PMOS" in LOGIC_FAMILIES
        pmos = LOGIC_FAMILIES["PMOS"]
        assert pmos["VCC"] == 0.0
        assert pmos["VDD"] == -12.0
        assert pmos["VIL_max"] == -3.0
        assert pmos["VIH_min"] == -9.0

    def test_nmos_family(self):
        """Test NMOS (positive voltage)."""
        assert "NMOS" in LOGIC_FAMILIES
        nmos = LOGIC_FAMILIES["NMOS"]
        assert nmos["VCC"] == 12.0
        assert nmos["VIL_max"] == 1.5
        assert nmos["VIH_min"] == 8.0


class TestGetLogicThreshold:
    """Test logic threshold calculation."""

    def test_ttl_midpoint(self):
        """Test TTL midpoint threshold."""
        threshold = get_logic_threshold("TTL", "midpoint")
        # Midpoint of 0.8V and 2.0V
        assert threshold == 1.4

    def test_ecl_midpoint(self):
        """Test ECL midpoint threshold."""
        threshold = get_logic_threshold("ECL", "midpoint")
        # Midpoint of -1.475V and -1.105V
        expected = (-1.475 + -1.105) / 2
        assert abs(threshold - expected) < 0.001

    def test_pmos_midpoint(self):
        """Test PMOS midpoint threshold."""
        threshold = get_logic_threshold("PMOS", "midpoint")
        # Should be negative voltage
        assert threshold < 0

    def test_vih_threshold(self):
        """Test VIH threshold type."""
        threshold = get_logic_threshold("TTL", "VIH")
        assert threshold == 2.0

    def test_vil_threshold(self):
        """Test VIL threshold type."""
        threshold = get_logic_threshold("TTL", "VIL")
        assert threshold == 0.8

    def test_unknown_family(self):
        """Test error with unknown family."""
        with pytest.raises(ValueError):
            get_logic_threshold("UNKNOWN", "midpoint")

    def test_unknown_threshold_type(self):
        """Test error with unknown threshold type."""
        with pytest.raises(ValueError):
            get_logic_threshold("TTL", "invalid")


class TestDetectLogicFamily:
    """Test automatic logic family detection."""

    def _create_signal(self, vlow: float, vhigh: float, num_samples: int = 1000) -> WaveformTrace:
        """Create a test signal with specified voltage levels."""
        # Create square wave
        data = np.zeros(num_samples)
        data[::2] = vlow
        data[1::2] = vhigh

        # Add some noise
        data += np.random.normal(0, 0.01, num_samples)

        metadata = TraceMetadata(
            sample_rate=1e6,
            channel_name="test",
        )
        return WaveformTrace(data=data, metadata=metadata)

    def test_detect_ttl(self):
        """Test TTL family detection."""
        # TTL levels: 0V to 5V
        trace = self._create_signal(0.0, 5.0)
        family, confidence = detect_logic_family(trace)

        assert family in ["TTL", "CMOS_5V"]  # Both use 5V
        assert confidence > 0.6

    def test_detect_lvcmos_3v3(self):
        """Test 3.3V LVCMOS detection."""
        trace = self._create_signal(0.0, 3.3)
        family, confidence = detect_logic_family(trace)

        assert "3V3" in family or "LVTTL" in family
        assert confidence > 0.6

    def test_detect_ecl(self):
        """Test ECL family detection."""
        # ECL levels: around -1.6V to -1.0V
        trace = self._create_signal(-1.63, -0.98)
        family, confidence = detect_logic_family(trace)

        assert "ECL" in family
        assert confidence > 0.5

    def test_detect_pmos(self):
        """Test PMOS detection."""
        # PMOS levels: 0V to -12V
        trace = self._create_signal(-11.5, -0.5)
        family, confidence = detect_logic_family(trace, confidence_threshold=0.4)

        assert family in ["PMOS", "MOS"]
        assert confidence > 0.4

    def test_insufficient_signal(self):
        """Test with insufficient signal."""
        trace = self._create_signal(0.0, 0.01, num_samples=5)
        family, confidence = detect_logic_family(trace)

        assert family == "unknown"
        assert confidence == 0.0

    def test_no_swing(self):
        """Test with DC signal (no swing)."""
        data = np.ones(1000) * 2.5  # Constant DC
        metadata = TraceMetadata(sample_rate=1e6, channel_name="test")
        trace = WaveformTrace(data=data, metadata=metadata)

        family, confidence = detect_logic_family(trace)
        assert family == "unknown"

    def test_confidence_threshold(self):
        """Test confidence threshold filtering."""
        # Signal with levels that don't match any family well
        trace = self._create_signal(1.5, 3.5)  # Right in the middle, not clear high/low

        family, confidence = detect_logic_family(trace, confidence_threshold=0.9)
        # Should return unknown if confidence too low
        assert confidence < 0.9 or family == "unknown"


class TestDetectOpenCollector:
    """Test open-collector/open-drain detection."""

    def _create_signal_with_edges(
        self, rise_time: float, fall_time: float, num_samples: int = 1000
    ) -> WaveformTrace:
        """Create signal with specified rise/fall times."""
        data = np.zeros(num_samples)
        sample_rate = 1e9  # 1GHz for nanosecond resolution

        # Create rising edges
        rise_samples = int(rise_time * sample_rate)
        fall_samples = int(fall_time * sample_rate)

        for i in range(0, num_samples, 200):
            # Rising edge
            if i + rise_samples < num_samples:
                data[i : i + rise_samples] = np.linspace(0, 5, rise_samples)
            # High level
            if i + rise_samples + 50 < num_samples:
                data[i + rise_samples : i + rise_samples + 50] = 5.0
            # Falling edge
            if i + rise_samples + 50 + fall_samples < num_samples:
                data[i + rise_samples + 50 : i + rise_samples + 50 + fall_samples] = np.linspace(
                    5, 0, fall_samples
                )

        metadata = TraceMetadata(sample_rate=sample_rate, channel_name="test")
        return WaveformTrace(data=data, metadata=metadata)

    def test_detect_open_collector_present(self):
        """Test detection of open-collector output."""
        # Open-collector: slow rise (100ns), fast fall (10ns)
        trace = self._create_signal_with_edges(rise_time=100e-9, fall_time=10e-9)
        is_oc, ratio = detect_open_collector(trace)

        assert is_oc is True
        assert ratio > 3.0  # Rise >> fall

    def test_detect_normal_output(self):
        """Test normal totem-pole output."""
        # Normal output: similar rise and fall
        trace = self._create_signal_with_edges(rise_time=10e-9, fall_time=10e-9)
        is_oc, ratio = detect_open_collector(trace)

        assert is_oc is False
        assert 0.5 < ratio < 2.0  # Symmetric

    def test_detect_with_custom_threshold(self):
        """Test with custom asymmetry threshold."""
        # Moderate asymmetry
        trace = self._create_signal_with_edges(rise_time=30e-9, fall_time=15e-9)

        # Should not detect with high threshold
        is_oc, ratio = detect_open_collector(trace, asymmetry_threshold=5.0)
        assert is_oc is False

        # Should detect with low threshold
        is_oc, ratio = detect_open_collector(trace, asymmetry_threshold=1.5)
        assert is_oc is True

    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = np.array([0, 1, 0])
        metadata = TraceMetadata(sample_rate=1e6, channel_name="test")
        trace = WaveformTrace(data=data, metadata=metadata)

        is_oc, ratio = detect_open_collector(trace)
        assert is_oc is False
        assert ratio == 1.0

    def test_ratio_calculation(self):
        """Test that asymmetry ratio is calculated correctly."""
        trace = self._create_signal_with_edges(rise_time=50e-9, fall_time=10e-9)
        is_oc, ratio = detect_open_collector(trace)

        # Ratio should be approximately 5:1
        assert 4.0 < ratio < 6.0
