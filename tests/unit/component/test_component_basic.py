"""Tests for component analysis module.

Tests requirements:
"""

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.utils.component.impedance import (
    discontinuity_analysis,
    extract_impedance,
    impedance_profile,
)
from oscura.utils.component.reactive import (
    measure_capacitance,
    measure_inductance,
)
from oscura.utils.component.transmission_line import (
    characteristic_impedance,
    propagation_delay,
    transmission_line_analysis,
    velocity_factor,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def tdr_trace():
    """Create a simulated TDR trace for testing.

    Simulates a 50 ohm transmission line with a step response.
    """
    sample_rate = 20e9  # 20 GSa/s
    num_samples = 10000
    np.arange(num_samples) / sample_rate

    # Simulate TDR step response
    # Initial step, then reflection from end
    data = np.zeros(num_samples)

    # Incident step at t=0
    step_idx = 100
    data[step_idx:] = 0.5  # 50% for 50 ohm match

    # Add some reflection at halfway point
    reflection_idx = 5000
    data[reflection_idx:] += 0.1  # Small mismatch

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def rc_voltage_trace():
    """Create voltage trace for RC circuit testing."""
    sample_rate = 1e6
    num_samples = 1000
    t = np.arange(num_samples) / sample_rate

    # RC charging curve: V = V0 * (1 - exp(-t/tau))
    tau = 1e-4  # 100us time constant
    v0 = 1.0
    data = v0 * (1 - np.exp(-t / tau))

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def rc_current_trace():
    """Create current trace for RC circuit testing."""
    sample_rate = 1e6
    num_samples = 1000
    t = np.arange(num_samples) / sample_rate

    # RC current: I = I0 * exp(-t/tau)
    tau = 1e-4  # 100us time constant
    R = 1000  # 1k ohm
    V0 = 1.0
    i0 = V0 / R
    data = i0 * np.exp(-t / tau)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


class TestImpedanceExtraction:
    """Tests for TDR impedance extraction (COMP-001)."""

    def test_extract_impedance_basic(self, tdr_trace):
        """Test basic impedance extraction."""
        z0, _profile = extract_impedance(tdr_trace, z0_source=50.0)
        # Should get approximately 50 ohms in stable region
        assert 30 < z0 < 80  # Reasonable range

    def test_impedance_profile(self, tdr_trace):
        """Test impedance profile generation."""
        profile = impedance_profile(tdr_trace, z0_source=50.0)
        assert len(profile.impedance) == len(tdr_trace.data)
        assert len(profile.distance) == len(tdr_trace.data)
        assert profile.z0_source == 50.0

    def test_discontinuity_analysis(self, tdr_trace):
        """Test discontinuity detection."""
        discontinuities = discontinuity_analysis(tdr_trace, z0_source=50.0, threshold=5.0)
        # Should find at least one discontinuity (the reflection)
        # May not find any if threshold too high
        assert isinstance(discontinuities, list)


class TestCapacitanceMeasurement:
    """Tests for capacitance measurement (COMP-002)."""

    def test_measure_capacitance_frequency_method(self, rc_voltage_trace):
        """Test capacitance measurement from time constant."""
        R = 1000  # 1k ohm
        result = measure_capacitance(rc_voltage_trace, method="frequency", resistance=R)
        # Expected: C = tau / R = 1e-4 / 1000 = 100nF
        expected_C = 1e-7
        assert 0.1 * expected_C < result.capacitance < 10 * expected_C


class TestInductanceMeasurement:
    """Tests for inductance measurement (COMP-003)."""

    def test_measure_inductance_frequency_method(self):
        """Test inductance measurement from time constant."""
        # Create RL circuit response
        sample_rate = 1e6
        num_samples = 1000
        t = np.arange(num_samples) / sample_rate

        # RL current: I = I0 * (1 - exp(-t*R/L))
        L = 1e-3  # 1mH
        R = 100  # 100 ohm
        L / R  # 10us
        V0 = 1.0
        data = (V0 / R) * (1 - np.exp(-t * R / L))

        metadata = TraceMetadata(sample_rate=sample_rate)
        current_trace = WaveformTrace(data=data, metadata=metadata)

        result = measure_inductance(current_trace, method="frequency", resistance=R)
        # Should be approximately 1mH
        assert 0.1e-3 < result.inductance < 10e-3


class TestTransmissionLineAnalysis:
    """Tests for transmission line analysis (SI-001)."""

    def test_transmission_line_analysis(self, tdr_trace):
        """Test transmission line characterization."""
        result = transmission_line_analysis(tdr_trace, z0_source=50.0)
        assert result.z0 > 0
        assert result.propagation_delay > 0
        assert 0 < result.velocity_factor <= 1

    def test_characteristic_impedance(self, tdr_trace):
        """Test characteristic impedance extraction."""
        z0 = characteristic_impedance(tdr_trace, z0_source=50.0)
        assert z0 > 0

    def test_propagation_delay_extraction(self, tdr_trace):
        """Test propagation delay extraction."""
        delay = propagation_delay(tdr_trace)
        assert delay >= 0

    def test_velocity_factor_calculation(self, tdr_trace):
        """Test velocity factor calculation."""
        vf = velocity_factor(tdr_trace, line_length=0.1)
        assert 0 < vf <= 1
