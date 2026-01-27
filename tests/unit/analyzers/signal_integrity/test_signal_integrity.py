"""Unit tests for signal integrity analyzers.

This module tests the signal integrity analysis modules:
- sparams.py: S-Parameter handling and Touchstone file support
- embedding.py: Channel embedding and de-embedding
- equalization.py: FFE, DFE, and CTLE equalization
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from oscura.core.types import WaveformTrace
from tests.utils.factories import make_waveform_trace

if TYPE_CHECKING:
    from oscura.analyzers.signal_integrity.sparams import SParameterData

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# Helper functions
def make_s_parameter_data(
    n_freq: int = 100,
    n_ports: int = 2,
    z0: float = 50.0,
) -> SParameterData:
    """Create synthetic S-parameter data for testing."""
    from oscura.analyzers.signal_integrity.sparams import SParameterData

    frequencies = np.linspace(1e6, 10e9, n_freq)
    s_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)

    # Create realistic S-parameters
    for i in range(n_freq):
        freq = frequencies[i]
        # S11 (return loss) - starts low, increases with frequency
        s11_mag = 0.05 + 0.1 * (freq / 10e9)
        s11_phase = -np.pi / 4 * (freq / 10e9)
        s_matrix[i, 0, 0] = s11_mag * np.exp(1j * s11_phase)

        # S21 (insertion loss) - decreases with frequency
        s21_mag = 0.9 - 0.2 * (freq / 10e9)
        s21_phase = -np.pi * freq / 5e9
        s_matrix[i, 1, 0] = s21_mag * np.exp(1j * s21_phase)

        # S12 (reverse transmission) - assume reciprocal
        s_matrix[i, 0, 1] = s_matrix[i, 1, 0]

        # S22 (output return loss)
        s_matrix[i, 1, 1] = s_matrix[i, 0, 0] * 0.9

    return SParameterData(
        frequencies=frequencies,
        s_matrix=s_matrix,
        n_ports=n_ports,
        z0=z0,
    )


# =============================================================================
# S-Parameter Tests (sparams.py)
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-001")
class TestSParameterData:
    """Test SParameterData dataclass."""

    def test_sparam_data_creation(self) -> None:
        """Test basic SParameterData creation."""
        from oscura.analyzers.signal_integrity.sparams import SParameterData

        frequencies = np.array([1e9, 2e9, 3e9])
        s_matrix = np.zeros((3, 2, 2), dtype=np.complex128)
        s_matrix[:, 0, 0] = 0.1  # S11
        s_matrix[:, 1, 0] = 0.9  # S21
        s_matrix[:, 0, 1] = 0.9  # S12
        s_matrix[:, 1, 1] = 0.1  # S22

        s_params = SParameterData(
            frequencies=frequencies,
            s_matrix=s_matrix,
            n_ports=2,
            z0=50.0,
        )

        assert len(s_params.frequencies) == 3
        assert s_params.n_ports == 2
        assert s_params.z0 == 50.0
        assert s_params.s_matrix.shape == (3, 2, 2)

    def test_sparam_data_validation_empty_frequencies(self) -> None:
        """Test that empty frequencies raise ValueError."""
        from oscura.analyzers.signal_integrity.sparams import SParameterData

        frequencies = np.array([])
        s_matrix = np.zeros((0, 2, 2), dtype=np.complex128)

        with pytest.raises(ValueError, match="frequencies cannot be empty"):
            SParameterData(
                frequencies=frequencies,
                s_matrix=s_matrix,
                n_ports=2,
            )

    def test_sparam_data_validation_shape_mismatch(self) -> None:
        """Test that mismatched shapes raise ValueError."""
        from oscura.analyzers.signal_integrity.sparams import SParameterData

        frequencies = np.array([1e9, 2e9, 3e9])
        # Wrong shape - 4 frequencies instead of 3
        s_matrix = np.zeros((4, 2, 2), dtype=np.complex128)

        with pytest.raises(ValueError, match="s_matrix shape"):
            SParameterData(
                frequencies=frequencies,
                s_matrix=s_matrix,
                n_ports=2,
            )

    def test_get_s_all_frequencies(self) -> None:
        """Test get_s returns all frequencies."""
        s_params = make_s_parameter_data(n_freq=50)

        s21 = s_params.get_s(2, 1)  # S21 (1-indexed)

        assert isinstance(s21, np.ndarray)
        assert len(s21) == 50
        assert np.all(np.isfinite(s21))

    def test_get_s_single_frequency(self) -> None:
        """Test get_s returns single interpolated value."""
        s_params = make_s_parameter_data(n_freq=100)

        # Get S21 at a specific frequency
        freq = 5e9
        s21 = s_params.get_s(2, 1, frequency=freq)

        assert isinstance(s21, complex | np.complexfloating)
        assert np.isfinite(s21)

    def test_get_s_interpolation(self) -> None:
        """Test that get_s interpolates correctly between points."""
        from oscura.analyzers.signal_integrity.sparams import SParameterData

        # Create simple linear S-parameters for testing interpolation
        frequencies = np.array([1e9, 2e9, 3e9])
        s_matrix = np.zeros((3, 2, 2), dtype=np.complex128)
        s_matrix[:, 1, 0] = np.array([0.9, 0.8, 0.7])  # Linear decrease

        s_params = SParameterData(
            frequencies=frequencies,
            s_matrix=s_matrix,
            n_ports=2,
        )

        # Interpolate at midpoint
        s21_mid = s_params.get_s(2, 1, frequency=1.5e9)

        # Should be between 0.9 and 0.8
        assert 0.79 <= np.real(s21_mid) <= 0.91


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-001")
class TestTouchstoneLoading:
    """Test Touchstone file loading."""

    def test_load_touchstone_s2p(self, tmp_path: Path) -> None:
        """Test loading a valid S2P Touchstone file."""
        from oscura.analyzers.signal_integrity.sparams import load_touchstone

        # Create a simple S2P file
        s2p_content = """! Simple test S2P file
# GHz S RI R 50
1.0  0.1 0.0  0.9 0.0  0.9 0.0  0.1 0.0
2.0  0.15 0.0  0.85 0.0  0.85 0.0  0.15 0.0
3.0  0.2 0.0  0.8 0.0  0.8 0.0  0.2 0.0
"""
        s2p_path = tmp_path / "test.s2p"
        s2p_path.write_text(s2p_content)

        s_params = load_touchstone(s2p_path)

        assert s_params.n_ports == 2
        assert len(s_params.frequencies) == 3
        assert s_params.z0 == 50.0
        # Check first frequency is 1 GHz
        assert np.isclose(s_params.frequencies[0], 1e9)

    def test_load_touchstone_ma_format(self, tmp_path: Path) -> None:
        """Test loading Touchstone file with MA (magnitude/angle) format."""
        from oscura.analyzers.signal_integrity.sparams import load_touchstone

        # Create S2P file with MA format
        s2p_content = """! Test S2P file with MA format
# MHz S MA R 50
1000  0.1 0  0.9 -10  0.9 -10  0.1 0
2000  0.15 -5  0.85 -20  0.85 -20  0.15 -5
"""
        s2p_path = tmp_path / "test_ma.s2p"
        s2p_path.write_text(s2p_content)

        s_params = load_touchstone(s2p_path)

        assert s_params.n_ports == 2
        assert s_params.format == "ma"
        # First frequency should be 1000 MHz = 1 GHz
        assert np.isclose(s_params.frequencies[0], 1e9)

    def test_load_touchstone_db_format(self, tmp_path: Path) -> None:
        """Test loading Touchstone file with dB format."""
        from oscura.analyzers.signal_integrity.sparams import load_touchstone

        s2p_content = """! Test S2P file with dB format
# GHz S DB R 50
1.0  -20 0  -1 -10  -1 -10  -20 0
2.0  -18 -5  -2 -20  -2 -20  -18 -5
"""
        s2p_path = tmp_path / "test_db.s2p"
        s2p_path.write_text(s2p_content)

        s_params = load_touchstone(s2p_path)

        assert s_params.n_ports == 2
        assert s_params.format == "db"
        # -20 dB should be about 0.1 magnitude
        s11_mag = np.abs(s_params.s_matrix[0, 0, 0])
        assert np.isclose(s11_mag, 0.1, rtol=0.01)

    def test_load_touchstone_file_not_found(self) -> None:
        """Test that missing file raises LoaderError."""
        from oscura.analyzers.signal_integrity.sparams import load_touchstone
        from oscura.core.exceptions import LoaderError

        with pytest.raises(LoaderError, match="File not found"):
            load_touchstone("/nonexistent/path/file.s2p")

    def test_load_touchstone_unsupported_extension(self, tmp_path: Path) -> None:
        """Test that unsupported extension raises FormatError."""
        from oscura.analyzers.signal_integrity.sparams import load_touchstone
        from oscura.core.exceptions import FormatError

        bad_path = tmp_path / "test.txt"
        bad_path.write_text("not a touchstone file")

        with pytest.raises(FormatError, match="Unsupported file extension"):
            load_touchstone(bad_path)

    def test_load_touchstone_s1p(self, tmp_path: Path) -> None:
        """Test loading 1-port Touchstone file."""
        from oscura.analyzers.signal_integrity.sparams import load_touchstone

        s1p_content = """! 1-port test file
# GHz S RI R 50
1.0  0.1 0.0
2.0  0.15 -0.05
"""
        s1p_path = tmp_path / "test.s1p"
        s1p_path.write_text(s1p_content)

        s_params = load_touchstone(s1p_path)

        assert s_params.n_ports == 1
        assert len(s_params.frequencies) == 2

    def test_load_touchstone_preserves_comments(self, tmp_path: Path) -> None:
        """Test that comments are preserved from Touchstone file."""
        from oscura.analyzers.signal_integrity.sparams import load_touchstone

        s2p_content = """! Comment line 1
! Comment line 2
# GHz S RI R 50
1.0  0.1 0.0  0.9 0.0  0.9 0.0  0.1 0.0
"""
        s2p_path = tmp_path / "test_comments.s2p"
        s2p_path.write_text(s2p_content)

        s_params = load_touchstone(s2p_path)

        assert len(s_params.comments) == 2
        assert "Comment line 1" in s_params.comments[0]


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-007")
class TestReturnLoss:
    """Test return loss calculation."""

    def test_return_loss_all_frequencies(self) -> None:
        """Test return loss calculation for all frequencies."""
        from oscura.analyzers.signal_integrity.sparams import return_loss

        s_params = make_s_parameter_data(n_freq=50)

        rl = return_loss(s_params)

        assert isinstance(rl, np.ndarray)
        assert len(rl) == 50
        # Return loss should be positive (in dB)
        assert np.all(rl > 0)

    def test_return_loss_single_frequency(self) -> None:
        """Test return loss at specific frequency."""
        from oscura.analyzers.signal_integrity.sparams import return_loss

        s_params = make_s_parameter_data(n_freq=100)

        rl = return_loss(s_params, frequency=5e9)

        assert isinstance(rl, float)
        assert rl > 0  # Return loss in dB is positive

    def test_return_loss_different_ports(self) -> None:
        """Test return loss at different ports."""
        from oscura.analyzers.signal_integrity.sparams import return_loss

        s_params = make_s_parameter_data(n_freq=50)

        rl_port1 = return_loss(s_params, port=1)
        rl_port2 = return_loss(s_params, port=2)

        # Both should be valid arrays
        assert len(rl_port1) == 50
        assert len(rl_port2) == 50

    def test_return_loss_handles_zero(self) -> None:
        """Test return loss handles zero magnitude gracefully."""
        from oscura.analyzers.signal_integrity.sparams import SParameterData, return_loss

        # Create S-parameters with zero S11
        frequencies = np.array([1e9, 2e9])
        s_matrix = np.zeros((2, 2, 2), dtype=np.complex128)
        s_matrix[:, 1, 0] = 0.9  # S21 is non-zero

        s_params = SParameterData(
            frequencies=frequencies,
            s_matrix=s_matrix,
            n_ports=2,
        )

        # Should not raise an error (uses regularization)
        rl = return_loss(s_params)
        assert np.all(np.isfinite(rl))


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-008")
class TestInsertionLoss:
    """Test insertion loss calculation."""

    def test_insertion_loss_all_frequencies(self) -> None:
        """Test insertion loss for all frequencies."""
        from oscura.analyzers.signal_integrity.sparams import insertion_loss

        s_params = make_s_parameter_data(n_freq=50)

        il = insertion_loss(s_params)

        assert isinstance(il, np.ndarray)
        assert len(il) == 50
        # Insertion loss is typically positive in dB
        assert np.all(np.isfinite(il))

    def test_insertion_loss_single_frequency(self) -> None:
        """Test insertion loss at specific frequency."""
        from oscura.analyzers.signal_integrity.sparams import insertion_loss

        s_params = make_s_parameter_data(n_freq=100)

        il = insertion_loss(s_params, frequency=5e9)

        assert isinstance(il, float)
        assert np.isfinite(il)

    def test_insertion_loss_increases_with_frequency(self) -> None:
        """Test that insertion loss increases with frequency (typical behavior)."""
        from oscura.analyzers.signal_integrity.sparams import insertion_loss

        s_params = make_s_parameter_data(n_freq=100)

        il = insertion_loss(s_params)

        # Our synthetic data has S21 decreasing with frequency
        # So insertion loss should generally increase
        # Check that the last value is greater than the first
        assert il[-1] > il[0]


@pytest.mark.unit
@pytest.mark.analyzer
class TestSToABCDConversion:
    """Test S-parameter to ABCD matrix conversion."""

    def test_s_to_abcd_single(self) -> None:
        """Test S to ABCD conversion for single frequency."""
        from oscura.analyzers.signal_integrity.sparams import s_to_abcd

        s_params = make_s_parameter_data(n_freq=10)

        abcd = s_to_abcd(s_params, frequency_idx=0)

        assert abcd.shape == (2, 2)
        assert abcd.dtype == np.complex128

    def test_s_to_abcd_all_frequencies(self) -> None:
        """Test S to ABCD conversion for all frequencies."""
        from oscura.analyzers.signal_integrity.sparams import s_to_abcd

        s_params = make_s_parameter_data(n_freq=50)

        abcd = s_to_abcd(s_params)

        assert abcd.shape == (50, 2, 2)
        assert np.all(np.isfinite(abcd))

    def test_s_to_abcd_requires_2port(self) -> None:
        """Test that S to ABCD conversion requires 2-port network."""
        from oscura.analyzers.signal_integrity.sparams import (
            SParameterData,
            s_to_abcd,
        )

        # Create 1-port network
        frequencies = np.array([1e9, 2e9])
        s_matrix = np.zeros((2, 1, 1), dtype=np.complex128)
        s_matrix[:, 0, 0] = 0.1

        s_params = SParameterData(
            frequencies=frequencies,
            s_matrix=s_matrix,
            n_ports=1,
        )

        with pytest.raises(ValueError, match="2-port"):
            s_to_abcd(s_params)


@pytest.mark.unit
@pytest.mark.analyzer
class TestABCDToSConversion:
    """Test ABCD to S-parameter conversion."""

    def test_abcd_to_s_single(self) -> None:
        """Test ABCD to S conversion for single matrix."""
        from oscura.analyzers.signal_integrity.sparams import abcd_to_s

        # Identity ABCD matrix (through connection)
        abcd = np.array([[1, 0], [0, 1]], dtype=np.complex128)

        s = abcd_to_s(abcd, z0=50.0)

        assert s.shape == (2, 2)
        # Through connection should have S11=S22=0, S12=S21=1
        assert np.isclose(np.abs(s[0, 0]), 0, atol=1e-10)
        assert np.isclose(np.abs(s[1, 0]), 1, atol=1e-10)

    def test_abcd_to_s_array(self) -> None:
        """Test ABCD to S conversion for array of matrices."""
        from oscura.analyzers.signal_integrity.sparams import abcd_to_s

        n_freq = 10
        abcd = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        abcd[:, 0, 0] = 1
        abcd[:, 1, 1] = 1  # Identity matrices

        s = abcd_to_s(abcd, z0=50.0)

        assert s.shape == (n_freq, 2, 2)
        assert np.all(np.isfinite(s))

    def test_s_to_abcd_roundtrip(self) -> None:
        """Test that S -> ABCD -> S returns original values."""
        from oscura.analyzers.signal_integrity.sparams import abcd_to_s, s_to_abcd

        s_params = make_s_parameter_data(n_freq=10)
        original_s = s_params.s_matrix

        abcd = s_to_abcd(s_params)
        recovered_s = abcd_to_s(abcd, z0=s_params.z0)

        # Should be close to original (within numerical precision)
        np.testing.assert_allclose(recovered_s, original_s, rtol=1e-6, atol=1e-10)


# =============================================================================
# Embedding/De-embedding Tests (embedding.py)
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-002")
class TestDeembed:
    """Test de-embedding functionality."""

    def test_deembed_frequency_domain(self) -> None:
        """Test frequency domain de-embedding."""
        from oscura.analyzers.signal_integrity.embedding import deembed

        # Create a simple test signal
        n_samples = 1000
        sample_rate = 10e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)  # 1 GHz sine wave

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        s_params = make_s_parameter_data(n_freq=100)

        result = deembed(trace, s_params, method="frequency_domain")

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == n_samples
        assert result.metadata.sample_rate == sample_rate

    def test_deembed_time_domain(self) -> None:
        """Test time domain de-embedding."""
        from oscura.analyzers.signal_integrity.embedding import deembed

        n_samples = 1000
        sample_rate = 10e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        s_params = make_s_parameter_data(n_freq=100)

        result = deembed(trace, s_params, method="time_domain")

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == n_samples

    def test_deembed_invalid_method(self) -> None:
        """Test that invalid method raises ValueError."""
        from oscura.analyzers.signal_integrity.embedding import deembed

        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        trace = make_waveform_trace(signal, sample_rate=1e9)
        s_params = make_s_parameter_data(n_freq=50)

        with pytest.raises(ValueError, match="Unknown method"):
            deembed(trace, s_params, method="invalid_method")

    def test_deembed_requires_2port(self) -> None:
        """Test that de-embedding requires 2-port S-parameters."""
        from oscura.analyzers.signal_integrity.embedding import deembed
        from oscura.analyzers.signal_integrity.sparams import SParameterData
        from oscura.core.exceptions import AnalysisError

        # Create 1-port S-parameters
        frequencies = np.array([1e9, 2e9, 3e9])
        s_matrix = np.zeros((3, 1, 1), dtype=np.complex128)

        s_params = SParameterData(
            frequencies=frequencies,
            s_matrix=s_matrix,
            n_ports=1,
        )

        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        trace = make_waveform_trace(signal, sample_rate=1e9)

        with pytest.raises(AnalysisError, match="2-port"):
            deembed(trace, s_params)


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-003")
class TestEmbed:
    """Test channel embedding functionality."""

    def test_embed_basic(self) -> None:
        """Test basic channel embedding."""
        from oscura.analyzers.signal_integrity.embedding import embed

        n_samples = 1000
        sample_rate = 10e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        s_params = make_s_parameter_data(n_freq=100)

        result = embed(trace, s_params)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == n_samples
        assert result.metadata.sample_rate == sample_rate

    def test_embed_attenuates_signal(self) -> None:
        """Test that embedding reduces signal amplitude (typical behavior)."""
        from oscura.analyzers.signal_integrity.embedding import embed

        n_samples = 1000
        sample_rate = 10e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        # Use S-parameters with significant loss
        s_params = make_s_parameter_data(n_freq=100)

        result = embed(trace, s_params)

        # RMS of embedded signal should be less than original
        original_rms = np.sqrt(np.mean(signal**2))
        embedded_rms = np.sqrt(np.mean(result.data**2))

        assert embedded_rms < original_rms

    def test_embed_requires_2port(self) -> None:
        """Test that embedding requires 2-port S-parameters."""
        from oscura.analyzers.signal_integrity.embedding import embed
        from oscura.analyzers.signal_integrity.sparams import SParameterData
        from oscura.core.exceptions import AnalysisError

        frequencies = np.array([1e9, 2e9, 3e9])
        s_matrix = np.zeros((3, 1, 1), dtype=np.complex128)

        s_params = SParameterData(
            frequencies=frequencies,
            s_matrix=s_matrix,
            n_ports=1,
        )

        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        trace = make_waveform_trace(signal, sample_rate=1e9)

        with pytest.raises(AnalysisError, match="2-port"):
            embed(trace, s_params)


@pytest.mark.unit
@pytest.mark.analyzer
class TestCascadeDeembed:
    """Test cascade de-embedding functionality."""

    def test_cascade_deembed_empty_list(self) -> None:
        """Test cascade de-embedding with empty fixture list."""
        from oscura.analyzers.signal_integrity.embedding import cascade_deembed

        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        trace = make_waveform_trace(signal, sample_rate=1e9)

        result = cascade_deembed(trace, [])

        # With no fixtures, should return original trace
        np.testing.assert_array_equal(result.data, trace.data)

    def test_cascade_deembed_single_fixture(self) -> None:
        """Test cascade de-embedding with single fixture."""
        from oscura.analyzers.signal_integrity.embedding import cascade_deembed

        n_samples = 1000
        sample_rate = 10e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        fixture = make_s_parameter_data(n_freq=100)

        result = cascade_deembed(trace, [fixture])

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == n_samples

    def test_cascade_deembed_multiple_fixtures(self) -> None:
        """Test cascade de-embedding with multiple fixtures."""
        from oscura.analyzers.signal_integrity.embedding import cascade_deembed

        n_samples = 1000
        sample_rate = 10e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        fixture1 = make_s_parameter_data(n_freq=100)
        fixture2 = make_s_parameter_data(n_freq=100)

        result = cascade_deembed(trace, [fixture1, fixture2])

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == n_samples


# =============================================================================
# Equalization Tests (equalization.py)
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-004")
class TestFFEEqualize:
    """Test Feed-Forward Equalization."""

    def test_ffe_equalize_basic(self) -> None:
        """Test basic FFE equalization."""
        from oscura.analyzers.signal_integrity.equalization import ffe_equalize

        # Create a simple test signal with ISI
        n_samples = 1000
        signal = np.zeros(n_samples)
        # Add impulses every 100 samples
        signal[::100] = 1.0

        trace = make_waveform_trace(signal, sample_rate=1e9)
        taps = [-0.1, 1.0, -0.1]  # 3-tap equalizer

        result = ffe_equalize(trace, taps)

        assert result.equalized_data is not None
        assert len(result.equalized_data) == n_samples
        assert len(result.taps) == 3
        assert result.n_precursor == 1
        assert result.n_postcursor == 1

    def test_ffe_equalize_preserves_energy(self) -> None:
        """Test that FFE approximately preserves signal energy."""
        from oscura.analyzers.signal_integrity.equalization import ffe_equalize

        # Create a signal
        n_samples = 1000
        np.random.seed(42)
        signal = np.random.randn(n_samples)

        trace = make_waveform_trace(signal, sample_rate=1e9)
        # Unity main cursor with small pre/post cursors
        taps = [-0.05, 1.0, -0.05]

        result = ffe_equalize(trace, taps)

        original_energy = np.sum(signal**2)
        equalized_energy = np.sum(result.equalized_data**2)

        # Energy should be similar (within 50%)
        assert equalized_energy > 0.5 * original_energy
        assert equalized_energy < 2.0 * original_energy

    def test_ffe_equalize_tap_array(self) -> None:
        """Test FFE with numpy array taps."""
        from oscura.analyzers.signal_integrity.equalization import ffe_equalize

        signal = np.sin(np.linspace(0, 10 * np.pi, 500))
        trace = make_waveform_trace(signal, sample_rate=1e9)

        taps = np.array([-0.1, -0.05, 1.0, -0.05, -0.1])  # 5-tap

        result = ffe_equalize(trace, taps)

        assert result.n_precursor == 2
        assert result.n_postcursor == 2


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-004")
class TestOptimizeFFE:
    """Test FFE tap optimization."""

    def test_optimize_ffe_basic(self) -> None:
        """Test basic FFE optimization."""
        from oscura.analyzers.signal_integrity.equalization import optimize_ffe

        # Create a signal with known ISI
        n_samples = 500
        np.random.seed(42)
        bits = np.random.choice([-1, 1], size=n_samples // 10)
        signal = np.repeat(bits, 10).astype(np.float64)

        # Add some ISI
        h = [0.1, 1.0, 0.3, 0.1]
        signal = np.convolve(signal, h, mode="same")

        trace = make_waveform_trace(signal, sample_rate=1e9)

        result = optimize_ffe(trace, n_taps=5, n_precursor=1)

        assert result.equalized_data is not None
        assert result.taps is not None
        assert result.mse is not None
        assert len(result.taps) == 5
        assert result.n_precursor == 1

    def test_optimize_ffe_with_target(self) -> None:
        """Test FFE optimization with explicit target."""
        from oscura.analyzers.signal_integrity.equalization import optimize_ffe

        n_samples = 500
        np.random.seed(42)
        target = np.random.choice([-1.0, 1.0], size=n_samples)

        # Create degraded signal
        h = [0.1, 1.0, 0.2]
        signal = np.convolve(target, h, mode="same")

        trace = make_waveform_trace(signal, sample_rate=1e9)

        result = optimize_ffe(trace, n_taps=5, target=target)

        assert result.mse is not None
        # MSE should be reasonably low since we're trying to recover target
        # (not necessarily very low due to optimization constraints)


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-005")
class TestDFEEqualize:
    """Test Decision Feedback Equalization."""

    def test_dfe_equalize_basic(self) -> None:
        """Test basic DFE equalization."""
        from oscura.analyzers.signal_integrity.equalization import dfe_equalize

        # Create a binary signal
        n_samples = 500
        np.random.seed(42)
        signal = np.random.choice([-1.0, 1.0], size=n_samples)

        trace = make_waveform_trace(signal, sample_rate=1e9)
        taps = [0.2, 0.1]  # 2-tap DFE

        result = dfe_equalize(trace, taps)

        assert result.equalized_data is not None
        assert result.decisions is not None
        assert len(result.taps) == 2
        assert result.n_taps == 2

    def test_dfe_equalize_makes_decisions(self) -> None:
        """Test that DFE makes bit decisions."""
        from oscura.analyzers.signal_integrity.equalization import dfe_equalize

        n_samples = 100
        # Create clean binary signal
        signal = np.array([1.0, -1.0, 1.0, 1.0, -1.0] * 20, dtype=np.float64)

        trace = make_waveform_trace(signal, sample_rate=1e9)
        taps = [0.1]  # 1-tap DFE

        result = dfe_equalize(trace, taps, samples_per_symbol=1)

        # Decisions should be 0 or 1
        assert np.all((result.decisions == 0) | (result.decisions == 1))

    def test_dfe_equalize_custom_threshold(self) -> None:
        """Test DFE with custom decision threshold."""
        from oscura.analyzers.signal_integrity.equalization import dfe_equalize

        n_samples = 100
        # Signal with DC offset
        signal = np.random.choice([0.5, 1.5], size=n_samples).astype(np.float64)

        trace = make_waveform_trace(signal, sample_rate=1e9)
        taps = [0.1]

        result = dfe_equalize(trace, taps, threshold=1.0)

        assert result.equalized_data is not None
        assert result.decisions is not None

    def test_dfe_equalize_samples_per_symbol(self) -> None:
        """Test DFE with oversampled data."""
        from oscura.analyzers.signal_integrity.equalization import dfe_equalize

        # 4x oversampled signal
        n_symbols = 50
        samples_per_symbol = 4
        symbol_values = np.random.choice([-1.0, 1.0], size=n_symbols)
        signal = np.repeat(symbol_values, samples_per_symbol)

        trace = make_waveform_trace(signal, sample_rate=4e9)
        taps = [0.1, 0.05]

        result = dfe_equalize(trace, taps, samples_per_symbol=samples_per_symbol)

        # Should have n_symbols decisions
        assert len(result.decisions) <= n_symbols


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-006")
class TestCTLEEqualize:
    """Test Continuous Time Linear Equalization."""

    def test_ctle_equalize_basic(self) -> None:
        """Test basic CTLE equalization."""
        from oscura.analyzers.signal_integrity.equalization import ctle_equalize

        n_samples = 1000
        sample_rate = 20e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)

        trace = make_waveform_trace(signal, sample_rate=sample_rate)

        result = ctle_equalize(
            trace,
            dc_gain=0.0,
            ac_gain=6.0,
            pole_frequency=5e9,
        )

        assert result.equalized_data is not None
        assert len(result.equalized_data) == n_samples
        assert result.dc_gain == 0.0
        assert result.ac_gain == 6.0
        assert result.pole_frequency == 5e9

    def test_ctle_equalize_boost_calculation(self) -> None:
        """Test CTLE boost calculation."""
        from oscura.analyzers.signal_integrity.equalization import ctle_equalize

        signal = np.sin(np.linspace(0, 20 * np.pi, 1000))
        trace = make_waveform_trace(signal, sample_rate=20e9)

        result = ctle_equalize(
            trace,
            dc_gain=-3.0,
            ac_gain=3.0,
            pole_frequency=5e9,
        )

        # Boost should be ac_gain - dc_gain
        expected_boost = 3.0 - (-3.0)
        assert result.boost == expected_boost

    def test_ctle_equalize_custom_zero(self) -> None:
        """Test CTLE with custom zero frequency."""
        from oscura.analyzers.signal_integrity.equalization import ctle_equalize

        signal = np.sin(np.linspace(0, 20 * np.pi, 1000))
        trace = make_waveform_trace(signal, sample_rate=20e9)

        result = ctle_equalize(
            trace,
            dc_gain=0.0,
            ac_gain=6.0,
            pole_frequency=5e9,
            zero_frequency=2e9,
        )

        assert result.zero_frequency == 2e9

    def test_ctle_equalize_computes_zero_frequency(self) -> None:
        """Test that CTLE computes zero frequency when not specified."""
        from oscura.analyzers.signal_integrity.equalization import ctle_equalize

        signal = np.sin(np.linspace(0, 20 * np.pi, 1000))
        trace = make_waveform_trace(signal, sample_rate=20e9)

        result = ctle_equalize(
            trace,
            dc_gain=0.0,
            ac_gain=6.0,
            pole_frequency=5e9,
        )

        # Zero frequency should be computed
        assert result.zero_frequency is not None
        assert result.zero_frequency > 0


@pytest.mark.unit
@pytest.mark.analyzer
class TestEqualizationResults:
    """Test equalization result dataclasses."""

    def test_ffe_result_attributes(self) -> None:
        """Test FFEResult dataclass attributes."""
        from oscura.analyzers.signal_integrity.equalization import FFEResult

        result = FFEResult(
            equalized_data=np.array([1.0, 2.0, 3.0]),
            taps=np.array([-0.1, 1.0, -0.1]),
            n_precursor=1,
            n_postcursor=1,
            mse=0.01,
        )

        assert len(result.equalized_data) == 3
        assert len(result.taps) == 3
        assert result.n_precursor == 1
        assert result.n_postcursor == 1
        assert result.mse == 0.01

    def test_dfe_result_attributes(self) -> None:
        """Test DFEResult dataclass attributes."""
        from oscura.analyzers.signal_integrity.equalization import DFEResult

        result = DFEResult(
            equalized_data=np.array([1.0, -1.0, 1.0]),
            taps=np.array([0.2, 0.1]),
            decisions=np.array([1, 0, 1]),
            n_taps=2,
            error_count=0,
        )

        assert len(result.equalized_data) == 3
        assert len(result.taps) == 2
        assert len(result.decisions) == 3
        assert result.n_taps == 2
        assert result.error_count == 0

    def test_ctle_result_attributes(self) -> None:
        """Test CTLEResult dataclass attributes."""
        from oscura.analyzers.signal_integrity.equalization import CTLEResult

        result = CTLEResult(
            equalized_data=np.array([1.0, 2.0, 3.0]),
            dc_gain=0.0,
            ac_gain=6.0,
            pole_frequency=5e9,
            zero_frequency=2e9,
            boost=6.0,
        )

        assert len(result.equalized_data) == 3
        assert result.dc_gain == 0.0
        assert result.ac_gain == 6.0
        assert result.pole_frequency == 5e9
        assert result.zero_frequency == 2e9
        assert result.boost == 6.0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestSignalIntegrityEdgeCases:
    """Test edge cases in signal integrity modules."""

    def test_empty_signal_ffe(self) -> None:
        """Test FFE with empty signal raises ValueError."""
        from oscura.analyzers.signal_integrity.equalization import ffe_equalize

        signal = np.array([], dtype=np.float64)
        trace = make_waveform_trace(signal, sample_rate=1e9)
        taps = [1.0]

        # numpy convolve raises ValueError for empty arrays
        with pytest.raises(ValueError):
            ffe_equalize(trace, taps)

    def test_single_sample_ffe(self) -> None:
        """Test FFE with single sample."""
        from oscura.analyzers.signal_integrity.equalization import ffe_equalize

        signal = np.array([1.0])
        trace = make_waveform_trace(signal, sample_rate=1e9)
        taps = [1.0]

        result = ffe_equalize(trace, taps)

        assert len(result.equalized_data) == 1

    def test_very_short_signal_embed(self) -> None:
        """Test embedding with very short signal."""
        from oscura.analyzers.signal_integrity.embedding import embed

        signal = np.array([1.0, -1.0, 1.0])
        trace = make_waveform_trace(signal, sample_rate=10e9)
        s_params = make_s_parameter_data(n_freq=50)

        result = embed(trace, s_params)

        assert len(result.data) == 3

    def test_high_frequency_content_deembed(self) -> None:
        """Test de-embedding with high frequency content."""
        from oscura.analyzers.signal_integrity.embedding import deembed

        n_samples = 1000
        sample_rate = 50e9
        t = np.arange(n_samples) / sample_rate
        # Signal with multiple frequency components
        signal = (
            np.sin(2 * np.pi * 1e9 * t)
            + 0.5 * np.sin(2 * np.pi * 5e9 * t)
            + 0.25 * np.sin(2 * np.pi * 10e9 * t)
        )

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        s_params = make_s_parameter_data(n_freq=200)

        result = deembed(trace, s_params)

        assert len(result.data) == n_samples
        assert np.all(np.isfinite(result.data))

    def test_single_tap_ffe(self) -> None:
        """Test FFE with single tap (identity)."""
        from oscura.analyzers.signal_integrity.equalization import ffe_equalize

        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        trace = make_waveform_trace(signal, sample_rate=1e9)
        taps = [1.0]  # Identity filter

        result = ffe_equalize(trace, taps)

        # With identity tap, output should be very close to input
        np.testing.assert_allclose(result.equalized_data, signal, rtol=1e-10)

    def test_large_tap_count_ffe(self) -> None:
        """Test FFE with large number of taps."""
        from oscura.analyzers.signal_integrity.equalization import ffe_equalize

        signal = np.random.randn(500)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        # 21-tap equalizer
        taps = np.zeros(21)
        taps[10] = 1.0  # Main cursor in middle
        taps[8:13] = [-0.05, -0.1, 1.0, -0.1, -0.05]

        result = ffe_equalize(trace, taps)

        assert result.n_precursor == 10
        assert result.n_postcursor == 10
        assert len(result.equalized_data) == 500
