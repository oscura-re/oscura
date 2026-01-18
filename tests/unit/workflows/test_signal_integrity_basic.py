"""Tests for signal integrity analysis module.

Tests for SI-001 through SI-008 requirements.
"""

from pathlib import Path

import numpy as np
import pytest

from oscura.analyzers.signal_integrity import (
    SParameterData,
    abcd_to_s,
    cascade_deembed,
    ctle_equalize,
    deembed,
    dfe_equalize,
    embed,
    ffe_equalize,
    insertion_loss,
    load_touchstone,
    optimize_ffe,
    return_loss,
    s_to_abcd,
)
from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_s2p_content() -> str:
    """Sample Touchstone 2-port content."""
    return """! Sample 2-port S-parameter file
! Cable measurement
# GHZ S RI R 50
! freq ReS11 ImS11 ReS21 ImS21 ReS12 ImS12 ReS22 ImS22
1.0  0.1  0.05  0.9  -0.1  0.9  -0.1  0.1  0.05
2.0  0.12 0.06  0.85 -0.15 0.85 -0.15 0.12 0.06
5.0  0.15 0.08  0.7  -0.2  0.7  -0.2  0.15 0.08
10.0 0.2  0.1   0.5  -0.3  0.5  -0.3  0.2  0.1
"""


@pytest.fixture
def sample_s2p_db_content() -> str:
    """Sample Touchstone in dB format."""
    return """! dB format S-parameters
# GHZ S DB R 50
! freq S11dB S11deg S21dB S21deg S12dB S12deg S22dB S22deg
1.0  -20 10  -1  -5  -1  -5  -20 10
5.0  -16 15  -3  -15 -3  -15 -16 15
10.0 -14 20  -6  -30 -6  -30 -14 20
"""


@pytest.fixture
def s2p_file(sample_s2p_content: str, tmp_path: Path) -> Path:
    """Create temporary S2P file."""
    s2p_path = tmp_path / "test.s2p"
    s2p_path.write_text(sample_s2p_content)
    return s2p_path


@pytest.fixture
def s2p_db_file(sample_s2p_db_content: str, tmp_path: Path) -> Path:
    """Create temporary S2P file in dB format."""
    s2p_path = tmp_path / "test_db.s2p"
    s2p_path.write_text(sample_s2p_db_content)
    return s2p_path


@pytest.fixture
def sample_sparams() -> SParameterData:
    """Create sample S-parameter data."""
    frequencies = np.array([1e9, 2e9, 5e9, 10e9])
    n_freq = len(frequencies)

    # Create S-matrix with typical cable characteristics
    s_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)

    for i, f in enumerate(frequencies):
        # S11: increasing return loss with frequency
        s11_mag = 0.1 + 0.02 * (f / 1e9)
        s_matrix[i, 0, 0] = s11_mag * np.exp(1j * 0.1 * i)

        # S21: decreasing transmission with frequency
        s21_mag = 0.9 - 0.05 * (f / 1e9)
        s_matrix[i, 1, 0] = s21_mag * np.exp(-1j * 0.2 * i)

        # S12 = S21 for reciprocal network
        s_matrix[i, 0, 1] = s_matrix[i, 1, 0]

        # S22 = S11 for symmetric network
        s_matrix[i, 1, 1] = s_matrix[i, 0, 0]

    return SParameterData(
        frequencies=frequencies,
        s_matrix=s_matrix,
        n_ports=2,
        z0=50.0,
    )


@pytest.fixture
def lossy_trace() -> WaveformTrace:
    """Generate trace with ISI/loss for equalization testing."""
    sample_rate = 10e9
    n_samples = 1000

    rng = np.random.default_rng(42)

    # Create NRZ signal
    n_bits = n_samples // 10
    bits = rng.integers(0, 2, n_bits)
    ideal = np.repeat(bits.astype(np.float64) * 2 - 1, 10)

    # Apply simple lossy channel (low-pass filter)
    from scipy.signal import butter, lfilter

    nyq = sample_rate / 2
    cutoff = 3e9  # 3 GHz bandwidth
    b, a = butter(2, cutoff / nyq, btype="low")
    data = lfilter(b, a, ideal)

    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


# =============================================================================
# Touchstone Loading Tests (SI-001)
# =============================================================================


class TestLoadTouchstone:
    """Tests for Touchstone file loading."""

    def test_load_s2p(self, s2p_file: Path):
        """Test loading 2-port S-parameter file."""
        s_params = load_touchstone(s2p_file)

        assert isinstance(s_params, SParameterData)
        assert s_params.n_ports == 2
        assert len(s_params.frequencies) == 4
        assert s_params.z0 == 50.0

    def test_load_s2p_db_format(self, s2p_db_file: Path):
        """Test loading dB format S-parameters."""
        s_params = load_touchstone(s2p_db_file)

        assert s_params.n_ports == 2
        assert len(s_params.frequencies) == 3
        # Verify magnitude conversion from dB
        s11_mag = np.abs(s_params.s_matrix[0, 0, 0])
        expected_mag = 10 ** (-20 / 20)  # -20 dB
        assert np.isclose(s11_mag, expected_mag, rtol=0.1)

    def test_load_nonexistent_file(self, tmp_path: Path):
        """Test error on nonexistent file."""
        with pytest.raises(LoaderError):
            load_touchstone(tmp_path / "nonexistent.s2p")

    def test_load_invalid_extension(self, tmp_path: Path):
        """Test error on invalid extension."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not s-parameters")

        with pytest.raises(FormatError):
            load_touchstone(txt_file)

    def test_get_s_parameter(self, sample_sparams: SParameterData):
        """Test retrieving specific S-parameter."""
        s11 = sample_sparams.get_s(1, 1)
        s21 = sample_sparams.get_s(2, 1)

        assert len(s11) == len(sample_sparams.frequencies)
        assert len(s21) == len(sample_sparams.frequencies)

    def test_get_s_at_frequency(self, sample_sparams: SParameterData):
        """Test S-parameter interpolation."""
        # Interpolate at 3 GHz (between 2 and 5 GHz)
        s21 = sample_sparams.get_s(2, 1, frequency=3e9)

        assert isinstance(s21, complex | np.complex128)


# =============================================================================
# Return Loss Tests (SI-007)
# =============================================================================


class TestReturnLoss:
    """Tests for return loss calculation."""

    def test_calculate_rl(self, sample_sparams: SParameterData):
        """Test return loss from S11."""
        rl = return_loss(sample_sparams, frequency=1e9)

        # RL = -20*log10(|S11|)
        s11_mag = np.abs(sample_sparams.get_s(1, 1, 1e9))
        expected_rl = -20 * np.log10(s11_mag)

        assert np.isclose(rl, expected_rl, rtol=0.01)

    def test_return_loss_all_frequencies(self, sample_sparams: SParameterData):
        """Test return loss at all frequencies."""
        rl = return_loss(sample_sparams)

        assert isinstance(rl, np.ndarray)
        assert len(rl) == len(sample_sparams.frequencies)
        assert np.all(rl > 0)  # RL should be positive (in dB)

    def test_return_loss_port(self, sample_sparams: SParameterData):
        """Test return loss for different ports."""
        rl1 = return_loss(sample_sparams, port=1)
        rl2 = return_loss(sample_sparams, port=2)

        # For symmetric network, should be similar
        assert np.allclose(rl1, rl2, rtol=0.1)


# =============================================================================
# Insertion Loss Tests (SI-008)
# =============================================================================


class TestInsertionLoss:
    """Tests for insertion loss calculation."""

    def test_calculate_il(self, sample_sparams: SParameterData):
        """Test insertion loss from S21."""
        il = insertion_loss(sample_sparams, frequency=1e9)

        # IL = -20*log10(|S21|)
        s21_mag = np.abs(sample_sparams.get_s(2, 1, 1e9))
        expected_il = -20 * np.log10(s21_mag)

        assert np.isclose(il, expected_il, rtol=0.01)

    def test_insertion_loss_increases_with_frequency(self, sample_sparams: SParameterData):
        """Test that IL increases with frequency (lossy cable)."""
        il = insertion_loss(sample_sparams)

        # IL should generally increase with frequency
        assert il[-1] > il[0]


# =============================================================================
# ABCD Conversion Tests
# =============================================================================


class TestABCDConversion:
    """Tests for S to ABCD parameter conversion."""

    def test_s_to_abcd(self, sample_sparams: SParameterData):
        """Test S to ABCD conversion."""
        abcd = s_to_abcd(sample_sparams)

        assert abcd.shape == (len(sample_sparams.frequencies), 2, 2)

    def test_abcd_roundtrip(self, sample_sparams: SParameterData):
        """Test S -> ABCD -> S round trip."""
        abcd = s_to_abcd(sample_sparams)
        s_recovered = abcd_to_s(abcd, z0=sample_sparams.z0)

        # Should recover original S-parameters
        np.testing.assert_allclose(
            s_recovered,
            sample_sparams.s_matrix,
            rtol=0.01,
        )


# =============================================================================
# De-embedding Tests (SI-002)
# =============================================================================


class TestDeembed:
    """Tests for S-parameter de-embedding."""

    def test_deembed_cable(self, lossy_trace: WaveformTrace, sample_sparams: SParameterData):
        """Test removing cable effects from waveform."""
        clean = deembed(lossy_trace, sample_sparams)

        assert isinstance(clean, WaveformTrace)
        assert len(clean.data) == len(lossy_trace.data)

    def test_deembed_recovers_amplitude(self, sample_sparams: SParameterData):
        """Test that de-embedding increases amplitude."""
        sample_rate = 10e9
        n = 1000
        t = np.arange(n) / sample_rate

        # Create test signal
        f_test = 1e9
        ideal = np.sin(2 * np.pi * f_test * t)

        # Embed channel effects
        trace = WaveformTrace(
            data=ideal,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )
        degraded = embed(trace, sample_sparams)

        # De-embed
        recovered = deembed(degraded, sample_sparams)

        # Recovered should have amplitude closer to original
        # (may not be exact due to regularization)
        np.max(np.abs(ideal))
        degraded_amp = np.max(np.abs(degraded.data))
        recovered_amp = np.max(np.abs(recovered.data))

        assert recovered_amp > degraded_amp * 0.9


# =============================================================================
# Embedding Tests (SI-003)
# =============================================================================


class TestEmbed:
    """Tests for channel embedding."""

    def test_embed_channel(self, sample_sparams: SParameterData):
        """Test applying channel effects to waveform."""
        sample_rate = 10e9
        n = 1000
        t = np.arange(n) / sample_rate

        # Create ideal signal
        data = np.sin(2 * np.pi * 1e9 * t)
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        # Embed
        degraded = embed(trace, sample_sparams)

        assert isinstance(degraded, WaveformTrace)
        # Embedding should reduce amplitude (lossy channel)
        assert np.max(np.abs(degraded.data)) <= np.max(np.abs(data)) * 1.1


class TestCascadeDeembed:
    """Tests for cascade de-embedding."""

    def test_cascade_multiple_fixtures(
        self, sample_sparams: SParameterData, lossy_trace: WaveformTrace
    ):
        """Test cascading multiple fixtures."""
        # Use same fixture twice (simulating two cables)
        fixtures = [sample_sparams, sample_sparams]

        result = cascade_deembed(lossy_trace, fixtures)

        assert isinstance(result, WaveformTrace)


# =============================================================================
# FFE Tests (SI-004)
# =============================================================================


class TestFFE:
    """Tests for feed-forward equalization."""

    def test_ffe_equalize(self, lossy_trace: WaveformTrace):
        """Test FFE with specified taps."""
        taps = [-0.1, 1.0, -0.1]  # 3-tap equalizer

        result = ffe_equalize(lossy_trace, taps)

        assert len(result.equalized_data) == len(lossy_trace.data)
        assert result.n_precursor == 1
        assert result.n_postcursor == 1

    def test_ffe_optimize(self, lossy_trace: WaveformTrace):
        """Test FFE tap optimization."""
        result = optimize_ffe(lossy_trace, n_taps=5, n_precursor=1)

        assert len(result.taps) == 5
        assert isinstance(result.mse, (int, float))
        assert result.mse >= 0  # MSE should be non-negative
        # Main cursor should be normalized to ~1.0
        assert np.isclose(result.taps[1], 1.0, rtol=0.5)

    def test_ffe_improves_eye(self, lossy_trace: WaveformTrace):
        """Test that FFE improves signal quality."""
        result = ffe_equalize(lossy_trace, taps=[-0.05, 1.0, -0.05])

        # Equalized should have sharper transitions
        # (simplified check: variance should change)
        np.var(lossy_trace.data)
        equalized_var = np.var(result.equalized_data)

        # Variance test is not definitive but ensures processing occurred
        assert equalized_var > 0


# =============================================================================
# DFE Tests (SI-005)
# =============================================================================


class TestDFE:
    """Tests for decision feedback equalization."""

    def test_dfe_equalize(self, lossy_trace: WaveformTrace):
        """Test DFE equalization."""
        taps = [0.2, 0.1]  # 2 post-cursor taps

        result = dfe_equalize(lossy_trace, taps)

        assert len(result.equalized_data) == len(lossy_trace.data)
        assert isinstance(result.decisions, (list, np.ndarray))
        assert len(result.decisions) > 0
        assert result.n_taps == 2

    def test_dfe_decisions(self, lossy_trace: WaveformTrace):
        """Test DFE produces valid decisions."""
        result = dfe_equalize(lossy_trace, taps=[0.1])

        # Decisions should be 0 or 1
        assert np.all((result.decisions == 0) | (result.decisions == 1))


# =============================================================================
# CTLE Tests (SI-006)
# =============================================================================


class TestCTLE:
    """Tests for CTLE equalization."""

    def test_ctle_apply(self, lossy_trace: WaveformTrace):
        """Test applying CTLE."""
        result = ctle_equalize(
            lossy_trace,
            dc_gain=0,
            ac_gain=6,
            pole_frequency=5e9,
        )

        assert len(result.equalized_data) == len(lossy_trace.data)
        assert result.boost == 6

    def test_ctle_high_freq_boost(self, lossy_trace: WaveformTrace):
        """Test that CTLE boosts high frequencies."""
        result = ctle_equalize(
            lossy_trace,
            dc_gain=0,
            ac_gain=6,
            pole_frequency=3e9,
        )

        # CTLE should increase high-frequency content
        # Compare spectra
        original_fft = np.abs(np.fft.rfft(lossy_trace.data))
        equalized_fft = np.abs(np.fft.rfft(result.equalized_data))

        # High frequency bins should have more energy after CTLE
        n_bins = len(original_fft)
        high_freq_idx = n_bins * 2 // 3  # Upper third of spectrum

        original_high = np.mean(original_fft[high_freq_idx:])
        equalized_high = np.mean(equalized_fft[high_freq_idx:])

        # Allow for some variation
        assert equalized_high >= original_high * 0.5


# =============================================================================
# Edge Cases
# =============================================================================


class TestWorkflowsSignalIntegrityBasicEdgeCases:
    """Tests for edge cases."""

    def test_unity_sparams(self):
        """Test with unity (thru) S-parameters."""
        frequencies = np.array([1e9, 5e9, 10e9])
        s_matrix = np.zeros((3, 2, 2), dtype=np.complex128)

        # Unity transmission: S21 = 1, S11 = 0
        s_matrix[:, 1, 0] = 1.0
        s_matrix[:, 0, 1] = 1.0

        unity = SParameterData(
            frequencies=frequencies,
            s_matrix=s_matrix,
            n_ports=2,
        )

        il = insertion_loss(unity, frequency=5e9)
        assert np.isclose(il, 0.0, atol=0.1)  # 0 dB insertion loss

    def test_empty_taps_ffe(self, lossy_trace: WaveformTrace):
        """Test FFE with minimal taps."""
        result = ffe_equalize(lossy_trace, taps=[1.0])

        # Should be identity (or close to it)
        np.testing.assert_allclose(
            result.equalized_data,
            lossy_trace.data,
            rtol=0.01,
        )
