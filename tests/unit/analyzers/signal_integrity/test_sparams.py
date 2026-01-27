"""Unit tests for S-parameter handling and Touchstone file support.

This module tests the sparams.py module functionality:
- SParameterData dataclass
- Touchstone file loading
- Return loss and insertion loss calculations
- S to ABCD conversion and vice versa
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

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
# S-Parameter Data Tests
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


# =============================================================================
# Touchstone Loading Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-001")
class TestTouchstoneLoading:
    """Test Touchstone file loading."""

    def test_load_touchstone_s2p(self, tmp_path: Path) -> None:
        """Test loading a valid S2P Touchstone file."""
        from oscura.loaders import load_touchstone

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
        from oscura.loaders import load_touchstone

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
        from oscura.loaders import load_touchstone

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
        from oscura.core.exceptions import LoaderError
        from oscura.loaders import load_touchstone

        with pytest.raises(LoaderError, match="File not found"):
            load_touchstone("/nonexistent/path/file.s2p")

    def test_load_touchstone_unsupported_extension(self, tmp_path: Path) -> None:
        """Test that unsupported extension raises FormatError."""
        from oscura.core.exceptions import FormatError
        from oscura.loaders import load_touchstone

        bad_path = tmp_path / "test.txt"
        bad_path.write_text("not a touchstone file")

        with pytest.raises(FormatError, match="Unsupported file extension"):
            load_touchstone(bad_path)

    def test_load_touchstone_s1p(self, tmp_path: Path) -> None:
        """Test loading 1-port Touchstone file."""
        from oscura.loaders import load_touchstone

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
        from oscura.loaders import load_touchstone

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


# =============================================================================
# Return Loss Tests
# =============================================================================


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


# =============================================================================
# Insertion Loss Tests
# =============================================================================


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


# =============================================================================
# S to ABCD Conversion Tests
# =============================================================================


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


# =============================================================================
# ABCD to S Conversion Tests
# =============================================================================


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
