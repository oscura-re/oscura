"""Unit tests for Touchstone S-parameter file loading.

Tests verify the touchstone loader correctly imports .s2p/.s4p files
and creates SParameterData structures.
"""

from pathlib import Path

import numpy as np
import pytest

from oscura.analyzers.signal_integrity.sparams import SParameterData, load_touchstone
from oscura.core.exceptions import FormatError, LoaderError
from oscura.loaders import load

pytestmark = [pytest.mark.unit, pytest.mark.loader]


@pytest.mark.unit
@pytest.mark.loader
class TestTouchstoneDirectLoad:
    """Test direct Touchstone loading via load_touchstone()."""

    def test_load_s2p_file(self):
        """Test loading .s2p file with 2-port data."""
        # Use test data file
        test_file = Path("test_data/signal_integrity/touchstone/transmission_line_50ohm.s2p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        s_params = load_touchstone(test_file)

        # Verify structure
        assert isinstance(s_params, SParameterData)
        assert s_params.n_ports == 2
        assert len(s_params.frequencies) > 0
        assert s_params.s_matrix.shape == (len(s_params.frequencies), 2, 2)

    def test_load_s2p_data_structure(self):
        """Test loaded .s2p has correct data structure."""
        test_file = Path("test_data/signal_integrity/touchstone/transmission_line_50ohm.s2p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        s_params = load_touchstone(test_file)

        # Check metadata
        assert s_params.z0 == 50.0  # Reference impedance
        assert s_params.source_file == str(test_file)
        assert isinstance(s_params.frequencies, np.ndarray)
        assert s_params.frequencies.dtype == np.float64
        assert isinstance(s_params.s_matrix, np.ndarray)
        assert s_params.s_matrix.dtype == np.complex128

    def test_load_s2p_frequencies_ascending(self):
        """Test frequencies are in ascending order."""
        test_file = Path("test_data/signal_integrity/touchstone/transmission_line_50ohm.s2p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        s_params = load_touchstone(test_file)

        # Verify frequencies are sorted
        assert np.all(np.diff(s_params.frequencies) > 0)

    def test_load_s2p_lossy_cable(self):
        """Test loading lossy cable S2P data."""
        test_file = Path("test_data/signal_integrity/touchstone/lossy_cable_10m.s2p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        s_params = load_touchstone(test_file)

        assert s_params.n_ports == 2
        assert len(s_params.frequencies) > 0

    def test_load_s4p_file(self):
        """Test loading .s4p file with 4-port data."""
        test_file = Path("test_data/signal_integrity/touchstone/differential_pair.s4p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        s_params = load_touchstone(test_file)

        # Verify 4-port structure
        assert isinstance(s_params, SParameterData)
        assert s_params.n_ports == 4
        assert s_params.s_matrix.shape == (len(s_params.frequencies), 4, 4)

    def test_load_touchstone_file_not_found(self):
        """Test load_touchstone raises error for missing file."""
        with pytest.raises(LoaderError, match="File not found"):
            load_touchstone("/nonexistent/file.s2p")

    def test_load_touchstone_invalid_extension(self, tmp_path: Path):
        """Test load_touchstone raises error for invalid extension."""
        # Create file with invalid extension to bypass file-not-found check
        invalid_file = tmp_path / "file.txt"
        invalid_file.write_text("dummy content")

        with pytest.raises(FormatError, match="Unsupported file extension"):
            load_touchstone(invalid_file)

    def test_s_parameter_get_s11(self):
        """Test get_s() method for S11 parameter."""
        test_file = Path("test_data/signal_integrity/touchstone/transmission_line_50ohm.s2p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        s_params = load_touchstone(test_file)

        # Get S11 for all frequencies
        s11 = s_params.get_s(1, 1)
        assert isinstance(s11, np.ndarray)
        assert len(s11) == len(s_params.frequencies)
        assert s11.dtype == np.complex128

    def test_s_parameter_get_s21(self):
        """Test get_s() method for S21 parameter."""
        test_file = Path("test_data/signal_integrity/touchstone/transmission_line_50ohm.s2p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        s_params = load_touchstone(test_file)

        # Get S21 for all frequencies
        s21 = s_params.get_s(2, 1)
        assert isinstance(s21, np.ndarray)
        assert len(s21) == len(s_params.frequencies)

    def test_s_parameter_get_s_at_frequency(self):
        """Test get_s() with frequency interpolation."""
        test_file = Path("test_data/signal_integrity/touchstone/transmission_line_50ohm.s2p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        s_params = load_touchstone(test_file)

        # Get S11 at specific frequency
        freq = 1e9  # 1 GHz
        s11_at_freq = s_params.get_s(1, 1, frequency=freq)
        assert isinstance(s11_at_freq, complex | np.complexfloating)


@pytest.mark.unit
@pytest.mark.loader
class TestTouchstoneUnifiedLoad:
    """Test Touchstone loading via unified load() interface."""

    def test_load_s2p_via_unified_interface(self):
        """Test loading .s2p through unified load() function."""
        test_file = Path("test_data/signal_integrity/touchstone/transmission_line_50ohm.s2p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        result = load(test_file)

        # Should return SParameterData
        assert isinstance(result, SParameterData)
        assert result.n_ports == 2

    def test_load_s4p_via_unified_interface(self):
        """Test loading .s4p through unified load() function."""
        test_file = Path("test_data/signal_integrity/touchstone/differential_pair.s4p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        result = load(test_file)

        # Should return SParameterData
        assert isinstance(result, SParameterData)
        assert result.n_ports == 4

    def test_load_touchstone_with_format_override(self):
        """Test loading Touchstone with explicit format override."""
        test_file = Path("test_data/signal_integrity/touchstone/transmission_line_50ohm.s2p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        result = load(test_file, format="touchstone")

        assert isinstance(result, SParameterData)
        assert result.n_ports == 2


@pytest.mark.unit
@pytest.mark.loader
class TestTouchstoneComments:
    """Test Touchstone comment and metadata parsing."""

    def test_load_extracts_comments(self):
        """Test that comments are extracted from Touchstone file."""
        test_file = Path("test_data/signal_integrity/touchstone/transmission_line_50ohm.s2p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        s_params = load_touchstone(test_file)

        # Comments should be a list
        assert isinstance(s_params.comments, list)
        # Should have at least one comment from the file
        assert len(s_params.comments) > 0

    def test_load_parses_format_line(self):
        """Test that option line format is parsed."""
        test_file = Path("test_data/signal_integrity/touchstone/transmission_line_50ohm.s2p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        s_params = load_touchstone(test_file)

        # Format should be one of the valid types
        assert s_params.format in ["ri", "ma", "db"]


@pytest.mark.unit
@pytest.mark.loader
class TestTouchstoneDataValidation:
    """Test SParameterData validation."""

    def test_empty_frequencies_raises_error(self):
        """Test that empty frequencies array raises ValueError."""
        with pytest.raises(ValueError, match="frequencies cannot be empty"):
            SParameterData(
                frequencies=np.array([]),
                s_matrix=np.array([]),
                n_ports=2,
            )

    def test_mismatched_shape_raises_error(self):
        """Test that mismatched s_matrix shape raises ValueError."""
        with pytest.raises(ValueError, match="does not match expected"):
            SParameterData(
                frequencies=np.array([1e9, 2e9]),
                s_matrix=np.zeros((3, 2, 2), dtype=np.complex128),  # Wrong length
                n_ports=2,
            )

    def test_valid_data_structure(self):
        """Test that valid data passes validation."""
        freqs = np.array([1e9, 2e9, 3e9])
        s_matrix = np.zeros((3, 2, 2), dtype=np.complex128)

        s_params = SParameterData(
            frequencies=freqs,
            s_matrix=s_matrix,
            n_ports=2,
        )

        assert s_params.n_ports == 2
        assert len(s_params.frequencies) == 3


@pytest.mark.unit
@pytest.mark.loader
class TestTouchstoneInMemoryParsing:
    """Test Touchstone parsing with synthetic data."""

    def test_parse_minimal_s2p(self, tmp_path: Path):
        """Test parsing minimal valid S2P content."""
        s2p_path = tmp_path / "test.s2p"
        content = """! Test S2P file
# Hz S RI R 50
1e9 0.1 0.0 0.9 0.0 0.9 0.0 0.1 0.0
2e9 0.15 0.0 0.85 0.0 0.85 0.0 0.15 0.0
"""
        s2p_path.write_text(content)

        s_params = load_touchstone(s2p_path)

        assert s_params.n_ports == 2
        assert len(s_params.frequencies) == 2
        assert s_params.frequencies[0] == 1e9
        assert s_params.frequencies[1] == 2e9
        assert s_params.z0 == 50.0
        assert s_params.format == "ri"

    def test_parse_db_format(self, tmp_path: Path):
        """Test parsing dB format S-parameters."""
        s2p_path = tmp_path / "test_db.s2p"
        content = """! Test S2P in dB format
# GHz S DB R 50
1.0 -20.0 0.0 -0.5 0.0 -0.5 0.0 -20.0 0.0
"""
        s2p_path.write_text(content)

        s_params = load_touchstone(s2p_path)

        assert s_params.format == "db"
        assert len(s_params.frequencies) == 1
        assert s_params.frequencies[0] == 1e9  # Converted from GHz

    def test_parse_ma_format(self, tmp_path: Path):
        """Test parsing magnitude/angle format S-parameters."""
        s2p_path = tmp_path / "test_ma.s2p"
        content = """! Test S2P in MA format
# MHz S MA R 50
100 0.1 0.0 0.9 0.0 0.9 0.0 0.1 0.0
"""
        s2p_path.write_text(content)

        s_params = load_touchstone(s2p_path)

        assert s_params.format == "ma"
        assert s_params.frequencies[0] == 100e6  # Converted from MHz

    def test_parse_multiline_s4p(self, tmp_path: Path):
        """Test parsing S4P with multi-line data per frequency."""
        s4p_path = tmp_path / "test.s4p"
        # S4P has 16 parameters per frequency (4x4 matrix)
        content = """! Test S4P file
# Hz S RI R 50
1e9 0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.1 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.1 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.1 0.0
"""
        s4p_path.write_text(content)

        s_params = load_touchstone(s4p_path)

        assert s_params.n_ports == 4
        assert len(s_params.frequencies) == 1
        assert s_params.s_matrix.shape == (1, 4, 4)


@pytest.mark.unit
@pytest.mark.loader
class TestTouchstoneSupport:
    """Test Touchstone format support in SUPPORTED_FORMATS."""

    def test_touchstone_formats_registered(self):
        """Test all Touchstone extensions are registered."""
        from oscura.loaders import SUPPORTED_FORMATS

        touchstone_exts = [".s1p", ".s2p", ".s3p", ".s4p", ".s5p", ".s6p", ".s7p", ".s8p"]

        for ext in touchstone_exts:
            assert ext in SUPPORTED_FORMATS
            assert SUPPORTED_FORMATS[ext] == "touchstone"

    def test_touchstone_in_supported_formats(self):
        """Test Touchstone formats appear in get_supported_formats()."""
        from oscura.loaders import get_supported_formats

        formats = get_supported_formats()

        assert ".s2p" in formats
        assert ".s4p" in formats
