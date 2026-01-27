"""Tests for Touchstone file loader.

Tests loading of S-parameter data from Touchstone 1.0 and 2.0 file formats
(.s1p through .s8p).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oscura.core.exceptions import FormatError, LoaderError
from oscura.loaders.touchstone import load_touchstone


@pytest.fixture
def temp_touchstone_file(tmp_path: Path) -> Path:
    """Create temporary directory for test files."""
    return tmp_path


class TestTouchstoneLoaderBasic:
    """Test basic Touchstone file loading."""

    def test_load_1port_ma_format(self, temp_touchstone_file: Path) -> None:
        """Test loading 1-port S-parameter file in MA format."""
        s1p_file = temp_touchstone_file / "test.s1p"
        s1p_file.write_text(
            """! Simple 1-port test
# GHz S MA R 50
1.0  -10.5 -45.0
2.0  -15.2 -90.0
3.0  -20.1 -135.0
"""
        )

        result = load_touchstone(s1p_file)

        assert result.n_ports == 1
        assert result.z0 == 50.0
        assert result.format == "ma"
        assert len(result.frequencies) == 3
        assert np.allclose(result.frequencies, [1e9, 2e9, 3e9])
        assert result.s_matrix.shape == (3, 1, 1)

    def test_load_2port_db_format(self, temp_touchstone_file: Path) -> None:
        """Test loading 2-port S-parameter file in DB format."""
        s2p_file = temp_touchstone_file / "test.s2p"
        s2p_file.write_text(
            """! 2-port test
# MHz S DB R 75
100  -20 0  -50 -90  -50 -90  -20 -180
"""
        )

        result = load_touchstone(s2p_file)

        assert result.n_ports == 2
        assert result.z0 == 75.0
        assert result.format == "db"
        assert len(result.frequencies) == 1
        assert np.isclose(result.frequencies[0], 100e6)
        assert result.s_matrix.shape == (1, 2, 2)

    def test_load_2port_ri_format(self, temp_touchstone_file: Path) -> None:
        """Test loading 2-port S-parameter file in RI format."""
        s2p_file = temp_touchstone_file / "test.s2p"
        s2p_file.write_text(
            """! 2-port RI format
# GHz S RI R 50
1.0  0.1 0.2  0.3 0.4  0.5 0.6  0.7 0.8
2.0  0.2 0.3  0.4 0.5  0.6 0.7  0.8 0.9
"""
        )

        result = load_touchstone(s2p_file)

        assert result.n_ports == 2
        assert result.format == "ri"
        assert len(result.frequencies) == 2
        assert result.s_matrix.shape == (2, 2, 2)

        # Verify complex values are correctly parsed
        assert result.s_matrix[0, 0, 0] == complex(0.1, 0.2)
        assert result.s_matrix[0, 0, 1] == complex(0.3, 0.4)

    def test_load_with_comments(self, temp_touchstone_file: Path) -> None:
        """Test loading file with comments."""
        s1p_file = temp_touchstone_file / "test.s1p"
        s1p_file.write_text(
            """! This is a comment
! Another comment
# GHz S MA R 50
1.0  -10 -45
"""
        )

        result = load_touchstone(s1p_file)

        assert len(result.comments) == 2
        assert "This is a comment" in result.comments
        assert "Another comment" in result.comments

    def test_load_multiline_data(self, temp_touchstone_file: Path) -> None:
        """Test loading 2-port data spread across multiple lines."""
        s2p_file = temp_touchstone_file / "test.s2p"
        s2p_file.write_text(
            """# GHz S MA R 50
1.0  -10 -45  -50 -90
     -50 -90  -10 -180
2.0  -15 -50  -55 -95
     -55 -95  -15 -185
"""
        )

        result = load_touchstone(s2p_file)

        assert result.n_ports == 2
        assert len(result.frequencies) == 2
        assert result.s_matrix.shape == (2, 2, 2)


class TestTouchstoneFrequencyUnits:
    """Test different frequency unit conversions."""

    @pytest.mark.parametrize(
        ("unit", "expected_hz"),
        [
            ("Hz", 1000.0),
            ("KHz", 1000e3),
            ("MHz", 1000e6),
            ("GHz", 1000e9),
        ],
    )
    def test_frequency_units(
        self,
        temp_touchstone_file: Path,
        unit: str,
        expected_hz: float,
    ) -> None:
        """Test all supported frequency units."""
        s1p_file = temp_touchstone_file / "test.s1p"
        s1p_file.write_text(
            f"""# {unit} S MA R 50
1000  -10 -45
"""
        )

        result = load_touchstone(s1p_file)

        assert np.isclose(result.frequencies[0], expected_hz)


class TestTouchstoneDataFormats:
    """Test different S-parameter data formats."""

    def test_ma_format_conversion(self, temp_touchstone_file: Path) -> None:
        """Test magnitude-angle format is correctly converted to complex."""
        s1p_file = temp_touchstone_file / "test.s1p"
        s1p_file.write_text(
            """# GHz S MA R 50
1.0  0.5 90.0
"""
        )

        result = load_touchstone(s1p_file)

        # 0.5 magnitude at 90 degrees = 0 + 0.5j
        expected = 0.5 * np.exp(1j * np.pi / 2)
        assert np.isclose(result.s_matrix[0, 0, 0], expected)

    def test_db_format_conversion(self, temp_touchstone_file: Path) -> None:
        """Test dB format is correctly converted to complex."""
        s1p_file = temp_touchstone_file / "test.s1p"
        s1p_file.write_text(
            """# GHz S DB R 50
1.0  -20 0
"""
        )

        result = load_touchstone(s1p_file)

        # -20 dB = 10^(-20/20) = 0.1 magnitude at 0 degrees
        expected = 10 ** (-20 / 20)
        assert np.isclose(result.s_matrix[0, 0, 0], expected)

    def test_ri_format_direct(self, temp_touchstone_file: Path) -> None:
        """Test real-imaginary format is directly converted."""
        s1p_file = temp_touchstone_file / "test.s1p"
        s1p_file.write_text(
            """# GHz S RI R 50
1.0  0.3 0.4
"""
        )

        result = load_touchstone(s1p_file)

        assert result.s_matrix[0, 0, 0] == complex(0.3, 0.4)


class TestTouchstonePortCounts:
    """Test loading files with different port counts."""

    @pytest.mark.parametrize("n_ports", [1, 2, 3, 4])
    def test_various_port_counts(
        self,
        temp_touchstone_file: Path,
        n_ports: int,
    ) -> None:
        """Test loading files with 1-4 ports."""
        filename = temp_touchstone_file / f"test.s{n_ports}p"

        # Generate data for n_ports^2 S-parameters
        n_sparams = n_ports * n_ports
        data_values = " ".join([f"{i} {i + 1}" for i in range(n_sparams)])

        filename.write_text(
            f"""# GHz S RI R 50
1.0  {data_values}
"""
        )

        result = load_touchstone(filename)

        assert result.n_ports == n_ports
        assert result.s_matrix.shape == (1, n_ports, n_ports)


class TestTouchstoneEdgeCases:
    """Test edge cases and error conditions."""

    def test_file_not_found(self) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(LoaderError, match="File not found"):
            load_touchstone("nonexistent.s1p")

    def test_invalid_extension(self, temp_touchstone_file: Path) -> None:
        """Test error with invalid file extension."""
        bad_file = temp_touchstone_file / "test.txt"
        bad_file.write_text("# GHz S MA R 50\n1.0 -10 -45\n")

        with pytest.raises(FormatError, match="Unsupported file extension"):
            load_touchstone(bad_file)

    def test_empty_file(self, temp_touchstone_file: Path) -> None:
        """Test error with empty file."""
        empty_file = temp_touchstone_file / "test.s1p"
        empty_file.write_text("")

        with pytest.raises(FormatError, match="No valid frequency points"):
            load_touchstone(empty_file)

    def test_no_data_lines(self, temp_touchstone_file: Path) -> None:
        """Test error when file has no data."""
        s1p_file = temp_touchstone_file / "test.s1p"
        s1p_file.write_text(
            """! Just comments
# GHz S MA R 50
! No data
"""
        )

        with pytest.raises(FormatError, match="No valid frequency points"):
            load_touchstone(s1p_file)

    def test_source_file_recorded(self, temp_touchstone_file: Path) -> None:
        """Test that source file path is recorded."""
        s1p_file = temp_touchstone_file / "test.s1p"
        s1p_file.write_text(
            """# GHz S MA R 50
1.0  -10 -45
"""
        )

        result = load_touchstone(s1p_file)

        assert result.source_file is not None
        assert "test.s1p" in result.source_file


class TestTouchstoneComplexScenarios:
    """Test complex real-world scenarios."""

    def test_multiple_frequencies(self, temp_touchstone_file: Path) -> None:
        """Test loading file with many frequency points."""
        s1p_file = temp_touchstone_file / "test.s1p"

        # Generate 100 frequency points
        lines = ["# GHz S MA R 50"]
        for i in range(1, 101):
            lines.append(f"{i * 0.1}  -{i * 0.2} -{i * 1.5}")

        s1p_file.write_text("\n".join(lines))

        result = load_touchstone(s1p_file)

        assert len(result.frequencies) == 100
        assert result.frequencies[0] == pytest.approx(0.1e9)
        assert result.frequencies[-1] == pytest.approx(10e9)

    def test_whitespace_handling(self, temp_touchstone_file: Path) -> None:
        """Test robust whitespace handling."""
        s1p_file = temp_touchstone_file / "test.s1p"
        s1p_file.write_text(
            """! Comments with spaces

# GHz  S  MA  R  50

1.0   -10   -45
2.0   -15   -50

"""
        )

        result = load_touchstone(s1p_file)

        assert len(result.frequencies) == 2

    def test_case_insensitive_options(self, temp_touchstone_file: Path) -> None:
        """Test that option parsing is case-insensitive."""
        s1p_file = temp_touchstone_file / "test.s1p"
        s1p_file.write_text(
            """# ghz s ma r 50
1.0  -10 -45
"""
        )

        result = load_touchstone(s1p_file)

        assert result.format == "ma"
        assert np.isclose(result.frequencies[0], 1e9)


class TestTouchstoneLargeFiles:
    """Test handling of large port counts."""

    def test_8port_file(self, temp_touchstone_file: Path) -> None:
        """Test loading 8-port S-parameter file."""
        s8p_file = temp_touchstone_file / "test.s8p"

        # 8x8 = 64 S-parameters
        data_values = " ".join([f"{i * 0.01} {i * 0.1}" for i in range(64)])

        lines = ["# GHz S RI R 50", f"1.0  {data_values}"]

        s8p_file.write_text("\n".join(lines))

        result = load_touchstone(s8p_file)

        assert result.n_ports == 8
        assert result.s_matrix.shape == (1, 8, 8)


class TestTouchstoneReferenceImpedance:
    """Test reference impedance handling."""

    @pytest.mark.parametrize("z0_value", [50.0, 75.0, 100.0, 25.0])
    def test_various_reference_impedances(
        self,
        temp_touchstone_file: Path,
        z0_value: float,
    ) -> None:
        """Test different reference impedances."""
        s1p_file = temp_touchstone_file / "test.s1p"
        s1p_file.write_text(
            f"""# GHz S MA R {z0_value}
1.0  -10 -45
"""
        )

        result = load_touchstone(s1p_file)

        assert result.z0 == z0_value

    def test_default_reference_impedance(self, temp_touchstone_file: Path) -> None:
        """Test default reference impedance when not specified."""
        s1p_file = temp_touchstone_file / "test.s1p"
        s1p_file.write_text(
            """# GHz S MA
1.0  -10 -45
"""
        )

        result = load_touchstone(s1p_file)

        assert result.z0 == 50.0  # Default value
