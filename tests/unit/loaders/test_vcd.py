"""Unit tests for VCD (Value Change Dump) loader.

Tests LOAD-009: VCD (Value Change Dump) Loader
"""

from pathlib import Path

import numpy as np
import pytest

from oscura.core.exceptions import FormatError, LoaderError
from oscura.loaders.vcd import load_vcd

pytestmark = [pytest.mark.unit, pytest.mark.loader]


class TestVCDLoader:
    """Test VCD file loader."""

    def create_vcd_file(
        self,
        path: Path,
        timescale: str = "1ns",
        signals: list[tuple[str, str, int]] | None = None,
        changes: list[tuple[int, str, str]] | None = None,
    ) -> None:
        """Create a minimal VCD file.

        Args:
            path: Output path.
            timescale: Timescale string (e.g., "1ns").
            signals: List of (type, name, width) tuples.
            changes: List of (time, identifier, value) tuples.
        """
        if signals is None:
            signals = [("wire", "clk", 1), ("wire", "data", 8)]

        if changes is None:
            changes = [
                (0, "!", "0"),
                (100, "!", "1"),
                (200, "!", "0"),
                (300, "!", "1"),
            ]

        content = []
        content.append("$version VCD Test $end")
        content.append(f"$timescale {timescale} $end")
        content.append("$scope module test $end")

        # Create variable definitions
        identifiers = "!@#$%^&*"
        for i, (var_type, name, width) in enumerate(signals):
            ident = identifiers[i] if i < len(identifiers) else f"v{i}"
            content.append(f"$var {var_type} {width} {ident} {name} $end")

        content.append("$upscope $end")
        content.append("$enddefinitions $end")
        content.append("$dumpvars")
        content.append("0!")
        content.append("$end")

        # Add value changes
        current_time = -1
        for time, ident, value in changes:
            if time != current_time:
                content.append(f"#{time}")
                current_time = time
            if len(value) == 1:
                content.append(f"{value}{ident}")
            else:
                content.append(f"b{value} {ident}")

        path.write_text("\n".join(content))

    def test_load_basic_vcd(self, tmp_path: Path) -> None:
        """Test loading a basic VCD file."""
        vcd_path = tmp_path / "test.vcd"
        self.create_vcd_file(vcd_path)

        trace = load_vcd(vcd_path)

        assert trace is not None
        assert len(trace.data) > 0
        assert trace.metadata.source_file == str(vcd_path)

    def test_load_specific_signal(self, tmp_path: Path) -> None:
        """Test loading a specific signal by name."""
        vcd_path = tmp_path / "test.vcd"
        self.create_vcd_file(vcd_path)

        trace = load_vcd(vcd_path, signal="clk")
        assert trace.metadata.channel == "clk"

    def test_timescale_parsing(self, tmp_path: Path) -> None:
        """Test that different timescales are parsed correctly."""
        for timescale, expected in [
            ("1ns", 1e-9),
            ("100ps", 100e-12),
            ("1us", 1e-6),
            ("10ns", 10e-9),
        ]:
            vcd_path = tmp_path / f"test_{timescale}.vcd"
            self.create_vcd_file(vcd_path, timescale=timescale)

            trace = load_vcd(vcd_path)
            assert trace.metadata.trigger_info is not None
            assert abs(trace.metadata.trigger_info["timescale"] - expected) < 1e-15

    def test_edge_detection(self, tmp_path: Path) -> None:
        """Test that edges are detected from value changes.

        In v0.9.0, DigitalTrace.edges was removed. We verify the data instead.
        """
        vcd_path = tmp_path / "edges.vcd"
        changes = [
            (0, "!", "0"),
            (100, "!", "1"),  # Rising
            (200, "!", "0"),  # Falling
            (300, "!", "1"),  # Rising
            (400, "!", "0"),  # Falling
        ]
        self.create_vcd_file(vcd_path, changes=changes)

        trace = load_vcd(vcd_path)

        # Verify we got a DigitalTrace with boolean data
        assert hasattr(trace, "data")
        assert trace.data is not None
        assert len(trace.data) > 0
        # Check for transitions in the data
        transitions = np.diff(trace.data.astype(int))
        rising = np.sum(transitions > 0)
        falling = np.sum(transitions < 0)

        assert rising >= 2
        assert falling >= 2

    def test_multibit_signals(self, tmp_path: Path) -> None:
        """Test loading multi-bit signals."""
        vcd_path = tmp_path / "multibit.vcd"
        signals = [("reg", "counter", 8)]
        changes = [
            (0, "!", "00000000"),
            (100, "!", "00000001"),
            (200, "!", "00000010"),
            (300, "!", "11111111"),
        ]
        self.create_vcd_file(vcd_path, signals=signals, changes=changes)

        trace = load_vcd(vcd_path)
        assert trace is not None

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error on missing file."""
        with pytest.raises(LoaderError, match="not found"):
            load_vcd(tmp_path / "nonexistent.vcd")

    def test_missing_enddefinitions(self, tmp_path: Path) -> None:
        """Test error on file without $enddefinitions."""
        vcd_path = tmp_path / "bad.vcd"
        vcd_path.write_text("$version test $end\n$timescale 1ns $end\n")

        with pytest.raises(FormatError, match="enddefinitions"):
            load_vcd(vcd_path)

    def test_no_variables(self, tmp_path: Path) -> None:
        """Test error when no variables defined."""
        vcd_path = tmp_path / "novars.vcd"
        vcd_path.write_text("$version test $end\n$timescale 1ns $end\n$enddefinitions $end\n")

        with pytest.raises(FormatError, match="No variables"):
            load_vcd(vcd_path)

    def test_signal_not_found(self, tmp_path: Path) -> None:
        """Test error when requested signal not found."""
        vcd_path = tmp_path / "test.vcd"
        self.create_vcd_file(vcd_path)

        with pytest.raises(LoaderError, match="not found"):
            load_vcd(vcd_path, signal="nonexistent")

    def test_custom_sample_rate(self, tmp_path: Path) -> None:
        """Test specifying custom sample rate."""
        vcd_path = tmp_path / "test.vcd"
        self.create_vcd_file(vcd_path)

        trace = load_vcd(vcd_path, sample_rate=10e6)
        assert trace.metadata.sample_rate == 10e6


class TestVCDValueParsing:
    """Test VCD value parsing edge cases."""

    def test_x_and_z_values(self, tmp_path: Path) -> None:
        """Test handling of X (unknown) and Z (high-impedance) values."""
        vcd_path = tmp_path / "xz.vcd"
        content = """$version Test $end
$timescale 1ns $end
$var wire 1 ! clk $end
$enddefinitions $end
#0
0!
#100
x!
#200
z!
#300
1!
"""
        vcd_path.write_text(content)

        trace = load_vcd(vcd_path)
        # X and Z should be treated as low (0)
        assert trace is not None

    def test_hierarchical_scope(self, tmp_path: Path) -> None:
        """Test parsing hierarchical scope."""
        vcd_path = tmp_path / "hier.vcd"
        content = """$version Test $end
$timescale 1ns $end
$scope module top $end
$scope module sub $end
$var wire 1 ! clk $end
$upscope $end
$upscope $end
$enddefinitions $end
#0
0!
#100
1!
"""
        vcd_path.write_text(content)

        trace = load_vcd(vcd_path)
        assert trace is not None
        assert trace.metadata.channel == "clk"
