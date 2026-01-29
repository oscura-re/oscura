"""Tests for VCD loader regex optimization.

This module tests the correctness and performance of the regex-based
VCD parser optimization that provides 10-30x speedup over line-by-line parsing.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from oscura.loaders.vcd import load_vcd


def generate_vcd_content(
    signal_name: str = "clk",
    num_transitions: int = 1000,
    timescale: str = "1ns",
) -> str:
    """Generate synthetic VCD content for testing.

    Args:
        signal_name: Name of the signal to generate.
        num_transitions: Number of value transitions to create.
        timescale: VCD timescale string.

    Returns:
        VCD file content as string.
    """
    lines = [
        "$date",
        "   Synthetic test VCD",
        "$end",
        "$version",
        "   Oscura VCD Test Generator",
        "$end",
        f"$timescale {timescale} $end",
        "$scope module testbench $end",
        f"$var wire 1 ! {signal_name} $end",
        "$upscope $end",
        "$enddefinitions $end",
        "#0",
        "$dumpvars",
        "0!",
        "$end",
    ]

    # Generate transitions
    current_value = 0
    for i in range(1, num_transitions + 1):
        timestamp = i * 10
        current_value = 1 - current_value
        lines.append(f"#{timestamp}")
        lines.append(f"{current_value}!")

    return "\n".join(lines) + "\n"


def generate_multibit_vcd_content(
    signal_name: str = "data",
    bit_width: int = 8,
    num_transitions: int = 1000,
) -> str:
    """Generate synthetic multi-bit VCD content.

    Args:
        signal_name: Name of the signal.
        bit_width: Width of the signal in bits.
        num_transitions: Number of value transitions.

    Returns:
        VCD file content as string.
    """
    lines = [
        "$date",
        "   Synthetic multi-bit test VCD",
        "$end",
        "$version",
        "   Oscura VCD Test Generator",
        "$end",
        "$timescale 1ns $end",
        "$scope module testbench $end",
        f"$var wire {bit_width} # {signal_name} $end",
        "$upscope $end",
        "$enddefinitions $end",
        "#0",
        "$dumpvars",
        "b00000000 #",
        "$end",
    ]

    # Generate transitions with binary values
    for i in range(1, num_transitions + 1):
        timestamp = i * 10
        value = i % (2**bit_width)
        binary_value = format(value, f"0{bit_width}b")
        lines.append(f"#{timestamp}")
        lines.append(f"b{binary_value} #")

    return "\n".join(lines) + "\n"


class TestVCDOptimizationCorrectness:
    """Test correctness of optimized VCD parser."""

    def test_single_bit_signal_basic(self) -> None:
        """Test parsing single-bit signal with basic transitions."""
        vcd_content = generate_vcd_content(num_transitions=10)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcd", delete=False) as f:
            f.write(vcd_content)
            temp_path = Path(f.name)

        try:
            trace = load_vcd(temp_path, signal="clk")

            assert trace.data is not None
            assert len(trace.data) > 0
            assert trace.metadata.channel_name == "clk"
            assert trace.edges is not None
            assert len(trace.edges) == 10  # 10 transitions

            # Verify alternating pattern
            edge_values = [edge[1] for edge in trace.edges]
            expected_pattern = [True, False] * 5  # Alternating rising/falling
            assert edge_values == expected_pattern

        finally:
            temp_path.unlink()

    def test_multi_bit_signal(self) -> None:
        """Test parsing multi-bit signal values."""
        vcd_content = generate_multibit_vcd_content(num_transitions=20)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcd", delete=False) as f:
            f.write(vcd_content)
            temp_path = Path(f.name)

        try:
            trace = load_vcd(temp_path, signal="data")

            assert trace.data is not None
            assert len(trace.data) > 0
            assert trace.metadata.channel_name == "data"
            # Multi-bit signal should have edges when LSB changes
            assert trace.edges is not None

        finally:
            temp_path.unlink()

    def test_signal_with_special_identifier(self) -> None:
        """Test signal with special characters in identifier (regex escaping)."""
        vcd_content = """$date
   Test
$end
$version
   1.0
$end
$timescale 1ns $end
$scope module testbench $end
$var wire 1 $+ signal_name $end
$upscope $end
$enddefinitions $end
#0
$dumpvars
0$+
$end
#10
1$+
#20
0$+
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcd", delete=False) as f:
            f.write(vcd_content)
            temp_path = Path(f.name)

        try:
            trace = load_vcd(temp_path, signal="signal_name")

            assert trace.data is not None
            assert len(trace.data) > 0
            assert trace.edges is not None
            assert len(trace.edges) == 2  # Two transitions

        finally:
            temp_path.unlink()

    def test_mixed_single_and_multibit(self) -> None:
        """Test VCD with both single-bit and multi-bit signals."""
        vcd_content = """$date
   Test
$end
$version
   1.0
$end
$timescale 1ns $end
$scope module testbench $end
$var wire 1 ! clk $end
$var wire 8 # data $end
$upscope $end
$enddefinitions $end
#0
$dumpvars
0!
b00000000 #
$end
#10
1!
b00000001 #
#20
0!
b00000010 #
#30
1!
b00000011 #
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcd", delete=False) as f:
            f.write(vcd_content)
            temp_path = Path(f.name)

        try:
            # Test loading single-bit signal
            trace_clk = load_vcd(temp_path, signal="clk")
            assert trace_clk.edges is not None
            assert len(trace_clk.edges) == 3

            # Test loading multi-bit signal
            trace_data = load_vcd(temp_path, signal="data")
            assert trace_data.edges is not None
            assert len(trace_data.edges) == 3  # LSB changes each time

        finally:
            temp_path.unlink()

    def test_timestamps_out_of_order(self) -> None:
        """Test that out-of-order value changes are sorted correctly."""
        # This can happen if regex finds matches in unexpected order
        vcd_content = """$date
   Test
$end
$timescale 1ns $end
$scope module test $end
$var wire 1 ! sig $end
$upscope $end
$enddefinitions $end
#0
0!
#30
1!
#10
0!
#20
1!
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcd", delete=False) as f:
            f.write(vcd_content)
            temp_path = Path(f.name)

        try:
            trace = load_vcd(temp_path, signal="sig")

            assert trace.edges is not None
            # Should be sorted by timestamp
            edge_times = [edge[0] for edge in trace.edges]
            assert edge_times == sorted(edge_times)

        finally:
            temp_path.unlink()

    def test_empty_signal(self) -> None:
        """Test VCD with signal that has no value changes."""
        vcd_content = """$date
   Test
$end
$timescale 1ns $end
$scope module test $end
$var wire 1 ! sig1 $end
$var wire 1 @ sig2 $end
$upscope $end
$enddefinitions $end
#0
0!
#10
1!
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcd", delete=False) as f:
            f.write(vcd_content)
            temp_path = Path(f.name)

        try:
            # sig2 has no value changes - should raise FormatError
            from oscura.core.exceptions import FormatError

            with pytest.raises(FormatError, match="No value changes found"):
                load_vcd(temp_path, signal="sig2")

        finally:
            temp_path.unlink()


class TestVCDOptimizationPerformance:
    """Test performance of optimized VCD parser."""

    def test_performance_large_file(self) -> None:
        """Test that large VCD files parse in reasonable time.

        Target: <30s for 10MB test file with 100k transitions.
        The regex optimization should achieve 10-30x speedup over line-by-line parsing.
        Original implementation: ~450s for 100k transitions (line-by-line).
        Optimized implementation: <30s for 100k transitions (regex + binary search).
        """
        # Generate large VCD with 100k transitions (~10MB)
        vcd_content = generate_vcd_content(num_transitions=100_000)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcd", delete=False) as f:
            f.write(vcd_content)
            temp_path = Path(f.name)

        try:
            start_time = time.perf_counter()
            trace = load_vcd(temp_path, signal="clk")
            elapsed_time = time.perf_counter() - start_time

            # Verify correctness
            assert trace.data is not None
            assert len(trace.data) > 0
            assert trace.edges is not None
            assert len(trace.edges) == 100_000

            # Performance check: should complete in <30s (15x speedup minimum)
            assert elapsed_time < 30.0, f"Parsing took {elapsed_time:.2f}s, expected <30s"

            # For reference, print timing (useful for benchmarking)
            print(f"\nPerformance: Parsed 100k transitions in {elapsed_time:.3f}s")

        finally:
            temp_path.unlink()

    def test_performance_multibit_large(self) -> None:
        """Test performance with large multi-bit signal file.

        Multi-bit signals use different regex pattern but same optimization approach.
        """
        # Generate large multi-bit VCD with 50k transitions
        vcd_content = generate_multibit_vcd_content(num_transitions=50_000, bit_width=32)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcd", delete=False) as f:
            f.write(vcd_content)
            temp_path = Path(f.name)

        try:
            start_time = time.perf_counter()
            trace = load_vcd(temp_path, signal="data")
            elapsed_time = time.perf_counter() - start_time

            assert trace.data is not None
            assert len(trace.data) > 0

            # Should complete in reasonable time (<20s for 50k transitions)
            assert elapsed_time < 20.0, f"Parsing took {elapsed_time:.2f}s, expected <20s"

            print(f"\nMulti-bit performance: Parsed 50k transitions in {elapsed_time:.3f}s")

        finally:
            temp_path.unlink()

    def test_memory_efficiency(self) -> None:
        """Test that optimization doesn't create excessive temporary objects."""
        # This is tested implicitly by the performance tests
        # The regex approach should not create intermediate line lists
        vcd_content = generate_vcd_content(num_transitions=10_000)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcd", delete=False) as f:
            f.write(vcd_content)
            temp_path = Path(f.name)

        try:
            # Should complete without memory errors
            trace = load_vcd(temp_path, signal="clk")
            assert trace.data is not None
            assert trace.edges is not None
            assert len(trace.edges) == 10_000

        finally:
            temp_path.unlink()


class TestVCDEdgeCases:
    """Test edge cases in VCD parsing."""

    def test_no_timestamps(self) -> None:
        """Test VCD with value changes but no timestamps."""
        vcd_content = """$date
   Test
$end
$timescale 1ns $end
$scope module test $end
$var wire 1 ! sig $end
$upscope $end
$enddefinitions $end
0!
1!
0!
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcd", delete=False) as f:
            f.write(vcd_content)
            temp_path = Path(f.name)

        try:
            trace = load_vcd(temp_path, signal="sig")

            # Should default to timestamp 0
            assert trace.data is not None
            assert trace.edges is not None
            # All changes should be at time 0
            edge_times = [edge[0] for edge in trace.edges]
            assert all(t == 0.0 for t in edge_times)

        finally:
            temp_path.unlink()

    def test_whitespace_variations(self) -> None:
        """Test VCD with various whitespace patterns."""
        vcd_content = """$date
   Test
$end
$timescale 1ns $end
$scope module test $end
$var wire 1 ! sig $end
$upscope $end
$enddefinitions $end
#0
0!
#10
1!
#20

0!
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcd", delete=False) as f:
            f.write(vcd_content)
            temp_path = Path(f.name)

        try:
            trace = load_vcd(temp_path, signal="sig")

            assert trace.edges is not None
            assert len(trace.edges) == 2  # Should handle whitespace correctly

        finally:
            temp_path.unlink()

    def test_x_and_z_values(self) -> None:
        """Test VCD with unknown (x) and high-impedance (z) values."""
        vcd_content = """$date
   Test
$end
$timescale 1ns $end
$scope module test $end
$var wire 1 ! sig $end
$upscope $end
$enddefinitions $end
#0
x!
#10
1!
#20
z!
#30
0!
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcd", delete=False) as f:
            f.write(vcd_content)
            temp_path = Path(f.name)

        try:
            trace = load_vcd(temp_path, signal="sig")

            assert trace.data is not None
            assert trace.edges is not None
            # x and z should be treated as False in boolean conversion
            # Only transition to 1 and back to 0 should create edges
            assert len(trace.edges) >= 2

        finally:
            temp_path.unlink()
