#!/usr/bin/env python3
"""VCD File Loading Demonstration.

This demo showcases Oscura's IEEE 1364 VCD (Value Change Dump) file
loading capabilities for digital waveform data.

**Features Demonstrated**:
- VCD file parsing
- Multi-signal extraction
- Timescale handling
- Scope navigation
- Value change event processing
- Conversion to sampled data
- Signal metadata extraction

**VCD File Format (IEEE 1364)**:
VCD files contain digital simulation/capture data as a series of
value change events with timestamps. Key sections:
- Header: Date, version, timescale
- Variable definitions: Signal names, types, widths
- Value changes: Timestamp and value pairs

**Supported Variable Types**:
- wire: Single-bit or multi-bit wires
- reg: Register/flip-flop values
- integer: Integer values
- parameter: Parameters
- event: Event triggers

**Common Sources**:
- Verilog/VHDL simulators
- Logic analyzers (sigrok)
- FPGA debuggers
- GTKWave captures

Usage:
    python vcd_loader_demo.py
    python vcd_loader_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, ValidationSuite, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import GREEN, RESET, print_subheader

# Oscura imports
from oscura.loaders.vcd import load_vcd


class VCDLoaderDemo(BaseDemo):
    """VCD File Loading Demonstration.

    This demo creates sample VCD files and demonstrates loading them
    with Oscura's VCD loader.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="vcd_file_loader",
            description="Demonstrates IEEE 1364 VCD file loading and analysis",
            capabilities=["oscura.loaders.vcd", "oscura.digital_analysis"],
            ieee_standards=["IEEE 1364"],
            **kwargs,
        )
        self.vcd_file = None
        self.traces = {}

    def _create_sample_vcd(self) -> Path:
        """Create a sample VCD file for demonstration.

        Returns:
            Path to created VCD file.
        """
        vcd_content = dedent(
            """\
            $date
               2026-01-16
            $end
            $version
               Oscura Demo VCD Generator
            $end
            $timescale
               1ns
            $end
            $scope module top $end
            $var wire 1 ! clk $end
            $var wire 1 " rst_n $end
            $var wire 8 # data [7:0] $end
            $var wire 1 $ valid $end
            $var wire 1 % ready $end
            $upscope $end
            $enddefinitions $end
            #0
            0!
            0"
            b00000000 #
            0$
            1%
            #5
            1!
            #10
            0!
            1"
            #15
            1!
            #20
            0!
            b00001010 #
            1$
            #25
            1!
            #30
            0!
            0$
            #35
            1!
            #40
            0!
            b00101011 #
            1$
            #45
            1!
            #50
            0!
            0$
            #55
            1!
            #60
            0!
            b11001100 #
            1$
            #65
            1!
            #70
            0!
            0$
            #75
            1!
            #80
            0!
            b11110000 #
            1$
            #85
            1!
            #90
            0!
            0$
            #95
            1!
            #100
            0!
            """
        )

        data_dir = self.get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        vcd_path = data_dir / "demo_signals.vcd"
        vcd_path.write_text(vcd_content)
        return vcd_path

    def generate_test_data(self) -> dict:
        """Create or load sample VCD files for demonstration.

        Returns:
            Dictionary containing vcd_file path.
        """
        print_info("Creating sample VCD file...")
        self.vcd_file = self._create_sample_vcd()
        print_result("VCD file created", self.vcd_file)

        # Show file content preview
        content = self.vcd_file.read_text()
        lines = content.split("\n")

        print_subheader("VCD File Preview")
        for i, line in enumerate(lines[:30]):
            print_info(f"  {i + 1:3d} | {line}")

        if len(lines) > 30:
            print_info(f"  ... ({len(lines) - 30} more lines)")

        print_result("File size", f"{self.vcd_file.stat().st_size} bytes")
        print_result("Total lines", len(lines))

        return {"vcd_file": self.vcd_file}

    def run_demonstration(self, data: dict) -> dict:
        """Load and analyze VCD file.

        Args:
            data: Dictionary containing vcd_file path.

        Returns:
            Dictionary containing analysis results.
        """
        vcd_file = data["vcd_file"]
        results = {}

        print_subheader("Loading VCD File")

        # Load signals one at a time
        signal_names = ["clk", "rst_n", "data", "valid", "ready"]
        traces = {}

        for signal_name in signal_names:
            try:
                trace = load_vcd(vcd_file, signal=signal_name)
                traces[signal_name] = trace
            except Exception:
                # Skip signals that can't be loaded
                pass

        print_result("Signals loaded", len(traces))

        # Store results
        results["signal_count"] = len(traces)
        results["signal_names"] = list(traces.keys())

        # Analyze each signal
        print_subheader("Signal Analysis")

        for signal_name, trace in traces.items():
            print_info(f"  Signal: {signal_name}")
            print_info(f"    Type: {type(trace).__name__}")
            print_info(f"    Samples: {len(trace.data)}")
            print_info(f"    Sample rate: {trace.metadata.sample_rate:.0f} Hz")

            # For digital traces, count edges
            if hasattr(trace, "data"):
                trace_data = trace.data
                if len(trace_data) > 1:
                    # Count transitions
                    edges = np.sum(trace_data[:-1] != trace_data[1:])
                    print_info(f"    Transitions: {edges}")

                    # Show first few values
                    first_vals = trace_data[:10].astype(int).tolist()
                    print_info(f"    First values: {first_vals}")

        # Signal relationships
        print_subheader("Signal Relationships")

        if "clk" in traces and "valid" in traces:
            clk_data = traces["clk"].data
            valid_data = traces["valid"].data

            # Find clk rising edges
            clk_rising = np.where(~clk_data[:-1] & clk_data[1:])[0] + 1

            # Check valid at rising edges
            valid_at_clk = []
            for edge in clk_rising[:20]:
                if edge < len(valid_data):
                    valid_at_clk.append(int(valid_data[edge]))

            print_info(f"  Clock rising edges: {len(clk_rising)}")
            print_info(f"  Valid at clock edges: {valid_at_clk[:10]}")

            valid_count = sum(valid_at_clk)
            print_result("Valid assertions", valid_count)
            results["valid_count"] = valid_count

        # Timing analysis
        print_subheader("Timing Analysis")

        if "clk" in traces:
            clk_trace = traces["clk"]
            clk_data = clk_trace.data
            sample_rate = clk_trace.metadata.sample_rate

            # Find clock period
            rising_edges = np.where(~clk_data[:-1] & clk_data[1:])[0]

            if len(rising_edges) >= 2:
                periods = np.diff(rising_edges) / sample_rate
                avg_period = np.mean(periods)
                clock_freq = 1.0 / avg_period

                print_result("Clock period", f"{avg_period * 1e9:.2f} ns")
                print_result("Clock frequency", f"{clock_freq / 1e6:.2f} MHz")

                results["clock_period_ns"] = avg_period * 1e9
                results["clock_freq_mhz"] = clock_freq / 1e6

        # Data bus analysis
        if "data" in traces:
            print_subheader("Data Bus Analysis")

            data_trace = traces["data"]
            data_values = data_trace.data

            # Get unique values
            unique_vals = np.unique(data_values)
            print_result("Unique data values", len(unique_vals))

            # Show data values at valid assertions
            if "clk" in traces and "valid" in traces:
                clk_data = traces["clk"].data
                valid_data = traces["valid"].data

                # Find when valid goes high
                valid_rising = np.where(~valid_data[:-1] & valid_data[1:])[0]

                captured_data = []
                for edge in valid_rising:
                    if edge < len(data_values):
                        captured_data.append(int(data_values[edge]))

                print_info("  Data values when valid asserted:")
                for i, val in enumerate(captured_data[:10]):
                    print_info(f"    Transfer {i + 1}: 0x{val:02X} ({val})")

                results["captured_data"] = captured_data

        # Summary
        print_subheader("Summary")
        print_result("VCD file", vcd_file.name)
        print_result("Signals extracted", results["signal_count"])
        print_info(f"Signal names: {', '.join(results['signal_names'])}")

        if results["signal_count"] > 0:
            print_info(f"  {GREEN}VCD loading successful!{RESET}")

        results["vcd_file"] = vcd_file
        results["traces"] = traces
        return results

    def validate(self, results: dict) -> bool:
        """Validate VCD loading results.

        Args:
            results: Dictionary containing analysis results.

        Returns:
            True if validation passed, False otherwise.
        """
        # Check signals were loaded
        signal_count = results.get("signal_count", 0)
        if signal_count <= 0:
            self.error("No signals loaded from VCD file")
            return False

        # Check clock was found
        signal_names = results.get("signal_names", [])
        if "clk" not in signal_names:
            self.error("Clock signal not found")
            return False

        # Check clock timing if available
        clock_freq = results.get("clock_freq_mhz", 0)
        if clock_freq <= 0:
            self.warning("Clock frequency not calculated")

        # Check VCD file exists
        vcd_file = results.get("vcd_file")
        if not vcd_file or not vcd_file.exists():
            self.error("VCD file does not exist")
            return False

        self.success("All validation checks passed")
        return True


if __name__ == "__main__":
    sys.exit(run_demo_main(VCDLoaderDemo))
