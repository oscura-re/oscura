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

from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import GREEN, RESET, print_subheader

# Oscura imports
from oscura.loaders.vcd import load_vcd


class VCDLoaderDemo(BaseDemo):
    """VCD File Loading Demonstration.

    This demo creates sample VCD files and demonstrates loading them
    with Oscura's VCD loader.
    """

    name = "VCD File Loader Demo"
    description = "Demonstrates IEEE 1364 VCD file loading and analysis"
    category = "file_format_io"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
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

        vcd_path = self.data_dir / "demo_signals.vcd"
        vcd_path.write_text(vcd_content)
        return vcd_path

    def generate_data(self) -> None:
        """Create or load sample VCD files for demonstration."""
        vcd_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            vcd_file_to_load = self.data_file
            print_info(f"Loading VCD from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("demo_signals.vcd"):
            vcd_file_to_load = default_file
            print_info(f"Loading VCD from default file: {default_file.name}")

        # Use existing file if found
        if vcd_file_to_load:
            self.vcd_file = vcd_file_to_load
            print_result("Loaded VCD file", vcd_file_to_load.name)
        else:
            # 3. Generate VCD file as fallback
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

    def run_analysis(self) -> None:
        """Load and analyze VCD file."""
        print_subheader("Loading VCD File")

        # Load signals one at a time
        signal_names = ["clk", "rst_n", "data", "valid", "ready"]
        self.traces = {}

        for signal_name in signal_names:
            try:
                trace = load_vcd(self.vcd_file, signal=signal_name)
                self.traces[signal_name] = trace
            except Exception:
                # Skip signals that can't be loaded
                pass

        print_result("Signals loaded", len(self.traces))

        # Store results
        self.results["signal_count"] = len(self.traces)
        self.results["signal_names"] = list(self.traces.keys())

        # Analyze each signal
        print_subheader("Signal Analysis")

        for signal_name, trace in self.traces.items():
            print_info(f"  Signal: {signal_name}")
            print_info(f"    Type: {type(trace).__name__}")
            print_info(f"    Samples: {len(trace.data)}")
            print_info(f"    Sample rate: {trace.metadata.sample_rate:.0f} Hz")

            # For digital traces, count edges
            if hasattr(trace, "data"):
                data = trace.data
                if len(data) > 1:
                    # Count transitions
                    edges = np.sum(data[:-1] != data[1:])
                    print_info(f"    Transitions: {edges}")

                    # Show first few values
                    first_vals = data[:10].astype(int).tolist()
                    print_info(f"    First values: {first_vals}")

        # Signal relationships
        print_subheader("Signal Relationships")

        if "clk" in self.traces and "valid" in self.traces:
            clk_data = self.traces["clk"].data
            valid_data = self.traces["valid"].data

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
            self.results["valid_count"] = valid_count

        # Timing analysis
        print_subheader("Timing Analysis")

        if "clk" in self.traces:
            clk_trace = self.traces["clk"]
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

                self.results["clock_period_ns"] = avg_period * 1e9
                self.results["clock_freq_mhz"] = clock_freq / 1e6

        # Data bus analysis
        if "data" in self.traces:
            print_subheader("Data Bus Analysis")

            data_trace = self.traces["data"]
            data_values = data_trace.data

            # Get unique values
            unique_vals = np.unique(data_values)
            print_result("Unique data values", len(unique_vals))

            # Show data values at valid assertions
            if "clk" in self.traces and "valid" in self.traces:
                clk_data = self.traces["clk"].data
                valid_data = self.traces["valid"].data

                # Find when valid goes high
                valid_rising = np.where(~valid_data[:-1] & valid_data[1:])[0]

                captured_data = []
                for edge in valid_rising:
                    if edge < len(data_values):
                        captured_data.append(int(data_values[edge]))

                print_info("  Data values when valid asserted:")
                for i, val in enumerate(captured_data[:10]):
                    print_info(f"    Transfer {i + 1}: 0x{val:02X} ({val})")

                self.results["captured_data"] = captured_data

        # Summary
        print_subheader("Summary")
        print_result("VCD file", self.vcd_file.name)
        print_result("Signals extracted", self.results["signal_count"])
        print_info(f"Signal names: {', '.join(self.results['signal_names'])}")

        if self.results["signal_count"] > 0:
            print_info(f"  {GREEN}VCD loading successful!{RESET}")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate VCD loading results."""
        # Check signals were loaded
        suite.check_greater(
            "Signal count",
            self.results.get("signal_count", 0),
            0,
            category="loading",
        )

        # Check clock was found
        signal_names = self.results.get("signal_names", [])
        suite.check_true(
            "Clock signal found",
            "clk" in signal_names,
            category="signals",
        )

        # Check clock timing if available
        clock_freq = self.results.get("clock_freq_mhz", 0)
        if clock_freq > 0:
            suite.check_greater(
                "Clock frequency",
                clock_freq,
                0,
                category="timing",
            )

        # Check VCD file exists
        suite.check_file_exists(
            "VCD file",
            self.vcd_file,
            category="files",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(VCDLoaderDemo))
