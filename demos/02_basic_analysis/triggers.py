#!/usr/bin/env python3
"""Trigger Detection: Oscilloscope-style trigger analysis.

This demo demonstrates trigger detection and analysis:
- Edge triggers (rising/falling/both)
- Level triggers (threshold crossing)
- Pulse width triggers (narrow/wide pulses)
- Pattern triggers (specific bit patterns)
- Trigger holdoff and count
- Pre/post-trigger capture

Related demos:
- 02_digital_basics.py - Edge detection
- ../03_protocol_decoding/01_uart_analysis.py - Serial triggers
- ../04_advanced_analysis/06_triggering_advanced.py - Advanced triggers

Usage:
    python demos/02_basic_analysis/06_triggers.py
    python demos/02_basic_analysis/06_triggers.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

import oscura as osc
from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.data_generation import generate_pulse_train, generate_square_wave
from demos.common.formatting import print_subheader


class TriggersDemo(BaseDemo):
    """Trigger detection and analysis demonstration."""

    name = "Trigger Detection and Analysis"
    description = "Edge, level, pulse width, and pattern triggers"
    category = "basic_analysis"

    capabilities: ClassVar[list[str]] = [
        "oscura.find_edges",
        "oscura.edge_trigger",
        "oscura.level_trigger",
        "oscura.pulse_trigger",
        "oscura.pattern_trigger",
    ]

    related_demos: ClassVar[list[str]] = [
        "02_digital_basics.py",
        "../03_protocol_decoding/01_uart_analysis.py",
        "../04_advanced_analysis/06_triggering_advanced.py",
    ]

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.test_signal = None
        self.pulse_signal = None
        self.pattern_signal = None

    def generate_data(self) -> None:
        """Generate test signals for trigger demonstrations."""
        print_info("Generating trigger test signals...")

        # 1. Regular square wave for edge and level triggers
        self.test_signal = generate_square_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=3.3,  # 3.3V logic
            duration=0.01,  # 10 ms
            sample_rate=1e6,  # 1 MHz sampling
            duty_cycle=0.5,
        )

        # 2. Variable pulse width signal for pulse triggers
        self.pulse_signal = generate_pulse_train(
            pulse_width=100e-6,  # 100 µs narrow pulses
            period=1000e-6,  # 1 ms period
            amplitude=5.0,  # 5V
            duration=0.02,  # 20 ms
            sample_rate=1e6,  # 1 MHz
            rise_time=1e-9,
            fall_time=1e-9,
        )

        # 3. Pattern signal (simulate UART-like pattern)
        # Create a specific bit pattern: 10110100
        sample_rate = 1e6
        bit_duration = 100  # samples per bit
        pattern: ClassVar[list[str]] = [1, 0, 1, 1, 0, 1, 0, 0]

        pattern_data: ClassVar[list[str]] = []
        for bit in pattern:
            pattern_data.extend([bit * 3.3] * bit_duration)

        # Add some repeats and padding
        full_pattern = np.concatenate(
            [
                np.zeros(500),
                pattern_data,
                np.zeros(200),
                pattern_data,
                np.zeros(500),
            ]
        )

        self.pattern_signal = osc.WaveformTrace(
            data=np.array(full_pattern),
            metadata=osc.TraceMetadata(sample_rate=sample_rate, channel_name="pattern"),
        )

        print_result("Square wave generated", "1 kHz, 3.3V, 50% duty")
        print_result("Pulse train generated", "100 µs pulses, 1 ms period")
        print_result("Pattern signal generated", "8-bit pattern: 10110100")

    def run_analysis(self) -> None:
        """Execute trigger detection demonstrations."""
        # ========== PART 1: EDGE TRIGGERS ==========
        print_subheader("Part 1: Edge Triggers")
        print_info("Detecting rising and falling edges")

        # Find all edges
        edges = osc.find_edges(self.test_signal)
        rising = edges["rising"]
        falling = edges["falling"]

        self.results["edge_rising_count"] = len(rising)
        self.results["edge_falling_count"] = len(falling)

        print_result("Rising edges detected", len(rising))
        print_result("Falling edges detected", len(falling))

        # Edge trigger with specific slope
        rising_trigger = osc.edge_trigger(self.test_signal, slope="rising")
        falling_trigger = osc.edge_trigger(self.test_signal, slope="falling")

        self.results["trigger_rising_index"] = rising_trigger["index"]
        self.results["trigger_falling_index"] = falling_trigger["index"]

        print_result("First rising edge", f"Sample {rising_trigger['index']}")
        print_result("First falling edge", f"Sample {falling_trigger['index']}")

        # Calculate trigger timing
        sample_rate = self.test_signal.metadata.sample_rate
        trigger_time_rising = rising_trigger["index"] / sample_rate
        trigger_time_falling = falling_trigger["index"] / sample_rate

        print_result("Rising edge time", f"{trigger_time_rising * 1e6:.2f} µs")
        print_result("Falling edge time", f"{trigger_time_falling * 1e6:.2f} µs")

        # ========== PART 2: LEVEL TRIGGERS ==========
        print_subheader("Part 2: Level Triggers")
        print_info("Triggering on voltage threshold crossings")

        # Set trigger level at 50% of amplitude
        trigger_level = osc.amplitude(self.test_signal) / 2

        level_trig = osc.level_trigger(self.test_signal, level=trigger_level, slope="rising")

        self.results["level_trigger_index"] = level_trig["index"]
        self.results["level_trigger_value"] = self.test_signal.data[level_trig["index"]]

        print_result("Trigger level", f"{trigger_level:.3f} V")
        print_result("Trigger at sample", level_trig["index"])
        print_result("Signal value at trigger", f"{self.results['level_trigger_value']:.3f} V")

        # ========== PART 3: PULSE WIDTH TRIGGERS ==========
        print_subheader("Part 3: Pulse Width Triggers")
        print_info("Triggering on pulse width conditions")

        # Measure all pulse widths
        pulse_widths: ClassVar[list[str]] = []
        pulse_start = None

        threshold = osc.amplitude(self.pulse_signal) / 2
        for i, val in enumerate(self.pulse_signal.data):
            if val > threshold and pulse_start is None:
                pulse_start = i
            elif val < threshold and pulse_start is not None:
                pulse_widths.append(i - pulse_start)
                pulse_start = None

        if pulse_widths:
            avg_pulse_width = np.mean(pulse_widths) / self.pulse_signal.metadata.sample_rate
            self.results["pulse_width_avg"] = avg_pulse_width
            self.results["pulse_count"] = len(pulse_widths)

            print_result("Pulses detected", len(pulse_widths))
            print_result("Average pulse width", f"{avg_pulse_width * 1e6:.2f} µs")

            # Trigger on narrow pulse (< 150 µs)
            narrow_trigger = osc.pulse_trigger(
                self.pulse_signal,
                min_width=50e-6,
                max_width=150e-6,
            )

            self.results["narrow_pulse_trigger"] = narrow_trigger["index"]
            print_result("Narrow pulse trigger", f"Sample {narrow_trigger['index']}")

        # ========== PART 4: PATTERN TRIGGERS ==========
        print_subheader("Part 4: Pattern Triggers")
        print_info("Triggering on specific bit patterns")

        # Convert signal to binary
        threshold_pattern = 1.65  # 50% of 3.3V
        binary_data = (self.pattern_signal.data > threshold_pattern).astype(int)

        # Look for pattern: 10110100
        target_pattern: ClassVar[list[str]] = [1, 0, 1, 1, 0, 1, 0, 0]
        pattern_length = len(target_pattern)

        # Find pattern matches (with decimation to account for samples per bit)
        matches: ClassVar[list[str]] = []
        bit_samples = 100  # samples per bit

        for i in range(0, len(binary_data) - pattern_length * bit_samples, bit_samples):
            # Extract one sample per bit period
            extracted: ClassVar[list[str]] = [
                binary_data[i + j * bit_samples] for j in range(pattern_length)
            ]
            if extracted == target_pattern:
                matches.append(i)

        self.results["pattern_matches"] = len(matches)
        self.results["pattern_indices"] = matches

        print_result("Pattern matches found", len(matches))
        if matches:
            for idx, match_idx in enumerate(matches):
                print_result(f"  Match {idx + 1} at sample", match_idx)

        # ========== PART 5: TRIGGER HOLDOFF ==========
        print_subheader("Part 5: Trigger Holdoff")
        print_info("Using holdoff to ignore subsequent triggers")

        # Find edges with minimum spacing (holdoff)
        all_rising = edges["rising"]
        holdoff_samples = int(0.001 * sample_rate)  # 1 ms holdoff

        triggers_with_holdoff: ClassVar[list[str]] = []
        last_trigger = -holdoff_samples

        for edge_idx in all_rising:
            if edge_idx - last_trigger >= holdoff_samples:
                triggers_with_holdoff.append(edge_idx)
                last_trigger = edge_idx

        self.results["triggers_without_holdoff"] = len(all_rising)
        self.results["triggers_with_holdoff"] = len(triggers_with_holdoff)

        print_result("Triggers without holdoff", len(all_rising))
        print_result("Triggers with 1ms holdoff", len(triggers_with_holdoff))
        print_result(
            "Holdoff reduction", f"{(1 - len(triggers_with_holdoff) / len(all_rising)) * 100:.1f}%"
        )

        # ========== MEASUREMENT INTERPRETATION ==========
        print_subheader("Trigger Analysis Summary")

        print_info("\n[Edge Triggers]")
        print_info(
            f"  Rising edges: {len(rising)} at ~{1000 / len(rising) if len(rising) > 0 else 0:.2f}ms spacing"
        )
        print_info(f"  Falling edges: {len(falling)}")
        print_info(f"  First trigger: {trigger_time_rising * 1e6:.2f}µs")

        print_info("\n[Level Triggers]")
        print_info(f"  Threshold: {trigger_level:.3f}V (50% of amplitude)")
        print_info(f"  Trigger at: {self.results['level_trigger_index']} samples")

        print_info("\n[Pulse Width Triggers]")
        if pulse_widths:
            print_info(f"  Average pulse width: {avg_pulse_width * 1e6:.2f}µs")
            print_info(f"  Pulses in range (50-150µs): {len(pulse_widths)}")

        print_info("\n[Pattern Triggers]")
        print_info("  Pattern: 10110100 (8 bits)")
        print_info(f"  Matches found: {len(matches)}")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate trigger detection results."""
        # Edge detection
        suite.check_range("Rising edges", self.results["edge_rising_count"], 8, 12)
        suite.check_range("Falling edges", self.results["edge_falling_count"], 8, 12)

        # Triggers should occur early in signal
        suite.check_range("Rising trigger index", self.results["trigger_rising_index"], 0, 2000)
        suite.check_range("Falling trigger index", self.results["trigger_falling_index"], 0, 2000)

        # Level trigger validation
        suite.check_range("Level trigger index", self.results["level_trigger_index"], 0, 2000)

        # Pulse width validation
        if "pulse_width_avg" in self.results:
            suite.check_range("Average pulse width", self.results["pulse_width_avg"], 95e-6, 105e-6)
            suite.check_range("Pulse count", self.results["pulse_count"], 15, 25)

        # Pattern matching
        suite.check_range("Pattern matches", self.results["pattern_matches"], 1, 3)

        # Holdoff validation
        suite.check_range("Triggers with holdoff", self.results["triggers_with_holdoff"], 8, 12)


if __name__ == "__main__":
    sys.exit(run_demo_main(TriggersDemo))
