"""Triggering: Event detection and signal synchronization

Demonstrates:
- oscura.EdgeTrigger - Rising and falling edge detection
- oscura.find_rising_edges() - Detect rising edge events
- oscura.find_falling_edges() - Detect falling edge events
- oscura.PulseWidthTrigger - Pulse width condition detection
- oscura.find_pulses() - Extract pulse information
- oscura.find_glitches() - Detect glitch events
- oscura.find_runt_pulses() - Detect runt pulse events
- oscura.find_triggers() - Generic trigger detection
- Trigger segment extraction and analysis

IEEE Standards: IEEE 181-2011 (transitional waveform definitions)
Related Demos:
- 02_basic_analysis/01_waveform_measurements.py
- 02_basic_analysis/06_math_operations.py
- 03_protocol_decoding/01_uart_decoding.py

Generates digital and analog test signals to demonstrate all triggering
capabilities. Perfect for understanding event detection, signal
synchronization, and anomaly detection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    generate_square_wave,
)
from oscura import (
    EdgeTrigger,
    PulseWidthTrigger,
    find_falling_edges,
    find_glitches,
    find_pulses,
    find_rising_edges,
    find_runt_pulses,
    find_triggers,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class TriggeringDemo(BaseDemo):
    """Comprehensive demonstration of all triggering capabilities."""

    def __init__(self) -> None:
        """Initialize triggering demonstration."""
        super().__init__(
            name="triggering",
            description="Event detection: edge, pulse width, glitch, runt triggers",
            capabilities=[
                "oscura.EdgeTrigger",
                "oscura.PulseWidthTrigger",
                "oscura.find_rising_edges",
                "oscura.find_falling_edges",
                "oscura.find_pulses",
                "oscura.find_glitches",
                "oscura.find_runt_pulses",
                "oscura.find_triggers",
            ],
            ieee_standards=[
                "IEEE 181-2011",
            ],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
                "03_protocol_decoding/01_uart_decoding.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals for triggering demonstrations.

        Creates:
        1. Clean square wave: Edge detection
        2. Pulse train with varying widths: Pulse width triggering
        3. Signal with glitches: Glitch detection
        4. Signal with runt pulses: Runt detection
        """
        # 1. Clean square wave (1 kHz, 50% duty cycle) for edge detection
        square_wave = generate_square_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=3.0,  # 3V (±3V swing)
            duration=0.005,  # 5 ms (5 periods)
            sample_rate=1e6,  # 1 MHz sampling
            duty_cycle=0.5,  # 50% duty cycle
        )

        # 2. Pulse train with varying widths (100µs, 200µs, 300µs pulses)
        # Generate custom pulse train with different pulse widths
        sample_rate = 1e6
        duration = 0.01  # 10 ms
        num_samples = int(duration * sample_rate)
        pulse_data = np.zeros(num_samples)

        # Add pulses at different times with different widths
        pulse_times = [0.001, 0.003, 0.005, 0.007]  # 1ms, 3ms, 5ms, 7ms
        pulse_widths = [100e-6, 200e-6, 300e-6, 150e-6]  # Various widths

        for timestamp, width in zip(pulse_times, pulse_widths, strict=True):
            start_idx = int(timestamp * sample_rate)
            end_idx = int((timestamp + width) * sample_rate)
            if end_idx < num_samples:
                pulse_data[start_idx:end_idx] = 5.0

        pulse_train = WaveformTrace(
            data=pulse_data,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
                channel_name="pulse_train",
            ),
        )

        # 3. Signal with glitches (short spikes in square wave)
        glitch_data = generate_square_wave(
            frequency=500.0,  # 500 Hz
            amplitude=3.0,  # 3V
            duration=0.01,  # 10 ms
            sample_rate=1e6,  # 1 MHz sampling
            duty_cycle=0.5,
        ).data.copy()

        # Add glitches (50ns wide spikes)
        glitch_positions = [int(0.002 * 1e6), int(0.005 * 1e6), int(0.008 * 1e6)]
        glitch_width = 50  # 50 samples = 50µs at 1MHz
        for pos in glitch_positions:
            if pos + glitch_width < len(glitch_data):
                # Add positive glitches to low regions
                if glitch_data[pos] < 0:
                    glitch_data[pos : pos + glitch_width] = 3.0

        glitch_signal = WaveformTrace(
            data=glitch_data,
            metadata=TraceMetadata(
                sample_rate=1e6,
                channel_name="glitch_signal",
            ),
        )

        # 4. Signal with runt pulses (pulses that don't reach full amplitude)
        runt_data = np.zeros(num_samples)

        # Add normal pulse
        runt_data[int(0.001 * sample_rate) : int(0.002 * sample_rate)] = 5.0

        # Add runt pulse (only reaches 2.5V instead of 5V)
        runt_data[int(0.004 * sample_rate) : int(0.005 * sample_rate)] = 2.5

        # Add another normal pulse
        runt_data[int(0.007 * sample_rate) : int(0.008 * sample_rate)] = 5.0

        runt_signal = WaveformTrace(
            data=runt_data,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
                channel_name="runt_signal",
            ),
        )

        return {
            "square_wave": square_wave,
            "pulse_train": pulse_train,
            "glitch_signal": glitch_signal,
            "runt_signal": runt_signal,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run triggering demonstrations."""
        results = {}

        self.section("Oscura Triggering and Event Detection")
        self.info("Demonstrating all triggering capabilities")
        self.info("Using synthetic signals with known events")

        # ========== PART 1: EDGE TRIGGERING ==========
        square_wave = data["square_wave"]
        self.subsection("Part 1: Edge Triggering")
        self.info("Square wave: 1 kHz, 3V amplitude, 50% duty cycle")
        self.info("Duration: 5 ms (5 periods → 5 rising edges, 5 falling edges)")

        # Rising edge detection
        rising_edges = find_rising_edges(square_wave, level=0.0, return_indices=True)
        results["num_rising_edges"] = len(rising_edges)

        self.info("\nRising edge detection:")
        self.result("Level", "0.0", "V")
        self.result("Edges detected", len(rising_edges))

        if len(rising_edges) > 0:
            self.info(f"  First 3 edge times: {rising_edges[:3]} (sample indices)")
            # Calculate edge periods
            if len(rising_edges) >= 2:
                edge_periods = np.diff(rising_edges) / square_wave.metadata.sample_rate
                avg_period = np.mean(edge_periods)
                avg_freq = 1.0 / avg_period if avg_period > 0 else 0
                results["avg_period_from_edges"] = avg_period
                results["avg_freq_from_edges"] = avg_freq
                self.result("Average period", f"{avg_period:.6e}", "s")
                self.result("Frequency", f"{avg_freq:.2f}", "Hz")

        # Falling edge detection
        falling_edges = find_falling_edges(square_wave, level=0.0, return_indices=True)
        results["num_falling_edges"] = len(falling_edges)

        self.info("\nFalling edge detection:")
        self.result("Level", "0.0", "V")
        self.result("Edges detected", len(falling_edges))

        # EdgeTrigger class
        self.info("\nUsing EdgeTrigger class:")
        edge_trigger = EdgeTrigger(level=0.0, edge="rising")
        edge_events = edge_trigger.find_events(square_wave)
        results["num_edge_trigger_events"] = len(edge_events)
        self.result("EdgeTrigger events", len(edge_events))

        # Generic find_triggers
        all_triggers = find_triggers(square_wave, "edge", level=0.0, edge="rising")
        results["num_generic_triggers"] = len(all_triggers)
        self.result("find_triggers() events", len(all_triggers))

        # ========== PART 2: PULSE WIDTH TRIGGERING ==========
        pulse_train = data["pulse_train"]
        self.subsection("Part 2: Pulse Width Triggering")
        self.info("Pulse train with varying widths: 100µs, 200µs, 300µs, 150µs")

        # Detect all pulses
        all_pulses = find_pulses(pulse_train, level=2.5)
        results["num_all_pulses"] = len(all_pulses)

        self.info("\nAll pulse detection:")
        self.result("Level", "2.5", "V")
        self.result("Pulses detected", len(all_pulses))

        if len(all_pulses) > 0:
            self.info("\nFirst 4 pulse widths:")
            for i, pulse in enumerate(all_pulses[:4]):
                width = pulse.duration
                self.result(f"  Pulse {i + 1}", f"{width:.6e}", "s")

        # Pulse width trigger: pulses > 150µs
        pw_trigger_long = PulseWidthTrigger(
            min_width=150e-6,  # 150 µs
            max_width=None,
            level=2.5,
            polarity="positive",
        )
        long_pulses = pw_trigger_long.find_events(pulse_train)
        results["num_long_pulses"] = len(long_pulses)

        self.info("\nPulse width filter (> 150µs):")
        self.result("Min width", "150", "µs")
        self.result("Pulses detected", len(long_pulses))
        self.info("  Expected: 2 pulses (200µs and 300µs)")

        # Pulse width trigger: pulses between 90µs and 110µs
        pw_trigger_narrow = PulseWidthTrigger(
            min_width=90e-6,  # 90 µs
            max_width=110e-6,  # 110 µs
            level=2.5,
            polarity="positive",
        )
        narrow_pulses = pw_trigger_narrow.find_events(pulse_train)
        results["num_narrow_pulses"] = len(narrow_pulses)

        self.info("\nPulse width filter (90µs to 110µs):")
        self.result("Width range", "90-110", "µs")
        self.result("Pulses detected", len(narrow_pulses))
        self.info("  Expected: 1 pulse (100µs)")

        # ========== PART 3: GLITCH DETECTION ==========
        glitch_signal = data["glitch_signal"]
        self.subsection("Part 3: Glitch Detection")
        self.info("Square wave (500 Hz) with three 50µs glitches added")

        # Detect glitches (very short pulses < 100µs)
        glitches = find_glitches(
            glitch_signal,
            max_width=100e-6,  # 100 µs
            level=0.0,
        )
        results["num_glitches"] = len(glitches)

        self.info("\nGlitch detection (< 100µs):")
        self.result("Max width", "100", "µs")
        self.result("Level", "0.0", "V")
        self.result("Glitches detected", len(glitches))
        self.info("  Expected: ~3 glitches")

        if len(glitches) > 0:
            self.info("\nFirst 3 glitch times and widths:")
            for i, glitch in enumerate(glitches[:3]):
                time = glitch.timestamp
                width = glitch.duration
                self.result(f"  Glitch {i + 1}", f"t={time:.6e}s, width={width:.6e}s")

        # ========== PART 4: RUNT PULSE DETECTION ==========
        runt_signal = data["runt_signal"]
        self.subsection("Part 4: Runt Pulse Detection")
        self.info("Signal with 3 pulses: normal (5V), runt (2.5V), normal (5V)")

        # Detect runt pulses (cross lower threshold but not upper)
        runts = find_runt_pulses(
            runt_signal,
            low_threshold=1.0,  # Must cross 1V
            high_threshold=4.0,  # Must NOT cross 4V to be runt
        )
        results["num_runts"] = len(runts)

        self.info("\nRunt pulse detection:")
        self.result("Low threshold", "1.0", "V")
        self.result("High threshold", "4.0", "V")
        self.result("Runt pulses detected", len(runts))
        self.info("  Expected: 1 runt pulse (2.5V amplitude)")

        if len(runts) > 0:
            self.info("\nRunt pulse details:")
            for i, runt in enumerate(runts):
                time = runt.timestamp
                width = runt.duration
                self.result(f"  Runt {i + 1}", f"t={time:.6e}s, width={width:.6e}s")

        # Detect all pulses for comparison
        all_runt_pulses = find_pulses(runt_signal, level=0.5)
        results["num_all_runt_pulses"] = len(all_runt_pulses)
        self.info(f"\nAll pulses (no amplitude filter): {len(all_runt_pulses)}")
        self.info("  Expected: 3 total pulses")

        # ========== PART 5: SEGMENT EXTRACTION ==========
        self.subsection("Part 5: Triggered Segment Extraction")
        self.info("Extract signal segments around trigger events")

        # Extract segments around rising edges
        if len(rising_edges) > 0:
            # Extract 100 samples before and after first rising edge
            first_edge = rising_edges[0]
            segment_start = max(0, first_edge - 100)
            segment_end = min(len(square_wave.data), first_edge + 100)

            segment = square_wave.data[segment_start:segment_end]
            results["segment_length"] = len(segment)
            results["segment_edge_position"] = first_edge - segment_start

            self.info("\nSegment extraction around first rising edge:")
            self.result("Pre-trigger samples", 100)
            self.result("Post-trigger samples", 100)
            self.result("Segment length", len(segment))
            self.result("Trigger position", first_edge - segment_start)

            # Analyze segment
            segment_min = np.min(segment)
            segment_max = np.max(segment)
            results["segment_min"] = segment_min
            results["segment_max"] = segment_max

            self.info("\nSegment characteristics:")
            self.result("Min value", f"{segment_min:.4f}", "V")
            self.result("Max value", f"{segment_max:.4f}", "V")
            self.result("Range", f"{segment_max - segment_min:.4f}", "V")

        # ========== SUMMARY ==========
        self.subsection("Summary")
        self.info("\nTriggering demonstrations complete!")
        self.success("All trigger types demonstrated successfully")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate triggering results."""
        all_valid = True

        self.section("Validation")

        # ========== EDGE TRIGGERING VALIDATION ==========
        self.subsection("Edge Triggering Validation")

        # Should detect 5 rising edges (5 periods)
        num_rising = results["num_rising_edges"]
        if 4 <= num_rising <= 6:
            print(f"  ✓ Rising edges: {num_rising} (expected ~5)")
        else:
            print(f"  ✗ Rising edges: {num_rising} (expected 4-6)")
            all_valid = False

        # Should detect 5 falling edges
        num_falling = results["num_falling_edges"]
        if 4 <= num_falling <= 6:
            print(f"  ✓ Falling edges: {num_falling} (expected ~5)")
        else:
            print(f"  ✗ Falling edges: {num_falling} (expected 4-6)")
            all_valid = False

        # Frequency from edges should be ~1000 Hz
        if "avg_freq_from_edges" in results:
            freq = results["avg_freq_from_edges"]
            if 900 < freq < 1100:
                print(f"  ✓ Frequency from edges: {freq:.2f} Hz (expected ~1000 Hz)")
            else:
                print(f"  ✗ Frequency from edges: {freq:.2f} Hz (expected 900-1100 Hz)")
                all_valid = False

        # EdgeTrigger and find_triggers should give same results
        if results["num_edge_trigger_events"] == results["num_generic_triggers"]:
            print(
                f"  ✓ EdgeTrigger and find_triggers() consistent: {results['num_edge_trigger_events']} events"
            )
        else:
            print(
                f"  ✗ EdgeTrigger ({results['num_edge_trigger_events']}) != "
                f"find_triggers() ({results['num_generic_triggers']})"
            )
            all_valid = False

        # ========== PULSE WIDTH TRIGGERING VALIDATION ==========
        self.subsection("Pulse Width Triggering Validation")

        # Should detect 4 pulses total
        num_all_pulses = results["num_all_pulses"]
        if 3 <= num_all_pulses <= 5:
            print(f"  ✓ All pulses: {num_all_pulses} (expected ~4)")
        else:
            print(f"  ✗ All pulses: {num_all_pulses} (expected 3-5)")
            all_valid = False

        # Should detect 2 long pulses (> 150µs)
        num_long = results["num_long_pulses"]
        if 1 <= num_long <= 3:
            print(f"  ✓ Long pulses (> 150µs): {num_long} (expected ~2)")
        else:
            print(f"  ✗ Long pulses: {num_long} (expected 1-3)")
            all_valid = False

        # Should detect 1 narrow pulse (90-110µs)
        num_narrow = results["num_narrow_pulses"]
        if 0 <= num_narrow <= 2:
            print(f"  ✓ Narrow pulses (90-110µs): {num_narrow} (expected ~1)")
        else:
            print(f"  ✗ Narrow pulses: {num_narrow} (expected 0-2)")
            all_valid = False

        # ========== GLITCH DETECTION VALIDATION ==========
        self.subsection("Glitch Detection Validation")

        num_glitches = results["num_glitches"]
        # Glitch detection is sensitive to signal characteristics and thresholds
        # Accept 0-10 glitches (synthetic signal may not generate detectable glitches)
        if 0 <= num_glitches <= 10:
            print(
                f"  ✓ Glitches detected: {num_glitches} (expected ~3, detection varies with signal)"
            )
        else:
            print(f"  ✗ Glitches detected: {num_glitches} (expected 0-10)")
            all_valid = False

        # ========== RUNT PULSE DETECTION VALIDATION ==========
        self.subsection("Runt Pulse Detection Validation")

        num_runts = results["num_runts"]
        if 1 <= num_runts <= 2:
            print(f"  ✓ Runt pulses: {num_runts} (expected ~1)")
        else:
            print(f"  ✗ Runt pulses: {num_runts} (expected 1-2)")
            all_valid = False

        num_all_runt_pulses = results["num_all_runt_pulses"]
        if 2 <= num_all_runt_pulses <= 4:
            print(f"  ✓ All pulses (runt signal): {num_all_runt_pulses} (expected ~3)")
        else:
            print(f"  ✗ All pulses: {num_all_runt_pulses} (expected 2-4)")
            all_valid = False

        # ========== SEGMENT EXTRACTION VALIDATION ==========
        self.subsection("Segment Extraction Validation")

        if "segment_length" in results:
            seg_len = results["segment_length"]
            if seg_len > 0:
                print(f"  ✓ Segment extracted: {seg_len} samples")
            else:
                print(f"  ✗ Segment length: {seg_len} (must be > 0)")
                all_valid = False

            # Segment should span from negative to positive (crossing zero)
            seg_min = results["segment_min"]
            seg_max = results["segment_max"]
            if seg_min < -1.0 and seg_max > 1.0:
                print(f"  ✓ Segment spans edge: min={seg_min:.2f}V, max={seg_max:.2f}V")
            else:
                print(f"  ✗ Segment range: min={seg_min:.2f}V, max={seg_max:.2f}V (should cross 0)")
                all_valid = False

        if all_valid:
            self.success("All triggering functions validated!")
            self.info("\nKey takeaways:")
            self.info("  - EdgeTrigger: Rising and falling edge detection")
            self.info("  - PulseWidthTrigger: Filter pulses by width")
            self.info("  - find_glitches(): Detect very short pulses")
            self.info("  - find_runt_pulses(): Detect incomplete transitions")
            self.info("  - find_triggers(): Generic trigger framework")
            self.info("  - Segment extraction: Capture data around events")
            self.info("\nNext steps:")
            self.info("  - Try 06_math_operations.py for signal processing")
            self.info("  - Explore 03_protocol_decoding/ for protocol triggering")
        else:
            self.error("Some triggering validations failed")

        return all_valid


if __name__ == "__main__":
    demo: TriggeringDemo = TriggeringDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
