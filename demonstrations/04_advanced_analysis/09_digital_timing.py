"""Digital Timing Analysis: Clock recovery, setup/hold time, timing verification

Demonstrates:
- oscura.analyzers.digital.clock.recover_clock() - Clock signal reconstruction
- oscura.analyzers.digital.clock.detect_clock_frequency() - Frequency detection
- oscura.analyzers.digital.timing.setup_time() - Setup time measurement
- oscura.analyzers.digital.timing.hold_time() - Hold time measurement
- oscura.analyzers.digital.edges.check_timing_constraints() - Constraint validation
- oscura.analyzers.digital.edges.measure_edge_timing() - Edge timing statistics

IEEE Standards: IEEE 181-2011 (Transitional Waveform Definitions)
JEDEC Standards: JEDEC No. 65B (High-Speed Interface Timing)

Related Demos:
- 04_advanced_analysis/01_jitter_analysis.py
- 04_advanced_analysis/04_eye_diagrams.py
- 03_protocol_decoding/01_uart.py

Uses clock and data signals with controlled timing variations to demonstrate
FPGA/ASIC timing verification and digital design validation techniques.
Perfect for understanding setup/hold time analysis and clock domain crossing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, generate_square_wave, validate_approximately
from oscura.analyzers.digital.clock import (
    ClockMetrics,
    detect_clock_frequency,
    measure_clock_jitter,
    recover_clock,
)
from oscura.analyzers.digital.edges import (
    EdgeTiming,
    TimingConstraint,
    check_timing_constraints,
    detect_edges,
    measure_edge_timing,
)
from oscura.analyzers.digital.timing import hold_time, setup_time
from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace


class DigitalTimingDemo(BaseDemo):
    """Comprehensive demonstration of digital timing analysis."""

    def __init__(self) -> None:
        """Initialize digital timing demonstration."""
        super().__init__(
            name="digital_timing",
            description=(
                "Digital timing analysis: clock recovery, setup/hold time, constraint checking"
            ),
            capabilities=[
                "oscura.analyzers.digital.clock.recover_clock",
                "oscura.analyzers.digital.clock.detect_clock_frequency",
                "oscura.analyzers.digital.timing.setup_time",
                "oscura.analyzers.digital.timing.hold_time",
                "oscura.analyzers.digital.edges.check_timing_constraints",
                "oscura.analyzers.digital.edges.measure_edge_timing",
            ],
            ieee_standards=[
                "IEEE 181-2011",
                "JEDEC No. 65B",
            ],
            related_demos=[
                "04_advanced_analysis/01_jitter_analysis.py",
                "04_advanced_analysis/04_eye_diagrams.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate clock and data signals with timing variations.

        Creates:
        1. Clean clock: Reference clock at 100 MHz
        2. Noisy clock: Clock with jitter for recovery testing
        3. Data signal: Synchronous data with timing variations
        4. Setup violation: Data with insufficient setup time
        5. Hold violation: Data with insufficient hold time
        """
        sample_rate = 10e9  # 10 GHz sampling (100 ps resolution)
        clock_freq = 100e6  # 100 MHz clock (10 ns period)
        duration = 1e-5  # 10 µs (1000 clock cycles)

        # 1. Clean reference clock
        clean_clock = generate_square_wave(
            frequency=clock_freq,
            amplitude=1.0,
            duration=duration,
            sample_rate=sample_rate,
            duty_cycle=0.5,
        )

        # 2. Noisy clock with jitter
        noisy_clock = self._generate_jittered_clock(
            clock_freq=clock_freq,
            duration=duration,
            sample_rate=sample_rate,
            period_jitter_rms=100e-12,  # 100 ps RMS jitter
            duty_cycle=0.5,
        )

        # 3. Data signal synchronized to clock
        good_data = self._generate_data_signal(
            clock_trace=clean_clock,
            sample_rate=sample_rate,
            setup_time=2e-9,  # 2 ns setup time
            hold_time=1e-9,  # 1 ns hold time
        )

        # 4. Data signal with setup violation
        setup_violation_data = self._generate_data_signal(
            clock_trace=clean_clock,
            sample_rate=sample_rate,
            setup_time=500e-12,  # 500 ps setup time (too short)
            hold_time=1e-9,
        )

        # 5. Data signal with hold violation
        hold_violation_data = self._generate_data_signal(
            clock_trace=clean_clock,
            sample_rate=sample_rate,
            setup_time=2e-9,
            hold_time=200e-12,  # 200 ps hold time (too short)
        )

        return {
            "clean_clock": clean_clock,
            "noisy_clock": noisy_clock,
            "good_data": good_data,
            "setup_violation_data": setup_violation_data,
            "hold_violation_data": hold_violation_data,
            "sample_rate": sample_rate,
            "clock_freq": clock_freq,
            "nominal_period": 1.0 / clock_freq,
        }

    def _generate_jittered_clock(
        self,
        clock_freq: float,
        duration: float,
        sample_rate: float,
        period_jitter_rms: float,
        duty_cycle: float = 0.5,
    ) -> WaveformTrace:
        """Generate clock with specified period jitter.

        Args:
            clock_freq: Nominal clock frequency
            duration: Signal duration in seconds
            sample_rate: Sample rate in Hz
            period_jitter_rms: RMS period jitter in seconds
            duty_cycle: Duty cycle (0.0 to 1.0)

        Returns:
            WaveformTrace with jittered clock
        """
        nominal_period = 1.0 / clock_freq
        num_samples = int(duration * sample_rate)

        # Generate jittered edge timestamps
        num_cycles = int(duration * clock_freq)
        edge_times = [0.0]

        for _ in range(num_cycles * 2):  # Rising and falling edges
            # Add nominal half-period with jitter
            jitter = np.random.normal(0, period_jitter_rms)
            if len(edge_times) % 2 == 0:
                # Rising edge: add high time
                edge_times.append(edge_times[-1] + nominal_period * duty_cycle + jitter)
            else:
                # Falling edge: add low time
                edge_times.append(edge_times[-1] + nominal_period * (1 - duty_cycle) + jitter)

        # Generate waveform from edge timestamps
        signal = np.zeros(num_samples)
        for i, edge_time in enumerate(edge_times):
            if edge_time > duration:
                break
            edge_idx = int(edge_time * sample_rate)
            if edge_idx >= num_samples:
                break

            # Rising edge (even index)
            if i % 2 == 0:
                signal[edge_idx:] = 1.0
            else:
                signal[edge_idx:] = 0.0

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="jittered_clock",
        )
        return WaveformTrace(data=signal, metadata=metadata)

    def _generate_data_signal(
        self,
        clock_trace: WaveformTrace,
        sample_rate: float,
        setup_time: float,
        hold_time: float,
    ) -> WaveformTrace:
        """Generate data signal with specified timing relative to clock.

        Args:
            clock_trace: Reference clock signal
            sample_rate: Sample rate in Hz
            setup_time: Data setup time before clock edge
            hold_time: Data hold time after clock edge

        Returns:
            WaveformTrace with data signal
        """
        clock_data = clock_trace.data
        num_samples = len(clock_data)

        # Find rising edges of clock
        threshold = 0.5
        rising_edges = np.where((clock_data[:-1] < threshold) & (clock_data[1:] >= threshold))[0]

        # Generate random data pattern
        data_signal = np.zeros(num_samples)

        # For each clock edge, place data transition before it
        for i, edge_idx in enumerate(rising_edges):
            # Skip first few edges
            if i < 2:
                continue

            # Random data value
            data_value = np.random.randint(0, 2)

            # Calculate data transition time (setup_time before clock edge)
            data_transition_idx = edge_idx - int(setup_time * sample_rate)

            if data_transition_idx > 0 and data_transition_idx < num_samples:
                # Hold data stable after clock edge
                hold_end_idx = edge_idx + int(hold_time * sample_rate)
                if hold_end_idx < num_samples:
                    data_signal[data_transition_idx:hold_end_idx] = data_value

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="data_signal",
        )
        return WaveformTrace(data=data_signal, metadata=metadata)

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run comprehensive digital timing analysis demonstration."""
        results = {}

        self.section("Oscura Digital Timing Analysis")
        self.info("Demonstrating clock recovery and setup/hold time analysis")
        self.info("IEEE 181-2011 and JEDEC No. 65B compliant measurements")

        # ========== PART 1: CLOCK FREQUENCY DETECTION ==========
        self.subsection("Part 1: Clock Frequency Detection")
        clean_clock = data["clean_clock"]
        self.info("Detecting clock frequency from clean 100 MHz clock")

        # Test different detection methods
        freq_edge = detect_clock_frequency(
            clean_clock.data, sample_rate=data["sample_rate"], method="edge"
        )
        freq_fft = detect_clock_frequency(
            clean_clock.data, sample_rate=data["sample_rate"], method="fft"
        )
        freq_autocorr = detect_clock_frequency(
            clean_clock.data, sample_rate=data["sample_rate"], method="autocorr"
        )

        self.result("Edge method", f"{freq_edge / 1e6:.3f}", "MHz")
        self.result("FFT method", f"{freq_fft / 1e6:.3f}", "MHz")
        self.result("Autocorrelation method", f"{freq_autocorr / 1e6:.3f}", "MHz")

        results["freq_edge"] = freq_edge
        results["freq_fft"] = freq_fft
        results["freq_autocorr"] = freq_autocorr

        # ========== PART 2: CLOCK RECOVERY ==========
        self.subsection("Part 2: Clock Recovery from Noisy Signal")
        noisy_clock = data["noisy_clock"]
        self.info("Recovering clock from signal with 100 ps RMS jitter")

        # Recover clock using different methods
        recovered_edge = recover_clock(
            noisy_clock.data, sample_rate=data["sample_rate"], method="edge"
        )
        recovered_pll = recover_clock(
            noisy_clock.data, sample_rate=data["sample_rate"], method="pll"
        )

        self.result("Recovered (edge method)", f"{len(recovered_edge)}", "samples")
        self.result("Recovered (PLL method)", f"{len(recovered_pll)}", "samples")

        results["recovered_edge"] = recovered_edge
        results["recovered_pll"] = recovered_pll

        # ========== PART 3: CLOCK JITTER MEASUREMENT ==========
        self.subsection("Part 3: Clock Jitter Measurement")

        # Measure jitter on clean clock
        clean_metrics: ClockMetrics = measure_clock_jitter(
            clean_clock.data, sample_rate=data["sample_rate"]
        )

        self.result("Clean clock frequency", f"{clean_metrics.frequency / 1e6:.3f}", "MHz")
        self.result("Clean clock period", f"{clean_metrics.period_seconds * 1e9:.6f}", "ns")
        self.result("Clean clock RMS jitter", f"{clean_metrics.jitter_rms * 1e12:.3f}", "ps")
        self.result("Clean clock P-P jitter", f"{clean_metrics.jitter_pp * 1e12:.3f}", "ps")
        self.result("Clean clock duty cycle", f"{clean_metrics.duty_cycle * 100:.3f}", "%")

        # Measure jitter on noisy clock
        noisy_metrics: ClockMetrics = measure_clock_jitter(
            noisy_clock.data, sample_rate=data["sample_rate"]
        )

        self.result("Noisy clock RMS jitter", f"{noisy_metrics.jitter_rms * 1e12:.3f}", "ps")
        self.result("Noisy clock P-P jitter", f"{noisy_metrics.jitter_pp * 1e12:.3f}", "ps")
        self.result("Noisy clock stability", f"{noisy_metrics.stability:.3f}", "")

        results["clean_metrics"] = clean_metrics
        results["noisy_metrics"] = noisy_metrics

        # ========== PART 4: SETUP/HOLD TIME MEASUREMENT ==========
        self.subsection("Part 4: Setup/Hold Time Measurement")
        good_data = data["good_data"]
        self.info("Measuring setup/hold times for properly timed data")

        # Measure setup time
        t_setup = setup_time(good_data, clean_clock, clock_edge="rising")

        # Measure hold time
        t_hold = hold_time(good_data, clean_clock, clock_edge="rising")

        self.result("Setup time", f"{t_setup * 1e9:.3f}", "ns" if not np.isnan(t_setup) else "")
        self.result("Hold time", f"{t_hold * 1e9:.3f}", "ns" if not np.isnan(t_hold) else "")

        results["setup_time_good"] = t_setup
        results["hold_time_good"] = t_hold

        # ========== PART 5: TIMING VIOLATIONS ==========
        self.subsection("Part 5: Timing Violation Detection")

        # Setup violation
        setup_violation_data = data["setup_violation_data"]
        t_setup_bad = setup_time(setup_violation_data, clean_clock, clock_edge="rising")

        self.info("Setup violation data (500 ps setup time):")
        self.result(
            "Measured setup time",
            f"{t_setup_bad * 1e9:.3f}" if not np.isnan(t_setup_bad) else "N/A",
            "ns",
        )

        # Hold violation
        hold_violation_data = data["hold_violation_data"]
        t_hold_bad = hold_time(hold_violation_data, clean_clock, clock_edge="rising")

        self.info("Hold violation data (200 ps hold time):")
        self.result(
            "Measured hold time",
            f"{t_hold_bad * 1e9:.3f}" if not np.isnan(t_hold_bad) else "N/A",
            "ns",
        )

        results["setup_time_bad"] = t_setup_bad
        results["hold_time_bad"] = t_hold_bad

        # ========== PART 6: EDGE TIMING STATISTICS ==========
        self.subsection("Part 6: Edge Timing Statistics")

        # Convert to DigitalTrace for edge detection
        digital_clock = DigitalTrace(
            data=clean_clock.data > 0.5,
            metadata=TraceMetadata(sample_rate=data["sample_rate"], channel_name="clock"),
        )

        # Detect all edges
        all_edges = detect_edges(
            digital_clock.data.astype(np.float64),
            edge_type="both",
            sample_rate=data["sample_rate"],
        )

        self.result("Total edges detected", len(all_edges))
        self.result("Rising edges", sum(1 for e in all_edges if e.edge_type == "rising"))
        self.result("Falling edges", sum(1 for e in all_edges if e.edge_type == "falling"))

        # Measure edge timing (uses all edges, so period is half-cycle)
        all_edge_timing: EdgeTiming = measure_edge_timing(
            all_edges, sample_rate=data["sample_rate"]
        )

        # For clock period, use only rising edges
        rising_edges = [e for e in all_edges if e.edge_type == "rising"]
        rising_edge_timing: EdgeTiming = measure_edge_timing(
            rising_edges, sample_rate=data["sample_rate"]
        )

        self.result(
            "Clock period (full cycle)", f"{rising_edge_timing.mean_period * 1e9:.6f}", "ns"
        )
        self.result("Period std dev", f"{rising_edge_timing.std_period * 1e12:.3f}", "ps")
        self.result("Min period", f"{rising_edge_timing.min_period * 1e9:.6f}", "ns")
        self.result("Max period", f"{rising_edge_timing.max_period * 1e9:.6f}", "ns")
        self.result("Mean duty cycle", f"{all_edge_timing.mean_duty_cycle * 100:.3f}", "%")
        self.result("RMS jitter", f"{rising_edge_timing.jitter_rms * 1e12:.3f}", "ps")
        self.result("Peak-to-peak jitter", f"{rising_edge_timing.jitter_pp * 1e12:.3f}", "ps")

        results["edge_timing"] = rising_edge_timing

        # ========== PART 7: TIMING CONSTRAINT CHECKING ==========
        self.subsection("Part 7: Timing Constraint Validation")

        # Define timing constraints for full clock period (rising edge to rising edge)
        constraints = [
            TimingConstraint(
                name="min_period",
                min_time=9e-9,
                reference="rising",  # 9 ns minimum period (relaxed for jitter)
            ),
            TimingConstraint(
                name="max_period",
                max_time=11e-9,
                reference="rising",  # 11 ns maximum period (relaxed for jitter)
            ),
        ]

        # Check constraints on rising edges only
        violations = check_timing_constraints(
            rising_edges, constraints, sample_rate=data["sample_rate"]
        )

        self.result("Constraints checked", len(constraints))
        self.result("Violations found", len(violations))

        if violations:
            self.warning(f"Found {len(violations)} timing violations:")
            for i, v in enumerate(violations[:5]):  # Show first 5
                self.info(
                    f"  #{i + 1}: {v.constraint.name} at edge {v.edge_index} "
                    f"({v.measured_time * 1e9:.3f} ns)"
                )
        else:
            self.success("All timing constraints satisfied!")

        results["violations"] = violations

        # ========== TIMING ANALYSIS INTERPRETATION ==========
        self.subsection("Timing Analysis Interpretation")

        self.info("\n[Clock Recovery]")
        self.info("  Edge-based: Best for clean digital signals with sharp transitions")
        self.info("  PLL-based: Tracks phase variations, robust to jitter")
        self.info("  FFT-based: Best for periodic signals with noise")

        self.info("\n[Setup/Hold Time]")
        self.info("  Setup time: Data must be stable BEFORE clock edge")
        self.info("  Hold time: Data must be stable AFTER clock edge")
        self.info("  Critical for: FPGA timing closure, ASIC design verification")

        self.info("\n[Edge Timing Statistics]")
        self.info("  Period jitter: Affects clock domain crossing reliability")
        self.info("  Duty cycle: Important for DDR interfaces and duty cycle correction")
        self.info("  Constraint checking: Validates timing margins for design signoff")

        self.success("All timing measurements complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate timing measurements."""
        self.info("Validating timing measurements...")

        all_valid = True

        # Validate frequency detection
        self.subsection("Frequency Detection Validation")

        expected_freq = 100e6  # 100 MHz
        tolerance = 0.01  # 1% tolerance

        if validate_approximately(
            results["freq_edge"], expected_freq, tolerance=tolerance, name="Edge method frequency"
        ):
            pass
        else:
            all_valid = False

        if validate_approximately(
            results["freq_fft"], expected_freq, tolerance=tolerance, name="FFT method frequency"
        ):
            pass
        else:
            all_valid = False

        # Validate clock metrics
        self.subsection("Clock Metrics Validation")

        clean_metrics: ClockMetrics = results["clean_metrics"]
        noisy_metrics: ClockMetrics = results["noisy_metrics"]

        # Clean clock should have low jitter
        if clean_metrics.jitter_rms < 500e-12:  # Less than 500 ps
            self.success(f"Clean clock RMS jitter: {clean_metrics.jitter_rms * 1e12:.3f} ps")
        else:
            self.warning(
                f"Clean clock jitter higher than expected: {clean_metrics.jitter_rms * 1e12:.3f} ps"
            )

        # Noisy clock should have higher jitter
        if noisy_metrics.jitter_rms > clean_metrics.jitter_rms:
            self.success(
                f"Noisy clock jitter ({noisy_metrics.jitter_rms * 1e12:.3f} ps) > "
                f"clean clock jitter ({clean_metrics.jitter_rms * 1e12:.3f} ps)"
            )
        else:
            self.warning("Noisy clock jitter not higher than clean clock (random variation)")

        # Validate duty cycle
        if validate_approximately(clean_metrics.duty_cycle, 0.5, tolerance=0.05, name="Duty cycle"):
            pass
        else:
            all_valid = False

        # Validate setup/hold times
        self.subsection("Setup/Hold Time Validation")

        # Good data should have reasonable setup/hold times
        if not np.isnan(results["setup_time_good"]):
            if results["setup_time_good"] > 1e-9:  # > 1 ns
                self.success(f"Good data setup time: {results['setup_time_good'] * 1e9:.3f} ns")
            else:
                self.warning(
                    f"Setup time lower than expected: {results['setup_time_good'] * 1e9:.3f} ns"
                )
        else:
            self.info("Setup time measurement returned NaN (may be due to signal characteristics)")

        if not np.isnan(results["hold_time_good"]):
            if results["hold_time_good"] > 0.5e-9:  # > 0.5 ns
                self.success(f"Good data hold time: {results['hold_time_good'] * 1e9:.3f} ns")
            else:
                self.warning(
                    f"Hold time lower than expected: {results['hold_time_good'] * 1e9:.3f} ns"
                )
        else:
            self.info("Hold time measurement returned NaN (may be due to signal characteristics)")

        # Validate edge timing
        self.subsection("Edge Timing Validation")

        edge_timing: EdgeTiming = results["edge_timing"]

        # Period should be close to 10 ns (100 MHz)
        expected_period = 10e-9
        if validate_approximately(
            edge_timing.mean_period, expected_period, tolerance=0.05, name="Mean period"
        ):
            pass
        else:
            all_valid = False

        # Duty cycle should be close to 50%
        if edge_timing.mean_duty_cycle > 0:
            if validate_approximately(
                edge_timing.mean_duty_cycle,
                0.5,
                tolerance=0.1,
                name="Edge timing duty cycle",
            ):
                pass
            else:
                self.warning("Duty cycle measurement may be affected by edge detection")

        if all_valid:
            self.success("All timing measurements validated!")
            self.info("\nKey takeaways:")
            self.info("  - Clock recovery: Multiple methods for different scenarios")
            self.info("  - Setup/hold: Critical for reliable data capture")
            self.info("  - Edge timing: Comprehensive jitter and period analysis")
            self.info("  - Constraints: Automated verification against specifications")
            self.info("\nNext steps:")
            self.info("  - Try 04_advanced_analysis/01_jitter_analysis.py for deep dive")
            self.info("  - Explore 04_advanced_analysis/04_eye_diagrams.py for BER analysis")
        else:
            self.error("Some timing measurements failed validation")

        return all_valid


if __name__ == "__main__":
    demo: DigitalTimingDemo = DigitalTimingDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
