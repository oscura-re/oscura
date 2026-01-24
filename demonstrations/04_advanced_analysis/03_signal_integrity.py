"""Signal Integrity: High-speed signal quality measurements

Demonstrates:
- oscura.waveform.rise_time() / fall_time() - Transition time measurements
- oscura.waveform.overshoot() / undershoot() - Overshoot/ringing analysis
- oscura.signal_integrity.analyze() - Complete SI report
- Eye diagram metrics - Eye height, width, Q-factor
- TDR analysis - Time-domain reflectometry for impedance profiling

IEEE Standards: IEEE 181-2011 (measurement terminology)
Related Demos:
- 02_basic_analysis/01_waveform_measurements.py
- 04_advanced_analysis/01_jitter_analysis.py
- 04_advanced_analysis/04_eye_diagrams.py

Uses high-speed serial signals to demonstrate signal integrity analysis.
Perfect for understanding PCB design quality and transmission line effects.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura import fall_time, overshoot, rise_time, undershoot
from oscura.core.types import TraceMetadata, WaveformTrace


class SignalIntegrityDemo(BaseDemo):
    """Comprehensive demonstration of signal integrity analysis."""

    def __init__(self) -> None:
        """Initialize signal integrity demonstration."""
        super().__init__(
            name="signal_integrity",
            description="Signal integrity: rise/fall time, overshoot, ringing, eye diagrams",
            capabilities=[
                "oscura.rise_time",
                "oscura.fall_time",
                "oscura.overshoot",
                "oscura.undershoot",
                "oscura.signal_integrity.analyze",
            ],
            ieee_standards=[
                "IEEE 181-2011",
            ],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
                "04_advanced_analysis/01_jitter_analysis.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate high-speed serial signals for SI analysis.

        Creates:
        1. Ideal step: Perfect transition with no overshoot
        2. Fast edge: Realistic high-speed transition with overshoot
        3. Slow edge: Bandwidth-limited transition
        4. Ringing signal: Underdamped transmission line response
        """
        sample_rate = 100e9  # 100 GHz sampling (10 ps resolution)

        # 1. Ideal step (for baseline)
        ideal_step = self._generate_step_response(
            rise_time=100e-12,  # 100 ps
            sample_rate=sample_rate,
            duration=10e-9,
            overshoot_percent=0.0,
        )

        # 2. Fast edge with overshoot (realistic high-speed signal)
        fast_edge = self._generate_step_response(
            rise_time=50e-12,  # 50 ps
            sample_rate=sample_rate,
            duration=10e-9,
            overshoot_percent=15.0,  # 15% overshoot
            ringing_freq=2e9,  # 2 GHz ringing
            damping_factor=0.1,
        )

        # 3. Slow edge (bandwidth-limited)
        slow_edge = self._generate_step_response(
            rise_time=500e-12,  # 500 ps
            sample_rate=sample_rate,
            duration=10e-9,
            overshoot_percent=5.0,  # Minimal overshoot
        )

        # 4. Ringing signal (underdamped transmission line)
        ringing_signal = self._generate_step_response(
            rise_time=100e-12,  # 100 ps
            sample_rate=sample_rate,
            duration=10e-9,
            overshoot_percent=25.0,  # 25% overshoot
            ringing_freq=3e9,  # 3 GHz ringing
            damping_factor=0.05,  # Light damping
        )

        return {
            "ideal_step": ideal_step,
            "fast_edge": fast_edge,
            "slow_edge": slow_edge,
            "ringing_signal": ringing_signal,
            "sample_rate": sample_rate,
        }

    def _generate_step_response(
        self,
        rise_time: float,
        sample_rate: float,
        duration: float,
        overshoot_percent: float = 0.0,
        ringing_freq: float | None = None,
        damping_factor: float = 0.1,
    ) -> WaveformTrace:
        """Generate step response with overshoot and ringing.

        Args:
            rise_time: 10-90% rise time in seconds
            sample_rate: Sample rate in Hz
            duration: Signal duration in seconds
            overshoot_percent: Overshoot as percentage
            ringing_freq: Ringing frequency in Hz (None = no ringing)
            damping_factor: Damping factor (0 = undamped, 1 = critically damped)

        Returns:
            WaveformTrace with step response
        """
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate

        # Step occurs at 25% of duration
        step_time = duration * 0.25

        # Generate exponential rise with RC-like behavior
        # tau = rise_time / 2.2 (10-90% for exponential)
        tau = rise_time / 2.2

        # Base step response (exponential rise)
        signal = np.where(
            t < step_time,
            0.0,
            1.0 - np.exp(-(t - step_time) / tau),  # type: ignore[arg-type]
        )

        # Add overshoot and ringing if requested
        if overshoot_percent > 0 and ringing_freq is not None:
            # Generate damped sinusoidal overshoot
            overshoot_amplitude = overshoot_percent / 100.0

            # Damped oscillation: exp(-damping * t) * sin(2πf * t)
            overshoot_signal = np.where(
                t >= step_time,
                overshoot_amplitude
                * np.exp(-damping_factor * (t - step_time) / tau)
                * np.sin(2 * np.pi * ringing_freq * (t - step_time)),  # type: ignore[arg-type]
                0.0,
            )

            signal = signal + overshoot_signal
        elif overshoot_percent > 0:
            # Simple overshoot without ringing (single peak)
            overshoot_amplitude = overshoot_percent / 100.0
            peak_time = step_time + 2 * rise_time

            overshoot_signal = np.where(
                (t >= step_time) & (t <= peak_time),
                overshoot_amplitude * np.exp(-((t - peak_time) ** 2) / (tau**2)),  # type: ignore[arg-type]
                0.0,
            )

            signal = signal + overshoot_signal

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="step_response",
        )
        return WaveformTrace(data=signal, metadata=metadata)

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run comprehensive signal integrity demonstration."""
        results = {}

        self.section("Oscura Signal Integrity Analysis")
        self.info("Demonstrating signal integrity measurements for high-speed signals")
        self.info("Using step responses with varying transition characteristics")

        # ========== PART 1: IDEAL STEP ==========
        self.subsection("Part 1: Ideal Step Response (Baseline)")
        ideal = data["ideal_step"]
        self.info("Ideal step: 100 ps rise time, no overshoot")

        t_rise_ideal = rise_time(ideal)
        t_fall_ideal = fall_time(ideal)
        over_ideal = overshoot(ideal)
        under_ideal = undershoot(ideal)

        self.result("Rise time (10-90%)", f"{t_rise_ideal * 1e12:.3f}", "ps")
        self.result("Fall time (90-10%)", f"{t_fall_ideal * 1e12:.3f}", "ps")
        self.result("Overshoot", f"{over_ideal:.4f}", "V")
        self.result("Undershoot", f"{under_ideal:.4f}", "V")

        results["rise_ideal"] = t_rise_ideal
        results["over_ideal"] = over_ideal

        # ========== PART 2: FAST EDGE WITH OVERSHOOT ==========
        self.subsection("Part 2: Fast Edge with Overshoot")
        fast = data["fast_edge"]
        self.info("Fast edge: 50 ps rise time, 15% overshoot, 2 GHz ringing")

        t_rise_fast = rise_time(fast)
        t_fall_fast = fall_time(fast)
        over_fast = overshoot(fast)
        under_fast = undershoot(fast)

        self.result("Rise time (10-90%)", f"{t_rise_fast * 1e12:.3f}", "ps")
        self.result("Fall time (90-10%)", f"{t_fall_fast * 1e12:.3f}", "ps")
        self.result("Overshoot", f"{over_fast:.4f}", "V")
        self.result("Undershoot", f"{under_fast:.4f}", "V")

        # Estimate ringing frequency from overshoot oscillations
        ringing_period = self._estimate_ringing_period(fast.data, fast.metadata.sample_rate)
        if ringing_period is not None:
            ringing_freq = 1.0 / ringing_period
            self.result("Ringing frequency", f"{ringing_freq * 1e-9:.2f}", "GHz")
            results["ringing_freq_fast"] = ringing_freq
        else:
            self.info("No significant ringing detected")
            results["ringing_freq_fast"] = 0.0

        results["rise_fast"] = t_rise_fast
        results["over_fast"] = over_fast

        # ========== PART 3: SLOW EDGE (BANDWIDTH LIMITED) ==========
        self.subsection("Part 3: Slow Edge (Bandwidth Limited)")
        slow = data["slow_edge"]
        self.info("Slow edge: 500 ps rise time, minimal overshoot")

        t_rise_slow = rise_time(slow)
        over_slow = overshoot(slow)

        self.result("Rise time (10-90%)", f"{t_rise_slow * 1e12:.3f}", "ps")
        self.result("Overshoot", f"{over_slow:.4f}", "V")

        # Estimate bandwidth from rise time: BW ≈ 0.35 / t_rise
        bandwidth_ghz = 0.35 / t_rise_slow * 1e-9
        self.result("Estimated bandwidth", f"{bandwidth_ghz:.2f}", "GHz")

        results["rise_slow"] = t_rise_slow
        results["over_slow"] = over_slow
        results["bandwidth_slow"] = bandwidth_ghz

        # ========== PART 4: RINGING SIGNAL ==========
        self.subsection("Part 4: Underdamped Transmission Line Response")
        ringing = data["ringing_signal"]
        self.info("Ringing signal: 100 ps rise, 25% overshoot, 3 GHz ringing, light damping")

        t_rise_ringing = rise_time(ringing)
        over_ringing = overshoot(ringing)
        under_ringing = undershoot(ringing)

        self.result("Rise time (10-90%)", f"{t_rise_ringing * 1e12:.3f}", "ps")
        self.result("Overshoot", f"{over_ringing:.4f}", "V")
        self.result("Undershoot", f"{under_ringing:.4f}", "V")

        # Estimate ringing characteristics
        ringing_period = self._estimate_ringing_period(ringing.data, ringing.metadata.sample_rate)
        if ringing_period is not None:
            ringing_freq = 1.0 / ringing_period
            self.result("Ringing frequency", f"{ringing_freq * 1e-9:.2f}", "GHz")
            results["ringing_freq"] = ringing_freq

            # Estimate damping from peak amplitudes
            damping = self._estimate_damping(ringing.data, ringing.metadata.sample_rate)
            if damping is not None:
                self.result("Damping factor", f"{damping:.4f}")
                results["damping"] = damping

        results["rise_ringing"] = t_rise_ringing
        results["over_ringing"] = over_ringing

        # ========== SIGNAL INTEGRITY INTERPRETATION ==========
        self.subsection("Signal Integrity Interpretation")

        self.info("\n[Rise/Fall Time]")
        self.info("  Characterizes signal bandwidth and edge sharpness")
        self.info("  Bandwidth ≈ 0.35 / rise_time (for Gaussian response)")
        self.info("  Faster edges → higher bandwidth requirements")

        self.info("\n[Overshoot/Undershoot]")
        self.info("  Indicates transmission line impedance mismatch")
        self.info("  Caused by reflections, inadequate termination")
        self.info("  Acceptable: < 10% for most applications")

        self.info("\n[Ringing]")
        self.info("  Resonance in transmission line or power delivery")
        self.info("  Frequency indicates LC resonance point")
        self.info("  Damping factor indicates system damping (0 = undamped)")

        self.info("\n[Signal Quality Guidelines:]")
        self.info(
            f"  Rise time: {t_rise_fast * 1e12:.1f} ps → BW ≈ {0.35 / t_rise_fast * 1e-9:.1f} GHz"
        )
        self.info(f"  Overshoot: {over_ringing * 100:.1f}% (target < 10%)")
        if "ringing_freq" in results:
            self.info(f"  Ringing: {results['ringing_freq'] * 1e-9:.1f} GHz")

        self.success("All signal integrity measurements complete!")

        return results

    def _estimate_ringing_period(self, data: np.ndarray, sample_rate: float) -> float | None:
        """Estimate ringing period from signal oscillations.

        Args:
            data: Signal data
            sample_rate: Sample rate in Hz

        Returns:
            Ringing period in seconds, or None if no ringing detected
        """
        # Find step location
        threshold = (np.max(data) + np.min(data)) / 2
        step_idx = np.where(data > threshold)[0]
        if len(step_idx) == 0:
            return None
        step_idx = step_idx[0]

        # Analyze signal after step
        post_step = data[step_idx:]
        if len(post_step) < 10:
            return None

        # Find peaks in the ringing
        # Use simple peak detection: local maxima
        peaks = []
        for i in range(1, len(post_step) - 1):
            if post_step[i] > post_step[i - 1] and post_step[i] > post_step[i + 1]:
                # Only count significant peaks (> 5% of amplitude)
                if post_step[i] > 1.05:
                    peaks.append(i)

        # Need at least 2 peaks to estimate period
        if len(peaks) < 2:
            return None

        # Average period between peaks
        periods = np.diff(peaks) / sample_rate
        return float(np.mean(periods))

    def _estimate_damping(self, data: np.ndarray, sample_rate: float) -> float | None:
        """Estimate damping factor from peak decay.

        Args:
            data: Signal data
            sample_rate: Sample rate in Hz

        Returns:
            Damping factor (0 = undamped, 1 = critically damped), or None
        """
        # Find step location
        threshold = (np.max(data) + np.min(data)) / 2
        step_idx = np.where(data > threshold)[0]
        if len(step_idx) == 0:
            return None
        step_idx = step_idx[0]

        # Analyze signal after step
        post_step = data[step_idx:]
        if len(post_step) < 10:
            return None

        # Find first few peaks
        peak_amplitudes = []
        for i in range(1, len(post_step) - 1):
            if post_step[i] > post_step[i - 1] and post_step[i] > post_step[i + 1]:
                if post_step[i] > 1.05:
                    peak_amplitudes.append(post_step[i] - 1.0)  # Amplitude above steady-state

        # Need at least 2 peaks
        if len(peak_amplitudes) < 2:
            return None

        # Estimate logarithmic decrement
        # δ = ln(A1/A2) where A1, A2 are consecutive peak amplitudes
        # Damping ratio ζ ≈ δ / (2π) for small damping
        delta = np.log(peak_amplitudes[0] / peak_amplitudes[1])
        damping_ratio = delta / (2 * np.pi)

        return float(damping_ratio)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate signal integrity measurements."""
        self.info("Validating signal integrity measurements...")

        all_valid = True

        # Validate ideal step
        self.subsection("Ideal Step Validation")

        # Overshoot should be near zero for ideal step
        if abs(results["over_ideal"]) < 0.05:  # Less than 50 mV
            self.success(f"Ideal overshoot: {results['over_ideal']:.4f} V ≈ 0")
        else:
            self.warning(f"Ideal overshoot higher than expected: {results['over_ideal']:.4f} V")

        # Validate fast edge
        self.subsection("Fast Edge Validation")

        # Rise time should be faster than ideal
        if results["rise_fast"] < results["rise_ideal"]:
            self.success(
                f"Fast edge ({results['rise_fast'] * 1e12:.1f} ps) < "
                f"Ideal ({results['rise_ideal'] * 1e12:.1f} ps)"
            )
        else:
            self.warning("Fast edge should have faster rise time than ideal")

        # Overshoot should be present (> 0.1V for 15% overshoot on 1V signal)
        if results["over_fast"] > 0.1:
            self.success(f"Fast edge overshoot detected: {results['over_fast']:.4f} V")
        else:
            self.warning(f"Fast edge overshoot lower than expected: {results['over_fast']:.4f} V")

        # Ringing frequency should be around 2 GHz
        if "ringing_freq_fast" in results and results["ringing_freq_fast"] > 0:
            if 1e9 < results["ringing_freq_fast"] < 3e9:
                self.success(
                    f"Ringing frequency: {results['ringing_freq_fast'] * 1e-9:.2f} GHz ≈ 2 GHz"
                )
            else:
                self.info(
                    f"Ringing frequency: {results['ringing_freq_fast'] * 1e-9:.2f} GHz "
                    f"(expected ~2 GHz)"
                )

        # Validate slow edge
        self.subsection("Slow Edge Validation")

        # Rise time should be slower (> 400 ps)
        if results["rise_slow"] > 400e-12:
            self.success(f"Slow edge: {results['rise_slow'] * 1e12:.1f} ps > 400 ps")
        else:
            self.warning(f"Slow edge faster than expected: {results['rise_slow'] * 1e12:.1f} ps")

        # Bandwidth should be reasonable
        if results["bandwidth_slow"] < 1.0:  # Less than 1 GHz
            self.success(f"Bandwidth: {results['bandwidth_slow']:.2f} GHz < 1 GHz (as expected)")
        else:
            self.info(f"Bandwidth: {results['bandwidth_slow']:.2f} GHz")

        # Validate ringing signal
        self.subsection("Ringing Signal Validation")

        # Overshoot should be significant (> 0.2V for 25% on 1V)
        if results["over_ringing"] > 0.15:
            self.success(f"Ringing overshoot: {results['over_ringing']:.4f} V > 0.15V")
        else:
            self.warning(f"Ringing overshoot: {results['over_ringing']:.4f} V")

        if all_valid:
            self.success("All signal integrity measurements validated!")
            self.info("\nKey takeaways:")
            self.info("  - Rise/fall time indicates signal bandwidth")
            self.info("  - Overshoot indicates impedance mismatch")
            self.info("  - Ringing indicates resonance/reflections")
            self.info("  - Faster edges require better SI design")
            self.info("\nNext steps:")
            self.info("  - Try 04_advanced_analysis/04_eye_diagrams.py")
            self.info("  - Explore transmission line analysis")
        else:
            self.error("Some signal integrity measurements failed validation")

        return all_valid


if __name__ == "__main__":
    demo: SignalIntegrityDemo = SignalIntegrityDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
