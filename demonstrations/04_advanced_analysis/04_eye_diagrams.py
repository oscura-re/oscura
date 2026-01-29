"""Eye Diagrams: Signal quality visualization and BER analysis

Demonstrates:
- oscura.eye.generate_eye() - Eye diagram construction from serial data
- oscura.eye.eye_height() - Vertical eye opening measurement
- oscura.eye.eye_width() - Horizontal eye opening measurement
- oscura.eye.q_factor() - Signal quality factor (Q)
- oscura.eye.crossing_percentage() - Duty cycle distortion indicator
- oscura.eye.measure_eye() - Comprehensive eye metrics
- BER contours - Bit error rate probability analysis

IEEE Standards: IEEE 802.3 (Ethernet PHY specifications)
Related Demos:
- 04_advanced_analysis/01_jitter_analysis.py
- 04_advanced_analysis/03_signal_integrity.py

Uses NRZ serial data to demonstrate eye diagram analysis.
Perfect for understanding high-speed link quality and BER margins.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.analyzers.eye.diagram import EyeDiagram, generate_eye
from oscura.analyzers.eye.metrics import (
    EyeMetrics,
    crossing_percentage,
    eye_height,
    eye_width,
    measure_eye,
    q_factor,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class EyeDiagramDemo(BaseDemo):
    """Comprehensive demonstration of eye diagram analysis."""

    def __init__(self) -> None:
        """Initialize eye diagram demonstration."""
        super().__init__(
            name="eye_diagrams",
            description="Eye diagram analysis: height, width, Q-factor, BER contours",
            capabilities=[
                "oscura.eye.generate_eye",
                "oscura.eye.eye_height",
                "oscura.eye.eye_width",
                "oscura.eye.q_factor",
                "oscura.eye.crossing_percentage",
                "oscura.eye.measure_eye",
            ],
            ieee_standards=[
                "IEEE 802.3",
            ],
            related_demos=[
                "04_advanced_analysis/01_jitter_analysis.py",
                "04_advanced_analysis/03_signal_integrity.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate serial data signals for eye diagram analysis.

        Creates:
        1. Clean NRZ: High-quality signal with minimal noise
        2. Noisy NRZ: Signal with additive Gaussian noise
        3. Jittered NRZ: Signal with timing jitter
        4. Degraded NRZ: Combined noise and jitter (realistic)
        """
        bit_rate = 1e9  # 1 Gbps
        unit_interval = 1.0 / bit_rate  # 1 ns
        sample_rate = 100e9  # 100 GHz (100 samples per UI)
        num_bits = 1000

        # Random bit pattern (PRBS-like)
        np.random.seed(42)  # For reproducibility
        bit_pattern = np.random.randint(0, 2, num_bits)

        # 1. Clean NRZ
        clean_nrz = self._generate_nrz_signal(
            bit_pattern=bit_pattern,
            bit_rate=bit_rate,
            sample_rate=sample_rate,
            rise_time=50e-12,  # 50 ps
            noise_amplitude=0.0,
            jitter_rms=0.0,
        )

        # 2. Noisy NRZ (SNR ~ 20 dB)
        noisy_nrz = self._generate_nrz_signal(
            bit_pattern=bit_pattern,
            bit_rate=bit_rate,
            sample_rate=sample_rate,
            rise_time=50e-12,
            noise_amplitude=0.05,  # 5% noise
            jitter_rms=0.0,
        )

        # 3. Jittered NRZ
        jittered_nrz = self._generate_nrz_signal(
            bit_pattern=bit_pattern,
            bit_rate=bit_rate,
            sample_rate=sample_rate,
            rise_time=50e-12,
            noise_amplitude=0.0,
            jitter_rms=10e-12,  # 10 ps RMS jitter
        )

        # 4. Degraded NRZ (realistic)
        degraded_nrz = self._generate_nrz_signal(
            bit_pattern=bit_pattern,
            bit_rate=bit_rate,
            sample_rate=sample_rate,
            rise_time=100e-12,  # Slower edges
            noise_amplitude=0.08,  # 8% noise
            jitter_rms=20e-12,  # 20 ps RMS jitter
        )

        return {
            "clean_nrz": clean_nrz,
            "noisy_nrz": noisy_nrz,
            "jittered_nrz": jittered_nrz,
            "degraded_nrz": degraded_nrz,
            "unit_interval": unit_interval,
            "bit_rate": bit_rate,
            "sample_rate": sample_rate,
        }

    def _generate_nrz_signal(
        self,
        bit_pattern: np.ndarray,
        bit_rate: float,
        sample_rate: float,
        rise_time: float,
        noise_amplitude: float = 0.0,
        jitter_rms: float = 0.0,
    ) -> WaveformTrace:
        """Generate NRZ serial data signal.

        Args:
            bit_pattern: Array of bits (0 or 1)
            bit_rate: Bit rate in bps
            sample_rate: Sample rate in Hz
            rise_time: 10-90% rise time in seconds
            noise_amplitude: Noise amplitude as fraction of signal
            jitter_rms: RMS timing jitter in seconds

        Returns:
            WaveformTrace with NRZ signal
        """
        unit_interval = 1.0 / bit_rate
        samples_per_bit = int(sample_rate * unit_interval)
        num_samples = len(bit_pattern) * samples_per_bit

        signal = np.zeros(num_samples)
        _t = np.arange(num_samples) / sample_rate  # Time vector for reference

        # Generate ideal NRZ
        for i, bit in enumerate(bit_pattern):
            start_sample = i * samples_per_bit

            # Add jitter to bit timing
            if jitter_rms > 0:
                jitter = np.random.normal(0, jitter_rms)
                jitter_samples = int(jitter * sample_rate)
                start_sample += jitter_samples

            start_sample = max(0, min(start_sample, num_samples - samples_per_bit))

            if bit == 1:
                # Rising edge
                tau = rise_time / 2.2
                for j in range(samples_per_bit):
                    idx = start_sample + j
                    if idx < num_samples:
                        t_rel = j / sample_rate
                        signal[idx] = 1.0 - np.exp(-t_rel / tau)
            else:
                # Falling edge (exponential decay)
                tau = rise_time / 2.2
                if start_sample > 0:
                    # Decay from previous high
                    for j in range(samples_per_bit):
                        idx = start_sample + j
                        if idx < num_samples:
                            t_rel = j / sample_rate
                            signal[idx] = signal[start_sample - 1] * np.exp(-t_rel / tau)

        # Add noise
        if noise_amplitude > 0:
            noise = np.random.normal(0, noise_amplitude, num_samples)
            signal = signal + noise

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="nrz_data",
        )
        return WaveformTrace(data=signal, metadata=metadata)

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run comprehensive eye diagram demonstration."""
        results = {}

        self.section("Oscura Eye Diagram Analysis")
        self.info("Demonstrating eye diagram construction and measurements")
        self.info("Using NRZ serial data at 1 Gbps")

        unit_interval = data["unit_interval"]

        # ========== PART 1: CLEAN NRZ EYE ==========
        self.subsection("Part 1: Clean NRZ Signal (High Quality)")
        clean = data["clean_nrz"]
        self.info("Clean signal: minimal noise and jitter")

        eye_clean: EyeDiagram = generate_eye(
            clean,
            unit_interval=unit_interval,
            n_ui=2,
            max_traces=500,
        )

        self.result("Eye traces", eye_clean.n_traces)
        self.result("Samples per UI", eye_clean.samples_per_ui)

        # Measure eye metrics
        height_clean = eye_height(eye_clean)
        width_clean = eye_width(eye_clean)
        q_clean = q_factor(eye_clean)
        xing_clean = crossing_percentage(eye_clean)

        self.result("Eye height", f"{height_clean * 1e3:.2f}", "mV")
        self.result("Eye width", f"{width_clean:.4f}", "UI")
        self.result("Q-factor", f"{q_clean:.2f}")
        self.result("Crossing %", f"{xing_clean:.2f}", "%")

        results["height_clean"] = height_clean
        results["width_clean"] = width_clean
        results["q_clean"] = q_clean
        results["xing_clean"] = xing_clean

        # ========== PART 2: NOISY NRZ EYE ==========
        self.subsection("Part 2: Noisy NRZ Signal")
        noisy = data["noisy_nrz"]
        self.info("Noisy signal: 5% additive Gaussian noise")

        eye_noisy: EyeDiagram = generate_eye(
            noisy,
            unit_interval=unit_interval,
            n_ui=2,
            max_traces=500,
        )

        height_noisy = eye_height(eye_noisy)
        width_noisy = eye_width(eye_noisy)
        q_noisy = q_factor(eye_noisy)

        self.result("Eye height", f"{height_noisy * 1e3:.2f}", "mV")
        self.result("Eye width", f"{width_noisy:.4f}", "UI")
        self.result("Q-factor", f"{q_noisy:.2f}")

        # Compare to clean
        height_degradation = (1 - height_noisy / height_clean) * 100
        self.result("Height degradation", f"{height_degradation:.1f}", "%")

        results["height_noisy"] = height_noisy
        results["q_noisy"] = q_noisy
        results["height_degradation_noisy"] = height_degradation

        # ========== PART 3: JITTERED NRZ EYE ==========
        self.subsection("Part 3: Jittered NRZ Signal")
        jittered = data["jittered_nrz"]
        self.info("Jittered signal: 10 ps RMS timing jitter")

        eye_jittered: EyeDiagram = generate_eye(
            jittered,
            unit_interval=unit_interval,
            n_ui=2,
            max_traces=500,
        )

        height_jittered = eye_height(eye_jittered)
        width_jittered = eye_width(eye_jittered)
        q_jittered = q_factor(eye_jittered)

        self.result("Eye height", f"{height_jittered * 1e3:.2f}", "mV")
        self.result("Eye width", f"{width_jittered:.4f}", "UI")
        self.result("Q-factor", f"{q_jittered:.2f}")

        # Compare to clean
        width_degradation = (1 - width_jittered / width_clean) * 100
        self.result("Width degradation", f"{width_degradation:.1f}", "%")

        results["width_jittered"] = width_jittered
        results["q_jittered"] = q_jittered
        results["width_degradation_jittered"] = width_degradation

        # ========== PART 4: DEGRADED NRZ EYE ==========
        self.subsection("Part 4: Degraded NRZ Signal (Realistic)")
        degraded = data["degraded_nrz"]
        self.info("Degraded signal: 8% noise + 20 ps jitter + slower edges")

        eye_degraded: EyeDiagram = generate_eye(
            degraded,
            unit_interval=unit_interval,
            n_ui=2,
            max_traces=500,
        )

        # Comprehensive eye metrics
        metrics: EyeMetrics = measure_eye(eye_degraded, ber=1e-12)

        self.result("Eye height", f"{metrics.height * 1e3:.2f}", "mV")
        self.result("Eye width", f"{metrics.width:.4f}", "UI")
        self.result("Q-factor", f"{metrics.q_factor:.2f}")
        self.result("SNR", f"{metrics.snr:.2f}", "dB")
        self.result("BER estimate", f"{metrics.ber_estimate:.2e}")
        self.result("Crossing %", f"{metrics.crossing_percent:.2f}", "%")

        # BER-extrapolated metrics
        if metrics.height_at_ber is not None:
            self.result("Eye height @ 1e-12 BER", f"{metrics.height_at_ber * 1e3:.2f}", "mV")
        if metrics.width_at_ber is not None:
            self.result("Eye width @ 1e-12 BER", f"{metrics.width_at_ber:.4f}", "UI")

        results["height_degraded"] = metrics.height
        results["width_degraded"] = metrics.width
        results["q_degraded"] = metrics.q_factor
        results["snr_degraded"] = metrics.snr
        results["ber_estimate"] = metrics.ber_estimate

        # ========== EYE DIAGRAM INTERPRETATION ==========
        self.subsection("Eye Diagram Interpretation")

        self.info("\n[Eye Height]")
        self.info("  Vertical eye opening (voltage margin)")
        self.info("  Reduced by noise, ISI, and vertical jitter")
        self.info("  Larger = better noise immunity")

        self.info("\n[Eye Width]")
        self.info("  Horizontal eye opening (timing margin)")
        self.info("  Reduced by timing jitter and ISI")
        self.info("  Wider = more timing margin for clock recovery")

        self.info("\n[Q-factor]")
        self.info("  Signal quality metric: Q = (mu_high - mu_low) / (sigma_high + sigma_low)")
        self.info("  Higher Q -> lower BER")
        self.info("  Q = 7 -> BER ~= 1e-12")

        self.info("\n[Crossing Percentage]")
        self.info("  Where eye crosses vertically (ideal = 50%)")
        self.info("  Deviation indicates duty cycle distortion")

        self.info("\n[Quality Comparison:]")
        self.info(f"  Clean:    Q = {q_clean:.2f}, Height = {height_clean * 1e3:.1f} mV")
        self.info(f"  Noisy:    Q = {q_noisy:.2f}, Height = {height_noisy * 1e3:.1f} mV")
        self.info(f"  Jittered: Q = {q_jittered:.2f}, Width = {width_jittered:.3f} UI")
        self.info(f"  Degraded: Q = {metrics.q_factor:.2f}, BER ≈ {metrics.ber_estimate:.1e}")

        self.success("All eye diagram measurements complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate eye diagram measurements."""
        self.info("Validating eye diagram measurements...")

        all_valid = True

        # Validate clean eye
        self.subsection("Clean Eye Validation")

        # Clean eye should have high Q-factor (> 10)
        if results["q_clean"] > 10:
            self.success(f"Clean Q-factor: {results['q_clean']:.2f} > 10")
        else:
            self.warning(f"Clean Q-factor lower than expected: {results['q_clean']:.2f}")

        # Eye width should be close to 1.0 UI for clean signal
        if results["width_clean"] > 0.8:
            self.success(f"Clean eye width: {results['width_clean']:.4f} UI > 0.8")
        else:
            self.warning(f"Clean eye width: {results['width_clean']:.4f} UI")

        # Crossing should be near 50%
        if 45 < results["xing_clean"] < 55:
            self.success(f"Crossing: {results['xing_clean']:.2f}% ≈ 50%")
        else:
            self.info(f"Crossing: {results['xing_clean']:.2f}% (target: 50%)")

        # Validate noise degradation
        self.subsection("Noise Impact Validation")

        # Noisy eye should have lower Q than clean
        if results["q_noisy"] < results["q_clean"]:
            self.success(f"Noise reduced Q: {results['q_noisy']:.2f} < {results['q_clean']:.2f}")
        else:
            self.warning("Noisy Q should be lower than clean")
            all_valid = False

        # Height degradation should be significant (> 5%)
        if results["height_degradation_noisy"] > 5:
            self.success(f"Noise reduced height by {results['height_degradation_noisy']:.1f}%")
        else:
            self.info(f"Height degradation: {results['height_degradation_noisy']:.1f}%")

        # Validate jitter degradation
        self.subsection("Jitter Impact Validation")

        # Jittered eye should have lower Q than clean
        if results["q_jittered"] < results["q_clean"]:
            self.success(
                f"Jitter reduced Q: {results['q_jittered']:.2f} < {results['q_clean']:.2f}"
            )
        else:
            self.warning("Jittered Q should be lower than clean")

        # Width degradation should be significant (> 5%)
        if results["width_degradation_jittered"] > 5:
            self.success(f"Jitter reduced width by {results['width_degradation_jittered']:.1f}%")
        else:
            self.info(f"Width degradation: {results['width_degradation_jittered']:.1f}%")

        # Validate degraded eye
        self.subsection("Degraded Eye Validation")

        # Degraded Q should be lowest
        if (
            results["q_degraded"] < results["q_noisy"]
            and results["q_degraded"] < results["q_jittered"]
        ):
            self.success(f"Degraded Q ({results['q_degraded']:.2f}) is lowest (as expected)")
        else:
            self.info(f"Degraded Q: {results['q_degraded']:.2f}")

        # BER estimate should be reasonable
        if 1e-15 < results["ber_estimate"] < 1e-3:
            self.success(f"BER estimate: {results['ber_estimate']:.2e} (reasonable)")
        else:
            self.info(f"BER estimate: {results['ber_estimate']:.2e}")

        # SNR should be positive
        if results["snr_degraded"] > 0:
            self.success(f"SNR: {results['snr_degraded']:.2f} dB > 0")
        else:
            self.warning(f"SNR too low: {results['snr_degraded']:.2f} dB")

        if all_valid:
            self.success("All eye diagram measurements validated!")
            self.info("\nKey takeaways:")
            self.info("  - Eye height measures vertical noise margin")
            self.info("  - Eye width measures horizontal timing margin")
            self.info("  - Q-factor relates to BER (Q=7 → BER≈1e-12)")
            self.info("  - Noise reduces height, jitter reduces width")
            self.info("\nNext steps:")
            self.info("  - Try 04_advanced_analysis/06_quality_assessment.py")
            self.info("  - Explore equalizer design for link optimization")
        else:
            self.error("Some eye diagram measurements failed validation")

        return all_valid


if __name__ == "__main__":
    demo: EyeDiagramDemo = EyeDiagramDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
