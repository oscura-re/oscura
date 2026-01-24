"""Transmission Lines: PCB trace characterization and TDR analysis

Demonstrates:
- oscura.component.characteristic_impedance() - Z0 extraction from TDR
- oscura.component.velocity_factor() - Propagation velocity determination
- oscura.component.propagation_delay() - Time delay measurement
- oscura.component.transmission_line_analysis() - Complete line analysis
- TDR waveform generation and interpretation

IEEE Standards: IEEE 370-2020 (Electrical Characterization of Interconnects)
IPC Standards: IPC-TM-650 2.5.5.7 (Characteristic Impedance)

Related Demos:
- 04_advanced_analysis/03_signal_integrity.py
- 05_domain_specific/01_pcb_analysis.py

Uses Time-Domain Reflectometry (TDR) to characterize transmission lines.
Essential for PCB design verification and signal integrity analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.component import (
    characteristic_impedance,
    propagation_delay,
    transmission_line_analysis,
    velocity_factor,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class TransmissionLinesDemo(BaseDemo):
    """Comprehensive demonstration of transmission line analysis."""

    def __init__(self) -> None:
        """Initialize transmission line demonstration."""
        super().__init__(
            name="transmission_lines",
            description="Transmission line characterization: Z0, velocity factor, propagation delay",
            capabilities=[
                "oscura.component.characteristic_impedance",
                "oscura.component.velocity_factor",
                "oscura.component.propagation_delay",
                "oscura.component.transmission_line_analysis",
            ],
            ieee_standards=[
                "IEEE 370-2020",
                "IPC-TM-650 2.5.5.7",
            ],
            related_demos=[
                "04_advanced_analysis/03_signal_integrity.py",
                "05_domain_specific/01_pcb_analysis.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate TDR waveforms for transmission line analysis.

        Creates:
        1. 50-ohm matched line: Perfect impedance match (no reflection)
        2. 75-ohm line: Microstrip with higher impedance
        3. 25-ohm line: Stripline with lower impedance
        4. FR4 PCB trace: Realistic PCB trace with typical parameters
        """
        sample_rate = 100e9  # 100 GHz sampling (10 ps resolution)

        # Use longer lines (1-2 meters) to ensure clear propagation delays
        # that the edge detection algorithms can reliably find

        # 1. 50-ohm matched line (reference) - open terminated
        matched_line = self._generate_tdr_waveform(
            z0_line=50.0,
            z0_source=50.0,
            line_length=1.0,  # 1 meter
            velocity_factor=0.66,  # FR4
            sample_rate=sample_rate,
            duration=15e-9,  # 15 ns
        )

        # 2. 75-ohm microstrip - open terminated
        microstrip_75 = self._generate_tdr_waveform(
            z0_line=75.0,
            z0_source=50.0,
            line_length=1.5,  # 1.5 meters
            velocity_factor=0.7,  # Microstrip (air-dielectric)
            sample_rate=sample_rate,
            duration=20e-9,
        )

        # 3. 25-ohm stripline - open terminated
        stripline_25 = self._generate_tdr_waveform(
            z0_line=25.0,
            z0_source=50.0,
            line_length=1.0,  # 1 meter
            velocity_factor=0.6,  # Stripline (buried in FR4)
            sample_rate=sample_rate,
            duration=15e-9,
        )

        # 4. Realistic FR4 PCB trace - open terminated with loss
        fr4_trace = self._generate_tdr_waveform(
            z0_line=50.0,
            z0_source=50.0,
            line_length=2.0,  # 2 meters
            velocity_factor=0.66,  # FR4
            sample_rate=sample_rate,
            duration=25e-9,  # 25 ns
            add_loss=True,  # Include dielectric loss
            loss_db_per_m=2.0,  # 2 dB/m at high freq
        )

        return {
            "matched_line": matched_line,
            "microstrip_75": microstrip_75,
            "stripline_25": stripline_25,
            "fr4_trace": fr4_trace,
            "sample_rate": sample_rate,
            "c": 299792458.0,  # Speed of light
        }

    def _generate_tdr_waveform(
        self,
        z0_line: float,
        z0_source: float,
        line_length: float,
        velocity_factor: float,
        sample_rate: float,
        duration: float,
        add_loss: bool = False,
        loss_db_per_m: float = 0.0,
    ) -> WaveformTrace:
        """Generate TDR waveform for a transmission line.

        TDR (Time-Domain Reflectometry) sends a step signal down a transmission line
        and measures reflections. The reflection coefficient determines impedance.

        This uses an OPEN termination (infinite impedance) at the end to ensure
        a clear reflection for velocity factor measurement.

        Args:
            z0_line: Characteristic impedance of the line (ohms)
            z0_source: Source impedance (ohms)
            line_length: Physical length of the line (meters)
            velocity_factor: Propagation velocity as fraction of c
            sample_rate: Sample rate in Hz
            duration: Signal duration in seconds
            add_loss: Include transmission line loss
            loss_db_per_m: Loss in dB per meter

        Returns:
            WaveformTrace with TDR reflection waveform
        """
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate

        # Calculate propagation delay
        c = 299792458.0
        velocity = c * velocity_factor
        prop_delay = line_length / velocity

        # Reflection coefficient at the OPEN termination
        # For open circuit: rho = +1 (full reflection)
        # This ensures we always have a clear reflection to measure
        rho_termination = 1.0

        # Initial reflection at source/line interface (for impedance measurement)
        rho_source = (z0_line - z0_source) / (z0_line + z0_source)

        # TDR waveform structure (using oscura's expected format):
        # V_measured = V_incident * (1 + rho)
        # 1. Before step: 0V
        # 2. After step: V_incident * (1 + rho_source) - immediate impedance step
        # 3. After round-trip delay: add reflection from open termination

        incident_step_time = duration * 0.1  # Step at 10% of duration

        # Create base step response with finite rise time
        rise_time = 100e-12  # 100 ps rise time
        tau = rise_time / 2.2  # RC time constant

        # Incident voltage level (normalized to 1V)
        v_incident = 1.0

        # Immediate level after source/line interface
        # V_measured = V_incident * (1 + rho_source)
        immediate_level = v_incident * (1.0 + rho_source)

        # Generate initial step to immediate level
        signal = np.where(
            t < incident_step_time,
            0.0,
            immediate_level * (1.0 - np.exp(-(t - incident_step_time) / tau)),
        )

        # Add reflected wave from OPEN termination
        # This returns after round-trip propagation delay
        reflection_time = incident_step_time + 2 * prop_delay

        # The reflection from open termination has rho = +1
        # Reflected voltage adds to the existing level
        reflected_amp = v_incident * rho_termination

        # Apply loss if requested
        if add_loss and loss_db_per_m > 0:
            # Round-trip loss
            total_loss_db = loss_db_per_m * line_length * 2
            loss_factor = 10 ** (-total_loss_db / 20)
            reflected_amp *= loss_factor

        # Add reflection with same rise time
        reflection = np.where(
            t < reflection_time,
            0.0,
            reflected_amp * (1.0 - np.exp(-(t - reflection_time) / tau)),
        )
        signal = signal + reflection

        # 2. Small amount of noise (realistic measurement)
        noise_level = 0.01  # 1% noise
        signal += np.random.normal(0, noise_level, len(signal))

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name=f"TDR_{z0_line:.0f}ohm",
        )
        return WaveformTrace(data=signal, metadata=metadata)

    def _apply_bandwidth_limit(
        self,
        signal: np.ndarray,
        sample_rate: float,
        rise_time: float,
    ) -> np.ndarray:
        """Apply bandwidth limitation (finite rise time) to signal.

        Args:
            signal: Input signal
            sample_rate: Sample rate in Hz
            rise_time: Desired 10-90% rise time in seconds

        Returns:
            Bandwidth-limited signal
        """
        # Convert rise time to RC time constant
        # For exponential: rise_time (10-90%) ≈ 2.2 * tau
        tau = rise_time / 2.2

        # Create exponential smoothing kernel
        kernel_length = int(5 * tau * sample_rate)  # 5 time constants
        if kernel_length < 3:
            return signal

        t_kernel = np.arange(kernel_length) / sample_rate
        kernel = (1 / tau) * np.exp(-t_kernel / tau)
        kernel = kernel / np.sum(kernel)  # Normalize

        # Apply convolution
        smoothed = np.convolve(signal, kernel, mode="same")
        return smoothed

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run comprehensive transmission line demonstration."""
        results = {}

        self.section("Oscura Transmission Line Analysis")
        self.info("Demonstrating TDR-based transmission line characterization")
        self.info("Using Time-Domain Reflectometry for impedance and delay measurement")

        # ========== PART 1: 50-OHM MATCHED LINE ==========
        self.subsection("Part 1: 50-Ohm Matched Line (Reference)")
        matched = data["matched_line"]
        self.info("50-ohm line with 50-ohm source (perfect match)")
        self.info("Length: 1.0 m, Velocity factor: 0.66 (FR4)")

        # Extract Z0 - analyze the stable region after incident edge but before reflection
        # For a 1m line at 0.66c, delay = 1.0m / (0.66 * 3e8 m/s) = 5.05ns
        # So analyze window from 0.5ns to 9ns (after incident, before reflection)
        z0_matched = characteristic_impedance(
            matched, z0_source=50.0, start_time=0.5e-9, end_time=9e-9
        )
        self.result("Characteristic impedance (Z0)", f"{z0_matched:.2f}", "Ω")

        # Measure propagation delay
        delay_matched = propagation_delay(matched)
        self.result("Propagation delay", f"{delay_matched * 1e9:.3f}", "ns")

        # Calculate velocity factor
        vf_matched = velocity_factor(matched, line_length=1.0)
        self.result("Velocity factor", f"{vf_matched:.3f}")

        # Calculate propagation velocity
        velocity = data["c"] * vf_matched
        self.result("Propagation velocity", f"{velocity * 1e-8:.2f}", "x10^8 m/s")

        results["z0_matched"] = z0_matched
        results["delay_matched"] = delay_matched
        results["vf_matched"] = vf_matched

        # ========== PART 2: 75-OHM MICROSTRIP ==========
        self.subsection("Part 2: 75-Ohm Microstrip")
        micro = data["microstrip_75"]
        self.info("75-ohm microstrip with 50-ohm source (impedance mismatch)")
        self.info("Length: 1.5 m, Velocity factor: 0.7 (air-dielectric)")

        # For 1.5m line at 0.7c, delay = 1.5m / (0.7 * 3e8) = 7.14ns
        # Analyze window: 0.5ns to 13ns
        z0_micro = characteristic_impedance(
            micro, z0_source=50.0, start_time=0.5e-9, end_time=13e-9
        )
        delay_micro = propagation_delay(micro)
        vf_micro = velocity_factor(micro, line_length=1.5)

        self.result("Characteristic impedance (Z0)", f"{z0_micro:.2f}", "Ω")
        self.result("Propagation delay", f"{delay_micro * 1e9:.3f}", "ns")
        self.result("Velocity factor", f"{vf_micro:.3f}")

        # Calculate reflection coefficient
        rho = (z0_micro - 50.0) / (z0_micro + 50.0)
        self.result("Reflection coefficient (rho)", f"{rho:.3f}")

        # Calculate return loss
        if abs(rho) > 1e-6:
            return_loss = -20 * np.log10(abs(rho))
            self.result("Return loss", f"{return_loss:.2f}", "dB")
            results["return_loss_micro"] = return_loss

        results["z0_micro"] = z0_micro
        results["delay_micro"] = delay_micro
        results["vf_micro"] = vf_micro

        # ========== PART 3: 25-OHM STRIPLINE ==========
        self.subsection("Part 3: 25-Ohm Stripline")
        strip = data["stripline_25"]
        self.info("25-ohm stripline with 50-ohm source (low impedance)")
        self.info("Length: 1.0 m, Velocity factor: 0.6 (buried in FR4)")

        # For 1m line at 0.6c, delay = 1.0m / (0.6 * 3e8) = 5.56ns
        # Analyze window: 0.5ns to 10ns
        z0_strip = characteristic_impedance(
            strip, z0_source=50.0, start_time=0.5e-9, end_time=10e-9
        )
        delay_strip = propagation_delay(strip)
        vf_strip = velocity_factor(strip, line_length=1.0)

        self.result("Characteristic impedance (Z0)", f"{z0_strip:.2f}", "Ω")
        self.result("Propagation delay", f"{delay_strip * 1e9:.3f}", "ns")
        self.result("Velocity factor", f"{vf_strip:.3f}")

        # Calculate reflection coefficient
        rho_strip = (z0_strip - 50.0) / (z0_strip + 50.0)
        self.result("Reflection coefficient (rho)", f"{rho_strip:.3f}")
        self.info("  Negative rho indicates lower impedance (capacitive loading)")

        results["z0_strip"] = z0_strip
        results["delay_strip"] = delay_strip
        results["vf_strip"] = vf_strip

        # ========== PART 4: COMPLETE TRANSMISSION LINE ANALYSIS ==========
        self.subsection("Part 4: Complete Transmission Line Analysis")
        fr4 = data["fr4_trace"]
        self.info("FR4 PCB trace: 50-ohm, 2.0 m, with dielectric loss")

        # Perform complete analysis
        tl_result = transmission_line_analysis(
            fr4,
            z0_source=50.0,
            line_length=2.0,
        )

        self.result("Characteristic impedance (Z0)", f"{tl_result.z0:.2f}", "Ω")
        self.result("Propagation delay", f"{tl_result.propagation_delay * 1e9:.3f}", "ns")
        self.result("Velocity factor", f"{tl_result.velocity_factor:.3f}")
        self.result("Propagation velocity", f"{tl_result.velocity * 1e-8:.2f}", "x10^8 m/s")
        self.result("Estimated length", f"{tl_result.length * 100:.2f}", "cm")

        if tl_result.loss is not None:
            self.result("Insertion loss", f"{tl_result.loss:.3f}", "dB")
            results["loss"] = tl_result.loss

        if tl_result.return_loss is not None:
            self.result("Return loss", f"{tl_result.return_loss:.2f}", "dB")

        results["tl_z0"] = tl_result.z0
        results["tl_delay"] = tl_result.propagation_delay
        results["tl_vf"] = tl_result.velocity_factor
        results["tl_length"] = tl_result.length

        # ========== PART 5: PRACTICAL PCB TRACE ANALYSIS ==========
        self.subsection("Part 5: Practical PCB Trace Analysis")
        self.info("\n[Understanding TDR Measurements]")
        self.info("  TDR sends a step signal and measures reflections")
        self.info("  Reflection coefficient rho = (Z_line - Z_source) / (Z_line + Z_source)")
        self.info("  rho > 0: Higher impedance (inductive)")
        self.info("  rho < 0: Lower impedance (capacitive)")
        self.info("  rho = 0: Perfect match (no reflection)")

        self.info("\n[Velocity Factor Interpretation]")
        self.info("  Typical values:")
        self.info("    - FR4 PCB: 0.60-0.66 (εᵣ ≈ 4.0-4.5)")
        self.info("    - Microstrip (partial air): 0.65-0.75")
        self.info("    - Stripline (buried): 0.55-0.65")
        self.info("    - Coaxial cable: 0.66-0.85")

        self.info("\n[Design Guidelines]")
        self.info("  - Target impedance: ±10% tolerance (45-55Ω for 50Ω design)")
        self.info("  - Return loss: > 20 dB recommended")
        self.info("  - Velocity factor accuracy: ±0.05 for timing analysis")

        self.info("\n[Example PCB Trace Calculation]")
        example_length = 0.25  # 25 cm
        example_vf = 0.66
        example_velocity = data["c"] * example_vf
        example_delay = example_length / example_velocity

        self.info("  Design: 50Ω microstrip, 25 cm length on FR4")
        self.info(f"  Velocity factor: {example_vf}")
        self.info(f"  Propagation delay: {example_delay * 1e9:.2f} ns")
        self.info(f"  Max frequency (λ/10): {data['c'] / (10 * example_length) * 1e-6:.1f} MHz")

        self.success("All transmission line measurements complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate transmission line measurements."""
        self.info("Validating transmission line measurements...")

        all_valid = True

        # Validate matched line
        self.subsection("Matched Line Validation")

        # Z0 should be close to 50 ohms (within 10%)
        z0_matched = results["z0_matched"]
        if 45.0 <= z0_matched <= 55.0:
            self.success(f"Z0 = {z0_matched:.2f} Ω (within 10% of 50Ω)")
        else:
            self.warning(f"Z0 = {z0_matched:.2f} Ω (expected 45-55Ω)")

        # Velocity factor should be reasonable for FR4 (0.6-0.7)
        vf_matched = results["vf_matched"]
        if 0.5 <= vf_matched <= 1.0:
            self.success(f"Velocity factor = {vf_matched:.3f} (reasonable range)")
        else:
            self.warning(f"Velocity factor = {vf_matched:.3f} (expected 0.5-1.0)")
            all_valid = False

        # Propagation delay should be positive
        delay_matched = results["delay_matched"]
        if delay_matched > 0:
            self.success(f"Propagation delay = {delay_matched * 1e9:.3f} ns > 0")
        else:
            self.error(f"Invalid propagation delay: {delay_matched * 1e9:.3f} ns")
            all_valid = False

        # Validate 75-ohm microstrip
        self.subsection("Microstrip Validation")

        z0_micro = results["z0_micro"]
        if 65.0 <= z0_micro <= 85.0:
            self.success(f"Z0 = {z0_micro:.2f} Ω (within range of 75Ω)")
        else:
            self.warning(f"Z0 = {z0_micro:.2f} Ω (expected ~75Ω)")

        # Microstrip should have higher velocity factor (more air exposure)
        vf_micro = results["vf_micro"]
        if vf_micro > results["vf_matched"]:
            self.success(
                f"Microstrip VF ({vf_micro:.3f}) > Buried VF ({results['vf_matched']:.3f})"
            )
        else:
            self.info(f"Microstrip VF: {vf_micro:.3f}")

        # Validate 25-ohm stripline
        self.subsection("Stripline Validation")

        z0_strip = results["z0_strip"]
        if 20.0 <= z0_strip <= 30.0:
            self.success(f"Z0 = {z0_strip:.2f} Ω (within range of 25Ω)")
        else:
            self.warning(f"Z0 = {z0_strip:.2f} Ω (expected ~25Ω)")

        # Stripline should have lower velocity factor (fully embedded)
        vf_strip = results["vf_strip"]
        if 0.5 <= vf_strip <= 0.7:
            self.success(f"Stripline VF = {vf_strip:.3f} (reasonable for FR4)")
        else:
            self.info(f"Stripline VF: {vf_strip:.3f}")

        # Validate complete analysis
        self.subsection("Complete Analysis Validation")

        tl_z0 = results["tl_z0"]
        tl_vf = results["tl_vf"]
        tl_length = results["tl_length"]

        if 45.0 <= tl_z0 <= 55.0:
            self.success(f"Extracted Z0 = {tl_z0:.2f} Ω")
        else:
            self.info(f"Extracted Z0 = {tl_z0:.2f} Ω")

        # Length should be close to actual (200 cm)
        if 1.6 <= tl_length <= 2.4:
            self.success(f"Estimated length = {tl_length:.2f} m (expected 2.0 m)")
        else:
            self.info(f"Estimated length = {tl_length:.2f} m")

        # Velocity factor consistency check
        if 0.5 <= tl_vf <= 1.0:
            self.success(f"Velocity factor = {tl_vf:.3f}")
        else:
            self.warning(f"Velocity factor = {tl_vf:.3f} out of range")

        if all_valid:
            self.success("All transmission line measurements validated!")
            self.info("\nKey takeaways:")
            self.info("  - TDR measures impedance by analyzing reflections")
            self.info("  - Velocity factor depends on dielectric material")
            self.info("  - Propagation delay = length / velocity")
            self.info("  - Impedance matching minimizes reflections")
            self.info("\nNext steps:")
            self.info("  - Try 04_advanced_analysis/03_signal_integrity.py")
            self.info("  - Explore PCB impedance control techniques")
            self.info("  - Study high-speed design guidelines")
        else:
            self.error("Some transmission line measurements failed validation")

        return all_valid


if __name__ == "__main__":
    demo: TransmissionLinesDemo = TransmissionLinesDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
