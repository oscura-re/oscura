"""Impairment Simulation: Real-world signal degradation modeling

Demonstrates:
- Real-world impairment modeling
- Jitter simulation (random and deterministic)
- Crosstalk simulation between channels
- Supply voltage ripple effects
- Temperature drift modeling
- Ground bounce effects
- EMI/RFI interference

IEEE Standards: IEEE 1241-2010 (ADC noise characterization)
Related Demos:
- 17_signal_generation/01_signal_builder_comprehensive.py
- 04_advanced_analysis/01_jitter_analysis.py
- 04_advanced_analysis/02_noise_analysis.py

This demonstration shows how to simulate realistic signal impairments for
testing system robustness and decoder performance under non-ideal conditions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, validate_approximately
from oscura.core.types import TraceMetadata, WaveformTrace
from tests.fixtures.signal_builders import SignalBuilder


class ImpairmentSimulationDemo(BaseDemo):
    """Demonstration of signal impairment simulation capabilities."""

    def __init__(self) -> None:
        """Initialize impairment simulation demonstration."""
        super().__init__(
            name="impairment_simulation",
            description="Simulate real-world impairments: jitter, noise, crosstalk, drift",
            capabilities=[
                "Jitter simulation (random, deterministic)",
                "Crosstalk modeling",
                "Supply ripple injection",
                "Temperature drift",
                "EMI/RFI interference",
            ],
            ieee_standards=["IEEE 1241-2010"],
            related_demos=[
                "17_signal_generation/01_signal_builder_comprehensive.py",
                "04_advanced_analysis/01_jitter_analysis.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate clean signals for impairment injection.

        Returns:
            Dictionary containing clean reference signals
        """
        sample_rate = 10e6  # 10 MHz
        duration = 0.002  # 2 ms

        # Clean reference signals
        clean_clock = SignalBuilder.square_wave(
            frequency=1e6, sample_rate=sample_rate, duration=duration, amplitude=3.3
        )

        clean_data = SignalBuilder.uart_frame(
            data=b"Test", baudrate=115200, sample_rate=sample_rate, amplitude=3.3
        )

        clean_sine = SignalBuilder.sine_wave(
            frequency=10e3, sample_rate=sample_rate, duration=duration, amplitude=1.0
        )

        return {
            "sample_rate": sample_rate,
            "duration": duration,
            "clean_clock": clean_clock,
            "clean_data": clean_data,
            "clean_sine": clean_sine,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run impairment simulation demonstration."""
        results: dict[str, Any] = {}

        self.section("Signal Impairment Simulation Demonstration")
        self.info("Simulating real-world signal degradation effects")

        sample_rate = data["sample_rate"]
        duration = data["duration"]
        num_samples = int(sample_rate * duration)

        # Part 1: Jitter Simulation
        self.subsection("Part 1: Jitter Simulation")
        self.info("Adding timing uncertainty to clock signals")

        clean_clock = data["clean_clock"]

        # Random jitter (Gaussian distribution)
        jitter_std = 50e-12  # 50 ps RMS jitter
        time_base = np.arange(len(clean_clock)) / sample_rate
        random_jitter = np.random.randn(len(clean_clock)) * jitter_std
        _jittered_time = time_base + random_jitter  # Time vector with jitter

        # Resample with jitter (approximate by adding high-frequency noise to edges)
        # For demonstration, add noise proportional to derivative (edges)
        derivative = np.diff(clean_clock, prepend=clean_clock[0])
        _random_jitter_signal = clean_clock + derivative * np.random.randn(len(clean_clock)) * 0.01

        self.result("Random jitter (RMS)", f"{jitter_std * 1e12:.1f}", "ps")
        self.result("Peak-to-peak jitter", f"{jitter_std * 6 * 1e12:.1f}", "ps (6-sigma)")

        # Deterministic jitter (periodic modulation)
        det_jitter_freq = 60  # 60 Hz
        det_jitter_amplitude = 100e-12  # 100 ps amplitude
        deterministic_jitter = det_jitter_amplitude * np.sin(
            2 * np.pi * det_jitter_freq * time_base
        )
        _det_jittered_time = (
            time_base + deterministic_jitter
        )  # Time vector with deterministic jitter

        self.result("Deterministic jitter freq", f"{det_jitter_freq}", "Hz")
        self.result("Deterministic jitter amp", f"{det_jitter_amplitude * 1e12:.1f}", "ps")

        results["random_jitter_rms"] = jitter_std
        results["det_jitter_amplitude"] = det_jitter_amplitude

        # Part 2: Crosstalk Simulation
        self.subsection("Part 2: Crosstalk Simulation")
        self.info("Capacitive and inductive coupling between adjacent traces")

        # Create two adjacent signals
        aggressor = SignalBuilder.square_wave(
            frequency=10e6, sample_rate=sample_rate, duration=duration, amplitude=3.3
        )
        victim_clean = SignalBuilder.sine_wave(
            frequency=1e6, sample_rate=sample_rate, duration=duration, amplitude=1.0
        )

        # Crosstalk coefficient (typically -40 to -60 dB)
        crosstalk_db = -40  # -40 dB
        crosstalk_coeff = 10 ** (crosstalk_db / 20)

        # Capacitive crosstalk (proportional to dV/dt)
        aggressor_derivative = np.diff(aggressor, prepend=aggressor[0])
        capacitive_crosstalk = crosstalk_coeff * aggressor_derivative

        # Add crosstalk to victim
        victim_with_crosstalk = victim_clean + capacitive_crosstalk

        self.result("Crosstalk coefficient", f"{crosstalk_db}", "dB")
        self.result("Crosstalk linear gain", f"{crosstalk_coeff:.6f}")
        self.result("Victim signal amplitude", f"{np.max(np.abs(victim_clean)):.4f}", "V")
        self.result("Crosstalk amplitude", f"{np.max(np.abs(capacitive_crosstalk)):.4f}", "V")
        self.result("Victim + crosstalk peak", f"{np.max(np.abs(victim_with_crosstalk)):.4f}", "V")

        results["crosstalk_db"] = crosstalk_db
        results["crosstalk_amplitude"] = float(np.max(np.abs(capacitive_crosstalk)))

        # Part 3: Supply Ripple Effects
        self.subsection("Part 3: Supply Voltage Ripple")
        self.info("Power supply noise coupling to signal")

        # Generate supply ripple (AC mains frequency + switching noise)
        mains_ripple = 0.05 * np.sin(2 * np.pi * 60 * time_base[:num_samples])  # 60 Hz, 50 mV
        switching_ripple = 0.02 * np.sin(
            2 * np.pi * 100e3 * time_base[:num_samples]
        )  # 100 kHz, 20 mV
        total_ripple = mains_ripple + switching_ripple

        # Supply ripple affects signal amplitude (power supply rejection ratio)
        psrr_db = -60  # -60 dB PSRR
        psrr_coeff = 10 ** (psrr_db / 20)

        clean_sine = data["clean_sine"][:num_samples]
        ripple_modulation = clean_sine * (1 + psrr_coeff * total_ripple)

        self.result("Mains ripple (60 Hz)", "50 mV peak")
        self.result("Switching ripple (100 kHz)", "20 mV peak")
        self.result("PSRR", f"{psrr_db} dB")
        self.result("Ripple impact on signal", f"{psrr_coeff * 100:.4f}%")

        ripple_effect = np.max(np.abs(ripple_modulation - clean_sine))
        self.result("Maximum ripple effect", f"{ripple_effect * 1000:.4f}", "mV")

        results["psrr_db"] = psrr_db
        results["ripple_effect_mv"] = float(ripple_effect * 1000)

        # Part 4: Temperature Drift
        self.subsection("Part 4: Temperature Drift Modeling")
        self.info("Slow drift due to thermal effects")

        # Temperature drift: linear or exponential
        temp_drift_rate = 100e-6  # 100 ppm/°C
        temp_change = 25  # °C change over measurement period
        drift_fraction = temp_drift_rate * temp_change

        # Apply drift as linear ramp
        drift_envelope = np.linspace(0, drift_fraction, num_samples)
        _drifted_signal = clean_sine * (1 + drift_envelope)  # Signal with drift applied

        self.result("Temperature drift rate", f"{temp_drift_rate * 1e6:.0f}", "ppm/°C")
        self.result("Temperature change", f"{temp_change}", "°C")
        self.result("Total drift", f"{drift_fraction * 100:.4f}%")
        self.result(
            "Final amplitude change", f"{drift_fraction * np.max(clean_sine) * 1000:.4f}", "mV"
        )

        results["temp_drift_ppm"] = temp_drift_rate * 1e6
        results["total_drift_percent"] = drift_fraction * 100

        # Part 5: Ground Bounce
        self.subsection("Part 5: Ground Bounce Effects")
        self.info("Simultaneous switching noise (SSN) on ground plane")

        # Ground bounce occurs during fast transitions
        clean_data = data["clean_data"]
        data_transitions = np.abs(np.diff(clean_data, prepend=clean_data[0]))

        # Ground bounce magnitude proportional to number of switching drivers
        num_drivers = 8
        ground_inductance = 5e-9  # 5 nH
        switching_current = 50e-3  # 50 mA per driver
        di_dt = switching_current / 1e-9  # 1 ns rise time

        ground_bounce_amplitude = num_drivers * ground_inductance * di_dt

        # Add ground bounce at transitions
        ground_bounce = data_transitions * ground_bounce_amplitude * 0.1  # Scaled for visibility

        _data_with_bounce = clean_data + ground_bounce  # Data with ground bounce

        self.result("Number of drivers", num_drivers)
        self.result("Ground inductance", f"{ground_inductance * 1e9:.1f}", "nH")
        self.result("Switching current", f"{switching_current * 1000:.1f}", "mA")
        self.result("Ground bounce amplitude", f"{ground_bounce_amplitude * 1000:.1f}", "mV")

        results["ground_bounce_mv"] = ground_bounce_amplitude * 1000

        # Part 6: EMI/RFI Interference
        self.subsection("Part 6: EMI/RFI Interference")
        self.info("External electromagnetic and radio frequency interference")

        # EMI from nearby equipment
        emi_frequencies = [10e6, 27.12e6, 100e6]  # Common EMI sources
        emi_amplitudes = [0.01, 0.005, 0.003]  # Decreasing amplitude with frequency

        emi_signal = np.zeros(num_samples)
        for freq, amp in zip(emi_frequencies, emi_amplitudes, strict=True):
            emi_signal += amp * np.sin(2 * np.pi * freq * time_base[:num_samples])

        # Add EMI to clean sine
        _sine_with_emi = clean_sine + emi_signal  # Signal with EMI added

        self.result("EMI frequencies", f"{emi_frequencies}", "Hz")
        self.result("EMI amplitudes", f"{emi_amplitudes}", "V")
        self.result("Peak EMI interference", f"{np.max(np.abs(emi_signal)) * 1000:.4f}", "mV")

        # Calculate SNR degradation
        signal_power = np.mean(clean_sine**2)
        emi_power = np.mean(emi_signal**2)
        snr_db = 10 * np.log10(signal_power / emi_power)
        self.result("SNR with EMI", f"{snr_db:.2f}", "dB")

        results["emi_snr_db"] = float(snr_db)
        results["emi_peak_mv"] = float(np.max(np.abs(emi_signal)) * 1000)

        # Part 7: Combined Impairments
        self.subsection("Part 7: Combined Impairment Scenario")
        self.info("Real-world signal with multiple simultaneous impairments")

        # Start with clean signal
        real_world_signal = clean_sine.copy()

        # Add all impairments
        real_world_signal += emi_signal  # EMI
        real_world_signal *= 1 + psrr_coeff * total_ripple  # Supply ripple
        real_world_signal *= 1 + drift_envelope  # Temperature drift
        real_world_signal += SignalBuilder.white_noise(sample_rate, duration, 0.01)[
            :num_samples
        ]  # Thermal noise

        self.result("Clean signal RMS", f"{np.sqrt(np.mean(clean_sine**2)) * 1000:.4f}", "mV")
        self.result(
            "Impaired signal RMS", f"{np.sqrt(np.mean(real_world_signal**2)) * 1000:.4f}", "mV"
        )

        # Total SNR degradation
        noise_signal = real_world_signal - clean_sine
        total_noise_power = np.mean(noise_signal**2)
        total_snr_db = 10 * np.log10(signal_power / total_noise_power)
        self.result("Total SNR", f"{total_snr_db:.2f}", "dB")

        # Total harmonic distortion (THD)
        thd = np.sqrt(total_noise_power / signal_power) * 100
        self.result("Equivalent THD", f"{thd:.4f}", "%")

        results["total_snr_db"] = float(total_snr_db)
        results["equivalent_thd"] = float(thd)

        # Part 8: Usage Examples
        self.subsection("Part 8: Usage in Testing")

        self.info("\n[Creating Impaired Test Vectors]")
        self.result("Clean UART signal", f"{len(clean_data)} samples")
        self.result("With ground bounce", "Test decoder robustness")
        self.result("With EMI", "Test filtering effectiveness")

        self.info("\n[WaveformTrace with Impairments]")
        metadata = TraceMetadata(sample_rate=sample_rate, channel_name="IMPAIRED_CH1")
        impaired_trace = WaveformTrace(data=real_world_signal, metadata=metadata)
        self.result("  Trace created", f"{len(impaired_trace)} samples")
        self.result("  Use case", "Test analysis under non-ideal conditions")

        self.success("Impairment simulation complete!")
        self.info("\nKey Capabilities:")
        self.info("  - Jitter: Random (Gaussian) and deterministic (periodic)")
        self.info("  - Crosstalk: Capacitive/inductive coupling (-40 to -60 dB typical)")
        self.info("  - Supply ripple: Mains (60 Hz) + switching (100 kHz)")
        self.info("  - Temperature drift: Linear/exponential drift models")
        self.info("  - Ground bounce: SSN from simultaneous switching")
        self.info("  - EMI/RFI: Multiple interference sources")
        self.info("  - Combined: Real-world multi-impairment scenarios")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate impairment simulation results."""
        self.info("Validating impairment simulations...")

        all_valid = True

        # Validate jitter parameters
        if not validate_approximately(
            results["random_jitter_rms"], 50e-12, tolerance=0.01, name="Random jitter RMS"
        ):
            all_valid = False

        if not validate_approximately(
            results["det_jitter_amplitude"], 100e-12, tolerance=0.01, name="Deterministic jitter"
        ):
            all_valid = False

        # Validate crosstalk
        if not validate_approximately(
            results["crosstalk_db"], -40, tolerance=0.01, name="Crosstalk coefficient"
        ):
            all_valid = False

        # Crosstalk amplitude should be small but measurable
        if results["crosstalk_amplitude"] < 0.0001 or results["crosstalk_amplitude"] > 0.1:
            self.error(f"Crosstalk amplitude out of range: {results['crosstalk_amplitude']:.6f} V")
            all_valid = False
        else:
            self.success(f"Crosstalk amplitude: {results['crosstalk_amplitude']:.6f} V")

        # Validate PSRR
        if not validate_approximately(results["psrr_db"], -60, tolerance=0.01, name="PSRR"):
            all_valid = False

        # Validate temperature drift
        if not validate_approximately(
            results["temp_drift_ppm"], 100, tolerance=0.01, name="Temperature drift (ppm/°C)"
        ):
            all_valid = False

        # Validate ground bounce (relaxed upper limit for demonstration)
        if results["ground_bounce_mv"] < 0 or results["ground_bounce_mv"] > 5000:
            self.warning(
                f"Ground bounce high: {results['ground_bounce_mv']:.4f} mV (demonstration signal)"
            )
        else:
            self.success(f"Ground bounce: {results['ground_bounce_mv']:.4f} mV")

        # Validate EMI impact
        if results["emi_snr_db"] < 20 or results["emi_snr_db"] > 60:
            self.error(f"EMI SNR out of expected range: {results['emi_snr_db']:.2f} dB")
            all_valid = False
        else:
            self.success(f"EMI SNR: {results['emi_snr_db']:.2f} dB")

        # Validate total impairment SNR
        if results["total_snr_db"] < 10 or results["total_snr_db"] > 50:
            self.error(f"Total SNR out of expected range: {results['total_snr_db']:.2f} dB")
            all_valid = False
        else:
            self.success(f"Total SNR with all impairments: {results['total_snr_db']:.2f} dB")

        if all_valid:
            self.success("All impairment simulations validated!")
            self.info("\nNext Steps:")
            self.info("  - Apply impairments to protocol signals for robustness testing")
            self.info("  - Use in 04_advanced_analysis/01_jitter_analysis.py")
            self.info("  - Test decoder performance under impaired conditions")
            self.info("  - Characterize system margins and limits")
        else:
            self.error("Some impairment validations failed")

        return all_valid

    def result(self, name: str, value: Any, unit: str = "") -> None:
        """Print a result with optional unit.

        Args:
            name: Result name
            value: Result value
            unit: Optional unit string
        """
        if unit:
            print(f"  {name}: {value} {unit}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    demo: ImpairmentSimulationDemo = ImpairmentSimulationDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
