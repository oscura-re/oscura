"""Signal Builder Comprehensive: Complete SignalBuilder API Coverage

Demonstrates:
- Complete SignalBuilder API coverage
- All waveform types (sine, square, triangle, sawtooth, pulse, noise)
- Impairment simulation (jitter, noise, distortion, offset)
- Multi-channel generation
- Advanced waveforms (chirp, multitone, AM/FM modulation)

IEEE Standards: IEEE 1241-2010 (ADC Terminology)
Related Demos:
- 02_basic_analysis/01_waveform_measurements.py
- 17_signal_generation/02_protocol_generation.py
- 17_signal_generation/03_impairment_simulation.py

This demonstration teaches how to use the SignalBuilder test fixture to generate
all types of signals for testing. SignalBuilder is the single source of truth
for test signal generation throughout Oscura's test suite.
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


class SignalBuilderDemo(BaseDemo):
    """Comprehensive demonstration of SignalBuilder capabilities."""

    def __init__(self) -> None:
        """Initialize signal builder demonstration."""
        super().__init__(
            name="signal_builder_comprehensive",
            description="Complete SignalBuilder API: all waveforms, impairments, multi-channel",
            capabilities=[
                "SignalBuilder.sine_wave",
                "SignalBuilder.square_wave",
                "SignalBuilder.triangle_wave",
                "SignalBuilder.sawtooth_wave",
                "SignalBuilder.pulse_train",
                "SignalBuilder.white_noise",
                "SignalBuilder.chirp",
                "SignalBuilder.multitone",
                "SignalBuilder.am_modulated",
                "SignalBuilder.fm_modulated",
                "SignalBuilder.dc_offset",
                "SignalBuilder.step_response",
                "SignalBuilder.exponential_decay",
            ],
            ieee_standards=["IEEE 1241-2010"],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
                "17_signal_generation/02_protocol_generation.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate all signal types using SignalBuilder.

        Returns:
            Dictionary containing all generated signal types
        """
        sample_rate = 1e6  # 1 MHz
        duration = 0.001  # 1 ms

        return {
            "sample_rate": sample_rate,
            "duration": duration,
            # Basic waveforms
            "sine": SignalBuilder.sine_wave(
                frequency=10e3, sample_rate=sample_rate, duration=duration, amplitude=1.0
            ),
            "square": SignalBuilder.square_wave(
                frequency=5e3, sample_rate=sample_rate, duration=duration, amplitude=2.0
            ),
            "triangle": SignalBuilder.triangle_wave(
                frequency=2e3, sample_rate=sample_rate, duration=duration, amplitude=1.5
            ),
            "sawtooth": SignalBuilder.sawtooth_wave(
                frequency=3e3, sample_rate=sample_rate, duration=duration, amplitude=1.0
            ),
            "pulse": SignalBuilder.pulse_train(
                frequency=1e3,
                sample_rate=sample_rate,
                duration=duration,
                pulse_width=0.2,
                amplitude=3.0,
            ),
            # Noise and special waveforms
            "noise": SignalBuilder.white_noise(
                sample_rate=sample_rate, duration=duration, amplitude=0.5
            ),
            "chirp": SignalBuilder.chirp(
                f0=1e3, f1=50e3, sample_rate=sample_rate, duration=duration, method="linear"
            ),
            "multitone": SignalBuilder.multitone(
                frequencies=[5e3, 10e3, 15e3],
                sample_rate=sample_rate,
                duration=duration,
                amplitudes=[1.0, 0.5, 0.25],
            ),
            # Modulated signals
            "am": SignalBuilder.am_modulated(
                carrier_freq=50e3,
                modulation_freq=2e3,
                modulation_index=0.8,
                sample_rate=sample_rate,
                duration=duration,
            ),
            "fm": SignalBuilder.fm_modulated(
                carrier_freq=50e3,
                modulation_freq=2e3,
                frequency_deviation=5e3,
                sample_rate=sample_rate,
                duration=duration,
            ),
            # DC and transient signals
            "dc": SignalBuilder.dc_offset(offset=2.5, sample_rate=sample_rate, duration=duration),
            "step": SignalBuilder.step_response(
                step_time=0.0005,
                sample_rate=sample_rate,
                duration=duration,
                amplitude=3.3,
            ),
            "decay": SignalBuilder.exponential_decay(
                time_constant=0.0002,
                sample_rate=sample_rate,
                duration=duration,
                amplitude=5.0,
            ),
            # Noisy sine wave
            "noisy_sine": SignalBuilder.noisy_sine(
                frequency=10e3, snr_db=20, sample_rate=sample_rate, duration=duration
            ),
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run comprehensive signal builder demonstration."""
        results: dict[str, Any] = {}

        self.section("SignalBuilder Comprehensive Demonstration")
        self.info("Demonstrating all SignalBuilder waveform types and capabilities")

        # Part 1: Basic Waveforms
        self.subsection("Part 1: Basic Waveforms")
        self.info("Core periodic waveforms for frequency and time domain testing")

        # Sine wave
        sine = data["sine"]
        self.result("Sine wave", f"{len(sine)} samples")
        self.result("  Peak amplitude", f"{np.max(sine):.4f}", "V")
        self.result("  Min amplitude", f"{np.min(sine):.4f}", "V")
        self.result("  RMS value", f"{np.sqrt(np.mean(sine**2)):.4f}", "V")
        results["sine_peak"] = float(np.max(sine))
        results["sine_rms"] = float(np.sqrt(np.mean(sine**2)))

        # Square wave
        square = data["square"]
        self.result("Square wave", f"{len(square)} samples")
        self.result("  Peak amplitude", f"{np.max(square):.4f}", "V")
        self.result("  Min amplitude", f"{np.min(square):.4f}", "V")
        results["square_peak"] = float(np.max(square))

        # Triangle wave
        triangle = data["triangle"]
        self.result("Triangle wave", f"{len(triangle)} samples")
        self.result("  Peak amplitude", f"{np.max(triangle):.4f}", "V")
        self.result("  Min amplitude", f"{np.min(triangle):.4f}", "V")
        results["triangle_peak"] = float(np.max(triangle))

        # Sawtooth wave
        sawtooth = data["sawtooth"]
        self.result("Sawtooth wave", f"{len(sawtooth)} samples")
        self.result("  Peak amplitude", f"{np.max(sawtooth):.4f}", "V")
        results["sawtooth_peak"] = float(np.max(sawtooth))

        # Pulse train
        pulse = data["pulse"]
        self.result("Pulse train", f"{len(pulse)} samples")
        self.result("  Peak amplitude", f"{np.max(pulse):.4f}", "V")
        self.result("  High samples", int(np.sum(pulse > 0)))
        results["pulse_peak"] = float(np.max(pulse))

        # Part 2: Noise and Special Waveforms
        self.subsection("Part 2: Noise and Specialized Waveforms")

        # White noise
        noise = data["noise"]
        self.result("White noise", f"{len(noise)} samples")
        self.result("  Standard deviation", f"{np.std(noise):.4f}", "V")
        self.result("  Mean value", f"{np.mean(noise):.6f}", "V")
        results["noise_std"] = float(np.std(noise))

        # Chirp (frequency sweep)
        chirp = data["chirp"]
        self.result("Chirp (1 kHz → 50 kHz)", f"{len(chirp)} samples")
        self.result("  Peak amplitude", f"{np.max(chirp):.4f}", "V")
        results["chirp_peak"] = float(np.max(chirp))

        # Multitone
        multitone = data["multitone"]
        self.result("Multitone (5k, 10k, 15k Hz)", f"{len(multitone)} samples")
        self.result("  Peak amplitude", f"{np.max(multitone):.4f}", "V")
        results["multitone_peak"] = float(np.max(multitone))

        # Part 3: Modulated Signals
        self.subsection("Part 3: Modulated Signals (AM/FM)")

        # AM modulation
        am = data["am"]
        self.result("AM modulated (50 kHz carrier)", f"{len(am)} samples")
        self.result("  Peak amplitude", f"{np.max(am):.4f}", "V")
        self.result("  Modulation index", "0.8")
        results["am_peak"] = float(np.max(am))

        # FM modulation
        fm = data["fm"]
        self.result("FM modulated (50 kHz carrier)", f"{len(fm)} samples")
        self.result("  Peak amplitude", f"{np.max(fm):.4f}", "V")
        self.result("  Frequency deviation", "5 kHz")
        results["fm_peak"] = float(np.max(fm))

        # Part 4: DC and Transient Signals
        self.subsection("Part 4: DC and Transient Responses")

        # DC offset
        dc = data["dc"]
        self.result("DC offset", f"{len(dc)} samples")
        self.result("  Value", f"{np.mean(dc):.4f}", "V")
        self.result("  Variation (std dev)", f"{np.std(dc):.6e}", "V")
        results["dc_mean"] = float(np.mean(dc))

        # Step response
        step = data["step"]
        self.result("Step response (3.3V)", f"{len(step)} samples")
        self.result("  Initial value", f"{step[0]:.4f}", "V")
        self.result("  Final value", f"{step[-1]:.4f}", "V")
        self.result("  Step amplitude", f"{step[-1] - step[0]:.4f}", "V")
        results["step_amplitude"] = float(step[-1] - step[0])

        # Exponential decay
        decay = data["decay"]
        self.result("Exponential decay (RC)", f"{len(decay)} samples")
        self.result("  Initial value", f"{decay[0]:.4f}", "V")
        self.result("  Final value", f"{decay[-1]:.4f}", "V")
        self.result("  Time constant", "200 µs")
        results["decay_initial"] = float(decay[0])

        # Part 5: Noise Injection
        self.subsection("Part 5: Noise Injection (Impairment Simulation)")

        # Noisy sine wave
        noisy_sine = data["noisy_sine"]
        self.result("Noisy sine (20 dB SNR)", f"{len(noisy_sine)} samples")
        self.result("  Peak amplitude", f"{np.max(noisy_sine):.4f}", "V")
        self.result("  Standard deviation", f"{np.std(noisy_sine):.4f}", "V")

        # Calculate actual SNR
        clean_sine = data["sine"]
        signal_power = np.mean(clean_sine**2)
        noise_power = np.mean((noisy_sine - clean_sine) ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power)
        self.result("  Measured SNR", f"{snr_db:.2f}", "dB")
        results["noisy_sine_snr"] = float(snr_db)

        # Part 6: Multi-Channel Generation
        self.subsection("Part 6: Multi-Channel Signal Generation")
        self.info("Creating differential pair (CH1/CH2) with common-mode noise")

        # Generate differential signals
        ch1_signal = SignalBuilder.sine_wave(frequency=10e3, amplitude=1.0, duration=0.001)
        ch2_signal = -ch1_signal  # Inverted for differential
        common_mode = SignalBuilder.sine_wave(frequency=60, amplitude=0.1, duration=0.001)

        ch1_with_cm = ch1_signal + common_mode
        ch2_with_cm = ch2_signal + common_mode

        self.result("CH1 samples", len(ch1_with_cm))
        self.result("CH2 samples", len(ch2_with_cm))
        self.result("Common-mode amplitude", f"{np.max(common_mode):.4f}", "V")

        # Differential signal (CH1 - CH2)
        differential = ch1_with_cm - ch2_with_cm
        common_mode_recovered = (ch1_with_cm + ch2_with_cm) / 2

        self.result("Differential amplitude", f"{np.max(differential):.4f}", "V")
        self.result("Common-mode rejection", f"{np.max(common_mode_recovered):.4f}", "V")

        results["differential_amplitude"] = float(np.max(differential))
        results["common_mode_amplitude"] = float(np.max(common_mode_recovered))

        # Part 7: Usage Examples
        self.subsection("Part 7: Usage Patterns and Best Practices")

        self.info("\n[Creating WaveformTrace from SignalBuilder]")
        metadata = TraceMetadata(sample_rate=1e6, channel_name="CH1")
        waveform = WaveformTrace(data=data["sine"], metadata=metadata)
        self.result("WaveformTrace created", f"{len(waveform)} samples")
        self.result("Duration", f"{waveform.duration:.6f}", "s")

        self.info("\n[Combining Multiple Signals]")
        # Ensure signals have same length (trim to minimum)
        min_len = min(len(data["sine"]), len(data["noise"]))
        composite = data["sine"][:min_len] + 0.1 * data["noise"][:min_len] + 0.5
        self.result("Composite signal", "sine + noise + DC offset")
        self.result("  Mean value (DC)", f"{np.mean(composite):.4f}", "V")
        self.result("  Peak amplitude", f"{np.max(composite):.4f}", "V")

        self.info("\n[Signal Comparison]")
        correlation = np.corrcoef(data["sine"], data["noisy_sine"])[0, 1]
        self.result("Correlation (clean vs noisy)", f"{correlation:.6f}")
        results["sine_correlation"] = float(correlation)

        self.success("SignalBuilder demonstration complete!")
        self.info("\nKey Capabilities:")
        self.info("  - 13+ waveform types (sine, square, triangle, sawtooth, pulse, etc.)")
        self.info("  - Noise injection with calibrated SNR")
        self.info("  - Modulated signals (AM/FM)")
        self.info("  - Transient responses (step, exponential decay)")
        self.info("  - Multi-channel generation")
        self.info("  - Single source of truth for all test signals")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate signal generation results."""
        self.info("Validating signal generation...")

        all_valid = True

        # Validate sine wave
        if not validate_approximately(
            results["sine_peak"], 1.0, tolerance=0.01, name="Sine peak amplitude"
        ):
            all_valid = False

        # RMS of 1V sine should be ~0.707
        if not validate_approximately(results["sine_rms"], 0.707, tolerance=0.01, name="Sine RMS"):
            all_valid = False

        # Validate square wave
        if not validate_approximately(
            results["square_peak"], 2.0, tolerance=0.01, name="Square peak amplitude"
        ):
            all_valid = False

        # Validate triangle wave
        if not validate_approximately(
            results["triangle_peak"], 1.5, tolerance=0.01, name="Triangle peak amplitude"
        ):
            all_valid = False

        # Validate DC offset
        if not validate_approximately(
            results["dc_mean"], 2.5, tolerance=0.01, name="DC offset mean"
        ):
            all_valid = False

        # Validate step response
        if not validate_approximately(
            results["step_amplitude"], 3.3, tolerance=0.01, name="Step amplitude"
        ):
            all_valid = False

        # Validate exponential decay
        if not validate_approximately(
            results["decay_initial"], 5.0, tolerance=0.01, name="Decay initial value"
        ):
            all_valid = False

        # Validate SNR is approximately 20 dB
        if not validate_approximately(
            results["noisy_sine_snr"], 20.0, tolerance=2.0, name="Noisy sine SNR"
        ):
            all_valid = False

        # Validate correlation between clean and noisy sine
        if not validate_approximately(
            results["sine_correlation"], 1.0, tolerance=0.1, name="Sine correlation"
        ):
            all_valid = False

        # Validate differential amplitude (should be ~2.0 for +1 and -1)
        if not validate_approximately(
            results["differential_amplitude"], 2.0, tolerance=0.1, name="Differential amplitude"
        ):
            all_valid = False

        if all_valid:
            self.success("All signal generation validations passed!")
            self.info("\nNext Steps:")
            self.info("  - Try 17_signal_generation/02_protocol_generation.py for protocol signals")
            self.info("  - Explore 17_signal_generation/03_impairment_simulation.py")
            self.info("  - Use SignalBuilder in your own tests via:")
            self.info("    from tests.fixtures.signal_builders import SignalBuilder")
        else:
            self.error("Some signal generation validations failed")

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
    demo: SignalBuilderDemo = SignalBuilderDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
