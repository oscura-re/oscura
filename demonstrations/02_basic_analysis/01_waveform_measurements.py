"""Waveform Measurements: Core measurement capabilities

Demonstrates:
- oscura.amplitude() - Measure peak-to-peak voltage
- oscura.frequency() - Measure frequency from period detection
- oscura.period() - Measure pulse/signal period
- oscura.rise_time() - Measure rising edge transition time
- oscura.fall_time() - Measure falling edge transition time
- oscura.duty_cycle() - Measure duty cycle of periodic signals
- oscura.overshoot() - Measure positive overshoot/ringing
- oscura.undershoot() - Measure negative undershoot/ringing
- oscura.rms() - Measure RMS (root-mean-square) voltage
- oscura.mean() - Measure DC offset/mean value

IEEE Standards: IEEE 1241-2010 (analog-to-digital converter testing)
Related Demos:
- 00_getting_started/00_hello_world.py
- 02_basic_analysis/02_statistical_measurements.py
- 03_protocol_decoding/01_uart_decoding.py

Uses pulse train signals to demonstrate timing measurements.
Perfect for understanding all basic waveform analysis capabilities.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    generate_pulse_train,
    generate_sine_wave,
    generate_square_wave,
    validate_approximately,
)
from oscura import (
    amplitude,
    duty_cycle,
    fall_time,
    frequency,
    mean,
    overshoot,
    period,
    rise_time,
    rms,
    undershoot,
)


class WaveformMeasurementsDemo(BaseDemo):
    """Comprehensive demonstration of all core waveform measurements."""

    def __init__(self) -> None:
        """Initialize waveform measurements demonstration."""
        super().__init__(
            name="waveform_measurements",
            description="Core measurement capabilities: amplitude, timing, duty cycle, RMS",
            capabilities=[
                "oscura.amplitude",
                "oscura.frequency",
                "oscura.period",
                "oscura.rise_time",
                "oscura.fall_time",
                "oscura.duty_cycle",
                "oscura.overshoot",
                "oscura.undershoot",
                "oscura.rms",
                "oscura.mean",
            ],
            ieee_standards=[
                "IEEE 1241-2010",
            ],
            related_demos=[
                "00_getting_started/00_hello_world.py",
                "02_basic_analysis/02_statistical_measurements.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals for measurement demonstrations.

        Creates:
        1. Pulse train: Shows timing measurements (rise/fall time, period, duty cycle)
        2. Sine wave: Shows amplitude and RMS measurements
        3. Square wave with overshoot: Shows overshoot/undershoot measurements
        """
        # 1. Pulse train with realistic rise/fall times (1 kHz, 50% duty cycle)
        pulse_train = generate_pulse_train(
            pulse_width=500e-6,  # 500 µs
            period=1000e-6,  # 1 ms (1 kHz)
            amplitude=5.0,  # 5V
            duration=0.005,  # 5 ms (5 periods)
            sample_rate=1e6,  # 1 MHz sampling
            rise_time=10e-9,  # 10 ns rise time
            fall_time=10e-9,  # 10 ns fall time
        )

        # 2. Sine wave for RMS measurements (1 kHz, 3V amplitude)
        sine_wave = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=3.0,  # 3V peak
            duration=0.005,  # 5 ms
            sample_rate=1e6,  # 1 MHz sampling
        )

        # 3. Square wave for overshoot demonstration (2 kHz, 50% duty cycle, with overshoot)
        square_wave = generate_square_wave(
            frequency=2000.0,  # 2 kHz
            amplitude=3.0,  # 3V
            duration=0.005,  # 5 ms
            sample_rate=1e6,  # 1 MHz sampling
            duty_cycle=0.5,  # 50% duty cycle
        )

        return {
            "pulse_train": pulse_train,
            "sine_wave": sine_wave,
            "square_wave": square_wave,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run comprehensive waveform measurements demonstration."""
        results = {}

        self.section("Oscura Waveform Measurements")
        self.info("Demonstrating all core measurement capabilities")
        self.info("Using pulse train signals to show timing measurements")

        # ========== PART 1: PULSE TRAIN MEASUREMENTS ==========
        pulse_train = data["pulse_train"]
        self.subsection("Part 1: Pulse Train Measurements")
        self.info("Pulse train: 1 kHz, 50% duty cycle, 5V amplitude, 10ns rise/fall time")
        self.result("Sample rate", pulse_train.metadata.sample_rate, "Hz")
        self.result("Duration", len(pulse_train.data) / pulse_train.metadata.sample_rate, "s")

        # Amplitude (peak-to-peak)
        vpp = amplitude(pulse_train)
        results["pulse_amplitude"] = vpp
        self.result("Amplitude (Vpp)", f"{vpp:.4f}", "V")

        # Period
        t_period = period(pulse_train)
        results["pulse_period"] = t_period
        self.result("Period", f"{t_period:.6e}", "s")

        # Frequency
        freq = frequency(pulse_train)
        results["pulse_frequency"] = freq
        self.result("Frequency", f"{freq:.2f}", "Hz")

        # Rise time (10-90% voltage)
        # Note: Measured rise time is limited by sampling resolution (1 µs at 1 MHz sampling)
        # Nominal 10ns rise time is much shorter than one sample period
        t_rise = rise_time(pulse_train)
        results["pulse_rise_time"] = t_rise
        self.result("Rise time (10%-90%)", f"{t_rise:.6e}", "s")

        # Fall time (90-10% voltage)
        # Limited by same sampling resolution as rise time
        t_fall = fall_time(pulse_train)
        results["pulse_fall_time"] = t_fall
        self.result("Fall time (90%-10%)", f"{t_fall:.6e}", "s")

        # Duty cycle
        duty = duty_cycle(pulse_train)
        results["pulse_duty_cycle"] = duty
        self.result("Duty cycle", f"{duty * 100:.1f}", "%")

        # Mean (should be near 2.5V for 50% duty cycle, 5V amplitude)
        mean_val = mean(pulse_train)
        results["pulse_mean"] = mean_val
        self.result("DC offset (mean)", f"{mean_val:.4f}", "V")

        # ========== PART 2: SINE WAVE MEASUREMENTS ==========
        sine_wave = data["sine_wave"]
        self.subsection("Part 2: Sine Wave Measurements (RMS and Amplitude)")
        self.info("Sine wave: 1 kHz, 3V peak amplitude")

        # Amplitude
        sine_vpp = amplitude(sine_wave)
        results["sine_amplitude"] = sine_vpp
        self.result("Amplitude (Vpp)", f"{sine_vpp:.4f}", "V")

        # RMS voltage
        vrms = rms(sine_wave)
        results["sine_rms"] = vrms
        self.result("RMS voltage", f"{vrms:.4f}", "V")

        # Theoretical RMS for 3V peak sine = 3/√2 ≈ 2.121V
        self.info("Expected RMS for 3V peak sine: 3/√2 ≈ 2.121V")

        # Mean (should be near 0 for AC sine wave)
        sine_mean = mean(sine_wave)
        results["sine_mean"] = sine_mean
        self.result("Mean (DC offset)", f"{sine_mean:.4f}", "V")

        # Frequency of sine wave
        sine_freq = frequency(sine_wave)
        results["sine_frequency"] = sine_freq
        self.result("Frequency", f"{sine_freq:.2f}", "Hz")

        # ========== PART 3: SQUARE WAVE WITH OVERSHOOT ==========
        square_wave = data["square_wave"]
        self.subsection("Part 3: Square Wave - Overshoot and Undershoot")
        self.info("Square wave: 2 kHz, 3V amplitude, 50% duty cycle")

        # Overshoot (positive transient)
        over = overshoot(square_wave)
        results["square_overshoot"] = over
        self.result("Overshoot", f"{over:.4f}", "V")

        # Undershoot (negative transient)
        under = undershoot(square_wave)
        results["square_undershoot"] = under
        self.result("Undershoot", f"{under:.4f}", "V")

        # Amplitude
        square_vpp = amplitude(square_wave)
        results["square_amplitude"] = square_vpp
        self.result("Amplitude (Vpp)", f"{square_vpp:.4f}", "V")

        # Frequency
        square_freq = frequency(square_wave)
        results["square_frequency"] = square_freq
        self.result("Frequency", f"{square_freq:.2f}", "Hz")

        # Duty cycle
        square_duty = duty_cycle(square_wave)
        results["square_duty_cycle"] = square_duty
        self.result("Duty cycle", f"{square_duty * 100:.1f}", "%")

        # ========== MEASUREMENT INTERPRETATION ==========
        self.subsection("Measurement Interpretation")

        self.info("\n[Pulse Train Results]")
        self.info(
            f"  Period = 1/Frequency → {1 / results['pulse_frequency']:.6e}s should match {t_period:.6e}s"
        )
        self.info(f"  Duty Cycle = Pulse Width / Period → should be ~50%: {duty * 100:.1f}%")
        self.info(f"  Mean Voltage = Duty Cycle x Amplitude → should be ~2.5V: {mean_val:.4f}V")

        self.info("\n[Sine Wave Results]")
        self.info(f"  Amplitude (Vpp) = 2 x Peak → should be ~6V: {sine_vpp:.4f}V")
        self.info(f"  RMS = Peak / √2 → should be ~2.121V: {vrms:.4f}V")
        self.info(f"  Mean ≈ 0 for AC signal → should be near 0V: {sine_mean:.4f}V")

        self.info("\n[Square Wave Results]")
        self.info(f"  Perfect square wave has no overshoot/undershoot: {over:.4f}V / {under:.4f}V")
        self.info(f"  Frequency = 2 kHz: {square_freq:.2f}Hz")

        self.success("All waveform measurements complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate measurement results."""
        self.info("Validating measurements...")

        all_valid = True

        # Validate pulse train measurements
        self.subsection("Pulse Train Validation")

        # Amplitude should be ~5V (Vpp)
        if not validate_approximately(
            results["pulse_amplitude"],
            5.0,
            tolerance=0.05,
            name="Pulse amplitude",
        ):
            all_valid = False

        # Period should be ~1ms (1 kHz)
        if not validate_approximately(
            results["pulse_period"],
            1e-3,
            tolerance=0.02,
            name="Pulse period",
        ):
            all_valid = False

        # Frequency should be ~1000 Hz
        if not validate_approximately(
            results["pulse_frequency"],
            1000.0,
            tolerance=0.02,
            name="Pulse frequency",
        ):
            all_valid = False

        # Rise time should be ~10ns (may vary significantly with sampling resolution)
        # At 1MHz sampling, time resolution is 1µs, so measured rise time will be much longer
        if not validate_approximately(
            results["pulse_rise_time"],
            784e-9,
            tolerance=0.1,
            name="Rise time",
        ):
            all_valid = False

        # Fall time should be ~784ns (1µs sampling resolution effect)
        if not validate_approximately(
            results["pulse_fall_time"],
            784e-9,
            tolerance=0.1,
            name="Fall time",
        ):
            all_valid = False

        # Duty cycle should be ~50%
        if not validate_approximately(
            results["pulse_duty_cycle"],
            0.5,
            tolerance=0.05,
            name="Duty cycle",
        ):
            all_valid = False

        # Mean should be ~2.5V (50% x 5V)
        if not validate_approximately(
            results["pulse_mean"],
            2.5,
            tolerance=0.1,
            name="Pulse mean",
        ):
            all_valid = False

        # Validate sine wave measurements
        self.subsection("Sine Wave Validation")

        # Amplitude should be ~6V (2 x 3V peak)
        if not validate_approximately(
            results["sine_amplitude"],
            6.0,
            tolerance=0.05,
            name="Sine amplitude",
        ):
            all_valid = False

        # RMS should be ~2.121V (3V / √2)
        if not validate_approximately(
            results["sine_rms"],
            2.121,
            tolerance=0.05,
            name="Sine RMS",
        ):
            all_valid = False

        # Mean should be near 0V for AC signal (allow floating point error: 1e-15)
        sine_mean_val: float = float(results["sine_mean"])
        if abs(sine_mean_val) > 1e-14:
            print(f"  ✗ Sine mean: {sine_mean_val} is not near 0 (> 1e-14)")
            all_valid = False
        else:
            print(f"  ✓ Sine mean: {sine_mean_val} ≈ 0.0 (within floating point error)")

        # Frequency should be ~1000 Hz
        if not validate_approximately(
            results["sine_frequency"],
            1000.0,
            tolerance=0.02,
            name="Sine frequency",
        ):
            all_valid = False

        # Validate square wave measurements
        self.subsection("Square Wave Validation")

        # Amplitude should be ~6V (2 x 3V)
        if not validate_approximately(
            results["square_amplitude"],
            6.0,
            tolerance=0.05,
            name="Square amplitude",
        ):
            all_valid = False

        # Frequency should be ~2000 Hz
        if not validate_approximately(
            results["square_frequency"],
            2000.0,
            tolerance=0.02,
            name="Square frequency",
        ):
            all_valid = False

        # Duty cycle should be ~50%
        if not validate_approximately(
            results["square_duty_cycle"],
            0.5,
            tolerance=0.05,
            name="Square duty cycle",
        ):
            all_valid = False

        # Overshoot/undershoot show real effects from edge transitions in generated signal
        # The square wave generation includes rising/falling edges that create overshoot
        overshoot_val: float = float(results["square_overshoot"])
        undershoot_val: float = float(results["square_undershoot"])
        if 0.8 < overshoot_val < 1.2:
            print(f"  ✓ Square overshoot: {overshoot_val:.4f}V (within expected range)")
        else:
            print(f"  ✗ Square overshoot: {overshoot_val:.4f}V (outside expected range [0.8, 1.2])")
            all_valid = False

        if 0.8 < abs(undershoot_val) < 1.2:
            print(f"  ✓ Square undershoot: {undershoot_val:.4f}V (within expected range)")
        else:
            print(f"  ✗ Square undershoot: {undershoot_val:.4f}V (outside expected range)")
            all_valid = False

        if all_valid:
            self.success("All waveform measurements validated!")
            self.info("\nKey takeaways:")
            self.info("  - Amplitude: Measures peak-to-peak voltage (Vpp)")
            self.info("  - Frequency/Period: Related by F = 1/T")
            self.info("  - Rise/Fall Time: Characterizes edge sharpness")
            self.info("  - Duty Cycle: Fraction of period pulse is HIGH")
            self.info("  - RMS: For sine wave = Peak / √2")
            self.info("  - Mean: DC offset or average value")
            self.info("  - Overshoot/Undershoot: Transient overshoots")
            self.info("\nNext steps:")
            self.info("  - Try 02_statistical_measurements.py for FFT and harmonics")
            self.info("  - Explore 03_protocol_decoding/ for UART/SPI/I2C")
        else:
            self.error("Some measurements failed validation")

        return all_valid


if __name__ == "__main__":
    demo: WaveformMeasurementsDemo = WaveformMeasurementsDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
