#!/usr/bin/env python3
"""Synthetic Tektronix WFM Test Data Generator.

This script generates synthetic Tektronix WFM#003 files for testing Oscura.
Creates legally safe, reproducible test data without dependencies on proprietary
or sensitive source material.

- Legal test data strategy
- Reproducible test datasets
- Multiple signal types for comprehensive testing

Usage:
    # Generate a single file
    python generate_synthetic_wfm.py --signal sine --output test.wfm

    # Generate full test suite
    python generate_synthetic_wfm.py --generate-suite --output-dir test_data/synthetic/

    # Generate specific scenario
    python generate_synthetic_wfm.py --signal pulse --frequency 10000 --duty-cycle 0.3

Examples:
    # 1 kHz sine wave
    python generate_synthetic_wfm.py --signal sine --frequency 1000 --amplitude 2.5

    # Square wave with harmonics
    python generate_synthetic_wfm.py --signal square --frequency 5000 --samples 10000

    # Noisy signal for edge case testing
    python generate_synthetic_wfm.py --signal noisy --snr 20 --samples 50000
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

try:
    from tm_data_types import AnalogWaveform, RawSample, write_file

    TM_DATA_TYPES_AVAILABLE = True
except ImportError:
    TM_DATA_TYPES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Supported signal types for generation."""

    SINE = "sine"
    SQUARE = "square"
    TRIANGLE = "triangle"
    SAWTOOTH = "sawtooth"
    PULSE = "pulse"
    DC = "dc"
    NOISY = "noisy"
    MIXED = "mixed"
    CHIRP = "chirp"
    PWM = "pwm"
    EXPONENTIAL = "exponential"
    DAMPED_SINE = "damped_sine"


@dataclass
class WaveformConfig:
    """Configuration for waveform generation."""

    signal_type: SignalType
    sample_rate: float = 1e6  # 1 MSa/s
    duration: float = 0.001  # 1 ms
    frequency: float = 1000.0  # 1 kHz
    amplitude: float = 1.0  # 1 V
    offset: float = 0.0  # 0 V DC offset
    phase: float = 0.0  # Phase in radians
    duty_cycle: float = 0.5  # For pulse/PWM signals
    snr_db: float | None = None  # Signal-to-noise ratio in dB
    num_samples: int | None = None  # Override calculated samples
    channel_name: str = "CH1"

    # Advanced parameters
    chirp_f0: float = 100.0  # Chirp start frequency
    chirp_f1: float = 10000.0  # Chirp end frequency
    damping_factor: float = 5.0  # For damped oscillations
    num_components: int = 3  # For mixed signals

    def __post_init__(self) -> None:
        """Calculate derived parameters."""
        if self.num_samples is None:
            self.num_samples = int(self.sample_rate * self.duration)


class SyntheticWFMGenerator:
    """Generate synthetic Tektronix WFM files with various signal types."""

    def __init__(self, config: WaveformConfig):
        """Initialize generator with configuration.

        Args:
            config: Waveform generation configuration.
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not TM_DATA_TYPES_AVAILABLE:
            msg = "tm_data_types library not available. Install with: pip install tm_data_types"
            raise ImportError(msg)

        if self.config.num_samples <= 0:
            msg = f"Invalid num_samples: {self.config.num_samples}"
            raise ValueError(msg)

        if self.config.sample_rate <= 0:
            msg = f"Invalid sample_rate: {self.config.sample_rate}"
            raise ValueError(msg)

        if self.config.amplitude < 0:
            msg = f"Invalid amplitude: {self.config.amplitude}"
            raise ValueError(msg)

    def generate(self) -> np.ndarray:
        """Generate waveform data based on configuration.

        Returns:
            Numpy array of waveform samples.
        """
        t = np.linspace(0, self.config.duration, self.config.num_samples, endpoint=False)

        # Generate base signal
        signal_func = {
            SignalType.SINE: self._generate_sine,
            SignalType.SQUARE: self._generate_square,
            SignalType.TRIANGLE: self._generate_triangle,
            SignalType.SAWTOOTH: self._generate_sawtooth,
            SignalType.PULSE: self._generate_pulse,
            SignalType.DC: self._generate_dc,
            SignalType.NOISY: self._generate_noisy,
            SignalType.MIXED: self._generate_mixed,
            SignalType.CHIRP: self._generate_chirp,
            SignalType.PWM: self._generate_pwm,
            SignalType.EXPONENTIAL: self._generate_exponential,
            SignalType.DAMPED_SINE: self._generate_damped_sine,
        }

        generator = signal_func.get(self.config.signal_type)
        if generator is None:
            msg = f"Unknown signal type: {self.config.signal_type}"
            raise ValueError(msg)

        y = generator(t)

        # Apply offset
        y = y + self.config.offset

        # Add noise if requested
        if self.config.snr_db is not None:
            y = self._add_noise(y, self.config.snr_db)

        return y

    def _generate_sine(self, t: np.ndarray) -> np.ndarray:
        """Generate sine wave."""
        return self.config.amplitude * np.sin(
            2 * np.pi * self.config.frequency * t + self.config.phase
        )

    def _generate_square(self, t: np.ndarray) -> np.ndarray:
        """Generate square wave using Fourier series approximation."""
        y = np.zeros_like(t)
        # Use first 10 harmonics for better square wave
        for n in range(1, 20, 2):
            y += (4 / (n * np.pi)) * np.sin(
                2 * np.pi * n * self.config.frequency * t + self.config.phase
            )
        return self.config.amplitude * y

    def _generate_triangle(self, t: np.ndarray) -> np.ndarray:
        """Generate triangle wave using Fourier series."""
        y = np.zeros_like(t)
        for n in range(1, 20, 2):
            sign = (-1) ** ((n - 1) // 2)
            y += (sign * 8 / (n * np.pi) ** 2) * np.sin(
                2 * np.pi * n * self.config.frequency * t + self.config.phase
            )
        return self.config.amplitude * y

    def _generate_sawtooth(self, t: np.ndarray) -> np.ndarray:
        """Generate sawtooth wave using Fourier series."""
        y = np.zeros_like(t)
        for n in range(1, 20):
            sign = (-1) ** (n + 1)
            y += (sign * 2 / (n * np.pi)) * np.sin(
                2 * np.pi * n * self.config.frequency * t + self.config.phase
            )
        return self.config.amplitude * y

    def _generate_pulse(self, t: np.ndarray) -> np.ndarray:
        """Generate pulse train with specified duty cycle."""
        period = 1.0 / self.config.frequency
        phase_time = self.config.phase / (2 * np.pi * self.config.frequency)
        t_shifted = (t - phase_time) % period
        pulse_width = period * self.config.duty_cycle
        return self.config.amplitude * (t_shifted < pulse_width).astype(float)

    def _generate_dc(self, t: np.ndarray) -> np.ndarray:
        """Generate DC signal (constant voltage)."""
        return np.full_like(t, self.config.amplitude)

    def _generate_noisy(self, t: np.ndarray) -> np.ndarray:
        """Generate random noise."""
        # Use seeded random for reproducibility
        rng = np.random.default_rng(seed=42)
        return self.config.amplitude * rng.normal(0, 1, len(t))

    def _generate_mixed(self, t: np.ndarray) -> np.ndarray:
        """Generate mixed signal with multiple frequency components."""
        y = np.zeros_like(t)
        for i in range(self.config.num_components):
            freq = self.config.frequency * (i + 1)
            amp = self.config.amplitude / (i + 1)  # Decreasing amplitudes
            phase = i * np.pi / 4  # Different phases
            y += amp * np.sin(2 * np.pi * freq * t + phase)
        return y

    def _generate_chirp(self, t: np.ndarray) -> np.ndarray:
        """Generate frequency chirp (sweep)."""
        # Linear chirp from f0 to f1
        f0 = self.config.chirp_f0
        f1 = self.config.chirp_f1
        T = self.config.duration
        # Instantaneous frequency: f(t) = f0 + (f1-f0)*t/T
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * T))
        return self.config.amplitude * np.sin(phase + self.config.phase)

    def _generate_pwm(self, t: np.ndarray) -> np.ndarray:
        """Generate PWM signal with varying duty cycle."""
        # Modulate duty cycle sinusoidally (10-90% duty cycle range)
        mod_freq = self.config.frequency / 10  # Modulation frequency

        period = 1.0 / self.config.frequency
        phase_time = self.config.phase / (2 * np.pi * self.config.frequency)
        t_shifted = (t - phase_time) % period

        # Compare against modulated duty cycle
        t_in_period = t_shifted / period
        duty_at_t = 0.5 + 0.4 * np.sin(2 * np.pi * mod_freq * t)
        return self.config.amplitude * (t_in_period < duty_at_t).astype(float)

    def _generate_exponential(self, t: np.ndarray) -> np.ndarray:
        """Generate exponential decay signal."""
        tau = 1.0 / (2 * np.pi * self.config.frequency)
        return self.config.amplitude * np.exp(-t / tau)

    def _generate_damped_sine(self, t: np.ndarray) -> np.ndarray:
        """Generate damped sinusoidal oscillation."""
        damping = self.config.damping_factor
        envelope = np.exp(-damping * t)
        return (
            self.config.amplitude
            * envelope
            * np.sin(2 * np.pi * self.config.frequency * t + self.config.phase)
        )

    def _add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add white Gaussian noise to signal.

        Args:
            signal: Clean signal.
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            Noisy signal.
        """
        signal_power = np.mean(signal**2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)

        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0, noise_std, len(signal))
        return signal + noise

    def save(self, output_path: str | Path) -> None:
        """Generate and save waveform to WFM file.

        Args:
            output_path: Path to output WFM file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate waveform data
        logger.info(
            f"Generating {self.config.signal_type.value} signal "
            f"({self.config.num_samples} samples @ {self.config.sample_rate / 1e6:.1f} MSa/s)..."
        )
        y = self.generate()

        # Create AnalogWaveform object
        wf = AnalogWaveform()
        wf.source_name = self.config.channel_name

        # Convert to 16-bit integer for WFM format
        # Scale to use full 16-bit range
        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min
        if y_range > 0:
            y_normalized = (y - y_min) / y_range  # Normalize to 0-1
            y_int16 = (y_normalized * 65535 - 32768).astype(np.int16)
            scale_factor = y_range / 65535
            offset = y_min
        else:
            # Constant signal
            y_int16 = np.zeros(len(y), dtype=np.int16)
            scale_factor = 1.0
            offset = y_min

        wf.y_axis_values = RawSample[np.int16](y_int16)
        wf.x_axis_spacing = 1.0 / self.config.sample_rate
        wf.x_axis_units = "s"
        wf.y_axis_units = "V"
        wf.y_axis_offset = offset
        wf.y_axis_spacing = scale_factor

        # Write to file
        logger.info(f"Writing to {output_path}...")
        write_file(str(output_path), wf)

        file_size = output_path.stat().st_size
        logger.info(f"✓ Created {output_path} ({file_size:,} bytes)")


class TestSuiteGenerator:
    """Generate comprehensive test suite of synthetic WFM files."""

    def __init__(self, output_dir: Path):
        """Initialize test suite generator.

        Args:
            output_dir: Directory to store generated files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_suite(self) -> dict[str, list[Path]]:
        """Generate complete test suite with various signal types and scenarios.

        Returns:
            Dictionary mapping category names to lists of generated file paths.
        """
        generated: dict[str, list[Path]] = {
            "basic": [],
            "edge_cases": [],
            "sizes": [],
            "frequencies": [],
            "advanced": [],
        }

        logger.info("=" * 60)
        logger.info("Generating Synthetic WFM Test Suite")
        logger.info("=" * 60)

        # Basic signal types
        logger.info("\n[1/5] Basic signal types...")
        basic_configs = [
            ("sine_1khz.wfm", SignalType.SINE, 1000, 1.0),
            ("square_5khz.wfm", SignalType.SQUARE, 5000, 2.0),
            ("triangle_2khz.wfm", SignalType.TRIANGLE, 2000, 1.5),
            ("sawtooth_3khz.wfm", SignalType.SAWTOOTH, 3000, 1.0),
            ("pulse_10khz.wfm", SignalType.PULSE, 10000, 3.3),
        ]
        for filename, sig_type, freq, amp in basic_configs:
            config = WaveformConfig(
                signal_type=sig_type,
                frequency=freq,
                amplitude=amp,
                duration=0.001,
            )
            gen = SyntheticWFMGenerator(config)
            path = self.output_dir / "basic" / filename
            gen.save(path)
            generated["basic"].append(path)

        # Edge cases
        logger.info("\n[2/5] Edge cases...")
        edge_configs = [
            ("dc_signal.wfm", SignalType.DC, 0, 2.5, {}),
            ("high_frequency_100khz.wfm", SignalType.SINE, 100000, 0.5, {}),
            ("low_amplitude_1mv.wfm", SignalType.SINE, 1000, 0.001, {}),
            ("high_amplitude_10v.wfm", SignalType.SINE, 1000, 10.0, {}),
            ("noisy_signal_snr20.wfm", SignalType.NOISY, 1000, 1.0, {"snr_db": 20}),
            (
                "sine_with_noise_snr30.wfm",
                SignalType.SINE,
                1000,
                1.0,
                {"snr_db": 30},
            ),
            ("dc_offset_positive.wfm", SignalType.SINE, 1000, 1.0, {"offset": 2.5}),
            ("dc_offset_negative.wfm", SignalType.SINE, 1000, 1.0, {"offset": -1.5}),
        ]
        for filename, sig_type, freq, amp, extra in edge_configs:
            config = WaveformConfig(
                signal_type=sig_type,
                frequency=freq,
                amplitude=amp,
                duration=0.001,
                **extra,
            )
            gen = SyntheticWFMGenerator(config)
            path = self.output_dir / "edge_cases" / filename
            gen.save(path)
            generated["edge_cases"].append(path)

        # Different file sizes
        logger.info("\n[3/5] Various file sizes...")
        size_configs = [
            ("small_100samples.wfm", 100),
            ("medium_10k_samples.wfm", 10000),
            ("large_100k_samples.wfm", 100000),
            ("very_large_1m_samples.wfm", 1000000),
        ]
        for filename, num_samples in size_configs:
            config = WaveformConfig(
                signal_type=SignalType.SINE,
                frequency=1000,
                amplitude=1.0,
                num_samples=num_samples,
                duration=num_samples / 1e6,  # Keep 1 MSa/s
            )
            gen = SyntheticWFMGenerator(config)
            path = self.output_dir / "sizes" / filename
            gen.save(path)
            generated["sizes"].append(path)

        # Different frequencies and sample rates
        logger.info("\n[4/5] Frequency variations...")
        freq_configs = [
            ("low_freq_10hz.wfm", 10, 1e6, 0.1),
            ("audio_freq_440hz.wfm", 440, 1e6, 0.01),
            ("ultrasonic_40khz.wfm", 40000, 1e6, 0.001),
            ("high_sample_rate_10msa.wfm", 1000, 10e6, 0.001),
            ("low_sample_rate_100ksa.wfm", 1000, 100e3, 0.01),
        ]
        for filename, freq, sr, dur in freq_configs:
            config = WaveformConfig(
                signal_type=SignalType.SINE,
                frequency=freq,
                amplitude=1.0,
                sample_rate=sr,
                duration=dur,
            )
            gen = SyntheticWFMGenerator(config)
            path = self.output_dir / "frequencies" / filename
            gen.save(path)
            generated["frequencies"].append(path)

        # Advanced signal types
        logger.info("\n[5/5] Advanced signal types...")
        advanced_configs = [
            ("mixed_harmonics.wfm", SignalType.MIXED, {"num_components": 5}),
            ("chirp_100hz_to_10khz.wfm", SignalType.CHIRP, {}),
            ("pwm_modulated.wfm", SignalType.PWM, {}),
            ("exponential_decay.wfm", SignalType.EXPONENTIAL, {}),
            ("damped_oscillation.wfm", SignalType.DAMPED_SINE, {}),
            ("pulse_train_10pct.wfm", SignalType.PULSE, {"duty_cycle": 0.1}),
            ("pulse_train_90pct.wfm", SignalType.PULSE, {"duty_cycle": 0.9}),
        ]
        for filename, sig_type, extra in advanced_configs:
            config = WaveformConfig(
                signal_type=sig_type,
                frequency=1000,
                amplitude=1.0,
                duration=0.001,
                **extra,
            )
            gen = SyntheticWFMGenerator(config)
            path = self.output_dir / "advanced" / filename
            gen.save(path)
            generated["advanced"].append(path)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Test Suite Generation Complete!")
        logger.info("=" * 60)
        total = sum(len(files) for files in generated.values())
        logger.info(f"Total files generated: {total}")
        for category, files in generated.items():
            logger.info(f"  {category}: {len(files)} files")
        logger.info(f"\nOutput directory: {self.output_dir.absolute()}")

        return generated


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Tektronix WFM test files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Generation mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--generate-suite",
        action="store_true",
        help="Generate complete test suite",
    )
    mode_group.add_argument(
        "--signal",
        type=str,
        choices=[s.value for s in SignalType],
        help="Signal type to generate",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (for single file generation)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_data/synthetic",
        help="Output directory (for test suite)",
    )

    # Signal parameters
    parser.add_argument(
        "--frequency",
        "-f",
        type=float,
        default=1000.0,
        help="Signal frequency in Hz (default: 1000)",
    )
    parser.add_argument(
        "--amplitude",
        "-a",
        type=float,
        default=1.0,
        help="Signal amplitude in V (default: 1.0)",
    )
    parser.add_argument(
        "--sample-rate",
        "-r",
        type=float,
        default=1e6,
        help="Sample rate in Sa/s (default: 1e6)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=0.001,
        help="Signal duration in s (default: 0.001)",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        help="Number of samples (overrides duration)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="DC offset in V (default: 0.0)",
    )
    parser.add_argument(
        "--phase",
        type=float,
        default=0.0,
        help="Phase in radians (default: 0.0)",
    )
    parser.add_argument(
        "--duty-cycle",
        type=float,
        default=0.5,
        help="Duty cycle for pulse/PWM (default: 0.5)",
    )
    parser.add_argument(
        "--snr",
        type=float,
        help="Signal-to-noise ratio in dB",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="CH1",
        help="Channel name (default: CH1)",
    )

    # Verbose output
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Check dependencies
    if not TM_DATA_TYPES_AVAILABLE:
        logger.error("ERROR: tm_data_types library not found")
        logger.error("Install with: pip install tm_data_types")
        raise SystemExit(1)

    # Generate test suite or single file
    if args.generate_suite:
        generator = TestSuiteGenerator(Path(args.output_dir))
        generator.generate_suite()
    else:
        if not args.output:
            logger.error("ERROR: --output required for single file generation")
            raise SystemExit(1)

        config = WaveformConfig(
            signal_type=SignalType(args.signal),
            frequency=args.frequency,
            amplitude=args.amplitude,
            sample_rate=args.sample_rate,
            duration=args.duration,
            offset=args.offset,
            phase=args.phase,
            duty_cycle=args.duty_cycle,
            snr_db=args.snr,
            num_samples=args.samples,
            channel_name=args.channel,
        )

        generator = SyntheticWFMGenerator(config)
        generator.save(args.output)


if __name__ == "__main__":
    main()
