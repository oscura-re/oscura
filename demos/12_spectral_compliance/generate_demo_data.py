#!/usr/bin/env python3
"""Generate optimal demo data for Spectral Compliance demo.

Creates three spectral test signals for IEEE-compliant analysis:
1. audio_amplifier_1khz.wfm - 1 kHz tone, THD=0.05%, SNR=90 dB
2. adc_characterization_12bit.wfm - Full-scale sine for 12-bit ADC, quantization, DNL/INL
3. power_line_harmonics.wfm - 60 Hz + harmonics, THD=8%

Usage:
    python generate_demo_data.py [--force]

Author: Oscura Development Team
Date: 2026-01-15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ANSI colors
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓{RESET} {msg}")


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"{BLUE}INFO:{RESET} {msg}")


def generate_audio_amplifier_1khz(output_file: Path) -> None:
    """Generate audio amplifier 1 kHz test tone WFM file.

    Signal: 1 kHz sine wave @ 1 Vrms (audio test tone)
    Characteristics: THD = 0.05% (2nd + 3rd harmonics), SNR = 90 dB
    Sampling: 192 kHz, 24-bit equivalent, 100 ms

    Sample rate: 192 kHz
    Duration: 100 ms
    """
    print_info("Generating audio_amplifier_1khz.wfm...")

    sample_rate = 192e3  # 192 kHz (high-quality audio)
    duration = 100e-3  # 100 ms
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    # Fundamental: 1 kHz @ 1 Vrms (peak = sqrt(2) * 1V)
    fundamental_freq = 1000  # Hz
    fundamental_amplitude = np.sqrt(2)  # 1 Vrms
    fundamental = fundamental_amplitude * np.sin(2 * np.pi * fundamental_freq * t)

    # Add harmonics for THD = 0.05%
    # THD = sqrt(V2^2 + V3^2 + ...) / V1
    # For THD = 0.05%, sum of harmonics^2 = (0.0005 * fundamental_amplitude)^2
    # Distribute between 2nd and 3rd harmonics
    thd_target = 0.0005
    harmonic2_amplitude = fundamental_amplitude * thd_target * 0.7  # 70% to 2nd
    harmonic3_amplitude = fundamental_amplitude * thd_target * 0.3  # 30% to 3rd

    harmonic2 = harmonic2_amplitude * np.sin(2 * np.pi * 2 * fundamental_freq * t)
    harmonic3 = harmonic3_amplitude * np.sin(2 * np.pi * 3 * fundamental_freq * t)

    # Add noise for SNR = 90 dB
    # SNR = 20 * log10(signal_rms / noise_rms)
    # 90 dB => noise_rms = signal_rms / 10^(90/20) = signal_rms / 31622.78
    signal_rms = fundamental_amplitude / np.sqrt(2)  # Convert peak to RMS
    noise_rms = signal_rms / (10 ** (90 / 20))
    noise = noise_rms * np.random.randn(num_samples)

    # Combine signal components
    signal = fundamental + harmonic2 + harmonic3 + noise

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        ch1=signal,
        sample_rate=sample_rate,
        duration=duration,
        fundamental_freq=fundamental_freq,
        channel_names=["Audio_Output"],
        metadata={
            "signal_type": "Audio test tone",
            "fundamental": "1 kHz @ 1 Vrms",
            "thd_target": "0.05%",
            "snr_target": "90 dB",
            "harmonics": "2nd (0.035%), 3rd (0.015%)",
        },
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated audio_amplifier_1khz.npz ({size_mb:.2f} MB)")


def generate_adc_characterization_12bit(output_file: Path) -> None:
    """Generate ADC characterization WFM file.

    Signal: Pure 1 kHz sine @ 4096 LSB p-p (full-scale 12-bit ADC)
    Characteristics: Quantization noise, DNL/INL errors, ENOB ≈ 11.2 bits
    Sampling: 100 MS/s, 10 ms

    Sample rate: 100 MS/s
    Duration: 10 ms
    """
    print_info("Generating adc_characterization_12bit.wfm...")

    sample_rate = 100e6  # 100 MS/s
    duration = 10e-3  # 10 ms
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    # Generate ideal sine wave
    fundamental_freq = 1000  # Hz
    adc_bits = 12
    adc_levels = 2**adc_bits  # 4096 levels
    full_scale = adc_levels  # Full-scale in LSB

    # Generate coherent sampling (exact integer number of cycles)
    # This ensures proper FFT analysis per IEEE 1241
    _num_cycles = int(fundamental_freq * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Ideal sine: -full_scale/2 to +full_scale/2
    ideal_signal = (full_scale / 2) * np.sin(2 * np.pi * fundamental_freq * t)

    # Apply 12-bit quantization
    # Add small dither before quantization (typical in real ADCs)
    dither = 0.5 * (np.random.rand(num_samples) - 0.5)
    quantized_signal = np.round(ideal_signal + dither)

    # Clip to ADC range
    quantized_signal = np.clip(quantized_signal, -full_scale / 2, full_scale / 2 - 1)

    # Add DNL/INL errors (differential and integral nonlinearity)
    # DNL: variation in step size (typically ±0.5 LSB for 12-bit ADC)
    dnl_error = 0.3 * np.random.randn(num_samples)

    # INL: cumulative error (bow shape, typically ±2 LSB for 12-bit ADC)
    # Model as low-frequency sine wave
    inl_error = 2.0 * np.sin(2 * np.pi * 0.1 * t / duration)  # Bow shape

    # Add ADC noise for realistic ENOB ~11.2 bits
    # ENOB = 11.2 => effective noise = quantization_noise * 2^(12-11.2)
    quantization_noise_rms = 1 / np.sqrt(12)  # LSB RMS for ideal quantization
    adc_noise_rms = quantization_noise_rms * (2 ** (adc_bits - 11.2))
    adc_noise = adc_noise_rms * np.random.randn(num_samples)

    # Combine all error sources
    adc_output = quantized_signal + dnl_error + inl_error + adc_noise

    # Convert to voltage (assume 0-3.3V range)
    voltage_per_lsb = 3.3 / adc_levels
    signal_voltage = adc_output * voltage_per_lsb

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        ch1=signal_voltage,
        adc_output_lsb=adc_output,
        sample_rate=sample_rate,
        duration=duration,
        fundamental_freq=fundamental_freq,
        adc_bits=adc_bits,
        channel_names=["ADC_Output"],
        metadata={
            "signal_type": "ADC characterization",
            "adc_resolution": "12 bits",
            "full_scale": "4096 LSB (3.3V)",
            "enob_target": "~11.2 bits",
            "test_standard": "IEEE 1241-2010",
            "errors": "DNL (±0.3 LSB), INL (±2 LSB), noise",
        },
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated adc_characterization_12bit.npz ({size_mb:.2f} MB)")


def generate_power_line_harmonics(output_file: Path) -> None:
    """Generate power line harmonics WFM file.

    Signal: 60 Hz fundamental + harmonics up to 13th
    Characteristics: THD = 8%, realistic power quality issues
    Sampling: 10 kHz, 1 second

    Sample rate: 10 kHz
    Duration: 1 second
    """
    print_info("Generating power_line_harmonics.wfm...")

    sample_rate = 10e3  # 10 kHz (sufficient for power line analysis)
    duration = 1.0  # 1 second (multiple cycles for good frequency resolution)
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    # Fundamental: 60 Hz @ 120V RMS (peak = 120 * sqrt(2))
    fundamental_freq = 60  # Hz
    fundamental_amplitude = 120 * np.sqrt(2)  # Peak voltage
    fundamental = fundamental_amplitude * np.sin(2 * np.pi * fundamental_freq * t)

    # Add harmonics for THD = 8%
    # Common harmonics in power systems (odd harmonics dominate)
    # THD = sqrt(sum(Vh^2)) / V1
    # THD = 8% => sum(Vh^2) = (0.08 * fundamental_amplitude)^2

    # Typical harmonic distribution for nonlinear loads
    harmonic_percentages = {
        3: 5.0,  # 3rd harmonic (most significant)
        5: 4.0,  # 5th harmonic
        7: 2.5,  # 7th harmonic
        9: 1.5,  # 9th harmonic
        11: 1.0,  # 11th harmonic
        13: 0.5,  # 13th harmonic
    }

    signal = fundamental.copy()
    for harmonic_num, percentage in harmonic_percentages.items():
        harmonic_amplitude = fundamental_amplitude * (percentage / 100)
        # Add some phase shift for realism
        phase = np.random.uniform(0, 2 * np.pi)
        harmonic = harmonic_amplitude * np.sin(
            2 * np.pi * harmonic_num * fundamental_freq * t + phase
        )
        signal += harmonic

    # Add measurement noise
    noise_rms = fundamental_amplitude * 0.001  # 0.1% noise
    noise = noise_rms * np.random.randn(num_samples)
    signal += noise

    # Add occasional transients (typical in power systems)
    # Small voltage sag at 0.3 seconds
    sag_start = int(0.3 * sample_rate)
    sag_duration = int(0.05 * sample_rate)  # 50 ms sag
    signal[sag_start : sag_start + sag_duration] *= 0.9

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        ch1=signal,
        sample_rate=sample_rate,
        duration=duration,
        fundamental_freq=fundamental_freq,
        channel_names=["AC_Line_Voltage"],
        metadata={
            "signal_type": "Power line harmonics",
            "fundamental": "60 Hz @ 120 Vrms",
            "thd_target": "~8%",
            "harmonics": "3rd (5%), 5th (4%), 7th (2.5%), 9th (1.5%), 11th (1%), 13th (0.5%)",
            "standard": "IEEE 1459 (Power quality)",
            "features": "Nonlinear load harmonics, voltage sag @ 0.3s",
        },
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated power_line_harmonics.npz ({size_mb:.2f} MB)")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate demo data for spectral compliance")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    # Create demo_data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "demo_data"
    data_dir.mkdir(exist_ok=True)

    # Define output files
    files_to_generate = [
        ("audio_amplifier_1khz.wfm", generate_audio_amplifier_1khz),
        ("adc_characterization_12bit.wfm", generate_adc_characterization_12bit),
        ("power_line_harmonics.wfm", generate_power_line_harmonics),
    ]

    print(f"\n{BOLD}{BLUE}Generating Spectral Compliance Demo Data{RESET}")
    print("=" * 80)

    for filename, generator_func in files_to_generate:
        output_file = data_dir / filename

        if output_file.with_suffix(".npz").exists() and not args.force:
            print_info(f"Skipping {filename} (already exists, use --force to overwrite)")
            continue

        generator_func(output_file)

    print(f"\n{GREEN}{BOLD}✓ Demo data generation complete!{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
