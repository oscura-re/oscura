#!/usr/bin/env python3
"""Generate optimal demo data for Waveform Analysis demo.

Creates three representative WFM files showcasing ALL waveform analysis capabilities:
1. multi_channel_mixed_signal.wfm - Multi-channel analog/digital mixed signals
2. power_supply_ripple.wfm - Power supply analysis (ripple, efficiency)
3. high_speed_serial.wfm - Eye diagram and jitter analysis

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


def generate_multi_channel_mixed_signal(output_file: Path) -> None:
    """Generate multi-channel mixed-signal WFM file.

    Channels:
        - CH1: 1 MHz sine wave (analog)
        - CH2: 500 kHz square wave (analog)
        - CH3: 2 MHz PWM signal (analog)
        - CH4: Noisy signal with harmonics (analog)
        - D0: UART TX (115200 baud, "Hello World")
        - D1: SPI clock

    Sample rate: 100 MS/s
    Duration: 10 ms
    """
    print_info("Generating multi_channel_mixed_signal.wfm...")

    sample_rate = 100e6  # 100 MS/s
    duration = 10e-3  # 10 ms
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    # CH1: Clean 1 MHz sine wave
    ch1 = np.sin(2 * np.pi * 1e6 * t)

    # CH2: 500 kHz square wave
    ch2 = np.sign(np.sin(2 * np.pi * 500e3 * t))

    # CH3: 2 MHz PWM (30% duty cycle)
    pwm_period = 1 / 2e6
    duty_cycle = 0.3
    ch3 = ((t % pwm_period) < (duty_cycle * pwm_period)).astype(float) * 2 - 1

    # CH4: Noisy signal with harmonics
    fundamental = 1e6
    ch4 = (
        1.0 * np.sin(2 * np.pi * fundamental * t)
        + 0.1 * np.sin(2 * np.pi * 2 * fundamental * t)  # 2nd harmonic
        + 0.05 * np.sin(2 * np.pi * 3 * fundamental * t)  # 3rd harmonic
        + 0.02 * np.random.randn(num_samples)  # Noise
    )

    # D0: UART TX (115200 baud, "Hello World")
    baud_rate = 115200
    samples_per_bit = int(sample_rate / baud_rate)
    message = "Hello World"

    uart_bits = [1] * (samples_per_bit * 10)  # Idle high
    for char in message:
        byte_val = ord(char)
        # START bit (0)
        uart_bits.extend([0] * samples_per_bit)
        # Data bits (LSB first)
        for i in range(8):
            bit = (byte_val >> i) & 1
            uart_bits.extend([bit] * samples_per_bit)
        # STOP bit (1)
        uart_bits.extend([1] * samples_per_bit)
        # Inter-frame gap
        uart_bits.extend([1] * (samples_per_bit * 2))

    # Pad or trim to match signal length
    uart_bits.extend([1] * (num_samples - len(uart_bits)))
    d0 = np.array(uart_bits[:num_samples], dtype=bool)

    # D1: SPI clock (10 MHz)
    spi_clock_freq = 10e6
    d1 = ((t * spi_clock_freq) % 1.0 < 0.5).astype(bool)

    # For now, save as NPZ (WFM export requires more complex serialization)
    # In production, would use actual Tektronix WFM writer
    np.savez_compressed(
        output_file.with_suffix(".npz"),
        ch1=ch1,
        ch2=ch2,
        ch3=ch3,
        ch4=ch4,
        d0=d0.astype(np.uint8),
        d1=d1.astype(np.uint8),
        sample_rate=sample_rate,
        duration=duration,
        channel_names=[
            "CH1_1MHz_Sine",
            "CH2_500kHz_Square",
            "CH3_2MHz_PWM",
            "CH4_Noisy",
            "D0_UART",
            "D1_SPI_CLK",
        ],
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated multi_channel_mixed_signal.npz ({size_mb:.2f} MB)")


def generate_power_supply_ripple(output_file: Path) -> None:
    """Generate power supply ripple analysis WFM file.

    Channels:
        - CH1: 5V rail with 120 Hz ripple
        - CH2: Current sense (synchronized with ripple)

    Sample rate: 10 MS/s
    Duration: 100 ms
    """
    print_info("Generating power_supply_ripple.wfm...")

    sample_rate = 10e6  # 10 MS/s
    duration = 100e-3  # 100 ms
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    # CH1: 5V rail with 120 Hz ripple (typical switching power supply)
    dc_level = 5.0  # 5V nominal
    ripple_freq = 120  # Hz (2x line frequency)
    ripple_amplitude = 0.05  # 50 mV peak-to-peak
    switching_noise_freq = 100e3  # 100 kHz switching frequency
    switching_noise_amplitude = 0.01  # 10 mV

    ch1 = (
        dc_level
        + ripple_amplitude * np.sin(2 * np.pi * ripple_freq * t)
        + switching_noise_amplitude * np.sin(2 * np.pi * switching_noise_freq * t)
        + 0.002 * np.random.randn(num_samples)  # Measurement noise
    )

    # CH2: Current sense (1A average, correlated with ripple)
    current_avg = 1.0  # 1A
    current_ripple = 0.1  # 100 mA ripple
    ch2 = (
        current_avg
        + current_ripple * np.sin(2 * np.pi * ripple_freq * t + np.pi / 4)  # Phase shift
        + 0.01 * np.random.randn(num_samples)
    )

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        ch1=ch1,
        ch2=ch2,
        sample_rate=sample_rate,
        duration=duration,
        channel_names=["CH1_5V_Rail", "CH2_Current_Sense"],
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated power_supply_ripple.npz ({size_mb:.2f} MB)")


def generate_high_speed_serial(output_file: Path) -> None:
    """Generate high-speed serial eye diagram test signal.

    Channel:
        - CH1: 1 Gbps NRZ serial data with jitter and noise

    Sample rate: 20 GS/s
    Duration: 1 μs (1000 bits)
    """
    print_info("Generating high_speed_serial.wfm...")

    data_rate = 1e9  # 1 Gbps
    sample_rate = 20e9  # 20 GS/s (20 samples per bit)
    duration = 1e-6  # 1 μs
    num_samples = int(sample_rate * duration)
    samples_per_bit = int(sample_rate / data_rate)

    # Generate PRBS-7 pattern (pseudo-random binary sequence)
    num_bits = num_samples // samples_per_bit
    prbs_state = 0b1111111  # 7-bit seed
    bits = []
    for _ in range(num_bits):
        # PRBS-7 polynomial: x^7 + x^6 + 1
        new_bit = ((prbs_state >> 6) ^ (prbs_state >> 5)) & 1
        bits.append(new_bit)
        prbs_state = ((prbs_state << 1) | new_bit) & 0x7F

    # Convert bits to NRZ signal
    signal = np.zeros(num_samples)
    for i, bit in enumerate(bits):
        start_idx = i * samples_per_bit
        end_idx = start_idx + samples_per_bit
        signal[start_idx:end_idx] = 1.0 if bit else -1.0

    # Add realistic jitter
    # Random jitter (RJ): Gaussian distributed
    rj_sigma = 5e-12  # 5 ps RMS
    # Deterministic jitter (DJ): Fixed offset
    dj_amplitude = 10e-12  # 10 ps peak

    t = np.arange(num_samples) / sample_rate
    jittered_signal = np.zeros_like(signal)

    for i in range(num_samples):
        # Add jitter by sampling at slightly offset time
        time_jitter = np.random.randn() * rj_sigma + dj_amplitude * np.sin(2 * np.pi * 1e6 * t[i])
        jittered_idx = int(i + time_jitter * sample_rate)
        if 0 <= jittered_idx < num_samples:
            jittered_signal[i] = signal[jittered_idx]
        else:
            jittered_signal[i] = signal[i]

    # Add ISI (inter-symbol interference)
    isi_signal = jittered_signal.copy()
    for i in range(samples_per_bit, num_samples):
        isi_signal[i] = 0.9 * jittered_signal[i] + 0.1 * jittered_signal[i - samples_per_bit]

    # Add noise
    noise = 0.05 * np.random.randn(num_samples)
    final_signal = isi_signal + noise

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        ch1=final_signal,
        sample_rate=sample_rate,
        duration=duration,
        data_rate=data_rate,
        channel_names=["CH1_1Gbps_NRZ"],
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated high_speed_serial.npz ({size_mb:.2f} MB)")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate demo data for waveform analysis")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    # Create demo_data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "demo_data"
    data_dir.mkdir(exist_ok=True)

    # Define output files
    files_to_generate = [
        ("multi_channel_mixed_signal.wfm", generate_multi_channel_mixed_signal),
        ("power_supply_ripple.wfm", generate_power_supply_ripple),
        ("high_speed_serial.wfm", generate_high_speed_serial),
    ]

    print(f"\n{BOLD}{BLUE}Generating Waveform Analysis Demo Data{RESET}")
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
