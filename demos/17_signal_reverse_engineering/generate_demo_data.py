#!/usr/bin/env python3
"""Generate optimal demo data for Signal Reverse Engineering demo.

Creates three mystery signal files requiring reverse engineering:
1. mystery_serial_protocol.wfm - Non-standard baud rate, 9-bit frames, custom parity
2. mixed_signal_embedded.wfm - 3 analog (power), 2 digital (I2C), correlated timing
3. rf_baseband_capture.wfm - FSK baseband with packets

All files contain intentionally unknown/non-standard characteristics that must be
discovered through signal analysis.

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


def generate_mystery_serial_protocol(output_file: Path) -> None:
    """Generate mystery serial protocol signal.

    Characteristics (to be discovered):
        - Baud rate: 117650 bps (non-standard, close to 115200)
        - Frame format: 9-bit (1 start, 8 data, 1 custom parity)
        - Parity: Custom algorithm (XOR of bits 0,2,4,6)
        - State machine: 5 states with timing-dependent transitions
        - Content: Command/response protocol

    Sample rate: 10 MS/s
    Duration: 100 ms
    File size: ~4 MB
    """
    print_info("Generating mystery_serial_protocol.wfm...")

    sample_rate = 10e6  # 10 MS/s
    duration = 100e-3  # 100 ms
    num_samples = int(sample_rate * duration)

    # Non-standard baud rate (to be discovered)
    baud_rate = 117650  # Not 115200!
    samples_per_bit = int(sample_rate / baud_rate)

    # State machine message sequence
    messages = [
        b"\x01\x00\x00\x00",  # INIT command
        b"\x81\x00\x00\x00",  # INIT ACK
        b"\x02\x12\x34\x56",  # DATA command
        b"\x82\x12\x34\x56",  # DATA ACK
        b"\x03\x00\x00\x00",  # STATUS request
        b"\x83\xaa\xbb\xcc",  # STATUS response
        b"\x04\xff\xff\xff",  # RESET command
        b"\x84\x00\x00\x00",  # RESET ACK
    ]

    signal = np.ones(num_samples)  # Start idle high
    current_sample = int(0.01 * sample_rate)  # Start after 10ms

    for msg_idx, message in enumerate(messages):
        for byte_val in message:
            # Custom parity: XOR of bits 0,2,4,6
            parity_bit = (
                ((byte_val >> 0) & 1)
                ^ ((byte_val >> 2) & 1)
                ^ ((byte_val >> 4) & 1)
                ^ ((byte_val >> 6) & 1)
            )

            # START bit (0)
            signal[current_sample : current_sample + samples_per_bit] = 0.0
            current_sample += samples_per_bit

            # Data bits (LSB first)
            for i in range(8):
                bit = (byte_val >> i) & 1
                signal[current_sample : current_sample + samples_per_bit] = float(bit)
                current_sample += samples_per_bit

            # Custom parity bit
            signal[current_sample : current_sample + samples_per_bit] = float(parity_bit)
            current_sample += samples_per_bit

            # Inter-byte gap (variable based on state)
            gap_samples = samples_per_bit * (2 + msg_idx % 3)  # Variable gap
            signal[current_sample : current_sample + gap_samples] = 1.0
            current_sample += gap_samples

            if current_sample >= num_samples:
                break

        # Inter-message gap (longer)
        gap_samples = int(0.005 * sample_rate)  # 5ms gap
        if current_sample + gap_samples < num_samples:
            signal[current_sample : current_sample + gap_samples] = 1.0
            current_sample += gap_samples

    # Add realistic noise and edge artifacts
    noise = 0.02 * np.random.randn(num_samples)
    signal += noise

    # Add rise/fall time (realistic edge transitions)
    for i in range(1, num_samples):
        if abs(signal[i] - signal[i - 1]) > 0.5:
            # Smooth transition over 5 samples
            for j in range(1, min(5, num_samples - i)):
                alpha = j / 5.0
                signal[i + j] = signal[i - 1] * (1 - alpha) + signal[i] * alpha

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        ch1=signal,
        sample_rate=sample_rate,
        duration=duration,
        channel_names=["CH1_Mystery_Serial"],
        # Hidden metadata (would not be in real capture)
        _true_baud_rate=baud_rate,
        _frame_format="9-bit (1 start, 8 data, 1 custom parity)",
        _parity_algorithm="XOR bits 0,2,4,6",
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated mystery_serial_protocol.npz ({size_mb:.2f} MB)")


def generate_mixed_signal_embedded(output_file: Path) -> None:
    """Generate mixed-signal embedded system capture.

    Channels:
        - CH1: 3.3V power rail (analog)
        - CH2: 5.0V power rail (analog)
        - CH3: Audio signal (analog)
        - D0: I2C SCL (digital)
        - D1: I2C SDA (digital)

    Correlations (to be discovered):
        - I2C transactions occur during power state transitions
        - Audio activity triggers I2C communication
        - Power rails show correlated ripple during I2C activity

    Sample rate: 10 MS/s
    Duration: 50 ms
    File size: ~6 MB
    """
    print_info("Generating mixed_signal_embedded.wfm...")

    sample_rate = 10e6  # 10 MS/s
    duration = 50e-3  # 50 ms
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    # Initialize signals
    ch1_3v3 = np.ones(num_samples) * 3.3  # 3.3V rail
    ch2_5v0 = np.ones(num_samples) * 5.0  # 5.0V rail
    ch3_audio = np.zeros(num_samples)  # Audio signal
    d0_scl = np.ones(num_samples, dtype=bool)  # I2C clock (idle high)
    d1_sda = np.ones(num_samples, dtype=bool)  # I2C data (idle high)

    # Generate correlated events
    i2c_clock_freq = 400e3  # 400 kHz I2C
    samples_per_i2c_bit = int(sample_rate / i2c_clock_freq)

    # Event 1: Audio burst at t=10ms triggers I2C transaction
    audio_start = int(0.01 * sample_rate)
    audio_duration = int(0.005 * sample_rate)
    ch3_audio[audio_start : audio_start + audio_duration] = 0.5 * np.sin(
        2 * np.pi * 1e3 * t[audio_start : audio_start + audio_duration]
    )

    # I2C transaction starts 1ms after audio
    i2c_start = audio_start + int(0.001 * sample_rate)
    i2c_addr = 0x48  # I2C address
    i2c_data = [0x01, 0x23, 0x45]  # Data bytes

    i2c_idx = i2c_start
    i2c_idx = generate_i2c_transaction(
        d0_scl, d1_sda, i2c_idx, samples_per_i2c_bit, i2c_addr, i2c_data, write=True
    )

    # Power rail shows ripple during I2C activity
    _i2c_duration = i2c_idx - i2c_start
    ripple_freq = i2c_clock_freq / 2
    ripple_amplitude_3v3 = 0.05  # 50mV ripple
    ripple_amplitude_5v0 = 0.08  # 80mV ripple

    ch1_3v3[i2c_start:i2c_idx] += ripple_amplitude_3v3 * np.sin(
        2 * np.pi * ripple_freq * t[i2c_start:i2c_idx]
    )
    ch2_5v0[i2c_start:i2c_idx] += ripple_amplitude_5v0 * np.sin(
        2 * np.pi * ripple_freq * t[i2c_start:i2c_idx] + np.pi / 4
    )

    # Event 2: Power state transition at t=30ms triggers I2C
    power_transition_start = int(0.03 * sample_rate)
    power_transition_duration = int(0.002 * sample_rate)

    # 3.3V rail drops briefly
    ch1_3v3[power_transition_start : power_transition_start + power_transition_duration] -= 0.2

    # I2C read transaction during power transition
    i2c_start2 = power_transition_start + int(0.0005 * sample_rate)
    _i2c_idx2 = generate_i2c_transaction(
        d0_scl, d1_sda, i2c_start2, samples_per_i2c_bit, 0x68, [0x00], write=False
    )

    # Add realistic noise to all channels
    ch1_3v3 += 0.01 * np.random.randn(num_samples)
    ch2_5v0 += 0.015 * np.random.randn(num_samples)
    ch3_audio += 0.01 * np.random.randn(num_samples)

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        ch1=ch1_3v3,
        ch2=ch2_5v0,
        ch3=ch3_audio,
        d0=d0_scl.astype(np.uint8),
        d1=d1_sda.astype(np.uint8),
        sample_rate=sample_rate,
        duration=duration,
        channel_names=["CH1_3V3_Rail", "CH2_5V0_Rail", "CH3_Audio", "D0_I2C_SCL", "D1_I2C_SDA"],
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated mixed_signal_embedded.npz ({size_mb:.2f} MB)")


def generate_i2c_transaction(
    scl: np.ndarray,
    sda: np.ndarray,
    start_idx: int,
    samples_per_bit: int,
    addr: int,
    data: list[int],
    write: bool,
) -> int:
    """Generate I2C transaction on SCL/SDA lines.

    Returns:
        Index after transaction completes
    """
    idx = start_idx

    # START condition: SDA falls while SCL high
    sda[idx : idx + samples_per_bit // 2] = False
    idx += samples_per_bit

    # Address byte (7 bits + R/W bit)
    addr_byte = (addr << 1) | (0 if write else 1)
    for i in range(8):
        # Clock low
        scl[idx : idx + samples_per_bit // 2] = False
        # Set data
        bit = (addr_byte >> (7 - i)) & 1
        sda[idx : idx + samples_per_bit] = bool(bit)
        idx += samples_per_bit // 2
        # Clock high
        scl[idx : idx + samples_per_bit // 2] = True
        idx += samples_per_bit // 2

    # ACK bit
    scl[idx : idx + samples_per_bit // 2] = False
    sda[idx : idx + samples_per_bit] = False  # ACK (low)
    idx += samples_per_bit // 2
    scl[idx : idx + samples_per_bit // 2] = True
    idx += samples_per_bit // 2

    # Data bytes
    for byte_val in data:
        for i in range(8):
            scl[idx : idx + samples_per_bit // 2] = False
            bit = (byte_val >> (7 - i)) & 1
            sda[idx : idx + samples_per_bit] = bool(bit)
            idx += samples_per_bit // 2
            scl[idx : idx + samples_per_bit // 2] = True
            idx += samples_per_bit // 2

        # ACK bit
        scl[idx : idx + samples_per_bit // 2] = False
        sda[idx : idx + samples_per_bit] = False
        idx += samples_per_bit // 2
        scl[idx : idx + samples_per_bit // 2] = True
        idx += samples_per_bit // 2

    # STOP condition: SDA rises while SCL high
    scl[idx : idx + samples_per_bit // 2] = False
    sda[idx : idx + samples_per_bit // 2] = False
    idx += samples_per_bit // 2
    scl[idx : idx + samples_per_bit // 2] = True
    idx += samples_per_bit // 2
    sda[idx : idx + samples_per_bit // 2] = True
    idx += samples_per_bit // 2

    return idx


def generate_rf_baseband_capture(output_file: Path) -> None:
    """Generate RF baseband FSK capture.

    Characteristics (to be discovered):
        - Modulation: FSK (Frequency Shift Keying)
        - Symbol rate: 9600 symbols/second (non-standard)
        - Frequency deviation: ±5 kHz from center
        - Packet structure: Preamble (32 bits), sync word (16 bits), data, CRC (16 bits)
        - Data content: 3 packets with timing

    Sample rate: 1 MS/s
    Duration: 100 ms
    File size: ~5 MB
    """
    print_info("Generating rf_baseband_capture.wfm...")

    sample_rate = 1e6  # 1 MS/s
    duration = 100e-3  # 100 ms
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate

    # FSK parameters
    center_freq = 0  # Baseband (DC)
    freq_deviation = 5e3  # ±5 kHz
    symbol_rate = 9600  # 9600 symbols/second
    samples_per_symbol = int(sample_rate / symbol_rate)

    signal = np.zeros(num_samples)

    # Generate 3 packets
    packet_starts = [int(0.01 * sample_rate), int(0.04 * sample_rate), int(0.07 * sample_rate)]

    for packet_start in packet_starts:
        # Preamble: Alternating 01010101 pattern (32 bits)
        preamble = [0, 1] * 16

        # Sync word: 0xAA55 (16 bits)
        sync_word = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]

        # Data: Random bytes (8 bytes = 64 bits)
        data_bits = []
        for _ in range(8):
            byte_val = np.random.randint(0, 256)
            for i in range(8):
                data_bits.append((byte_val >> (7 - i)) & 1)

        # CRC: 0xFFFF (16 bits) - simplified
        crc = [1] * 16

        # Combine packet
        packet_bits = preamble + sync_word + data_bits + crc

        # Generate FSK signal
        idx = packet_start
        for bit in packet_bits:
            if idx + samples_per_symbol >= num_samples:
                break
            freq = center_freq + (freq_deviation if bit else -freq_deviation)
            signal[idx : idx + samples_per_symbol] = np.sin(
                2 * np.pi * freq * t[idx : idx + samples_per_symbol]
            )
            idx += samples_per_symbol

    # Add noise and carrier drift
    noise = 0.1 * np.random.randn(num_samples)
    carrier_drift = 0.05 * np.sin(2 * np.pi * 100 * t)  # 100 Hz drift
    signal = signal + noise + carrier_drift

    # Normalize
    signal = signal / np.max(np.abs(signal))

    np.savez_compressed(
        output_file.with_suffix(".npz"),
        ch1=signal,
        sample_rate=sample_rate,
        duration=duration,
        channel_names=["CH1_RF_Baseband"],
        # Hidden metadata (would not be in real capture)
        _modulation="FSK",
        _symbol_rate=symbol_rate,
        _freq_deviation=freq_deviation,
    )

    size_mb = output_file.with_suffix(".npz").stat().st_size / (1024 * 1024)
    print_success(f"Generated rf_baseband_capture.npz ({size_mb:.2f} MB)")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate demo data for signal reverse engineering"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    # Create demo_data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "demo_data"
    data_dir.mkdir(exist_ok=True)

    # Define output files
    files_to_generate = [
        ("mystery_serial_protocol.wfm", generate_mystery_serial_protocol),
        ("mixed_signal_embedded.wfm", generate_mixed_signal_embedded),
        ("rf_baseband_capture.wfm", generate_rf_baseband_capture),
    ]

    print(f"\n{BOLD}{BLUE}Generating Signal Reverse Engineering Demo Data{RESET}")
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
