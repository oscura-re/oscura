#!/usr/bin/env python3
"""Generate optimal demo data for Custom DAQ demo.

Creates two binary DAQ files showcasing YAML-driven custom format loading:
1. multi_lane_daq_10M.bin (~80MB) - 10M samples, 4 lanes with different signal types
2. continuous_acquisition_100M.bin (~800MB) - 100M samples for streaming capability

Format: 8 bytes/sample matching custom_daq_continuous.yml config:
    - 4 lanes x 2 bytes each (16 bits per lane)
    - Lane 1: 10 MHz sine wave (12-bit ADC)
    - Lane 2: 5 MHz triangle wave (12-bit ADC)
    - Lane 3: Random noise (full 16-bit)
    - Lane 4: Digital counter pattern

Usage:
    python generate_demo_data.py [--force] [--skip-large]

Author: Oscura Development Team
Date: 2026-01-15
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# ANSI colors
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓{RESET} {msg}")


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"{BLUE}INFO:{RESET} {msg}")


def print_warning(msg: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}⚠{RESET} {msg}")


def quantize_12bit(signal: np.ndarray) -> np.ndarray:
    """Quantize signal to 12-bit ADC resolution.

    Args:
        signal: Input signal (normalized to [-1, 1])

    Returns:
        Quantized signal in 16-bit format (12-bit data in upper bits)
    """
    # Scale to 12-bit range [0, 4095]
    scaled = ((signal + 1.0) / 2.0) * 4095.0
    quantized = np.clip(scaled, 0, 4095).astype(np.uint16)

    # Shift to upper 12 bits of 16-bit word (common ADC output format)
    return quantized << 4


def generate_multi_lane_daq_10M(output_file: Path) -> None:
    """Generate 10M sample multi-lane DAQ file (~80MB).

    Format: 8 bytes/sample (4 lanes x 16 bits)
        - Lane 1: 10 MHz sine wave (12-bit ADC)
        - Lane 2: 5 MHz triangle wave (12-bit ADC)
        - Lane 3: Random noise (full 16-bit)
        - Lane 4: Digital counter pattern

    Sample rate: 100 MS/s
    Duration: 100 ms
    File size: ~80 MB
    """
    print_info("Generating multi_lane_daq_10M.bin...")

    sample_rate = 100e6  # 100 MS/s
    num_samples = 10_000_000  # 10M samples
    duration = num_samples / sample_rate
    t = np.arange(num_samples) / sample_rate

    print_info(f"  Duration: {duration * 1000:.1f} ms")
    print_info(f"  Samples: {num_samples:,}")

    # Lane 1: 10 MHz sine wave (12-bit ADC)
    print_info("  Generating Lane 1: 10 MHz sine wave...")
    lane1_analog = np.sin(2 * np.pi * 10e6 * t)
    lane1 = quantize_12bit(lane1_analog)

    # Lane 2: 5 MHz triangle wave (12-bit ADC)
    print_info("  Generating Lane 2: 5 MHz triangle wave...")
    triangle_freq = 5e6
    phase = (t * triangle_freq) % 1.0
    lane2_analog = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0  # Triangle [-1, 1]
    lane2 = quantize_12bit(lane2_analog)

    # Lane 3: Random noise (full 16-bit)
    print_info("  Generating Lane 3: Random noise...")
    # Generate in chunks to manage memory
    chunk_size = 1_000_000
    lane3 = np.zeros(num_samples, dtype=np.uint16)
    for i in range(0, num_samples, chunk_size):
        end = min(i + chunk_size, num_samples)
        lane3[i:end] = np.random.randint(0, 65536, size=end - i, dtype=np.uint16)

    # Lane 4: Digital counter pattern
    print_info("  Generating Lane 4: Counter pattern...")
    lane4 = np.arange(num_samples, dtype=np.uint16)

    # Pack into binary format: 8 bytes per sample (little-endian)
    print_info("  Writing binary file...")
    with open(output_file, "wb") as f:
        # Write in chunks to manage memory
        chunk_size = 100_000
        for i in range(0, num_samples, chunk_size):
            end = min(i + chunk_size, num_samples)
            chunk_bytes = b""
            for j in range(i, end):
                # Pack 4 uint16 values in little-endian format
                chunk_bytes += struct.pack("<HHHH", lane1[j], lane2[j], lane3[j], lane4[j])
            f.write(chunk_bytes)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print_success(f"Generated multi_lane_daq_10M.bin ({size_mb:.2f} MB)")


def generate_continuous_acquisition_100M(output_file: Path) -> None:
    """Generate 100M sample continuous acquisition file (~800MB).

    Format: 8 bytes/sample (4 lanes x 16 bits)
        - Lane 1-4: Synthetic continuous waveforms

    Sample rate: 100 MS/s
    Duration: 1 second
    File size: ~800 MB

    This demonstrates optimal chunked processing for large files.
    """
    print_info("Generating continuous_acquisition_100M.bin...")
    print_warning("  This is a large file (~800MB) and will take a few minutes...")

    sample_rate = 100e6  # 100 MS/s
    num_samples = 100_000_000  # 100M samples
    duration = num_samples / sample_rate

    print_info(f"  Duration: {duration:.1f} seconds")
    print_info(f"  Samples: {num_samples:,}")

    # Generate and write in chunks to avoid memory issues
    chunk_size = 1_000_000  # 1M samples per chunk (~8MB)
    num_chunks = num_samples // chunk_size

    with open(output_file, "wb") as f:
        for chunk_idx in range(num_chunks):
            if chunk_idx % 10 == 0:
                progress = (chunk_idx / num_chunks) * 100
                print_info(f"  Progress: {progress:.1f}% ({chunk_idx}/{num_chunks} chunks)")

            # Time vector for this chunk
            t_start = chunk_idx * chunk_size / sample_rate
            t = t_start + np.arange(chunk_size) / sample_rate

            # Lane 1: Slow sine wave (1 MHz)
            lane1_analog = np.sin(2 * np.pi * 1e6 * t)
            lane1 = quantize_12bit(lane1_analog)

            # Lane 2: Medium frequency sine (5 MHz)
            lane2_analog = np.sin(2 * np.pi * 5e6 * t)
            lane2 = quantize_12bit(lane2_analog)

            # Lane 3: Fast sine (10 MHz)
            lane3_analog = np.sin(2 * np.pi * 10e6 * t)
            lane3 = quantize_12bit(lane3_analog)

            # Lane 4: Counter
            lane4 = np.arange(chunk_idx * chunk_size, (chunk_idx + 1) * chunk_size, dtype=np.uint16)

            # Pack and write chunk
            chunk_bytes = b""
            for i in range(chunk_size):
                chunk_bytes += struct.pack("<HHHH", lane1[i], lane2[i], lane3[i], lane4[i])
            f.write(chunk_bytes)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print_success(f"Generated continuous_acquisition_100M.bin ({size_mb:.2f} MB)")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate demo data for custom DAQ")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="Skip generating the large 100M sample file (800MB)",
    )
    args = parser.parse_args()

    # Create demo_data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "demo_data"
    data_dir.mkdir(exist_ok=True)

    print(f"\n{BOLD}{BLUE}Generating Custom DAQ Demo Data{RESET}")
    print("=" * 80)

    # Generate small file (10M samples, ~80MB)
    small_file = data_dir / "multi_lane_daq_10M.bin"
    if small_file.exists() and not args.force:
        print_info("Skipping multi_lane_daq_10M.bin (already exists, use --force to overwrite)")
    else:
        generate_multi_lane_daq_10M(small_file)

    # Generate large file (100M samples, ~800MB)
    if not args.skip_large:
        large_file = data_dir / "continuous_acquisition_100M.bin"
        if large_file.exists() and not args.force:
            print_info(
                "Skipping continuous_acquisition_100M.bin (already exists, use --force to overwrite)"
            )
        else:
            print_info("")
            generate_continuous_acquisition_100M(large_file)
    else:
        print_warning("Skipping large file generation (--skip-large specified)")

    print(f"\n{GREEN}{BOLD}✓ Demo data generation complete!{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
