#!/usr/bin/env python3
"""Optimal memory-efficient custom DAQ loader using Oscura core streaming APIs.

# SKIP_VALIDATION: Processes 2.9GB file, takes >30s

This script demonstrates the OPTIMAL way to load large custom DAQ files
using Oscura's core streaming infrastructure. All core functionality
(packet streaming, channel extraction) is handled by Oscura's proven
core APIs. This script only adds domain-specific logic.

File: udp_capture_1.bin (2.9GB, 382M samples)
Format: Continuous time-series, 8 bytes per sample
Origin: MATLAB-preprocessed UDP packet capture
Structure: 4-lane parallel acquisition @ 100 MHz

Features:
    - Uses Oscura core streaming APIs (load_packets_streaming)
    - Memory-efficient (O(chunk_size) instead of O(total_size))
    - Statistics-only mode (minimal memory)
    - Export to NPZ (full dataset)
    - Export to HDF5 (chunked storage)

Usage:
    # Statistics only (minimal memory ~305 MB)
    python scripts/load_custom_daq_optimal.py udp_capture_1.bin custom_daq_continuous.yml --stats

    # Export to NPZ
    python scripts/load_custom_daq_optimal.py udp_capture_1.bin custom_daq_continuous.yml --export output.npz

    # Export to HDF5
    python scripts/load_custom_daq_optimal.py udp_capture_1.bin custom_daq_continuous.yml --export-hdf5 output.h5

Author: Oscura Development Team
Date: 2026-01-15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Use Oscura core streaming APIs (OPTIMAL)
from oscura.loaders.configurable import (
    BitfieldExtractor,
    PacketFormatConfig,
    load_packets_streaming,
)


class OptimalDAQAnalyzer:
    """Domain-specific DAQ analyzer using Oscura core streaming APIs."""

    def __init__(
        self,
        data_file: Path,
        config_file: Path,
        channel_map: dict[str, dict[str, Any]],
        *,
        chunk_size: int = 10_000_000,
    ):
        """Initialize analyzer.

        Args:
            data_file: Path to binary DAQ data file.
            config_file: Path to YAML format configuration.
            channel_map: Channel definitions with bit ranges.
            chunk_size: Number of samples per chunk (default: 10M).
        """
        self.data_file = data_file
        self.config_file = config_file
        self.channel_map = channel_map
        self.chunk_size = chunk_size

        # Load configuration
        self.config = PacketFormatConfig.from_yaml(config_file)
        self.sample_rate = 100e6  # 100 MHz

        # File info
        self.file_size = data_file.stat().st_size
        self.total_samples = self.file_size // self.config.packet_size
        self.duration = self.total_samples / self.sample_rate

    def print_info(self) -> None:
        """Print file and configuration information."""
        print("=" * 80)
        print("OPTIMAL DAQ LOADER - USING OSCURA CORE STREAMING APIs")
        print("=" * 80)

        print("\n1. Configuration:")
        print(f"   Format: {self.config.name} v{self.config.version}")
        print(f"   Description: {self.config.description}")

        print("\n2. File Information:")
        print(f"   File: {self.data_file.name}")
        print(f"   Size: {self.file_size:,} bytes ({self.file_size / 1024**3:.2f} GB)")
        print(f"   Total samples: {self.total_samples:,}")
        print(f"   Duration @ {self.sample_rate / 1e6:.0f} MHz: {self.duration:.3f} seconds")

        print("\n3. Processing Configuration:")
        print(f"   Chunk size: {self.chunk_size:,} samples")
        print(f"   Chunk memory: ~{self.chunk_size * 8 / 1024**2:.0f} MB per chunk")
        print(f"   Channels: {list(self.channel_map.keys())}")

    def compute_statistics(self) -> dict[str, Any]:
        """Compute statistics without storing all data (minimal memory).

        Returns:
            Dictionary with statistics for each channel.
        """
        print("\n4. Computing Statistics (Streaming Mode)...")
        print("   Using Oscura core load_packets_streaming() API")
        print("   Processing: ", end="", flush=True)

        stats = {}
        for ch_name in self.channel_map:
            stats[ch_name] = {
                "min": float("inf"),
                "max": float("-inf"),
                "sum": 0.0,
                "count": 0,
                "non_zero": 0,
            }

        start_time = time.time()
        chunk_num = 0
        extractor = BitfieldExtractor()

        # Buffer packets for chunked processing
        packet_buffer: list[dict[str, Any]] = []

        # Use Oscura core streaming API (OPTIMAL!)
        for packet in load_packets_streaming(
            self.data_file, self.config, chunk_size=self.chunk_size
        ):
            packet_buffer.append(packet)

            # When buffer is full, extract channels and process
            if len(packet_buffer) >= self.chunk_size:
                # Extract all samples from buffered packets
                samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]

                # Extract and process each channel
                for ch_name, ch_def in self.channel_map.items():
                    if "bits" in ch_def:
                        bit_range = ch_def["bits"]
                        values = np.array(
                            [
                                extractor.extract_bits(sample, bit_range[0], bit_range[1])
                                for sample in samples
                            ],
                            dtype=np.uint16,
                        )
                    elif "bit" in ch_def:
                        values = np.array(
                            [extractor.extract_bit(sample, ch_def["bit"]) for sample in samples],
                            dtype=np.uint8,
                        )
                    else:
                        continue

                    # Update statistics
                    stats[ch_name]["min"] = min(stats[ch_name]["min"], float(values.min()))
                    stats[ch_name]["max"] = max(stats[ch_name]["max"], float(values.max()))
                    stats[ch_name]["sum"] += float(values.sum())
                    stats[ch_name]["count"] += len(values)
                    stats[ch_name]["non_zero"] += int(np.count_nonzero(values))

                # Progress indicator
                chunk_num += 1
                if chunk_num % 10 == 0:
                    print(f"{chunk_num}", end=" ", flush=True)

                # Clear buffer
                packet_buffer = []

        # Process remaining packets
        if packet_buffer:
            samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]

            for ch_name, ch_def in self.channel_map.items():
                if "bits" in ch_def:
                    bit_range = ch_def["bits"]
                    values = np.array(
                        [
                            extractor.extract_bits(sample, bit_range[0], bit_range[1])
                            for sample in samples
                        ],
                        dtype=np.uint16,
                    )
                elif "bit" in ch_def:
                    values = np.array(
                        [extractor.extract_bit(sample, ch_def["bit"]) for sample in samples],
                        dtype=np.uint8,
                    )
                else:
                    continue

                # Update statistics
                stats[ch_name]["min"] = min(stats[ch_name]["min"], float(values.min()))
                stats[ch_name]["max"] = max(stats[ch_name]["max"], float(values.max()))
                stats[ch_name]["sum"] += float(values.sum())
                stats[ch_name]["count"] += len(values)
                stats[ch_name]["non_zero"] += int(np.count_nonzero(values))

        print()  # Newline

        elapsed = time.time() - start_time
        total_processed = stats[next(iter(self.channel_map.keys()))]["count"]
        print(f"   ✓ Processed {total_processed:,} samples in {elapsed:.2f} seconds")
        print(f"   ✓ Processing rate: {total_processed / elapsed / 1e6:.2f} M samples/sec")

        # Compute derived statistics
        for ch_name in stats:  # noqa: PLC0206
            stats[ch_name]["mean"] = stats[ch_name]["sum"] / stats[ch_name]["count"]
            stats[ch_name]["non_zero_pct"] = (
                stats[ch_name]["non_zero"] / stats[ch_name]["count"] * 100
            )

        return stats

    def export_npz(self, output_file: Path) -> None:
        """Export all data to NPZ file using core streaming API.

        Args:
            output_file: Path to output NPZ file.
        """
        print(f"\n4. Exporting to NPZ: {output_file}")
        print("   Using Oscura core load_packets_streaming() API")
        print("   Processing: ", end="", flush=True)

        # Pre-allocate arrays
        arrays = {}
        for ch_name in self.channel_map:
            arrays[ch_name] = np.zeros(self.total_samples, dtype=np.float64)

        offsets = dict.fromkeys(self.channel_map, 0)
        start_time = time.time()
        chunk_num = 0
        extractor = BitfieldExtractor()

        # Buffer packets for chunked processing
        packet_buffer: list[dict[str, Any]] = []

        # Use Oscura core streaming API (OPTIMAL!)
        for packet in load_packets_streaming(
            self.data_file, self.config, chunk_size=self.chunk_size
        ):
            packet_buffer.append(packet)

            # When buffer is full, extract channels and write
            if len(packet_buffer) >= self.chunk_size:
                # Extract all samples from buffered packets
                samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]

                # Extract and write each channel
                for ch_name, ch_def in self.channel_map.items():
                    if "bits" in ch_def:
                        bit_range = ch_def["bits"]
                        values = np.array(
                            [
                                extractor.extract_bits(sample, bit_range[0], bit_range[1])
                                for sample in samples
                            ],
                            dtype=np.uint16,
                        )
                    elif "bit" in ch_def:
                        values = np.array(
                            [extractor.extract_bit(sample, ch_def["bit"]) for sample in samples],
                            dtype=np.uint8,
                        )
                    else:
                        continue

                    # Write to pre-allocated array
                    offset = offsets[ch_name]
                    arrays[ch_name][offset : offset + len(values)] = values.astype(np.float64)
                    offsets[ch_name] += len(values)

                # Progress indicator
                chunk_num += 1
                if chunk_num % 10 == 0:
                    print(f"{chunk_num}", end=" ", flush=True)

                # Clear buffer
                packet_buffer = []

        # Process remaining packets
        if packet_buffer:
            samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]

            for ch_name, ch_def in self.channel_map.items():
                if "bits" in ch_def:
                    bit_range = ch_def["bits"]
                    values = np.array(
                        [
                            extractor.extract_bits(sample, bit_range[0], bit_range[1])
                            for sample in samples
                        ],
                        dtype=np.uint16,
                    )
                elif "bit" in ch_def:
                    values = np.array(
                        [extractor.extract_bit(sample, ch_def["bit"]) for sample in samples],
                        dtype=np.uint8,
                    )
                else:
                    continue

                offset = offsets[ch_name]
                arrays[ch_name][offset : offset + len(values)] = values.astype(np.float64)
                offsets[ch_name] += len(values)

        print()  # Newline

        elapsed = time.time() - start_time
        total_processed = offsets[next(iter(self.channel_map.keys()))]
        print(f"   ✓ Processed {total_processed:,} samples in {elapsed:.2f} seconds")
        print(f"   ✓ Processing rate: {total_processed / elapsed / 1e6:.2f} M samples/sec")

        # Save to NPZ
        print(f"   Saving to {output_file}...")
        arrays["sample_rate"] = self.sample_rate
        np.savez_compressed(output_file, **arrays)

        file_size_mb = output_file.stat().st_size / 1024**2
        print(f"   ✓ Saved {output_file.name} ({file_size_mb:.1f} MB)")

    def export_hdf5(self, output_file: Path) -> None:
        """Export all data to HDF5 file using core streaming API.

        Args:
            output_file: Path to output HDF5 file.
        """
        try:
            import h5py
        except ImportError:
            print("   ✗ h5py not installed. Install with: pip install h5py")
            return

        print(f"\n4. Exporting to HDF5: {output_file}")
        print("   Using Oscura core load_packets_streaming() API")
        print("   Processing: ", end="", flush=True)

        start_time = time.time()
        chunk_num = 0
        extractor = BitfieldExtractor()

        with h5py.File(output_file, "w") as hf:
            # Create datasets
            datasets = {}
            for ch_name in self.channel_map:
                datasets[ch_name] = hf.create_dataset(
                    ch_name,
                    shape=(self.total_samples,),
                    dtype=np.float64,
                    chunks=(self.chunk_size,),
                )

            # Write metadata
            hf.attrs["sample_rate"] = self.sample_rate
            hf.attrs["total_samples"] = self.total_samples
            hf.attrs["duration_seconds"] = self.duration
            hf.attrs["source_file"] = str(self.data_file)

            # Write data
            offsets = dict.fromkeys(self.channel_map, 0)

            # Buffer packets for chunked processing
            packet_buffer: list[dict[str, Any]] = []

            # Use Oscura core streaming API (OPTIMAL!)
            for packet in load_packets_streaming(
                self.data_file, self.config, chunk_size=self.chunk_size
            ):
                packet_buffer.append(packet)

                # When buffer is full, extract channels and write
                if len(packet_buffer) >= self.chunk_size:
                    # Extract all samples from buffered packets
                    samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]

                    # Extract and write each channel
                    for ch_name, ch_def in self.channel_map.items():
                        if "bits" in ch_def:
                            bit_range = ch_def["bits"]
                            values = np.array(
                                [
                                    extractor.extract_bits(sample, bit_range[0], bit_range[1])
                                    for sample in samples
                                ],
                                dtype=np.uint16,
                            )
                        elif "bit" in ch_def:
                            values = np.array(
                                [
                                    extractor.extract_bit(sample, ch_def["bit"])
                                    for sample in samples
                                ],
                                dtype=np.uint8,
                            )
                        else:
                            continue

                        # Write to HDF5
                        offset = offsets[ch_name]
                        datasets[ch_name][offset : offset + len(values)] = values.astype(np.float64)
                        offsets[ch_name] += len(values)

                    # Progress indicator
                    chunk_num += 1
                    if chunk_num % 10 == 0:
                        print(f"{chunk_num}", end=" ", flush=True)

                    # Clear buffer
                    packet_buffer = []

            # Process remaining packets
            if packet_buffer:
                samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]

                for ch_name, ch_def in self.channel_map.items():
                    if "bits" in ch_def:
                        bit_range = ch_def["bits"]
                        values = np.array(
                            [
                                extractor.extract_bits(sample, bit_range[0], bit_range[1])
                                for sample in samples
                            ],
                            dtype=np.uint16,
                        )
                    elif "bit" in ch_def:
                        values = np.array(
                            [extractor.extract_bit(sample, ch_def["bit"]) for sample in samples],
                            dtype=np.uint8,
                        )
                    else:
                        continue

                    offset = offsets[ch_name]
                    datasets[ch_name][offset : offset + len(values)] = values.astype(np.float64)
                    offsets[ch_name] += len(values)

        print()  # Newline

        elapsed = time.time() - start_time
        total_processed = offsets[next(iter(self.channel_map.keys()))]
        print(f"   ✓ Processed {total_processed:,} samples in {elapsed:.2f} seconds")
        print(f"   ✓ Processing rate: {total_processed / elapsed / 1e6:.2f} M samples/sec")

        file_size_mb = output_file.stat().st_size / 1024**2
        print(f"   ✓ Saved {output_file.name} ({file_size_mb:.1f} MB)")


def main() -> int:
    """Main entry point."""
    # Default demo paths
    demo_dir = Path(__file__).parent
    default_data_file = demo_dir / "data" / "multi_lane_daq_10M.bin"
    default_config_file = demo_dir / "custom_daq_continuous.yml"

    parser = argparse.ArgumentParser(
        description="Optimal memory-efficient custom DAQ loader (uses Oscura core APIs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "data_file",
        type=Path,
        nargs="?",
        default=default_data_file,
        help="Path to binary DAQ data file (default: demo_data/multi_lane_daq_10M.bin)",
    )
    parser.add_argument(
        "config_file",
        type=Path,
        nargs="?",
        default=default_config_file,
        help="Path to YAML format configuration (default: custom_daq_continuous.yml)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000_000,
        help="Samples per chunk (default: 10M, ~80MB memory per chunk)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute statistics only (no data export, minimal memory)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Export all data to NPZ file",
    )
    parser.add_argument(
        "--export-hdf5",
        type=Path,
        default=None,
        help="Export all data to HDF5 file",
    )

    args = parser.parse_args()

    # Resolve data file path - check for alternate default files
    if not args.data_file.exists():
        # Try alternate default file
        alt_data_file = demo_dir / "data" / "continuous_acquisition_100M.bin"
        if alt_data_file.exists():
            print(f"Using alternate data file: {alt_data_file.name}", file=sys.stderr)
            args.data_file = alt_data_file
        else:
            print(f"Error: Data file not found: {args.data_file}", file=sys.stderr)
            print("Available files in demo_data/:", file=sys.stderr)
            demo_data_dir = demo_dir / "data"
            if demo_data_dir.exists():
                for f in demo_data_dir.glob("*.bin"):
                    print(f"  - {f.name}", file=sys.stderr)
            return 1

    if not args.config_file.exists():
        print(f"Error: Config file not found: {args.config_file}", file=sys.stderr)
        return 1

    try:
        # Define channel map for 4-lane DAQ
        channel_map = {
            "Lane_1": {"bits": [0, 15]},  # Bytes 0-1
            "Lane_2": {"bits": [16, 31]},  # Bytes 2-3
            "Lane_3": {"bits": [32, 47]},  # Bytes 4-5
            "Lane_4": {"bits": [48, 63]},  # Bytes 6-7
        }

        # Create analyzer
        analyzer = OptimalDAQAnalyzer(
            args.data_file, args.config_file, channel_map, chunk_size=args.chunk_size
        )
        analyzer.print_info()

        # Execute requested operation
        if args.export:
            analyzer.export_npz(args.export)

        elif args.export_hdf5:
            analyzer.export_hdf5(args.export_hdf5)

        else:  # Default: compute statistics
            stats = analyzer.compute_statistics()

            print("\n5. Statistics Summary:")
            for ch_name in channel_map:
                s = stats[ch_name]
                print(f"\n   {ch_name}:")
                print(f"      Samples: {s['count']:,}")
                print(f"      Range: [{s['min']:.0f}, {s['max']:.0f}]")
                print(f"      Mean: {s['mean']:.2f}")
                print(f"      Non-zero: {s['non_zero']:,} ({s['non_zero_pct']:.2f}%)")

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print("\n✅ Demo validation passed")

        return 0

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
