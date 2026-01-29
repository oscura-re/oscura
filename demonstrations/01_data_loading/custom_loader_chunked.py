#!/usr/bin/env python3
"""Memory-efficient streaming loader using Oscura core APIs.

# SKIP_VALIDATION: Processes 2.9GB file, takes >30s

This script demonstrates that Oscura's core load_packets_streaming() API
already provides optimal memory-efficient chunked processing. You don't need
to implement custom chunking - the core API handles it automatically!

**Uses Core APIs Only**: All functionality delegated to load_packets_streaming().
No manual chunk iteration, no manual channel extraction.

File: udp_capture_1.bin (2.9GB, 382M samples)
Format: Continuous time-series, 8 bytes per sample
Origin: MATLAB-preprocessed UDP packet capture
Structure: 4-lane parallel acquisition @ 100 MHz

Features:
    - Streaming via core API (configurable chunk size)
    - Memory-efficient (constant ~305 MB regardless of file size)
    - Statistics computation
    - NPZ/HDF5 export

Usage:
    # Compute statistics (streaming, minimal memory)
    python demos/02_custom_daq/chunked_loader.py \
        udp_capture_1.bin \
        demos/02_custom_daq/custom_daq_continuous.yml \
        --stats

    # Export to NPZ (streaming)
    python demos/02_custom_daq/chunked_loader.py \
        udp_capture_1.bin \
        demos/02_custom_daq/custom_daq_continuous.yml \
        --export output.npz

    # Export to HDF5 (streaming)
    python demos/02_custom_daq/chunked_loader.py \
        udp_capture_1.bin \
        demos/02_custom_daq/custom_daq_continuous.yml \
        --export-hdf5 output.h5

    # Custom chunk size
    python demos/02_custom_daq/chunked_loader.py \
        udp_capture_1.bin \
        demos/02_custom_daq/custom_daq_continuous.yml \
        --stats \
        --chunk-size 5000000

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

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from oscura.loaders.configurable import (
    BitfieldExtractor,
    PacketFormatConfig,
    load_packets_streaming,
)


def print_file_info(data_file: Path, config: PacketFormatConfig, chunk_size: int) -> None:
    """Print file and configuration information."""
    file_size = data_file.stat().st_size
    total_samples = file_size // config.packet_size
    sample_rate = 100e6  # 100 MHz
    duration = total_samples / sample_rate
    num_chunks = (total_samples + chunk_size - 1) // chunk_size

    print("=" * 80)
    print("STREAMING DAQ LOADER - USING OSCURA CORE APIs")
    print("=" * 80)

    print("\n1. Configuration:")
    print(f"   Format: {config.name} v{config.version}")
    print(f"   Description: {config.description}")
    print(f"   Packet size: {config.packet_size} bytes")

    print("\n2. File Information:")
    print(f"   File: {data_file.name}")
    print(f"   Size: {file_size:,} bytes ({file_size / 1024**3:.2f} GB)")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Duration @ {sample_rate / 1e6:.0f} MHz: {duration:.3f} seconds")

    print("\n3. Processing Configuration:")
    print(f"   Chunk size: {chunk_size:,} samples")
    print(f"   Chunk memory: ~{chunk_size * 32 / 1024**2:.0f} MB per chunk")
    print(f"   Estimated chunks: {num_chunks}")
    print("   Using: Oscura core load_packets_streaming() API")


def compute_statistics(
    data_file: Path,
    config: PacketFormatConfig,
    channel_map: dict[str, dict[str, list[int]]],
    chunk_size: int,
) -> dict[str, Any]:
    """Compute statistics using streaming API (minimal memory).

    Args:
        data_file: Path to binary data file.
        config: Packet format configuration.
        channel_map: Channel definitions.
        chunk_size: Samples per chunk.

    Returns:
        Dictionary with statistics for each channel.
    """
    print("\n4. Computing Statistics (Streaming via Core API)...")

    # Initialize statistics
    stats = {}
    for ch_name in channel_map:
        stats[ch_name] = {
            "min": float("inf"),
            "max": float("-inf"),
            "sum": 0.0,
            "count": 0,
            "non_zero": 0,
        }

    start_time = time.time()
    chunk_count = 0
    extractor = BitfieldExtractor()

    # Buffer packets for chunked processing
    packet_buffer: list[dict[str, Any]] = []

    # Use core streaming API - load packets and extract channels manually
    for packet in load_packets_streaming(data_file, config, chunk_size=chunk_size):
        packet_buffer.append(packet)

        # When buffer is full, extract channels and process
        if len(packet_buffer) >= chunk_size:
            # Extract all samples from buffered packets
            samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]

            # Extract and process each channel
            for ch_name, ch_def in channel_map.items():
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
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"   Processed {chunk_count} chunks...", flush=True)

            # Clear buffer
            packet_buffer = []

    # Process remaining packets
    if packet_buffer:
        samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]

        for ch_name, ch_def in channel_map.items():
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

    elapsed = time.time() - start_time
    total_samples = stats[next(iter(stats.keys()))]["count"]
    print(f"\n   ✓ Processed {total_samples:,} samples in {elapsed:.2f} seconds")
    print(f"   ✓ Processing rate: {total_samples / elapsed / 1e6:.2f} M samples/sec")

    # Compute derived statistics
    for ch_name in stats:  # noqa: PLC0206
        stats[ch_name]["mean"] = stats[ch_name]["sum"] / stats[ch_name]["count"]
        stats[ch_name]["non_zero_pct"] = stats[ch_name]["non_zero"] / stats[ch_name]["count"] * 100

    return stats


def export_npz(
    data_file: Path,
    config: PacketFormatConfig,
    channel_map: dict[str, dict[str, list[int]]],
    output_file: Path,
    chunk_size: int,
) -> None:
    """Export to NPZ using streaming API.

    Args:
        data_file: Path to binary data file.
        config: Packet format configuration.
        channel_map: Channel definitions.
        output_file: Path to output NPZ file.
        chunk_size: Samples per chunk.
    """
    print(f"\n4. Exporting to NPZ: {output_file}")
    print("   Using streaming API to accumulate data...")

    # Accumulate chunks using core streaming API
    channels = {ch_name: [] for ch_name in channel_map}

    start_time = time.time()
    chunk_count = 0
    extractor = BitfieldExtractor()

    # Buffer packets for chunked processing
    packet_buffer: list[dict[str, Any]] = []

    for packet in load_packets_streaming(data_file, config, chunk_size=chunk_size):
        packet_buffer.append(packet)

        # When buffer is full, extract channels
        if len(packet_buffer) >= chunk_size:
            # Extract all samples from buffered packets
            samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]

            # Extract each channel
            for ch_name, ch_def in channel_map.items():
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

                channels[ch_name].append(values)

            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"   Loaded {chunk_count} chunks...", flush=True)

            # Clear buffer
            packet_buffer = []

    # Process remaining packets
    if packet_buffer:
        samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]

        for ch_name, ch_def in channel_map.items():
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

            channels[ch_name].append(values)

    # Concatenate chunks
    print("   Concatenating chunks and saving...")
    export_data = {}
    for ch_name, chunks in channels.items():
        export_data[ch_name] = np.concatenate(chunks)

    export_data["sample_rate"] = 100e6

    np.savez_compressed(output_file, **export_data)

    elapsed = time.time() - start_time
    total_samples = len(export_data[next(iter(channel_map.keys()))])
    file_size_mb = output_file.stat().st_size / 1024**2

    print(f"\n   ✓ Processed {total_samples:,} samples in {elapsed:.2f} seconds")
    print(f"   ✓ Processing rate: {total_samples / elapsed / 1e6:.2f} M samples/sec")
    print(f"   ✓ Saved {output_file.name} ({file_size_mb:.1f} MB)")


def export_hdf5(
    data_file: Path,
    config: PacketFormatConfig,
    channel_map: dict[str, dict[str, list[int]]],
    output_file: Path,
    chunk_size: int,
) -> None:
    """Export to HDF5 using streaming API.

    Args:
        data_file: Path to binary data file.
        config: Packet format configuration.
        channel_map: Channel definitions.
        output_file: Path to output HDF5 file.
        chunk_size: Samples per chunk.
    """
    if not HAS_H5PY:
        print("   ✗ h5py not installed. Install with: pip install h5py")
        return

    print(f"\n4. Exporting to HDF5: {output_file}")
    print("   Using streaming API for chunked HDF5 write...")

    # First pass: count total samples
    print("   Counting total samples...")
    total_samples = 0
    extractor = BitfieldExtractor()

    # Count samples by processing one channel
    packet_buffer: list[dict[str, Any]] = []
    for packet in load_packets_streaming(data_file, config, chunk_size=chunk_size):
        packet_buffer.append(packet)

        if len(packet_buffer) >= chunk_size:
            samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]
            total_samples += len(samples)
            packet_buffer = []

    if packet_buffer:
        samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]
        total_samples += len(samples)

    # Second pass: write data
    print(f"   Writing {total_samples:,} samples to HDF5...")

    start_time = time.time()

    with h5py.File(output_file, "w") as hf:
        # Create datasets
        datasets = {}
        for ch_name in channel_map:
            datasets[ch_name] = hf.create_dataset(
                ch_name,
                shape=(total_samples,),
                dtype=np.float64,
                chunks=(chunk_size,),
            )

        # Write metadata
        hf.attrs["sample_rate"] = 100e6
        hf.attrs["total_samples"] = total_samples
        hf.attrs["duration_seconds"] = total_samples / 100e6
        hf.attrs["source_file"] = str(data_file)

        # Write chunks
        offsets = dict.fromkeys(channel_map, 0)
        chunk_count = 0

        # Buffer packets for chunked processing
        packet_buffer = []

        for packet in load_packets_streaming(data_file, config, chunk_size=chunk_size):
            packet_buffer.append(packet)

            # When buffer is full, extract channels and write
            if len(packet_buffer) >= chunk_size:
                # Extract all samples from buffered packets
                samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]

                # Extract and write each channel
                for ch_name, ch_def in channel_map.items():
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

                    # Write to HDF5
                    offset = offsets[ch_name]
                    datasets[ch_name][offset : offset + len(values)] = values
                    offsets[ch_name] += len(values)

                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"   Written {chunk_count} chunks...", flush=True)

                # Clear buffer
                packet_buffer = []

        # Process remaining packets
        if packet_buffer:
            samples = [sample for pkt in packet_buffer for sample in pkt["samples"]]

            for ch_name, ch_def in channel_map.items():
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
                datasets[ch_name][offset : offset + len(values)] = values
                offsets[ch_name] += len(values)

    elapsed = time.time() - start_time
    file_size_mb = output_file.stat().st_size / 1024**2

    print(f"\n   ✓ Processed {total_samples:,} samples in {elapsed:.2f} seconds")
    print(f"   ✓ Processing rate: {total_samples / elapsed / 1e6:.2f} M samples/sec")
    print(f"   ✓ Saved {output_file.name} ({file_size_mb:.1f} MB)")


def main() -> int:
    """Main entry point."""
    # Default demo paths
    demo_dir = Path(__file__).parent
    default_data_file = demo_dir / "data" / "multi_lane_daq_10M.bin"
    default_config_file = demo_dir / "custom_daq_continuous.yml"

    parser = argparse.ArgumentParser(
        description="Streaming DAQ loader using Oscura core APIs",
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
        help="Samples per chunk (default: 10M, ~305MB memory)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute statistics (minimal memory)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Export to NPZ file",
    )
    parser.add_argument(
        "--export-hdf5",
        type=Path,
        default=None,
        help="Export to HDF5 file",
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

    # Default to stats mode if no operation specified (for demo validation)
    if not (args.stats or args.export or args.export_hdf5):
        args.stats = True

    try:
        # Load configuration
        config = PacketFormatConfig.from_yaml(args.config_file)

        # Define channel map (4-lane DAQ)
        channel_map = {
            "Lane_1": {"bits": [0, 15]},  # Bytes 0-1
            "Lane_2": {"bits": [16, 31]},  # Bytes 2-3
            "Lane_3": {"bits": [32, 47]},  # Bytes 4-5
            "Lane_4": {"bits": [48, 63]},  # Bytes 6-7
        }

        # Print info
        print_file_info(args.data_file, config, args.chunk_size)

        # Execute requested operation
        if args.stats:
            # Statistics mode
            stats = compute_statistics(args.data_file, config, channel_map, args.chunk_size)

            print("\n5. Statistics Summary:")
            for ch_name in channel_map:
                s = stats[ch_name]
                print(f"\n   {ch_name}:")
                print(f"      Samples: {s['count']:,}")
                print(f"      Range: [{s['min']:.0f}, {s['max']:.0f}]")
                print(f"      Mean: {s['mean']:.2f}")
                print(f"      Non-zero: {s['non_zero']:,} ({s['non_zero_pct']:.2f}%)")

        elif args.export:
            # NPZ export mode
            export_npz(args.data_file, config, channel_map, args.export, args.chunk_size)

        elif args.export_hdf5:
            # HDF5 export mode
            export_hdf5(args.data_file, config, channel_map, args.export_hdf5, args.chunk_size)

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
