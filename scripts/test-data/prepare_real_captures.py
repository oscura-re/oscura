#!/usr/bin/env python3
"""Prepare real capture data for integration testing and demos.

This script processes the minimal_testset directory and creates an optimized
subset suitable for version control while preserving test coverage.

For the large UDP binary file (2.9GB), it extracts representative subsets:
- Head: First N packets (startup/initialization)
- Middle: Middle N packets (steady-state behavior)
- Tail: Last N packets (shutdown/completion)

Usage:
    python scripts/prepare_real_captures.py /path/to/minimal_testset
    python scripts/prepare_real_captures.py --analyze-only /path/to/udp_capture.bin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Target directory for real captures
REAL_CAPTURES_DIR = "test_data/real_captures"

# UDP extraction parameters
UDP_PACKETS_PER_SEGMENT = 2000  # Packets per segment (head/middle/tail)
ESTIMATED_PACKET_SIZE = 1500  # Typical MTU-sized UDP packets


@dataclass
class FileMetadata:
    """Metadata for a captured file."""

    filename: str
    size_bytes: int
    md5_hash: str
    category: str
    subcategory: str = ""
    description: str = ""
    capture_info: dict | None = None


@dataclass
class UDPAnalysis:
    """Analysis results for UDP binary file."""

    file_size: int
    estimated_packet_count: int
    packet_size_detected: int | None
    has_consistent_framing: bool
    sample_header: bytes | None
    recommended_extraction: dict


def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def analyze_udp_binary(filepath: Path, sample_size: int = 1024 * 1024) -> UDPAnalysis:
    """Analyze UDP binary file to determine optimal extraction strategy.

    Args:
        filepath: Path to the UDP binary file.
        sample_size: Bytes to read for analysis.

    Returns:
        UDPAnalysis with file characteristics and recommendations.
    """
    file_size = filepath.stat().st_size

    # Read sample from beginning
    with open(filepath, "rb") as f:
        sample = f.read(min(sample_size, file_size))

    # Try to detect packet framing
    # Common patterns: length-prefixed, fixed-size, delimiter-based
    packet_size_detected = None
    has_consistent_framing = False
    sample_header = None

    # Check for length-prefixed packets (first 2 or 4 bytes as length)
    if len(sample) >= 4:
        # Try 2-byte big-endian length
        potential_len_2be = struct.unpack(">H", sample[0:2])[0]
        # Try 2-byte little-endian length
        potential_len_2le = struct.unpack("<H", sample[0:2])[0]
        # Try 4-byte big-endian length
        potential_len_4be = struct.unpack(">I", sample[0:4])[0]
        # Try 4-byte little-endian length
        potential_len_4le = struct.unpack("<I", sample[0:4])[0]

        # Check which interpretation makes sense
        for potential_len, prefix_size in [
            (potential_len_2be, 2),
            (potential_len_2le, 2),
            (potential_len_4be, 4),
            (potential_len_4le, 4),
        ]:
            if 64 <= potential_len <= 65535:  # Reasonable packet size
                # Verify by checking next packet
                next_offset = prefix_size + potential_len
                if next_offset < len(sample) - prefix_size:
                    next_potential_len = struct.unpack(
                        ">H" if prefix_size == 2 else ">I",
                        sample[next_offset : next_offset + prefix_size],
                    )[0]
                    if 64 <= next_potential_len <= 65535:
                        packet_size_detected = potential_len + prefix_size
                        has_consistent_framing = True
                        sample_header = sample[: min(64, potential_len + prefix_size)]
                        break

    # If no length prefix found, try fixed-size detection
    if not has_consistent_framing:
        # Look for repeating patterns at common MTU sizes
        for test_size in [1024, 1460, 1472, 1500, 2048]:
            if file_size % test_size == 0 or file_size % test_size < 100:
                # Check if pattern repeats
                if len(sample) >= test_size * 3:
                    # Look for similar headers
                    h1 = sample[0:16]
                    h2 = sample[test_size : test_size + 16]
                    h3 = sample[test_size * 2 : test_size * 2 + 16]
                    # Check for structural similarity (not exact match)
                    if h1[:4] == h2[:4] == h3[:4]:
                        packet_size_detected = test_size
                        has_consistent_framing = True
                        sample_header = sample[:64]
                        break

    # Calculate recommendations
    if packet_size_detected:
        estimated_packets = file_size // packet_size_detected
        packets_per_segment = min(UDP_PACKETS_PER_SEGMENT, estimated_packets // 3)
    else:
        # Fallback: assume ~1500 byte packets
        estimated_packets = file_size // ESTIMATED_PACKET_SIZE
        packets_per_segment = UDP_PACKETS_PER_SEGMENT
        packet_size_detected = ESTIMATED_PACKET_SIZE

    bytes_per_segment = packets_per_segment * (packet_size_detected or ESTIMATED_PACKET_SIZE)

    return UDPAnalysis(
        file_size=file_size,
        estimated_packet_count=estimated_packets,
        packet_size_detected=packet_size_detected,
        has_consistent_framing=has_consistent_framing,
        sample_header=sample_header,
        recommended_extraction={
            "packets_per_segment": packets_per_segment,
            "bytes_per_segment": bytes_per_segment,
            "total_extracted_bytes": bytes_per_segment * 3,
            "segments": ["head", "middle", "tail"],
        },
    )


def extract_udp_segments(
    source_path: Path,
    output_dir: Path,
    analysis: UDPAnalysis,
) -> list[FileMetadata]:
    """Extract head, middle, and tail segments from UDP binary.

    Args:
        source_path: Path to the source UDP binary file.
        output_dir: Directory to write extracted segments.
        analysis: UDPAnalysis results from analyze_udp_binary.

    Returns:
        List of FileMetadata for extracted files.
    """
    # Create udp subdirectory
    udp_dir = output_dir / "udp"
    udp_dir.mkdir(parents=True, exist_ok=True)
    metadata_list = []

    file_size = analysis.file_size
    bytes_per_segment = analysis.recommended_extraction["bytes_per_segment"]

    segments = {
        "head": 0,  # Start of file
        "middle": (file_size // 2) - (bytes_per_segment // 2),  # Center of file
        "tail": file_size - bytes_per_segment,  # End of file
    }

    with open(source_path, "rb") as src:
        for segment_name, offset in segments.items():
            output_path = udp_dir / f"udp_{segment_name}.bin"

            src.seek(offset)
            data = src.read(bytes_per_segment)

            with open(output_path, "wb") as dst:
                dst.write(data)

            metadata_list.append(
                FileMetadata(
                    filename=output_path.name,
                    size_bytes=len(data),
                    md5_hash=compute_md5(output_path),
                    category="packets",
                    subcategory="udp",
                    description=f"UDP packet segment ({segment_name}): {analysis.recommended_extraction['packets_per_segment']} packets",
                    capture_info={
                        "source_file": source_path.name,
                        "source_size": file_size,
                        "segment": segment_name,
                        "offset": offset,
                        "packet_count_estimate": analysis.recommended_extraction[
                            "packets_per_segment"
                        ],
                        "extraction_date": datetime.now().isoformat(),
                    },
                )
            )

            print(f"  Extracted {segment_name}: {output_path} ({len(data):,} bytes)")

    return metadata_list


def copy_waveform_files(
    source_dir: Path,
    output_dir: Path,
    size_categories: dict[str, tuple[int, int]],
) -> list[FileMetadata]:
    """Copy waveform files organized by size category.

    Args:
        source_dir: Source directory containing waveform files.
        output_dir: Output directory for organized files.
        size_categories: Dict mapping category name to (min_size, max_size) in bytes.

    Returns:
        List of FileMetadata for copied files.
    """
    metadata_list = []
    wfm_dir = source_dir / "waveform"

    if not wfm_dir.exists():
        print(f"  Warning: {wfm_dir} not found")
        return metadata_list

    for wfm_file in wfm_dir.glob("*.wfm"):
        file_size = wfm_file.stat().st_size

        # Determine category
        category = "other"
        for cat_name, (min_size, max_size) in size_categories.items():
            if min_size <= file_size <= max_size:
                category = cat_name
                break

        # Create category directory and copy
        cat_dir = output_dir / "waveforms" / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        dest_path = cat_dir / wfm_file.name
        shutil.copy2(wfm_file, dest_path)

        metadata_list.append(
            FileMetadata(
                filename=dest_path.name,
                size_bytes=file_size,
                md5_hash=compute_md5(dest_path),
                category="waveforms",
                subcategory=category,
                description=f"Tektronix WFM file ({category} size tier)",
                capture_info={
                    "format": "WFM#003",
                    "source": "Tektronix oscilloscope",
                },
            )
        )

        print(f"  Copied {wfm_file.name} -> {category}/ ({file_size:,} bytes)")

    return metadata_list


def copy_session_files(source_dir: Path, output_dir: Path) -> list[FileMetadata]:
    """Copy session files preserving channel configuration info."""
    metadata_list = []
    session_dir = source_dir / "session"

    if not session_dir.exists():
        print(f"  Warning: {session_dir} not found")
        return metadata_list

    dest_dir = output_dir / "sessions"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for tss_file in session_dir.glob("*.tss"):
        dest_path = dest_dir / tss_file.name
        shutil.copy2(tss_file, dest_path)

        file_size = tss_file.stat().st_size
        metadata_list.append(
            FileMetadata(
                filename=dest_path.name,
                size_bytes=file_size,
                md5_hash=compute_md5(dest_path),
                category="sessions",
                description="Tektronix session file (compressed ZIP)",
            )
        )

        print(f"  Copied {tss_file.name} ({file_size:,} bytes)")

    return metadata_list


def copy_settings_files(source_dir: Path, output_dir: Path) -> list[FileMetadata]:
    """Copy settings files."""
    metadata_list = []
    settings_dir = source_dir / "settings"

    if not settings_dir.exists():
        print(f"  Warning: {settings_dir} not found")
        return metadata_list

    dest_dir = output_dir / "settings"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for set_file in settings_dir.glob("*.set"):
        dest_path = dest_dir / set_file.name
        shutil.copy2(set_file, dest_path)

        file_size = set_file.stat().st_size
        metadata_list.append(
            FileMetadata(
                filename=dest_path.name,
                size_bytes=file_size,
                md5_hash=compute_md5(dest_path),
                category="settings",
                description="Tektronix settings file (SCPI commands)",
            )
        )

        print(f"  Copied {set_file.name} ({file_size:,} bytes)")

    return metadata_list


def write_manifest(output_dir: Path, metadata_list: list[FileMetadata]) -> None:
    """Write manifest.json with all file metadata."""
    manifest = {
        "version": "1.0",
        "generated": datetime.now().isoformat(),
        "description": "Real capture data for Oscura integration testing and demos",
        "total_files": len(metadata_list),
        "total_size_bytes": sum(m.size_bytes for m in metadata_list),
        "categories": {},
        "files": [asdict(m) for m in metadata_list],
    }

    # Group by category
    for m in metadata_list:
        if m.category not in manifest["categories"]:
            manifest["categories"][m.category] = {"count": 0, "size_bytes": 0}
        manifest["categories"][m.category]["count"] += 1
        manifest["categories"][m.category]["size_bytes"] += m.size_bytes

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {manifest_path}")
    print(f"Total: {len(metadata_list)} files, {manifest['total_size_bytes']:,} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare real capture data for integration testing and demos"
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to minimal_testset directory or UDP binary file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(REAL_CAPTURES_DIR),
        help=f"Output directory (default: {REAL_CAPTURES_DIR})",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze UDP file, don't extract",
    )
    parser.add_argument(
        "--udp-packets-per-segment",
        type=int,
        default=UDP_PACKETS_PER_SEGMENT,
        help=f"Packets per segment for UDP extraction (default: {UDP_PACKETS_PER_SEGMENT})",
    )

    args = parser.parse_args()
    source = args.source
    output = args.output

    # Handle analyze-only mode for UDP file
    if args.analyze_only:
        if not source.is_file():
            print(f"Error: {source} is not a file")
            return 1

        print(f"Analyzing UDP binary: {source}")
        analysis = analyze_udp_binary(source)

        print("\nFile Analysis Results:")
        print(
            f"  File size: {analysis.file_size:,} bytes ({analysis.file_size / (1024**3):.2f} GB)"
        )
        print(f"  Estimated packet count: {analysis.estimated_packet_count:,}")
        print(f"  Detected packet size: {analysis.packet_size_detected}")
        print(f"  Consistent framing: {analysis.has_consistent_framing}")
        print("\nRecommended Extraction:")
        print(f"  Packets per segment: {analysis.recommended_extraction['packets_per_segment']:,}")
        print(f"  Bytes per segment: {analysis.recommended_extraction['bytes_per_segment']:,}")
        print(
            f"  Total extracted: {analysis.recommended_extraction['total_extracted_bytes']:,} bytes"
        )

        if analysis.sample_header:
            print("\nSample header (first 64 bytes):")
            print(f"  {analysis.sample_header[:64].hex()}")

        return 0

    # Full processing mode
    if not source.is_dir():
        print(f"Error: {source} is not a directory")
        return 1

    print(f"Preparing real captures from: {source}")
    print(f"Output directory: {output}")

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    all_metadata: list[FileMetadata] = []

    # 1. Process waveform files by size
    print("\n[1/5] Processing waveform files...")
    size_categories = {
        "small": (0, 1_500_000),  # 0 - 1.5 MB
        "medium": (1_500_001, 6_000_000),  # 1.5 - 6 MB
        "large": (6_000_001, 50_000_000),  # 6 - 50 MB
    }
    all_metadata.extend(copy_waveform_files(source, output, size_categories))

    # 2. Process session files
    print("\n[2/5] Processing session files...")
    all_metadata.extend(copy_session_files(source, output))

    # 3. Process settings files
    print("\n[3/5] Processing settings files...")
    all_metadata.extend(copy_settings_files(source, output))

    # 4. Process UDP binary file
    print("\n[4/5] Processing UDP packet capture...")
    udp_dir = source / "udp_packet"
    if udp_dir.exists():
        for bin_file in udp_dir.glob("*.bin"):
            print(f"  Analyzing {bin_file.name}...")
            analysis = analyze_udp_binary(bin_file)

            print(f"    File size: {analysis.file_size:,} bytes")
            print(f"    Estimated packets: {analysis.estimated_packet_count:,}")
            print(
                f"    Extracting {analysis.recommended_extraction['packets_per_segment'] * 3:,} packets total..."
            )

            packets_dir = output / "packets"
            all_metadata.extend(extract_udp_segments(bin_file, packets_dir, analysis))
    else:
        print("  No UDP packet directory found")

    # 5. Write manifest
    print("\n[5/5] Writing manifest...")
    write_manifest(output, all_metadata)

    print("\nDone! Real captures prepared successfully.")
    return 0


if __name__ == "__main__":
    exit(main())
