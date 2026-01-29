"""Oscura command-line interface.

This module provides the main entry point for Oscura CLI operations,
including sample data download.


Example:
    python -m oscura download_samples
    python -m oscura --help
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any


def get_samples_dir() -> Path:
    """Get the samples directory path (~/.oscura/samples/).

    Returns:
        Path to the samples directory.
    """
    return Path.home() / ".oscura" / "samples"


def get_sample_files() -> dict[str, dict[str, Any]]:
    """Get the list of sample files to download.

    Returns:
        Dictionary mapping filename to file metadata.
    """
    # Sample files configuration
    # In production, these would be hosted on a public CDN or GitHub releases
    return {
        "sine_1khz.csv": {
            "description": "1 kHz sine wave, 100 kS/s, CSV format",
            "format": "csv",
            "size": 1024 * 50,  # ~50 KB
            "checksum": None,  # Would be populated with actual checksum
            "url": "https://raw.githubusercontent.com/oscura/sample-data/main/sine_1khz.csv",
        },
        "square_wave.csv": {
            "description": "10 kHz square wave with ringing, CSV format",
            "format": "csv",
            "size": 1024 * 100,  # ~100 KB
            "checksum": None,
            "url": "https://raw.githubusercontent.com/oscura/sample-data/main/square_wave.csv",
        },
        "uart_9600.bin": {
            "description": "UART signal at 9600 baud, binary format",
            "format": "binary",
            "size": 1024 * 20,  # ~20 KB
            "checksum": None,
            "url": "https://raw.githubusercontent.com/oscura/sample-data/main/uart_9600.bin",
        },
        "i2c_capture.bin": {
            "description": "I2C bus capture with multiple devices",
            "format": "binary",
            "size": 1024 * 50,  # ~50 KB
            "checksum": None,
            "url": "https://raw.githubusercontent.com/oscura/sample-data/main/i2c_capture.bin",
        },
        "spi_flash.bin": {
            "description": "SPI flash read operation",
            "format": "binary",
            "size": 1024 * 30,  # ~30 KB
            "checksum": None,
            "url": "https://raw.githubusercontent.com/oscura/sample-data/main/spi_flash.bin",
        },
        "noisy_signal.csv": {
            "description": "Noisy analog signal for filtering examples",
            "format": "csv",
            "size": 1024 * 80,  # ~80 KB
            "checksum": None,
            "url": "https://raw.githubusercontent.com/oscura/sample-data/main/noisy_signal.csv",
        },
        "eye_diagram.npz": {
            "description": "High-speed serial data for eye diagram",
            "format": "npz",
            "size": 1024 * 200,  # ~200 KB
            "checksum": None,
            "url": "https://raw.githubusercontent.com/oscura/sample-data/main/eye_diagram.npz",
        },
    }


def download_file(url: str, dest: Path, checksum: str | None = None) -> bool:
    """Download a file from URL to destination.

    Args:
        url: URL to download from.
        dest: Destination file path.
        checksum: Optional SHA256 checksum to verify.

    Returns:
        True if download successful, False otherwise.
    """
    try:
        import ssl
        import urllib.request

        # Create SSL context that works in most environments
        context = ssl.create_default_context()

        print(f"  Downloading: {url}")

        with urllib.request.urlopen(url, context=context, timeout=30) as response:
            data = response.read()

        # Verify checksum if provided
        if checksum:
            computed = hashlib.sha256(data).hexdigest()
            if computed != checksum:
                print(f"  ERROR: Checksum mismatch for {dest.name}")
                print(f"    Expected: {checksum}")
                print(f"    Got: {computed}")
                return False

        # Write to destination
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

        print(f"  Saved: {dest}")
        return True

    except Exception as e:
        print(f"  ERROR: Failed to download {url}: {e}")
        return False


def generate_sample_file(filename: str, dest: Path) -> bool:
    """Generate a sample file locally when download is not available.

    This is a fallback when the remote repository is not available.

    Args:
        filename: Name of the sample file to generate.
        dest: Destination file path.

    Returns:
        True if generation successful, False otherwise.
    """
    try:
        import numpy as np

        dest.parent.mkdir(parents=True, exist_ok=True)

        if filename == "sine_1khz.csv":
            # Generate 1 kHz sine wave at 100 kS/s, 1000 samples
            sample_rate = 100_000
            duration = 0.01  # 10 ms
            t = np.arange(0, duration, 1 / sample_rate)
            signal = 0.5 * np.sin(2 * np.pi * 1000 * t)
            np.savetxt(
                dest,
                np.column_stack([t, signal]),
                delimiter=",",
                header="time,voltage",
                comments="",
            )
            return True

        elif filename == "square_wave.csv":
            # Generate 10 kHz square wave with some ringing
            sample_rate = 1_000_000
            duration = 0.001  # 1 ms
            t = np.arange(0, duration, 1 / sample_rate)
            signal = 0.5 * np.sign(np.sin(2 * np.pi * 10000 * t))
            # Add some ringing/noise
            signal += 0.05 * np.random.randn(len(signal))
            np.savetxt(
                dest,
                np.column_stack([t, signal]),
                delimiter=",",
                header="time,voltage",
                comments="",
            )
            return True

        elif filename == "noisy_signal.csv":
            # Generate noisy sine wave
            sample_rate = 10_000
            duration = 0.1  # 100 ms
            t = np.arange(0, duration, 1 / sample_rate)
            signal = 0.5 * np.sin(2 * np.pi * 100 * t)
            signal += 0.1 * np.random.randn(len(signal))
            np.savetxt(
                dest,
                np.column_stack([t, signal]),
                delimiter=",",
                header="time,voltage",
                comments="",
            )
            return True

        elif filename.endswith(".bin"):
            # Generate placeholder binary data
            data = np.random.randint(0, 256, 1000, dtype=np.uint8)
            data.tofile(dest)
            return True

        elif filename.endswith(".npz"):
            # Generate high-speed signal for eye diagram
            sample_rate = 10_000_000  # 10 MS/s
            samples_per_ui = 100
            num_ui = 100
            t = np.arange(samples_per_ui * num_ui) / sample_rate
            # Generate random bit pattern
            bits = np.random.randint(0, 2, num_ui)
            signal = np.repeat(bits.astype(float), samples_per_ui)
            # Add some jitter and noise
            signal += 0.1 * np.random.randn(len(signal))
            np.savez(dest, time=t, signal=signal, sample_rate=sample_rate)
            return True

        else:
            print(f"  WARNING: Unknown file type: {filename}")
            return False

    except Exception as e:
        print(f"  ERROR: Failed to generate {filename}: {e}")
        return False


def download_samples(force: bool = False, generate: bool = True) -> int:
    """Download sample waveform files for testing and tutorials.

    Args:
        force: Force re-download even if files exist.
        generate: Generate files locally if download fails.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    samples_dir = get_samples_dir()
    sample_files = get_sample_files()

    print("Oscura Sample Data Download")
    print("==============================")
    print(f"Destination: {samples_dir}")
    print()

    success_count = 0
    fail_count = 0

    for filename, info in sample_files.items():
        dest = samples_dir / filename

        if dest.exists() and not force:
            print(f"[SKIP] {filename} (already exists)")
            success_count += 1
            continue

        print(f"[DOWNLOAD] {filename}")
        print(f"  Description: {info['description']}")

        # Try to download first
        url = info.get("url")
        checksum = info.get("checksum")

        if url and download_file(url, dest, checksum):
            success_count += 1
            continue

        # Fall back to local generation
        if generate:
            print("  Falling back to local generation...")
            if generate_sample_file(filename, dest):
                print(f"  Generated: {dest}")
                success_count += 1
                continue

        fail_count += 1
        print(f"  FAILED: {filename}")

    print()
    print(f"Summary: {success_count} succeeded, {fail_count} failed")

    if fail_count > 0:
        print()
        print("Some downloads failed. Sample files are optional and used for")
        print("tutorials and testing. You can proceed without them.")
        return 1

    print()
    print("Sample files downloaded successfully!")
    print()
    print("Example usage:")
    print("  >>> import oscura as osc")
    print(f"  >>> trace = osc.load('{samples_dir / 'sine_1khz.csv'}')")
    print("  >>> trace.plot()")

    return 0


def list_samples() -> int:
    """List available sample files.

    Returns:
        Exit code.
    """
    samples_dir = get_samples_dir()
    sample_files = get_sample_files()

    print("Available sample files:")
    print()

    for filename, info in sample_files.items():
        dest = samples_dir / filename
        status = "[EXISTS]" if dest.exists() else "[NOT DOWNLOADED]"
        print(f"  {status} {filename}")
        print(f"           {info['description']}")
        print()

    return 0


def main() -> int:
    """Main entry point for Oscura CLI.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="oscura",
        description="Oscura signal analysis toolkit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # download_samples command
    download_parser = subparsers.add_parser(
        "download_samples",
        aliases=["download"],
        help="Download sample waveform files",
    )
    download_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if files exist",
    )
    download_parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Do not generate files locally if download fails",
    )

    # list_samples command
    subparsers.add_parser(
        "list_samples",
        aliases=["list"],
        help="List available sample files",
    )

    # version command
    subparsers.add_parser(
        "version",
        help="Show version information",
    )

    args = parser.parse_args()

    if args.command in ("download_samples", "download"):
        return download_samples(
            force=args.force,
            generate=not args.no_generate,
        )

    elif args.command in ("list_samples", "list"):
        return list_samples()

    elif args.command == "version":
        try:
            from oscura import __version__

            print(f"Oscura version {__version__}")
        except ImportError:
            print("Oscura version unknown")
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
