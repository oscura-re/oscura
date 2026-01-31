"""Custom Binary Format Loading

Demonstrates creating custom binary loaders for proprietary formats:
- Binary loader API usage with oscura.loaders.binary
- Custom format parsing with struct module
- Header extraction and validation
- Endianness handling (big-endian vs little-endian)
- Multi-channel binary data extraction

IEEE Standards: IEEE 1057-2017 (Digitizing Waveform Recorders)
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    ValidationSuite,
    format_size,
    generate_sine_wave,
    generate_square_wave,
)

from oscura.loaders.binary import load_binary


class CustomBinaryDemo(BaseDemo):
    """Demonstrate custom binary format loading and parsing."""

    def __init__(self) -> None:
        """Initialize custom binary demonstration."""
        super().__init__(
            name="custom_binary",
            description="Load and parse custom binary data formats",
            capabilities=[
                "oscura.loaders.binary.load_binary",
                "Custom binary header parsing",
                "Endianness handling",
                "Multi-channel extraction",
            ],
            ieee_standards=["IEEE 1057-2017"],
            related_demos=[
                "01_oscilloscopes.py",
                "04_scientific_formats.py",
            ],
        )
        self.temp_dir = Path(tempfile.mkdtemp(prefix="oscura_binary_"))

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic custom binary formats."""
        self.info("Creating synthetic binary format files...")

        # Simple raw binary
        simple_file = self._create_simple_binary()
        self.info("  ✓ Simple raw binary (float64, 1 channel)")

        # Custom header with endianness variants
        custom_le = self._create_custom_header_binary(little_endian=True)
        self.info("  ✓ Custom header format (little-endian)")

        custom_be = self._create_custom_header_binary(little_endian=False)
        self.info("  ✓ Custom header format (big-endian)")

        # Interleaved multi-channel
        interleaved = self._create_interleaved_binary()
        self.info("  ✓ Interleaved multi-channel (4 channels)")

        return {
            "simple": simple_file,
            "custom_le": custom_le,
            "custom_be": custom_be,
            "interleaved": interleaved,
        }

    def _create_simple_binary(self) -> dict[str, Any]:
        """Create simple raw binary file (no header)."""
        sample_rate = 1e6
        duration = 0.001

        signal = generate_sine_wave(
            frequency=10e3, amplitude=1.0, duration=duration, sample_rate=sample_rate
        )

        filepath = self.temp_dir / "simple.bin"
        signal.data.astype(np.float64).tofile(filepath)

        return {
            "filepath": filepath,
            "sample_rate": sample_rate,
            "dtype": "float64",
            "channels": 1,
            "num_samples": len(signal.data),
        }

    def _create_custom_header_binary(self, little_endian: bool = True) -> dict[str, Any]:
        """Create binary file with custom 40-byte header."""
        sample_rate = 5e6
        duration = 0.0005
        channels = 2

        ch1 = generate_sine_wave(100e3, 1.5, duration, sample_rate)
        ch2 = generate_square_wave(50e3, 1.0, duration, sample_rate)

        samples_per_channel = len(ch1.data)

        # Build header: Magic(4) Ver(2) Ch(2) Rate(8) Samples(4) Type(2) Reserved(18)
        endian = "<" if little_endian else ">"
        header = struct.pack(
            f"{endian}I H H d I H 18x",
            0x4F534355,  # 'OSCU'
            1,  # Version
            channels,
            sample_rate,
            samples_per_channel,
            3,  # float64
        )

        # Interleave channels
        data = np.empty(samples_per_channel * channels, dtype=np.float64)
        data[0::2] = ch1.data
        data[1::2] = ch2.data

        suffix = "le" if little_endian else "be"
        filepath = self.temp_dir / f"custom_{suffix}.bin"
        with open(filepath, "wb") as f:
            f.write(header)
            data.astype(f"{endian}f8").tofile(f)

        return {
            "filepath": filepath,
            "sample_rate": sample_rate,
            "channels": channels,
            "samples_per_channel": samples_per_channel,
            "little_endian": little_endian,
            "header_size": 40,
        }

    def _create_interleaved_binary(self) -> dict[str, Any]:
        """Create interleaved multi-channel binary file (int16)."""
        sample_rate = 100e3
        duration = 0.01
        channels = 4
        frequencies = [1e3, 2e3, 5e3, 10e3]

        channel_data = []
        for freq in frequencies:
            signal = generate_sine_wave(freq, 1.0, duration, sample_rate)
            scaled = (signal.data * 32767 * 0.8).astype(np.int16)
            channel_data.append(scaled)

        samples_per_channel = len(channel_data[0])

        # Interleave
        interleaved = np.empty(samples_per_channel * channels, dtype=np.int16)
        for i in range(channels):
            interleaved[i::channels] = channel_data[i]

        filepath = self.temp_dir / "interleaved.bin"
        interleaved.tofile(filepath)

        return {
            "filepath": filepath,
            "sample_rate": sample_rate,
            "dtype": "int16",
            "channels": channels,
            "samples_per_channel": samples_per_channel,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the custom binary format demonstration."""
        results = {}

        self.subsection("Custom Binary Format Loading")
        self.info("Binary files are common in embedded systems and custom DAQ hardware.")
        self.info("Challenges: No standard metadata, variable data types, endianness")
        self.info("")

        # Simple binary
        self.subsection("1. Simple Raw Binary (No Header)")
        results["simple"] = self._load_simple_binary(data["simple"])

        # Custom headers
        self.subsection("2. Custom Header Format (Little-Endian)")
        results["custom_le"] = self._load_custom_header(data["custom_le"])

        self.subsection("3. Endianness Handling (Big-Endian)")
        results["custom_be"] = self._load_custom_header(data["custom_be"])

        # Multi-channel
        self.subsection("4. Interleaved Multi-Channel Data")
        results["interleaved"] = self._load_interleaved_binary(data["interleaved"])

        # Best practices
        self.subsection("Custom Binary Loader Best Practices")
        self._show_best_practices()

        return results

    def _load_simple_binary(self, file_info: dict[str, Any]) -> dict[str, Any]:
        """Load and analyze simple raw binary file."""
        filepath = file_info["filepath"]
        trace = load_binary(
            filepath, dtype=file_info["dtype"], sample_rate=file_info["sample_rate"]
        )

        self.result("File Size", format_size(filepath.stat().st_size))
        self.result("Samples Loaded", len(trace.data))
        self.result("RMS Value", f"{np.sqrt(np.mean(trace.data**2)):.4f}", "V")

        return {
            "num_samples": len(trace.data),
            "rms": float(np.sqrt(np.mean(trace.data**2))),
        }

    def _load_custom_header(self, file_info: dict[str, Any]) -> dict[str, Any]:
        """Load and parse binary file with custom header."""
        filepath = file_info["filepath"]
        little_endian = file_info["little_endian"]

        # Parse header
        header_info = self._parse_custom_header(filepath, little_endian)

        self.result("Magic Number", f"0x{header_info['magic']:08X}")
        self.result("Channels", header_info["channels"])
        self.result("Sample Rate", f"{header_info['sample_rate'] / 1e6:.1f}", "MHz")

        return {
            "header": header_info,
            "num_channels": header_info["channels"],
        }

    def _parse_custom_header(self, filepath: Path, little_endian: bool) -> dict[str, Any]:
        """Parse custom binary header format."""
        endian = "<" if little_endian else ">"
        with open(filepath, "rb") as f:
            header_bytes = f.read(40)

        unpacked = struct.unpack(f"{endian}I H H d I H 18x", header_bytes)
        data_type_map = {1: "int16", 2: "float32", 3: "float64"}

        return {
            "magic": unpacked[0],
            "version": unpacked[1],
            "channels": unpacked[2],
            "sample_rate": unpacked[3],
            "samples_per_channel": unpacked[4],
            "data_type": data_type_map.get(unpacked[5], "unknown"),
        }

    def _load_interleaved_binary(self, file_info: dict[str, Any]) -> dict[str, Any]:
        """Load and analyze interleaved multi-channel data."""
        filepath = file_info["filepath"]
        channels = file_info["channels"]

        self.result("Channels", channels)
        self.result("File Size", format_size(filepath.stat().st_size))

        # Load first channel as example
        trace = load_binary(
            filepath,
            dtype=file_info["dtype"],
            sample_rate=file_info["sample_rate"],
            channels=channels,
            channel=0,
        )

        self.result("Samples per Channel", len(trace.data))

        return {
            "num_channels": channels,
            "samples_per_channel": len(trace.data),
        }

    def _show_best_practices(self) -> None:
        """Show best practices for custom binary loaders."""
        self.info("""
Custom Binary Loader Best Practices:

1. HEADER DESIGN
   - Include magic number for format identification
   - Store version number for format evolution
   - Document sample rate, channels, data type
   - Use padding to align header to power-of-2

2. ENDIANNESS
   - Document byte order explicitly
   - Use little-endian for PC/ARM compatibility
   - Use big-endian for network protocols
   - Test on both architectures

3. MULTI-CHANNEL DATA
   - Interleaved: [ch1[0], ch2[0], ch1[1], ch2[1], ...]
   - Planar: [ch1[0..N], ch2[0..N], ...]
   - Document channel order clearly
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate custom binary loading results."""
        suite = ValidationSuite()

        # Validate simple binary
        if "simple" in results:
            suite.check_approximately(
                results["simple"]["rms"], 0.707, tolerance=0.1, name="Simple binary RMS"
            )

        # Validate custom headers
        if "custom_le" in results:
            suite.check_equal(
                results["custom_le"]["header"]["magic"], 0x4F534355, "Custom LE magic number"
            )

        if "custom_be" in results:
            suite.check_equal(results["custom_be"]["num_channels"], 2, "Custom BE channels")

        # Validate interleaved
        if "interleaved" in results:
            suite.check_equal(results["interleaved"]["num_channels"], 4, "Interleaved channels")

        if suite.all_passed():
            self.success("All binary format validations passed!")
        else:
            self.error("Some validations failed")

        return suite.all_passed()


if __name__ == "__main__":
    demo = CustomBinaryDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
