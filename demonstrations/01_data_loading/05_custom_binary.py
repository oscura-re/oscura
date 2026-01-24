"""Custom Binary Format Loading

Demonstrates creating custom binary loaders for proprietary formats:
- Binary loader API usage
- Custom format parsing with struct
- Header extraction and validation
- Endianness handling
- Multi-channel binary data extraction

IEEE Standards: IEEE 1057-2017 (Digitizing Waveform Recorders)
Related Demos:
- 01_data_loading/01_oscilloscopes.py
- 01_data_loading/04_scientific_formats.py
- 08_extensibility/01_plugin_basics.py

This demonstration shows:
1. How to use the binary loader API for raw data files
2. How to parse custom binary headers with struct
3. How to handle different endianness (big-endian vs little-endian)
4. How to extract multi-channel interleaved data
5. How to validate custom binary formats and extract metadata
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    format_size,
    format_table,
    generate_sine_wave,
    generate_square_wave,
    validate_approximately,
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
                "Binary format validation",
            ],
            ieee_standards=["IEEE 1057-2017"],
            related_demos=[
                "01_data_loading/01_oscilloscopes.py",
                "01_data_loading/04_scientific_formats.py",
                "08_extensibility/01_plugin_basics.py",
            ],
        )
        self.temp_dir = Path(tempfile.mkdtemp(prefix="oscura_binary_"))

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic custom binary formats."""
        self.info("Creating synthetic binary format files...")

        # Format 1: Simple raw binary (no header)
        simple_file = self._create_simple_binary()
        self.info("  ✓ Simple raw binary (float64, 1 channel)")

        # Format 2: Custom header + data (little-endian)
        custom_le_file = self._create_custom_header_binary(little_endian=True)
        self.info("  ✓ Custom header format (little-endian, 2 channels)")

        # Format 3: Custom header + data (big-endian)
        custom_be_file = self._create_custom_header_binary(little_endian=False)
        self.info("  ✓ Custom header format (big-endian, 2 channels)")

        # Format 4: Interleaved multi-channel
        interleaved_file = self._create_interleaved_binary()
        self.info("  ✓ Interleaved multi-channel (4 channels, int16)")

        return {
            "simple": simple_file,
            "custom_le": custom_le_file,
            "custom_be": custom_be_file,
            "interleaved": interleaved_file,
        }

    def _create_simple_binary(self) -> dict[str, Any]:
        """Create simple raw binary file (no header)."""
        sample_rate = 1e6  # 1 MHz
        duration = 0.001  # 1 ms

        # Generate sine wave
        signal = generate_sine_wave(
            frequency=10e3, amplitude=1.0, duration=duration, sample_rate=sample_rate
        )

        # Write to file
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
        """Create binary file with custom header format.

        Header format (40 bytes):
        - Magic number (4 bytes): 0x4F534355 ('OSCU')
        - Version (2 bytes): uint16
        - Channels (2 bytes): uint16
        - Sample rate (8 bytes): double
        - Samples per channel (4 bytes): uint32
        - Data type (2 bytes): uint16 (1=int16, 2=float32, 3=float64)
        - Reserved (18 bytes): padding
        """
        sample_rate = 5e6  # 5 MHz
        duration = 0.0005  # 500 μs
        channels = 2

        # Generate two channels
        ch1 = generate_sine_wave(
            frequency=100e3, amplitude=1.5, duration=duration, sample_rate=sample_rate
        )
        ch2 = generate_square_wave(
            frequency=50e3, amplitude=1.0, duration=duration, sample_rate=sample_rate
        )

        samples_per_channel = len(ch1.data)

        # Build header
        endian_char = "<" if little_endian else ">"
        header_format = f"{endian_char}I H H d I H 18x"
        header = struct.pack(
            header_format,
            0x4F534355,  # Magic: 'OSCU'
            1,  # Version
            channels,  # Number of channels
            sample_rate,  # Sample rate
            samples_per_channel,  # Samples per channel
            3,  # Data type: float64
        )

        # Interleave channel data
        data = np.empty(samples_per_channel * channels, dtype=np.float64)
        data[0::2] = ch1.data
        data[1::2] = ch2.data

        # Write to file
        endian_suffix = "le" if little_endian else "be"
        filepath = self.temp_dir / f"custom_{endian_suffix}.bin"
        with open(filepath, "wb") as f:
            f.write(header)
            data.astype(f"{endian_char}f8").tofile(f)

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
        sample_rate = 100e3  # 100 kHz
        duration = 0.01  # 10 ms
        channels = 4

        # Generate 4 channels with different frequencies
        frequencies = [1e3, 2e3, 5e3, 10e3]
        channel_data = []

        for freq in frequencies:
            signal = generate_sine_wave(
                frequency=freq, amplitude=1.0, duration=duration, sample_rate=sample_rate
            )
            # Scale to int16 range
            scaled = (signal.data * 32767 * 0.8).astype(np.int16)
            channel_data.append(scaled)

        samples_per_channel = len(channel_data[0])

        # Interleave: [ch1[0], ch2[0], ch3[0], ch4[0], ch1[1], ...]
        interleaved = np.empty(samples_per_channel * channels, dtype=np.int16)
        for i in range(channels):
            interleaved[i::channels] = channel_data[i]

        # Write to file
        filepath = self.temp_dir / "interleaved.bin"
        interleaved.tofile(filepath)

        return {
            "filepath": filepath,
            "sample_rate": sample_rate,
            "dtype": "int16",
            "channels": channels,
            "samples_per_channel": samples_per_channel,
            "frequencies": frequencies,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the custom binary format demonstration."""
        results = {}

        self.section("Custom Binary Format Loading")
        self.info("Binary files are common in embedded systems and custom DAQ hardware.")
        self.info("Key challenges:")
        self.info("  • No standard metadata (sample rate, channels, units)")
        self.info("  • Variable data types (int8, int16, float32, float64)")
        self.info("  • Endianness differences (little vs big-endian)")
        self.info("  • Custom headers with proprietary structures")
        self.info("")

        # Simple binary loading
        self.section("1. Simple Raw Binary (No Header)")
        results["simple"] = self._load_simple_binary(data["simple"])

        # Custom header parsing
        self.section("2. Custom Header Format (Little-Endian)")
        results["custom_le"] = self._load_custom_header(data["custom_le"])

        # Big-endian handling
        self.section("3. Endianness Handling (Big-Endian)")
        results["custom_be"] = self._load_custom_header(data["custom_be"])

        # Multi-channel interleaved
        self.section("4. Interleaved Multi-Channel Data")
        results["interleaved"] = self._load_interleaved_binary(data["interleaved"])

        # Comparison
        self.section("Binary Format Comparison")
        self._display_format_comparison(data)

        # Best practices
        self.section("Custom Binary Loader Best Practices")
        self._show_best_practices()

        return results

    def _load_simple_binary(self, file_info: dict[str, Any]) -> dict[str, Any]:
        """Load and analyze simple raw binary file."""
        self.subsection("Loading Raw Binary Data")

        filepath = file_info["filepath"]
        sample_rate = file_info["sample_rate"]
        dtype = file_info["dtype"]

        self.result("File Path", str(filepath.name))
        self.result("File Size", format_size(filepath.stat().st_size))
        self.result("Data Type", dtype)
        self.result("Sample Rate", f"{sample_rate / 1e6:.1f}", "MHz")

        # Load using binary loader
        trace = load_binary(filepath, dtype=dtype, sample_rate=sample_rate)

        self.result("Samples Loaded", len(trace.data))
        self.result("Duration", f"{len(trace.data) / sample_rate * 1e3:.2f}", "ms")
        self.result("Min Value", f"{np.min(trace.data):.4f}", "V")
        self.result("Max Value", f"{np.max(trace.data):.4f}", "V")
        self.result("RMS Value", f"{np.sqrt(np.mean(trace.data**2)):.4f}", "V")
        self.info("")

        return {
            "num_samples": len(trace.data),
            "sample_rate": sample_rate,
            "rms": float(np.sqrt(np.mean(trace.data**2))),
        }

    def _load_custom_header(self, file_info: dict[str, Any]) -> dict[str, Any]:
        """Load and parse binary file with custom header."""
        filepath = file_info["filepath"]
        little_endian = file_info["little_endian"]
        endian_name = "Little-Endian" if little_endian else "Big-Endian"

        self.subsection(f"Parsing Custom Header ({endian_name})")

        # Read and parse header
        header_info = self._parse_custom_header(filepath, little_endian)

        self.result("Magic Number", f"0x{header_info['magic']:08X}")
        self.result("Format Version", header_info["version"])
        self.result("Channels", header_info["channels"])
        self.result("Sample Rate", f"{header_info['sample_rate'] / 1e6:.1f}", "MHz")
        self.result("Samples/Channel", header_info["samples_per_channel"])
        self.result("Data Type", header_info["data_type_name"])
        self.info("")

        # Load channel data
        self.subsection("Extracting Channels")
        channels = []
        for ch in range(header_info["channels"]):
            # Load channel from interleaved data
            endian_char = "<" if little_endian else ">"
            trace = load_binary(
                filepath,
                dtype=f"{endian_char}f8",
                sample_rate=header_info["sample_rate"],
                channels=header_info["channels"],
                channel=ch,
                offset=file_info["header_size"] // 8,  # Skip header (in float64 units)
            )
            channels.append(trace)

            rms = float(np.sqrt(np.mean(trace.data**2)))
            peak = float(np.max(np.abs(trace.data)))
            self.result(f"  Channel {ch + 1}", f"RMS={rms:.3f} V, Peak={peak:.3f} V")

        self.info("")

        return {
            "header": header_info,
            "num_channels": len(channels),
            "endianness": endian_name,
        }

    def _parse_custom_header(self, filepath: Path, little_endian: bool) -> dict[str, Any]:
        """Parse custom binary header format."""
        endian_char = "<" if little_endian else ">"
        header_format = f"{endian_char}I H H d I H 18x"

        with open(filepath, "rb") as f:
            header_bytes = f.read(40)

        unpacked = struct.unpack(header_format, header_bytes)

        data_type_map = {1: "int16", 2: "float32", 3: "float64"}

        return {
            "magic": unpacked[0],
            "version": unpacked[1],
            "channels": unpacked[2],
            "sample_rate": unpacked[3],
            "samples_per_channel": unpacked[4],
            "data_type": unpacked[5],
            "data_type_name": data_type_map.get(unpacked[5], "unknown"),
        }

    def _load_interleaved_binary(self, file_info: dict[str, Any]) -> dict[str, Any]:
        """Load and analyze interleaved multi-channel data."""
        self.subsection("Loading Interleaved Channels")

        filepath = file_info["filepath"]
        sample_rate = file_info["sample_rate"]
        channels = file_info["channels"]
        dtype = file_info["dtype"]

        self.result("File Path", str(filepath.name))
        self.result("File Size", format_size(filepath.stat().st_size))
        self.result("Number of Channels", channels)
        self.result("Data Type", dtype)
        self.result("Sample Rate", f"{sample_rate / 1e3:.1f}", "kHz")
        self.info("")

        # Load each channel
        channel_info = []
        for ch in range(channels):
            trace = load_binary(
                filepath, dtype=dtype, sample_rate=sample_rate, channels=channels, channel=ch
            )

            # Convert int16 to voltage
            if dtype == "int16":
                trace.data = trace.data / 32767.0

            # Analyze
            rms = float(np.sqrt(np.mean(trace.data**2)))
            peak = float(np.max(np.abs(trace.data)))

            # Estimate frequency
            fft = np.fft.rfft(trace.data)
            freqs = np.fft.rfftfreq(len(trace.data), 1 / sample_rate)
            peak_idx = np.argmax(np.abs(fft[1:])) + 1
            peak_freq = freqs[peak_idx]

            channel_info.append(
                [
                    f"Channel {ch + 1}",
                    f"{len(trace.data)}",
                    f"{rms:.3f}",
                    f"{peak:.3f}",
                    f"{peak_freq / 1e3:.2f} kHz",
                ]
            )

        headers = ["Channel", "Samples", "RMS", "Peak", "Frequency"]
        self.info(format_table(channel_info, headers))
        self.info("")

        return {
            "num_channels": channels,
            "sample_rate": sample_rate,
            "samples_per_channel": file_info["samples_per_channel"],
        }

    def _display_format_comparison(self, data: dict[str, Any]) -> None:
        """Display comparison of binary formats."""
        comparison = []

        # Simple binary
        simple = data["simple"]
        comparison.append(
            [
                "Raw Binary",
                "None",
                f"{simple['num_samples']}",
                format_size(simple["filepath"].stat().st_size),
                "Simplest, no metadata",
            ]
        )

        # Custom little-endian
        custom_le = data["custom_le"]
        comparison.append(
            [
                "Custom (LE)",
                "40 bytes",
                f"{custom_le['samples_per_channel'] * custom_le['channels']}",
                format_size(custom_le["filepath"].stat().st_size),
                "Metadata-rich, multi-channel",
            ]
        )

        # Custom big-endian
        custom_be = data["custom_be"]
        comparison.append(
            [
                "Custom (BE)",
                "40 bytes",
                f"{custom_be['samples_per_channel'] * custom_be['channels']}",
                format_size(custom_be["filepath"].stat().st_size),
                "Cross-platform compatible",
            ]
        )

        # Interleaved
        interleaved = data["interleaved"]
        comparison.append(
            [
                "Interleaved",
                "None",
                f"{interleaved['samples_per_channel'] * interleaved['channels']}",
                format_size(interleaved["filepath"].stat().st_size),
                "Efficient multi-channel",
            ]
        )

        headers = ["Format", "Header Size", "Total Samples", "File Size", "Notes"]
        self.info(format_table(comparison, headers))
        self.info("")

    def _show_best_practices(self) -> None:
        """Show best practices for custom binary loaders."""
        self.info("""
When creating custom binary loaders, follow these guidelines:

1. HEADER DESIGN
   ✓ Include magic number for format identification (4 bytes)
   ✓ Store version number for format evolution (2 bytes)
   ✓ Document sample rate, channels, data type explicitly
   ✓ Use padding to align header to power-of-2 boundary (32, 64, 128 bytes)
   ✓ Include CRC/checksum for data integrity validation

2. ENDIANNESS
   ✓ Document byte order explicitly in specification
   ✓ Use little-endian for PC/ARM compatibility (most common)
   ✓ Use big-endian for network protocols (IEEE 802.3)
   ✓ Test on both architectures during development
   ✓ Use struct module with '<' or '>' prefix

3. DATA TYPES
   ✓ Document bit depth and signed/unsigned format
   ✓ Use standard sizes: int8, int16, int32, float32, float64
   ✓ Include scaling factors (ADC_SCALE, VOLTAGE_SCALE)
   ✓ Consider compression for large datasets (zlib, lz4)

4. MULTI-CHANNEL DATA
   ✓ Interleaved: [ch1[0], ch2[0], ch1[1], ch2[1], ...] (better cache locality)
   ✓ Planar: [ch1[0..N], ch2[0..N], ...] (simpler indexing)
   ✓ Document channel order and interleaving pattern
   ✓ Align channels to cache line boundaries for performance

5. VALIDATION
   ✓ Verify magic number matches expected value
   ✓ Check version compatibility (fail gracefully on mismatch)
   ✓ Validate sample count matches file size
   ✓ Verify CRC/checksum if present
   ✓ Range-check metadata values (sample rate, channels)

EXAMPLE LOADER IMPLEMENTATION:

```python
from oscura.loaders.binary import load_binary
import struct

def load_custom_format(filepath):
    # Parse header
    with open(filepath, 'rb') as f:
        header = f.read(40)

    magic, version, channels, rate, samples, dtype = \\
        struct.unpack('<I H H d I H 18x', header)

    # Validate
    assert magic == 0x4F534355, "Invalid magic number"
    assert version == 1, f"Unsupported version {version}"

    # Load data
    trace = load_binary(
        filepath,
        dtype='<f8',
        sample_rate=rate,
        channels=channels,
        channel=0,
        offset=40 // 8  # Skip header
    )

    return trace
```
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate custom binary loading results."""
        self.info("Validating binary format loading...")

        all_valid = True

        # Validate simple binary
        if "simple" in results:
            simple = results["simple"]
            if not validate_approximately(
                simple["sample_rate"],
                1e6,
                tolerance=0.01,
                name="Simple binary sample rate",
            ):
                all_valid = False

            # Sine wave RMS should be ~0.707 (1.0 / sqrt(2))
            if not validate_approximately(
                simple["rms"], 0.707, tolerance=0.1, name="Simple binary RMS"
            ):
                all_valid = False

        # Validate custom header parsing
        if "custom_le" in results:
            custom = results["custom_le"]
            if custom["num_channels"] != 2:
                self.error(f"Custom LE channel count: expected 2, got {custom['num_channels']}")
                all_valid = False

            if custom["header"]["magic"] != 0x4F534355:
                self.error(f"Custom LE magic mismatch: {custom['header']['magic']:08X}")
                all_valid = False

        # Validate big-endian parsing
        if "custom_be" in results:
            custom = results["custom_be"]
            if custom["num_channels"] != 2:
                self.error(f"Custom BE channel count: expected 2, got {custom['num_channels']}")
                all_valid = False

        # Validate interleaved loading
        if "interleaved" in results:
            interleaved = results["interleaved"]
            if interleaved["num_channels"] != 4:
                self.error(
                    f"Interleaved channel count: expected 4, got {interleaved['num_channels']}"
                )
                all_valid = False

            if not validate_approximately(
                interleaved["sample_rate"],
                100e3,
                tolerance=0.01,
                name="Interleaved sample rate",
            ):
                all_valid = False

        if all_valid:
            self.success("All binary format validations passed!")
            self.info("""
Next steps for custom binary formats:

1. SIMPLE RAW BINARY
   from oscura.loaders.binary import load_binary
   trace = load_binary("data.bin", dtype="float64", sample_rate=1e6)

2. SKIP HEADER BYTES
   # Skip 128-byte header
   trace = load_binary("data.bin", dtype="int16", sample_rate=1e6,
                      offset=128 // 2)  # offset in int16 units

3. MULTI-CHANNEL INTERLEAVED
   # Load channel 2 from 4-channel interleaved file
   trace = load_binary("data.bin", dtype="float32", sample_rate=1e6,
                      channels=4, channel=1)

4. CREATE CUSTOM LOADER
   # See demonstration code for full example
   # Extend BaseLoader for plugin integration
            """)
        else:
            self.error("Some binary format validations failed!")

        return all_valid


if __name__ == "__main__":
    demo = CustomBinaryDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
