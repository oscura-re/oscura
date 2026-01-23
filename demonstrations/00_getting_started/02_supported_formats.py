"""Supported Formats: Explore all 21+ file formats Oscura loads

Demonstrates:
- oscura.get_supported_formats() - List all supported formats
- Format categories (oscilloscopes, logic analyzers, automotive, scientific)
- Format detection and loader mapping
- Example usage patterns for each major category

IEEE Standards: N/A (informational reference)
Related Demos:
- 00_getting_started/00_hello_world.py
- 01_data_loading/basic_file_loading.py

This demonstration showcases the breadth of Oscura's data loading capabilities,
organizing 21+ file formats into logical categories and showing how to work with
each type. Perfect for understanding what data sources Oscura can ingest.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo


class SupportedFormatsDemo(BaseDemo):
    """Comprehensive showcase of all supported file formats."""

    def __init__(self) -> None:
        """Initialize supported formats demonstration."""
        super().__init__(
            name="supported_formats",
            description="Complete guide to all 21+ file formats Oscura supports",
            capabilities=[
                "oscura.get_supported_formats()",
                "oscura.load() with auto-detection",
                "Format categories and organization",
            ],
        )

    def generate_test_data(self) -> dict[str, object]:
        """No test data needed - this is an informational demo."""
        return {}

    def run_demonstration(self, data: dict[str, object]) -> dict[str, object]:
        """Run the supported formats demonstration."""
        import oscura

        self.section("Oscura Supported Formats Guide")
        self.info("Oscura supports 21+ file formats from test & measurement equipment")
        self.info("All formats are automatically detected - just use oscura.load(filename)")

        # Get all supported formats
        supported_formats = oscura.get_supported_formats()

        self.subsection("Quick Reference")
        self.result("Total supported formats", len(supported_formats))
        self.result("Auto-detection", "Enabled (no format parameter needed)")
        self.result("Multi-channel loading", "Supported for oscilloscope formats")
        self.result("Lazy loading", "Available for large files (>100MB)")

        # Display all formats
        self.subsection("All Supported File Extensions")
        format_list = sorted(supported_formats)
        formats_str = " | ".join(format_list)
        self.info(f"Formats: {formats_str}")

        # Categorize formats
        format_categories = self._categorize_formats(supported_formats)

        # Display by category
        self.subsection("Formats by Category")

        for category, formats in format_categories.items():
            self.subsubsection(category)
            for fmt in sorted(formats):
                info = self._get_format_info(fmt)
                multi_ch = "Yes" if info["multi_channel"] else "No"
                equipment_str = str(info["equipment"])
                type_str = str(info["type"])
                self.info(f"  {fmt:8} | {equipment_str:20} | {type_str:15} | {multi_ch}")

        # Show format details
        self.subsection("Format Details & Usage")

        # Oscilloscopes section
        self.subsubsection("Oscilloscope Formats (.wfm)")
        self.info("Standard format for digital and mixed-signal oscilloscopes")
        self.result("Format", ".wfm")
        self.result("Vendors", "Tektronix, Rigol")
        self.result("Auto-Detection", "Yes (magic bytes)")
        self.result("Analog Channels", "Yes")
        self.result("Digital Channels", "Yes (mixed-signal)")
        self.result("Sample Rate Range", "DC to 8 GHz")
        self.result("Resolution", "8-16 bits")

        self.info("")
        self.info("Usage Examples:")
        self.code("""# Load oscilloscope capture (auto-detects Tektronix vs Rigol)
trace = oscura.load("TEK00001.wfm")
print(f"Sample rate: {trace.metadata.sample_rate} Hz")

# Load all channels from multi-channel capture
channels = oscura.load_all_channels("multi_channel.wfm")
for name, trace in channels.items():
    print(f"{name}: {len(trace.data)} samples")

# Handle mixed-signal (analog + digital)
trace = oscura.load("mixed_signal.wfm")
if isinstance(trace, oscura.DigitalTrace):
    print("Digital trace loaded")
else:
    print("Analog trace loaded")""")

        # Logic analyzer section
        self.subsubsection("Logic Analyzer Formats")
        self.info("Digital signal capture formats")
        logic_formats: dict[str, str] = {
            ".sr": "sigrok (PulseView)",
            ".vcd": "VCD (Value Change Dump)",
        }

        for ext, desc in logic_formats.items():
            self.result(f"Format {ext}", desc)

        self.info("")
        self.info("Usage Examples:")
        self.code("""# Load sigrok (PulseView) capture
digital = oscura.load("capture.sr")
print(f"Digital trace: {len(digital.data)} samples")

# Load VCD (Verilog Value Change Dump)
vcd_trace = oscura.load("simulation.vcd")

# Load network packets (PCAP)
packets = oscura.load("network_capture.pcap")""")

        # Scientific/Generic formats
        self.subsubsection("Scientific & Generic Formats")
        self.info("Cross-platform, vendor-neutral data formats")
        sci_formats: dict[str, str] = {
            ".csv": "CSV (spreadsheet/text)",
            ".npz": "NumPy binary",
            ".h5 / .hdf5": "HDF5 (hierarchical)",
            ".wav": "Audio waveform",
            ".tdms": "TDM (National Instruments)",
        }

        for ext, desc in sci_formats.items():
            self.result(f"Format {ext}", desc)

        self.info("")
        self.info("Usage Examples:")
        self.code("""# Load CSV with automatic column detection
data = oscura.load("measurements.csv")
print(f"Loaded {len(data.data)} samples")

# Load NumPy archive
trace = oscura.load("signals.npz")

# Load HDF5 with multiple datasets
waveforms = oscura.load("experiment.h5")

# Load WAV audio file (scientific use)
audio = oscura.load("signal.wav")

# Load National Instruments TDMS
tdms = oscura.load("channels.tdms")""")

        # RF/Network section
        self.subsubsection("RF & Network Formats")
        self.info("Specialized formats for RF and network analysis")
        rf_formats: dict[str, str] = {
            ".s1p to .s8p": "Touchstone S-parameters",
            ".pcap / .pcapng": "Network packets (PCAP)",
        }

        for ext, desc in rf_formats.items():
            self.result(f"Format {ext}", desc)

        self.info("")
        self.info("Usage Examples:")
        self.code("""# Load Touchstone S-parameter files
s_params = oscura.load("filter.s2p")
print(f"Frequency points: {len(s_params.data)}")

# S1p through S8p supported
s1p = oscura.load("single_port.s1p")
s4p = oscura.load("4port_network.s4p")""")

        # Security/Embedded section
        self.subsubsection("Security & Embedded Formats")
        self.info("Specialized formats for security research and embedded systems")
        emb_formats: dict[str, str] = {
            "ChipWhisperer": ".npz (specialized)",
            "Custom Binary": "Configurable parsing",
        }

        for tool, ext in emb_formats.items():
            self.result(f"{tool}", ext)

        self.info("")
        self.info("Usage Examples:")
        self.code("""# Load ChipWhisperer power traces
power = oscura.load("power_capture.npz")

# Load custom binary with configurable format
custom = oscura.load("raw_data.bin", format="binary",
                     sample_rate=1e6, dtype="float32")""")

        # Feature showcase
        self.subsection("Advanced Loading Features")

        self.subsubsection("Format Auto-Detection")
        self.info("Simply pass a filename - no format parameter needed!")
        self.code("""# Auto-detect from extension
trace = oscura.load("data.wfm")        # Auto -> Tektronix or Rigol
trace = oscura.load("data.csv")        # Auto -> CSV
trace = oscura.load("data.pcap")       # Auto -> PCAP

# Explicit format override (optional)
trace = oscura.load("data.bin", format="tektronix")""")

        self.subsubsection("Multi-Channel Loading")
        self.info("Load all channels at once for efficiency")
        self.code("""# Oscilloscope with 4 analog + 8 digital channels
channels = oscura.load_all_channels("capture.wfm")

for channel_name, trace in channels.items():
    print(f"{channel_name}: {len(trace.data)} samples")
    # Output:
    # ch1: 10000 samples
    # ch2: 10000 samples
    # ch3: 10000 samples
    # ch4: 10000 samples
    # d1: 10000 samples  (digital channels)
    # d2: 10000 samples
    # ...""")

        self.subsubsection("Lazy Loading for Large Files")
        self.info("Load huge files (>100MB) without full memory consumption")
        self.code("""# Automatic warning for files >100MB
trace = oscura.load("100gb_capture.wfm")
# Warning: File is large. Consider using lazy=True

# Enable lazy loading
lazy_trace = oscura.load("100gb_capture.wfm", lazy=True)
# Memory efficient - loads data on demand

# Works with most formats
large_csv = oscura.load("massive_dataset.csv", lazy=True)""")

        self.subsubsection("Channel Selection")
        self.info("Load specific channels from multi-channel files")
        self.code("""# Load specific analog channel
ch2 = oscura.load("capture.wfm", channel="ch2")

# By index
ch3 = oscura.load("capture.wfm", channel=2)  # 0-indexed

# Works with load_all_channels for filtering
all_ch = oscura.load_all_channels("capture.wfm")
ch2_only = all_ch["ch2"]""")

        # Format matrix
        self.subsection("Format Feature Matrix")

        self.info("Legend: checkmark = supported, dash = not applicable")
        self.info("")
        self.info(
            "  Format                    | Multi-Ch | Analog | Digital | Lazy Load | Auto-Detect"
        )
        self.info("  " + "-" * 75)

        matrix_data: list[tuple[str, str, str, str, str, str]] = [
            ("Oscilloscope (.wfm)", "Y", "Y", "Y", "Y", "Y"),
            ("Logic Analyzer (.sr/.vcd)", "Y", "-", "Y", "Y", "Y"),
            ("Generic (.csv/.npz/.h5)", "Y", "Y", "-", "Y", "Y"),
            ("Network (.pcap)", "-", "-", "Y", "Y", "Y"),
            ("RF (.s1p-.s8p)", "Y", "Y", "-", "Y", "Y"),
        ]

        for fmt, multi_ch, analog, digital, lazy, auto_detect in matrix_data:
            self.info(
                f"  {fmt:26} | {multi_ch:8} | {analog:6} | {digital:7} | {lazy:9} | {auto_detect}"
            )

        # Next steps
        self.subsection("Next Steps")
        self.info("Now that you know what formats Oscura supports:")
        self.info("")
        self.info("1. Choose your demo based on your format:")
        self.info("   - Oscilloscopes: 01_data_loading/basic_file_loading.py")
        self.info("   - Logic Analyzer: 01_data_loading/vcd_loading.py")
        self.info("   - Custom formats: 08_extensibility/custom_loader.py")
        self.info("")
        self.info("2. Try loading your own files:")
        self.code("""import oscura
trace = oscura.load("your_file.wfm")
print(f"Loaded: {len(trace.data)} samples at {trace.metadata.sample_rate} Hz")""")
        self.info("")
        self.info("3. Combine with analysis:")
        self.code("""import oscura
trace = oscura.load("data.wfm")
amplitude = oscura.amplitude(trace)
frequency = oscura.frequency(trace)
print(f"Vpp: {amplitude}V, Freq: {frequency}Hz")""")

        self.success("Supported formats overview complete!")

        return {
            "total_formats": len(supported_formats),
            "categories": len(format_categories),
            "formats_by_category": format_categories,
        }

    def validate(self, results: dict[str, object]) -> bool:
        """Validate the results."""
        self.info("Validating format availability...")

        # Check we found formats
        total = int(results["total_formats"]) if isinstance(results["total_formats"], int) else 0
        if total < 15:
            self.error(f"Expected 15+ formats, found {total}")
            return False

        self.success(f"Found {total} supported formats")

        # Check categorization
        categories = int(results["categories"]) if isinstance(results["categories"], int) else 0
        if categories < 5:
            self.error("Expected 5+ format categories")
            return False

        self.success(f"Formats organized into {categories} categories")

        self.info("\nFormat coverage summary:")
        formats_by_cat = results.get("formats_by_category", {})
        if isinstance(formats_by_cat, dict):
            for category, formats in formats_by_cat.items():
                if isinstance(formats, list):
                    self.info(f"  • {category}: {len(formats)} formats")

        self.success("All format information verified!")

        return True

    def _categorize_formats(self, formats: list[str]) -> dict[str, list[str]]:
        """Categorize file formats by type."""
        categories: dict[str, list[str]] = {
            "Oscilloscopes": [".wfm"],
            "Logic Analyzers": [".sr", ".vcd"],
            "Scientific/Generic": [".csv", ".npz", ".h5", ".hdf5", ".wav", ".tdms"],
            "Network/Packet": [".pcap", ".pcapng"],
            "RF/S-Parameters": [".s1p", ".s2p", ".s3p", ".s4p", ".s5p", ".s6p", ".s7p", ".s8p"],
        }

        # Return only categories with formats that exist
        result: dict[str, list[str]] = {}
        for category, format_list in categories.items():
            supported_in_cat = [f for f in format_list if f in formats]
            if supported_in_cat:
                result[category] = supported_in_cat

        return result

    def _get_format_info(self, fmt: str) -> dict[str, str | bool]:
        """Get information about a specific format."""
        format_db: dict[str, dict[str, str | bool]] = {
            ".wfm": {
                "equipment": "Tektronix/Rigol",
                "type": "Oscilloscope",
                "multi_channel": True,
            },
            ".sr": {
                "equipment": "PulseView/sigrok",
                "type": "Logic Analyzer",
                "multi_channel": True,
            },
            ".vcd": {
                "equipment": "Generic/Simulation",
                "type": "Logic Analyzer",
                "multi_channel": True,
            },
            ".csv": {
                "equipment": "Generic/Spreadsheet",
                "type": "Generic Data",
                "multi_channel": True,
            },
            ".npz": {
                "equipment": "NumPy/Python",
                "type": "Scientific",
                "multi_channel": True,
            },
            ".h5": {
                "equipment": "HDF5/Scientific",
                "type": "Scientific",
                "multi_channel": True,
            },
            ".hdf5": {
                "equipment": "HDF5/Scientific",
                "type": "Scientific",
                "multi_channel": True,
            },
            ".wav": {
                "equipment": "Audio/Generic",
                "type": "Audio",
                "multi_channel": True,
            },
            ".tdms": {
                "equipment": "National Instruments",
                "type": "Scientific",
                "multi_channel": True,
            },
            ".pcap": {
                "equipment": "Network",
                "type": "Packet Capture",
                "multi_channel": False,
            },
            ".pcapng": {
                "equipment": "Network",
                "type": "Packet Capture",
                "multi_channel": False,
            },
            ".s1p": {
                "equipment": "RF/Touchstone",
                "type": "S-Parameters",
                "multi_channel": False,
            },
            ".s2p": {
                "equipment": "RF/Touchstone",
                "type": "S-Parameters",
                "multi_channel": False,
            },
            ".s3p": {
                "equipment": "RF/Touchstone",
                "type": "S-Parameters",
                "multi_channel": False,
            },
            ".s4p": {
                "equipment": "RF/Touchstone",
                "type": "S-Parameters",
                "multi_channel": False,
            },
            ".s5p": {
                "equipment": "RF/Touchstone",
                "type": "S-Parameters",
                "multi_channel": False,
            },
            ".s6p": {
                "equipment": "RF/Touchstone",
                "type": "S-Parameters",
                "multi_channel": False,
            },
            ".s7p": {
                "equipment": "RF/Touchstone",
                "type": "S-Parameters",
                "multi_channel": False,
            },
            ".s8p": {
                "equipment": "RF/Touchstone",
                "type": "S-Parameters",
                "multi_channel": False,
            },
        }

        return format_db.get(
            fmt,
            {
                "equipment": "Unknown",
                "type": "Unknown",
                "multi_channel": False,
            },
        )

    def code(self, code_str: str) -> None:
        """Display code example."""
        lines = code_str.strip().split("\n")
        for line in lines:
            self.info(f"  {line}")

    def subsubsection(self, title: str) -> None:
        """Display subsubsection header."""
        self.info("")
        self.info(f"• {title}")
        self.info("  " + "-" * (len(title) + 2))


if __name__ == "__main__":
    demo = SupportedFormatsDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
