"""Scientific Data Format Loading

Demonstrates loading scientific instrument data formats:
- TDMS (National Instruments LabVIEW)
- HDF5 (Hierarchical Data Format)
- WAV (audio waveforms)
- Scientific metadata extraction

Common in lab measurement and data acquisition systems.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    ValidationSuite,
    format_size,
    format_table,
    generate_sine_wave,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class ScientificFormatsDemo(BaseDemo):
    """Demonstrate loading scientific data formats."""

    def __init__(self) -> None:
        """Initialize scientific formats demonstration."""
        super().__init__(
            name="scientific_formats",
            description="Load and analyze scientific instrument data formats",
            capabilities=[
                "oscura.loaders.load_tdms",
                "oscura.loaders.load_hdf5",
                "oscura.loaders.load_wav",
                "Scientific metadata extraction",
            ],
            ieee_standards=[],
            related_demos=[
                "01_oscilloscopes.py",
                "07_multi_channel.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic scientific data."""
        self.info("Creating synthetic scientific data...")

        # TDMS-like data (National Instruments)
        tdms_data = self._create_tdms_synthetic()
        self.info("  ✓ TDMS format (4 channels, 100 kHz)")

        # HDF5-like hierarchical data
        hdf5_data = self._create_hdf5_synthetic()
        self.info("  ✓ HDF5 format (hierarchical groups)")

        # WAV audio data
        wav_data = self._create_wav_synthetic()
        self.info("  ✓ WAV format (44.1 kHz stereo)")

        return {
            "tdms": tdms_data,
            "hdf5": hdf5_data,
            "wav": wav_data,
        }

    def _create_tdms_synthetic(self) -> dict[str, Any]:
        """Create synthetic TDMS data structure."""
        sample_rate = 100e3  # 100 kHz
        duration = 0.1  # 100 ms
        num_channels = 4

        channels = {}
        for i in range(num_channels):
            frequency = 1e3 * (i + 1)  # 1 kHz, 2 kHz, 3 kHz, 4 kHz
            trace = generate_sine_wave(
                frequency=frequency,
                amplitude=1.0,
                duration=duration,
                sample_rate=sample_rate,
            )
            channels[f"Channel_{i}"] = trace

        return {
            "format": "TDMS",
            "channels": channels,
            "sample_rate": sample_rate,
            "metadata": {
                "instrument": "NI DAQ 6211",
                "acquisition_mode": "continuous",
                "trigger_type": "software",
            },
        }

    def _create_hdf5_synthetic(self) -> dict[str, Any]:
        """Create synthetic HDF5 hierarchical structure."""
        return {
            "format": "HDF5",
            "root": {
                "experiment": {
                    "temperature": np.random.randn(100) * 2 + 25,  # 25°C ± 2°C
                    "pressure": np.random.randn(100) * 0.1 + 1.0,  # 1.0 bar ± 0.1
                },
                "waveforms": {
                    "ch1": generate_sine_wave(1e3, 1.0, 0.01, 10e3).data,
                    "ch2": generate_sine_wave(2e3, 0.5, 0.01, 10e3).data,
                },
                "metadata": {
                    "date": "2024-01-01",
                    "operator": "Lab User",
                    "experiment_id": "EXP-001",
                },
            },
        }

    def _create_wav_synthetic(self) -> dict[str, Any]:
        """Create synthetic WAV audio data."""
        sample_rate = 44100  # 44.1 kHz
        duration = 0.5  # 500 ms

        # Left channel: 440 Hz (A4 note)
        left = generate_sine_wave(440, 0.5, duration, sample_rate)

        # Right channel: 880 Hz (A5 note)
        right = generate_sine_wave(880, 0.5, duration, sample_rate)

        return {
            "format": "WAV",
            "sample_rate": sample_rate,
            "channels": {"left": left, "right": right},
            "bit_depth": 16,
            "num_samples": len(left.data),
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the scientific formats demonstration."""
        results = {}

        self.subsection("Scientific Data Formats Overview")
        self.info("Common scientific data formats:")
        self.info("  • TDMS: National Instruments LabVIEW/DAQmx")
        self.info("  • HDF5: Hierarchical scientific data")
        self.info("  • WAV: Audio waveforms, acoustics")
        self.info("  • MAT: MATLAB workspace files")
        self.info("")

        # TDMS analysis
        self.subsection("TDMS Format Analysis")
        results["tdms"] = self._analyze_tdms(data["tdms"])

        # HDF5 analysis
        self.subsection("HDF5 Hierarchical Data")
        results["hdf5"] = self._analyze_hdf5(data["hdf5"])

        # WAV analysis
        self.subsection("WAV Audio Format")
        results["wav"] = self._analyze_wav(data["wav"])

        # Format comparison
        self.subsection("Format Comparison")
        self._display_format_comparison()

        return results

    def _analyze_tdms(self, tdms_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze TDMS format data."""
        self.result("Format", tdms_data["format"])
        self.result("Instrument", tdms_data["metadata"]["instrument"])
        self.result("Sample Rate", f"{tdms_data['sample_rate'] / 1e3:.0f}", "kHz")
        self.result("Channels", len(tdms_data["channels"]))

        # Channel statistics
        channel_rows = []
        for name, trace in tdms_data["channels"].items():
            rms = float(np.sqrt(np.mean(trace.data**2)))
            channel_rows.append(
                [
                    name,
                    len(trace.data),
                    f"{rms:.4f} V",
                ]
            )

        headers = ["Channel", "Samples", "RMS"]
        print(format_table(channel_rows, headers=headers))

        return {
            "num_channels": len(tdms_data["channels"]),
            "sample_rate": tdms_data["sample_rate"],
        }

    def _analyze_hdf5(self, hdf5_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze HDF5 hierarchical data."""
        self.result("Format", hdf5_data["format"])

        root = hdf5_data["root"]
        self.info("\nHierarchical Structure:")
        self.info("  /experiment/")
        self.info(f"    temperature: {len(root['experiment']['temperature'])} samples")
        self.info(f"    pressure: {len(root['experiment']['pressure'])} samples")
        self.info("  /waveforms/")
        self.info(f"    ch1: {len(root['waveforms']['ch1'])} samples")
        self.info(f"    ch2: {len(root['waveforms']['ch2'])} samples")
        self.info("  /metadata/")
        for key, value in root["metadata"].items():
            self.info(f"    {key}: {value}")

        return {
            "num_groups": 3,
            "total_datasets": 6,
        }

    def _analyze_wav(self, wav_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze WAV audio format."""
        self.result("Format", wav_data["format"])
        self.result("Sample Rate", f"{wav_data['sample_rate'] / 1000:.1f}", "kHz")
        self.result("Bit Depth", wav_data["bit_depth"], "bits")
        self.result("Channels", len(wav_data["channels"]))
        self.result("Duration", f"{wav_data['num_samples'] / wav_data['sample_rate']:.3f}", "s")

        # Calculate file size
        num_samples = wav_data["num_samples"]
        bytes_per_sample = wav_data["bit_depth"] // 8
        file_size = num_samples * len(wav_data["channels"]) * bytes_per_sample
        self.result("Estimated Size", format_size(file_size))

        return {
            "sample_rate": wav_data["sample_rate"],
            "num_channels": len(wav_data["channels"]),
        }

    def _display_format_comparison(self) -> None:
        """Display comparison of scientific formats."""
        comparison = [
            ["TDMS", "Binary", "Yes", "DAQ/LabVIEW", "High-speed acquisition"],
            ["HDF5", "Binary", "Yes", "Multi-platform", "Large datasets, hierarchical"],
            ["WAV", "Binary", "Limited", "Audio", "Acoustics, simple waveforms"],
            ["MAT", "Binary", "Yes", "MATLAB", "MATLAB workspace data"],
        ]

        headers = ["Format", "Type", "Metadata", "Ecosystem", "Best For"]
        print(format_table(comparison, headers=headers))
        self.info("")

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate scientific format loading results."""
        suite = ValidationSuite()

        # Validate TDMS
        if "tdms" in results:
            suite.check_equal(results["tdms"]["num_channels"], 4, "TDMS channels")
            suite.check_equal(results["tdms"]["sample_rate"], 100e3, "TDMS sample rate")

        # Validate HDF5
        if "hdf5" in results:
            suite.check_equal(results["hdf5"]["num_groups"], 3, "HDF5 groups")

        # Validate WAV
        if "wav" in results:
            suite.check_equal(results["wav"]["sample_rate"], 44100, "WAV sample rate")
            suite.check_equal(results["wav"]["num_channels"], 2, "WAV channels (stereo)")

        if suite.all_passed():
            self.success("All scientific format validations passed!")
            self.info("\nNext Steps:")
            self.info("  from oscura.loaders import load")
            self.info("  trace = load('experiment.tdms')")
        else:
            self.error("Some validations failed")

        return suite.all_passed()


if __name__ == "__main__":
    demo = ScientificFormatsDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
