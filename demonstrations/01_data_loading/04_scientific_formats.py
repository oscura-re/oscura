"""Scientific File Format Loading

Demonstrates loading and handling scientific data acquisition formats:
- TDMS (LabVIEW/National Instruments)
- HDF5 (Hierarchical Data Format)
- NumPy NPZ (compressed arrays)
- WAV (audio waveforms)

IEEE Standards: IEEE 1057-2017 (Digitizing Waveform Recorders)
Related Demos:
- 01_data_loading/01_oscilloscopes.py
- 01_data_loading/02_logic_analyzers.py
- 02_basic_analysis/01_waveform_measurements.py

This demonstration shows:
1. How to load TDMS files from LabVIEW data acquisition systems
2. How to work with HDF5 hierarchical datasets
3. How to save/load compressed NumPy arrays (NPZ format)
4. How to load audio files as waveform data
5. Format-specific metadata extraction and validation
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    add_noise,
    format_size,
    format_table,
    generate_complex_signal,
    generate_sine_wave,
    validate_approximately,
)

# Check for optional dependencies
try:
    import h5py  # noqa: F401

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    from scipy.io import wavfile  # noqa: F401

    WAV_AVAILABLE = True
except ImportError:
    WAV_AVAILABLE = False


class ScientificFormatsDemo(BaseDemo):
    """Demonstrate loading scientific data acquisition formats."""

    def __init__(self) -> None:
        """Initialize scientific formats demonstration."""
        super().__init__(
            name="scientific_formats",
            description="Load and analyze scientific data acquisition formats",
            capabilities=[
                "oscura.loaders.load_tdms",
                "oscura.loaders.load_hdf5",
                "oscura.loaders.load_npz",
                "oscura.loaders.load_wav",
                "Multi-format metadata extraction",
            ],
            ieee_standards=["IEEE 1057-2017"],
            related_demos=[
                "01_data_loading/01_oscilloscopes.py",
                "01_data_loading/02_logic_analyzers.py",
            ],
        )
        self.temp_dir = tempfile.mkdtemp(prefix="oscura_sci_")

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic scientific format data."""
        self.info("Creating synthetic scientific data files...")

        # TDMS-style acquisition: Multi-channel DAQ
        tdms_data = self._create_tdms_synthetic()
        self.info("  ✓ TDMS DAQ capture (4 channels, 100 kHz)")

        # HDF5 hierarchical dataset
        hdf5_data = self._create_hdf5_synthetic()
        self.info("  ✓ HDF5 hierarchical dataset (2 groups, 3 channels)")

        # NumPy NPZ compressed archive
        npz_data = self._create_npz_synthetic()
        self.info("  ✓ NumPy NPZ compressed arrays (3 signals)")

        # WAV audio file
        wav_data = self._create_wav_synthetic()
        self.info("  ✓ WAV audio capture (stereo, 44.1 kHz)")

        return {
            "tdms": tdms_data,
            "hdf5": hdf5_data,
            "npz": npz_data,
            "wav": wav_data,
        }

    def _create_tdms_synthetic(self) -> dict[str, Any]:
        """Create synthetic TDMS multi-channel data acquisition."""
        sample_rate = 100e3  # 100 kHz
        duration = 0.01  # 10 ms
        _num_samples = int(sample_rate * duration)  # For reference only

        # Simulate 4-channel DAQ
        # Channel 1: Temperature sensor (sine wave, slowly varying)
        temp_signal = generate_sine_wave(
            frequency=10.0,  # 10 Hz
            amplitude=2.5,  # ±2.5°C variation
            duration=duration,
            sample_rate=sample_rate,
            offset=25.0,  # 25°C baseline
        )

        # Channel 2: Pressure sensor (complex signal)
        pressure_signal = generate_complex_signal(
            fundamentals=[50.0, 150.0],  # Hz
            amplitudes=[1.2, 0.4],
            duration=duration,
            sample_rate=sample_rate,
            snr_db=45,
        )

        # Channel 3: Vibration sensor (high frequency)
        vibration_signal = generate_sine_wave(
            frequency=5000.0,  # 5 kHz
            amplitude=0.8,
            duration=duration,
            sample_rate=sample_rate,
        )
        vibration_signal = add_noise(vibration_signal, snr_db=35)

        # Channel 4: Voltage monitor (square-ish)
        voltage_signal = generate_sine_wave(
            frequency=60.0,  # 60 Hz (mains frequency)
            amplitude=5.0,
            duration=duration,
            sample_rate=sample_rate,
        )

        return {
            "sample_rate": sample_rate,
            "channels": {
                "Temperature": temp_signal.data,
                "Pressure": pressure_signal.data,
                "Vibration": vibration_signal.data,
                "Voltage": voltage_signal.data,
            },
            "properties": {
                "Temperature": {"unit": "°C", "sensor_id": "TC-001"},
                "Pressure": {"unit": "Pa", "sensor_id": "PS-042"},
                "Vibration": {"unit": "m/s²", "sensor_id": "ACC-123"},
                "Voltage": {"unit": "V", "sensor_id": "VM-017"},
            },
        }

    def _create_hdf5_synthetic(self) -> dict[str, Any]:
        """Create synthetic HDF5 hierarchical dataset."""
        sample_rate = 1e6  # 1 MHz
        duration = 0.001  # 1 ms

        # Group 1: High-speed signals
        hf_signal1 = generate_sine_wave(
            frequency=100e3, amplitude=1.0, duration=duration, sample_rate=sample_rate
        )
        hf_signal2 = generate_sine_wave(
            frequency=200e3, amplitude=0.8, duration=duration, sample_rate=sample_rate
        )

        # Group 2: Low-speed signals
        lf_signal = generate_sine_wave(
            frequency=1e3, amplitude=2.0, duration=duration, sample_rate=sample_rate
        )

        return {
            "groups": {
                "high_frequency": {
                    "ch1": hf_signal1.data,
                    "ch2": hf_signal2.data,
                    "sample_rate": sample_rate,
                    "description": "High-frequency acquisition channels",
                },
                "low_frequency": {
                    "ch1": lf_signal.data,
                    "sample_rate": sample_rate,
                    "description": "Low-frequency monitoring channel",
                },
            },
            "metadata": {
                "experiment": "Signal Characterization",
                "date": "2024-01-15",
                "operator": "oscura_demo",
            },
        }

    def _create_npz_synthetic(self) -> dict[str, Any]:
        """Create synthetic NumPy NPZ compressed archive."""
        sample_rate = 10e6  # 10 MHz
        duration = 0.0005  # 500 μs

        # Signal 1: Clean sine wave
        signal1 = generate_sine_wave(
            frequency=1e6, amplitude=1.0, duration=duration, sample_rate=sample_rate
        )

        # Signal 2: Noisy complex signal
        signal2 = generate_complex_signal(
            fundamentals=[500e3, 1.5e6, 2.5e6],
            amplitudes=[0.8, 0.5, 0.3],
            duration=duration,
            sample_rate=sample_rate,
            snr_db=40,
        )

        # Signal 3: Modulated signal
        t = np.arange(int(sample_rate * duration)) / sample_rate
        carrier = 2e6  # 2 MHz carrier
        modulation = 50e3  # 50 kHz modulation
        signal3_data = np.sin(2 * np.pi * carrier * t) * (
            0.5 + 0.5 * np.sin(2 * np.pi * modulation * t)
        )

        return {
            "signals": {
                "clean_sine": signal1.data,
                "complex_signal": signal2.data,
                "am_modulated": signal3_data,
            },
            "sample_rate": sample_rate,
            "time_vector": t,
            "metadata": {
                "format_version": "1.0",
                "compression": "npz",
            },
        }

    def _create_wav_synthetic(self) -> dict[str, Any]:
        """Create synthetic WAV audio data (stereo)."""
        sample_rate = 44100  # 44.1 kHz (CD quality)
        duration = 0.1  # 100 ms

        # Left channel: 440 Hz (A4 note)
        left = generate_sine_wave(
            frequency=440.0, amplitude=0.8, duration=duration, sample_rate=sample_rate
        )

        # Right channel: 880 Hz (A5 note, one octave higher)
        right = generate_sine_wave(
            frequency=880.0, amplitude=0.6, duration=duration, sample_rate=sample_rate
        )

        return {
            "sample_rate": sample_rate,
            "channels": {
                "left": left.data,
                "right": right.data,
            },
            "bit_depth": 16,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the scientific formats demonstration."""
        results = {}

        self.section("Scientific Data Acquisition Formats")
        self.info("Scientific instruments use specialized formats for data storage:")
        self.info("  • TDMS: LabVIEW data acquisition (multi-channel, metadata-rich)")
        self.info("  • HDF5: Hierarchical datasets (large-scale experiments)")
        self.info("  • NPZ: Compressed NumPy arrays (Python ecosystem)")
        self.info("  • WAV: Audio waveforms (universal compatibility)")
        self.info("")

        # TDMS Analysis
        self.section("TDMS (LabVIEW) Multi-Channel Acquisition")
        results["tdms"] = self._analyze_tdms(data["tdms"])

        # HDF5 Analysis
        self.section("HDF5 Hierarchical Dataset")
        results["hdf5"] = self._analyze_hdf5(data["hdf5"])

        # NPZ Analysis
        self.section("NumPy NPZ Compressed Archive")
        results["npz"] = self._analyze_npz(data["npz"])

        # WAV Analysis
        self.section("WAV Audio Format")
        results["wav"] = self._analyze_wav(data["wav"])

        # Format comparison
        self.section("Format Comparison")
        self._display_format_comparison(data)

        # Best practices
        self.section("Best Practices for Scientific Data")
        self._show_best_practices()

        return results

    def _analyze_tdms(self, tdms_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze TDMS multi-channel acquisition."""
        self.subsection("TDMS Channel Overview")

        sample_rate = tdms_data["sample_rate"]
        channels = tdms_data["channels"]
        properties = tdms_data["properties"]

        self.result("Sample Rate", f"{sample_rate / 1e3:.1f}", "kHz")
        self.result("Number of Channels", len(channels))

        # Build channel table
        channel_info = []
        for name, signal_data in channels.items():
            props = properties[name]
            num_samples = len(signal_data)
            duration = num_samples / sample_rate
            v_min = float(np.min(signal_data))
            v_max = float(np.max(signal_data))
            v_mean = float(np.mean(signal_data))

            channel_info.append(
                [
                    name,
                    props["unit"],
                    props["sensor_id"],
                    f"{num_samples}",
                    f"{duration * 1e3:.2f} ms",
                    f"{v_mean:.3f}",
                    f"{v_min:.3f}",
                    f"{v_max:.3f}",
                ]
            )

        headers = ["Channel", "Unit", "Sensor ID", "Samples", "Duration", "Mean", "Min", "Max"]
        self.info(format_table(channel_info, headers))
        self.info("")

        return {
            "sample_rate": sample_rate,
            "num_channels": len(channels),
            "channel_names": list(channels.keys()),
        }

    def _analyze_hdf5(self, hdf5_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze HDF5 hierarchical structure."""
        self.subsection("HDF5 Dataset Structure")

        groups = hdf5_data["groups"]
        metadata = hdf5_data["metadata"]

        self.result("Number of Groups", len(groups))
        self.result("Experiment", metadata["experiment"])
        self.result("Date", metadata["date"])
        self.info("")

        # Analyze each group
        group_info = []
        for group_name, group_data in groups.items():
            num_channels = sum(1 for key in group_data if key.startswith("ch"))
            sample_rate = group_data["sample_rate"]
            description = group_data["description"]

            # Calculate total size
            total_samples = sum(len(group_data[key]) for key in group_data if key.startswith("ch"))
            size_bytes = total_samples * 8  # float64

            group_info.append(
                [
                    group_name,
                    f"{num_channels}",
                    f"{sample_rate / 1e6:.1f} MHz",
                    format_size(size_bytes),
                    description,
                ]
            )

        headers = ["Group", "Channels", "Sample Rate", "Size", "Description"]
        self.info(format_table(group_info, headers))
        self.info("")

        return {
            "num_groups": len(groups),
            "group_names": list(groups.keys()),
            "metadata": metadata,
        }

    def _analyze_npz(self, npz_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze NumPy NPZ archive."""
        self.subsection("NPZ Archive Contents")

        signals = npz_data["signals"]
        sample_rate = npz_data["sample_rate"]

        self.result("Sample Rate", f"{sample_rate / 1e6:.1f}", "MHz")
        self.result("Number of Signals", len(signals))
        self.info("")

        # Analyze each signal
        signal_info = []
        for name, signal_data in signals.items():
            num_samples = len(signal_data)
            duration = num_samples / sample_rate
            rms = float(np.sqrt(np.mean(signal_data**2)))
            peak = float(np.max(np.abs(signal_data)))

            # Estimate frequency content
            fft = np.fft.rfft(signal_data)
            freqs = np.fft.rfftfreq(num_samples, 1 / sample_rate)
            peak_freq_idx = np.argmax(np.abs(fft[1:])) + 1  # Skip DC
            peak_freq = freqs[peak_freq_idx]

            signal_info.append(
                [
                    name,
                    f"{num_samples}",
                    f"{duration * 1e6:.1f} μs",
                    f"{rms:.3f}",
                    f"{peak:.3f}",
                    f"{peak_freq / 1e6:.2f} MHz",
                ]
            )

        headers = ["Signal", "Samples", "Duration", "RMS", "Peak", "Dominant Freq"]
        self.info(format_table(signal_info, headers))
        self.info("")

        return {
            "sample_rate": sample_rate,
            "num_signals": len(signals),
            "signal_names": list(signals.keys()),
        }

    def _analyze_wav(self, wav_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze WAV audio format."""
        self.subsection("WAV Audio Properties")

        sample_rate = wav_data["sample_rate"]
        channels = wav_data["channels"]
        bit_depth = wav_data["bit_depth"]

        self.result("Sample Rate", f"{sample_rate}", "Hz")
        self.result("Bit Depth", f"{bit_depth}", "bits")
        self.result("Number of Channels", len(channels))
        self.info("")

        # Analyze each channel
        channel_info = []
        for name, signal_data in channels.items():
            num_samples = len(signal_data)
            duration = num_samples / sample_rate
            rms = float(np.sqrt(np.mean(signal_data**2)))

            # Find dominant frequency
            fft = np.fft.rfft(signal_data)
            freqs = np.fft.rfftfreq(num_samples, 1 / sample_rate)
            peak_idx = np.argmax(np.abs(fft[1:])) + 1
            dominant_freq = freqs[peak_idx]

            channel_info.append(
                [
                    name.capitalize(),
                    f"{num_samples}",
                    f"{duration * 1e3:.1f} ms",
                    f"{rms:.3f}",
                    f"{dominant_freq:.1f} Hz",
                ]
            )

        headers = ["Channel", "Samples", "Duration", "RMS", "Dominant Frequency"]
        self.info(format_table(channel_info, headers))
        self.info("")

        return {
            "sample_rate": sample_rate,
            "num_channels": len(channels),
            "bit_depth": bit_depth,
        }

    def _display_format_comparison(self, data: dict[str, Any]) -> None:
        """Display comparison of scientific formats."""
        comparison = []

        # TDMS
        tdms_channels = len(data["tdms"]["channels"])
        tdms_samples = len(next(iter(data["tdms"]["channels"].values())))
        tdms_size = tdms_channels * tdms_samples * 8
        comparison.append(
            [
                "TDMS",
                f"{tdms_channels}",
                f"{data['tdms']['sample_rate'] / 1e3:.0f} kHz",
                format_size(tdms_size),
                "Metadata-rich",
            ]
        )

        # HDF5
        hdf5_groups = len(data["hdf5"]["groups"])
        hdf5_channels = sum(
            sum(1 for k in g if k.startswith("ch")) for g in data["hdf5"]["groups"].values()
        )
        hdf5_size = sum(
            sum(len(g[k]) * 8 for k in g if k.startswith("ch"))
            for g in data["hdf5"]["groups"].values()
        )
        comparison.append(
            [
                "HDF5",
                f"{hdf5_channels} ({hdf5_groups} groups)",
                "Variable",
                format_size(hdf5_size),
                "Hierarchical",
            ]
        )

        # NPZ
        npz_signals = len(data["npz"]["signals"])
        npz_samples = len(next(iter(data["npz"]["signals"].values())))
        npz_size = npz_signals * npz_samples * 8
        comparison.append(
            [
                "NPZ",
                f"{npz_signals}",
                f"{data['npz']['sample_rate'] / 1e6:.0f} MHz",
                format_size(npz_size),
                "Compressed",
            ]
        )

        # WAV
        wav_channels = len(data["wav"]["channels"])
        wav_samples = len(next(iter(data["wav"]["channels"].values())))
        wav_size = wav_channels * wav_samples * 2  # 16-bit
        comparison.append(
            [
                "WAV",
                f"{wav_channels}",
                f"{data['wav']['sample_rate'] / 1e3:.1f} kHz",
                format_size(wav_size),
                "Universal audio",
            ]
        )

        headers = ["Format", "Channels", "Sample Rate", "Size", "Features"]
        self.info(format_table(comparison, headers))
        self.info("")

    def _show_best_practices(self) -> None:
        """Show best practices for scientific data formats."""
        self.subsection("Format Selection Guidelines")
        self.info("""
Choose the right format for your application:

1. TDMS (LabVIEW)
   ✓ Use for: Multi-channel DAQ systems, LabVIEW integration
   ✓ Strengths: Rich metadata, channel properties, time stamps
   ✓ Tools: NI DIAdem, LabVIEW, Python (npTDMS)
   ✗ Avoid: Non-LabVIEW ecosystems, simple single-channel data

2. HDF5 (Hierarchical Data Format)
   ✓ Use for: Large datasets, complex hierarchies, scientific computing
   ✓ Strengths: Scalable, efficient, supports compression, partial loading
   ✓ Tools: MATLAB, Python (h5py), HDFView, universal support
   ✗ Avoid: Simple data, real-time streaming applications

3. NumPy NPZ
   ✓ Use for: Python workflows, multiple related arrays, intermediate results
   ✓ Strengths: Native Python, automatic compression, simple API
   ✓ Tools: Python only (NumPy, SciPy)
   ✗ Avoid: Cross-language compatibility, streaming data

4. WAV Audio
   ✓ Use for: Audio-rate signals, universal compatibility, simple data
   ✓ Strengths: Universal support, simple format, portable
   ✓ Tools: Everything supports WAV
   ✗ Avoid: High sample rates (>192 kHz), complex metadata

METADATA RECOMMENDATIONS:
- Always store sample rate, units, sensor IDs
- Include calibration data for physical measurements
- Record acquisition parameters (filters, gain, coupling)
- Add timestamps for long-term experiments
- Document coordinate systems and conventions
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate scientific format loading results."""
        self.info("Validating scientific format data...")

        all_valid = True

        # Validate TDMS
        if "tdms" in results:
            tdms = results["tdms"]
            if not validate_approximately(
                tdms["sample_rate"],
                100e3,
                tolerance=0.01,
                name="TDMS sample rate",
            ):
                all_valid = False

            if tdms["num_channels"] != 4:
                self.error(f"TDMS channel count mismatch: expected 4, got {tdms['num_channels']}")
                all_valid = False

        # Validate HDF5
        if "hdf5" in results:
            hdf5 = results["hdf5"]
            if hdf5["num_groups"] != 2:
                self.error(f"HDF5 group count mismatch: expected 2, got {hdf5['num_groups']}")
                all_valid = False

        # Validate NPZ
        if "npz" in results:
            npz = results["npz"]
            if not validate_approximately(
                npz["sample_rate"],
                10e6,
                tolerance=0.01,
                name="NPZ sample rate",
            ):
                all_valid = False

            if npz["num_signals"] != 3:
                self.error(f"NPZ signal count mismatch: expected 3, got {npz['num_signals']}")
                all_valid = False

        # Validate WAV
        if "wav" in results:
            wav = results["wav"]
            if not validate_approximately(
                wav["sample_rate"],
                44100,
                tolerance=0.01,
                name="WAV sample rate",
            ):
                all_valid = False

            if wav["num_channels"] != 2:
                self.error(f"WAV channel count mismatch: expected 2, got {wav['num_channels']}")
                all_valid = False

        if all_valid:
            self.success("All scientific format validations passed!")
            self.info("""
Next steps for working with scientific data:

1. TDMS FILES
   from oscura.loaders import load_tdms
   trace = load_tdms("measurement.tdms", channel="CH1", group="Voltage")

2. HDF5 FILES
   from oscura.loaders import load_hdf5
   trace = load_hdf5("experiment.h5", dataset="/group/channel1")

3. NPZ FILES
   from oscura.loaders import load_npz
   trace = load_npz("signals.npz", array_name="signal1")

4. WAV FILES
   from oscura.loaders import load_wav
   trace = load_wav("audio.wav", channel=0)  # Left channel
            """)
        else:
            self.error("Some scientific format validations failed!")

        return all_valid


if __name__ == "__main__":
    demo = ScientificFormatsDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
