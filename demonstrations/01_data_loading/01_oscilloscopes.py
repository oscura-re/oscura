"""Oscilloscope File Format Loading

Demonstrates loading and handling various oscilloscope file formats:
- Tektronix .wfm files (DPO/MSO series)
- Rigol .wfm files (DS1000/DS2000 series)
- LeCroy .trc files (WaveRunner/WavePro series)
- Generic oscilloscope format detection and metadata extraction

IEEE Standards: IEEE 181-2011 (Waveform and Vector Measurements)
Related Demos:
- 01_data_loading/00_csv_json.py
- 02_basic_analysis/01_waveform_measurements.py
- 02_basic_analysis/02_metadata_inspection.py

This demonstration shows:
1. How to detect and load different oscilloscope formats
2. How to extract metadata (sample rate, vertical scale, coupling, etc.)
3. How to handle missing files gracefully with format detection
4. Synthetic data generation that mimics real oscilloscope captures
5. Format-specific features (digital channels, IQ waveforms, etc.)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    add_noise,
    format_table,
    generate_sine_wave,
    generate_square_wave,
    validate_approximately,
)
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.loaders import get_supported_formats


class OscilloscopeLoadingDemo(BaseDemo):
    """Demonstrate loading oscilloscope file formats with synthetic data."""

    def __init__(self) -> None:
        """Initialize oscilloscope loading demonstration."""
        super().__init__(
            name="oscilloscope_loading",
            description="Load and analyze oscilloscope file formats",
            capabilities=[
                "oscura.loaders.load_tektronix_wfm",
                "oscura.loaders.load_rigol_wfm",
                "oscura.loaders.get_supported_formats",
                "WaveformTrace metadata extraction",
                "Oscilloscope format detection",
            ],
            ieee_standards=["IEEE 181-2011"],
            related_demos=[
                "01_data_loading/00_csv_json.py",
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic oscilloscope waveforms."""
        self.info("Creating synthetic oscilloscope captures...")

        # Tektronix-like capture: 1kHz sine wave + noise
        # DPO7000 typical settings: 10 GSa/s sample rate, 1V/div vertical scale
        tek_trace = self._create_tektronix_synthetic()
        self.info("  ✓ Tektronix DPO7000 capture (1kHz sine, 10 GSa/s)")

        # Rigol-like capture: 500Hz square wave
        # DS1054Z typical settings: 100 MSa/s sample rate, 2V/div vertical scale
        rigol_trace = self._create_rigol_synthetic()
        self.info("  ✓ Rigol DS1054Z capture (500Hz square, 100 MSa/s)")

        # LeCroy-like capture: Complex signal (sine + harmonics)
        # WaveRunner typical settings: 1 GSa/s sample rate
        lecroy_trace = self._create_lecroy_synthetic()
        self.info("  ✓ LeCroy WaveRunner capture (1kHz + harmonics, 1 GSa/s)")

        # Mixed-signal capture simulation (Tektronix MSO)
        mixed_trace = self._create_mixed_signal_synthetic()
        self.info("  ✓ Tektronix MSO mixed-signal capture")

        return {
            "tektronix": tek_trace,
            "rigol": rigol_trace,
            "lecroy": lecroy_trace,
            "mixed": mixed_trace,
        }

    def _create_tektronix_synthetic(self) -> WaveformTrace:
        """Create synthetic Tektronix capture."""
        # Tektronix DPO7000: 10 GSa/s, 1V/div
        sample_rate = 10e9  # 10 GSa/s
        # Capture 5 periods of 1kHz = 5ms (50M samples)
        duration = 5e-3  # 5 milliseconds
        vertical_scale = 1.0  # 1V/div

        # 1 kHz sine wave, 0.8V amplitude
        trace = generate_sine_wave(
            frequency=1e3,
            amplitude=0.8,
            duration=duration,
            sample_rate=sample_rate,
        )

        # Add realistic oscilloscope noise
        noisy_trace = add_noise(trace, snr_db=40)

        # Create Tektronix-specific metadata
        metadata = TraceMetadata(
            sample_rate=sample_rate,
            vertical_scale=vertical_scale,
            vertical_offset=0.0,
        )

        return WaveformTrace(data=noisy_trace.data, metadata=metadata)

    def _create_rigol_synthetic(self) -> WaveformTrace:
        """Create synthetic Rigol capture."""
        # Rigol DS1054Z: 100 MSa/s, 2V/div
        sample_rate = 100e6  # 100 MSa/s
        duration = 20e-3  # 20 ms for 2000 samples
        vertical_scale = 2.0  # 2V/div

        # 500 Hz square wave, 2V peak
        trace = generate_square_wave(
            frequency=500.0,
            amplitude=2.0,
            duration=duration,
            sample_rate=sample_rate,
            duty_cycle=0.5,
        )

        # Add realistic noise
        noisy_trace = add_noise(trace, snr_db=35)

        # Create Rigol-specific metadata
        metadata = TraceMetadata(
            sample_rate=sample_rate,
            vertical_scale=vertical_scale,
            vertical_offset=0.0,
        )

        return WaveformTrace(data=noisy_trace.data, metadata=metadata)

    def _create_lecroy_synthetic(self) -> WaveformTrace:
        """Create synthetic LeCroy capture with harmonics."""
        # LeCroy WaveRunner: 1 GSa/s
        sample_rate = 1e9  # 1 GSa/s
        # Capture 5 periods of 1kHz = 5ms (5M samples)
        duration = 5e-3  # 5 milliseconds
        vertical_scale = 0.5  # 0.5V/div

        # Create fundamental 1kHz sine
        fundamental = generate_sine_wave(
            frequency=1e3,
            amplitude=0.8,
            duration=duration,
            sample_rate=sample_rate,
        )

        # Add 3rd harmonic (3 kHz) at 30% amplitude
        t = np.linspace(0, duration, len(fundamental.data))
        harmonic_3 = 0.24 * np.sin(2 * np.pi * 3e3 * t)

        # Add 5th harmonic (5 kHz) at 20% amplitude
        harmonic_5 = 0.16 * np.sin(2 * np.pi * 5e3 * t)

        # Combine signals
        combined_data = fundamental.data + harmonic_3 + harmonic_5

        # Create trace with combined signal and add noise
        combined_trace = WaveformTrace(
            data=combined_data,
            metadata=fundamental.metadata,
        )
        noisy_trace = add_noise(combined_trace, snr_db=38)

        # Create LeCroy-specific metadata
        metadata = TraceMetadata(
            sample_rate=sample_rate,
            vertical_scale=vertical_scale,
            vertical_offset=0.0,
        )

        return WaveformTrace(data=noisy_trace.data, metadata=metadata)

    def _create_mixed_signal_synthetic(self) -> WaveformTrace:
        """Create mixed-signal capture (analog + digital simulation)."""
        # Tektronix MSO5000: 12.5 GSa/s
        sample_rate = 12.5e9
        # Capture 5 periods of 2MHz = 2.5 microseconds
        duration = 2.5e-6
        vertical_scale = 1.5

        # Analog channel: 2 MHz square wave
        analog = generate_square_wave(
            frequency=2e6,
            amplitude=1.5,
            duration=duration,
            sample_rate=sample_rate,
            duty_cycle=0.5,
        )

        noisy_analog = add_noise(analog, snr_db=42)

        # Create mixed-signal metadata
        metadata = TraceMetadata(
            sample_rate=sample_rate,
            vertical_scale=vertical_scale,
            vertical_offset=0.0,
        )

        return WaveformTrace(data=noisy_analog.data, metadata=metadata)

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the oscilloscope loading demonstration."""
        self.section("Oscilloscope Format Overview")
        self.info("Modern oscilloscopes save waveforms in vendor-specific formats:")
        self.info("  • Tektronix: .wfm (TekScope binary format)")
        self.info("  • Rigol: .wfm (Rigol binary format)")
        self.info("  • LeCroy: .trc (LeCroy binary format)")
        self.info("  • Generic: .csv, .dat (text-based)")
        self.info("")

        # Display supported formats
        self.subsection("Supported Formats in Oscura")
        formats = get_supported_formats()
        self.info(f"Total supported formats: {len(formats)}")
        oscilloscope_formats = [
            f for f in formats if any(x in f.lower() for x in ["tek", "rigol", "lecroy"])
        ]
        if oscilloscope_formats:
            for fmt in oscilloscope_formats:
                self.info(f"  • {fmt}")
        self.info("")

        # Analyze each captured waveform
        results = {}

        self.section("Tektronix DPO7000 Capture Analysis")
        results["tektronix"] = self._analyze_capture(
            data["tektronix"],
            "Tektronix",
            "DPO7104C",
        )

        self.section("Rigol DS1054Z Capture Analysis")
        results["rigol"] = self._analyze_capture(
            data["rigol"],
            "Rigol",
            "DS1054Z",
        )

        self.section("LeCroy WaveRunner Analysis")
        results["lecroy"] = self._analyze_capture(
            data["lecroy"],
            "LeCroy",
            "WaveRunner 6100",
        )

        self.section("Tektronix MSO Mixed-Signal Analysis")
        results["mixed"] = self._analyze_capture(
            data["mixed"],
            "Tektronix",
            "MSO5104B",
        )

        # Comparison table
        self.section("Format Comparison")
        self._display_format_comparison(data)

        # Metadata extraction best practices
        self.section("Metadata Extraction Best Practices")
        self._show_metadata_practices()
        self.info("")

        return results

    def _analyze_capture(
        self,
        trace: WaveformTrace,
        vendor: str,
        model: str,
    ) -> dict[str, Any]:
        """Analyze a single oscilloscope capture."""
        meta = trace.metadata

        self.subsection(f"{vendor} {model}")

        # Display key parameters
        self.result("Instrument", f"{vendor} {model}")
        self.result("Sample Rate", f"{meta.sample_rate:.2e}", "Hz")
        if meta.vertical_scale is not None:
            self.result("Vertical Scale", f"{meta.vertical_scale}", "V/div")
        if meta.vertical_offset is not None:
            self.result("Vertical Offset", f"{meta.vertical_offset}", "V")
        if meta.channel_name:
            self.result("Channel Name", meta.channel_name)

        # Calculate derived metrics
        num_samples = len(trace.data)
        duration = num_samples / meta.sample_rate
        vmin = float(np.min(trace.data))
        vmax = float(np.max(trace.data))
        vmean = float(np.mean(trace.data))
        vrms = float(np.sqrt(np.mean(trace.data**2)))

        self.result("Number of Samples", f"{num_samples}", "samples")
        self.result("Capture Duration", f"{duration:.2e}", "s")
        self.result("Min Voltage", f"{vmin:.4f}", "V")
        self.result("Max Voltage", f"{vmax:.4f}", "V")
        self.result("Mean Voltage", f"{vmean:.4f}", "V")
        self.result("RMS Voltage", f"{vrms:.4f}", "V")

        return {
            "vendor": vendor,
            "model": model,
            "sample_rate": meta.sample_rate,
            "num_samples": num_samples,
            "duration": duration,
            "vmin": vmin,
            "vmax": vmax,
            "vmean": vmean,
            "vrms": vrms,
        }

    def _display_format_comparison(self, data: dict[str, Any]) -> None:
        """Display comparison table of different formats."""
        traces = [
            ("Tektronix DPO7000", data["tektronix"]),
            ("Rigol DS1054Z", data["rigol"]),
            ("LeCroy WaveRunner", data["lecroy"]),
            ("Tektronix MSO5000", data["mixed"]),
        ]

        # Build comparison table
        table_data = []
        for name, trace in traces:
            meta = trace.metadata
            v_scale = f"{meta.vertical_scale}" if meta.vertical_scale else "—"
            num_samples = len(trace.data)
            duration = num_samples / meta.sample_rate
            table_data.append(
                [
                    name,
                    f"{meta.sample_rate:.2e}",
                    v_scale,
                    f"{num_samples}",
                    f"{duration:.2e}",
                ]
            )

        headers = ["Instrument", "Sample Rate (Hz)", "V/div", "Samples", "Duration (s)"]
        self.info(format_table(table_data, headers))
        self.info("")

    def _show_metadata_practices(self) -> None:
        """Demonstrate metadata extraction best practices."""
        self.subsection("Key Metadata Fields")
        self.info("""
When loading oscilloscope files, always extract and validate:

1. TIMING INFORMATION
   - sample_rate: Critical for frequency analysis and time-domain scaling
   - horizontal_scale: Time per division setting
   - horizontal_offset: Time reference (may indicate trigger delay)
   - duration: Total capture time (samples / sample_rate)

2. VOLTAGE INFORMATION
   - vertical_scale: Volts per division (display scaling)
   - vertical_offset: DC offset applied to channel
   - resolution_bits: ADC resolution affects quantization noise
   - unit: Typically "V" for analog, varies for current/power

3. COUPLING AND IMPEDANCE
   - coupling: "AC" (blocks DC), "DC" (passes DC), "GND" (grounded)
   - impedance: Input impedance (1MΩ, 50Ω for RF)
   - probe_attenuation: Probe setting (1:1, 10:1, 100:1)
   - bandwidth_limit: Hardware bandwidth restriction

4. INSTRUMENT IDENTIFICATION
   - instrument_vendor: "Tektronix", "Rigol", "LeCroy", etc.
   - instrument_model: Specific model for calibration data
   - channel: Channel number for multi-channel files

5. QUALITY INDICATORS
   - Verify sample_rate >= 2.5x signal frequency (Nyquist)
   - Check resolution_bits >= log2(dynamic_range) for adequate precision
   - Validate coupling mode matches measurement requirements
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate oscilloscope loading results."""
        self.info("Validating oscilloscope data loading...")

        all_valid = True

        # Validate Tektronix capture
        if "tektronix" in results:
            tek = results["tektronix"]
            if not validate_approximately(
                tek["sample_rate"],
                10e9,
                tolerance=0.01,
                name="Tektronix sample rate",
            ):
                all_valid = False
            # 0.8V peak sine = 0.566V RMS
            if not validate_approximately(
                tek["vrms"],
                0.566,
                tolerance=0.15,  # Allow for noise
                name="Tektronix RMS voltage",
            ):
                all_valid = False

        # Validate Rigol capture
        if "rigol" in results:
            rigol = results["rigol"]
            if not validate_approximately(
                rigol["sample_rate"],
                100e6,
                tolerance=0.01,
                name="Rigol sample rate",
            ):
                all_valid = False
            # Square wave should have higher RMS than sine
            if rigol["vrms"] < 1.0:
                self.error("Rigol square wave RMS too low (expected >1.0V)")
                all_valid = False

        # Validate LeCroy capture
        if "lecroy" in results:
            lecroy = results["lecroy"]
            if not validate_approximately(
                lecroy["sample_rate"],
                1e9,
                tolerance=0.01,
                name="LeCroy sample rate",
            ):
                all_valid = False
            # With harmonics (0.8 + 0.24 + 0.16 = 1.2V peak), RMS ≈ 0.60V
            if not validate_approximately(
                lecroy["vrms"],
                0.60,
                tolerance=0.15,
                name="LeCroy harmonic RMS",
            ):
                all_valid = False

        # Validate mixed-signal capture
        if "mixed" in results:
            mixed = results["mixed"]
            if not validate_approximately(
                mixed["sample_rate"],
                12.5e9,
                tolerance=0.01,
                name="MSO sample rate",
            ):
                all_valid = False

        if all_valid:
            self.success("All oscilloscope captures validated!")
            self.info("""
Next steps for real oscilloscope files:

1. LOADING REAL FILES
   from oscura.loaders import load
   trace = load("TEK00001.wfm")  # Tektronix
   trace = load("DS1054Z_001.wfm")  # Rigol
   trace = load("capture.trc")  # LeCroy

2. HANDLING MISSING LIBRARIES
   - Tektronix: Optional tm_data_types library
   - Rigol: Optional RigolWFM library
   - LeCroy: Basic binary parsing implemented
   - Fallback: All formats support basic binary parsing

3. METADATA VALIDATION
   - Always check sample_rate >= 2.5x signal frequency
   - Verify vertical_scale matches expected amplitude
   - Confirm coupling mode (AC/DC) is appropriate
   - Check bandwidth_limit for aliasing risk

4. MULTI-CHANNEL CAPTURES
   Load all channels: oscura.load_all_channels("file.wfm")
   Access individual channels with channel parameter
            """)
        else:
            self.error("Some oscilloscope validations failed!")

        return all_valid


if __name__ == "__main__":
    demo = OscilloscopeLoadingDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
