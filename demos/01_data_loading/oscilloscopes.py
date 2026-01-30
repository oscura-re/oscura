"""Oscilloscope File Format Loading

Demonstrates loading and handling various oscilloscope file formats:
- Tektronix .wfm files (DPO/MSO series)
- Rigol .wfm files (DS1000/DS2000 series)
- Multiple channels and metadata extraction
- Format-specific features

IEEE Standards: IEEE 181-2011 (Waveform and Vector Measurements)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path for common utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    ValidationSuite,
    add_noise,
    format_table,
    generate_sine_wave,
    generate_square_wave,
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
            ],
            ieee_standards=["IEEE 181-2011"],
            related_demos=[
                "02_logic_analyzers.py",
                "07_multi_channel.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic oscilloscope waveforms."""
        self.info("Creating synthetic oscilloscope captures...")

        # Tektronix-like capture: 1kHz sine wave + noise
        tek_trace = self._create_tektronix_synthetic()
        self.info("  ✓ Tektronix DPO7000 capture (1kHz sine, 10 GSa/s)")

        # Rigol-like capture: 500Hz square wave
        rigol_trace = self._create_rigol_synthetic()
        self.info("  ✓ Rigol DS1054Z capture (500Hz square, 100 MSa/s)")

        # Mixed-signal capture simulation (Tektronix MSO)
        mixed_trace = self._create_mixed_signal_synthetic()
        self.info("  ✓ Tektronix MSO mixed-signal capture")

        return {
            "tektronix": tek_trace,
            "rigol": rigol_trace,
            "mixed": mixed_trace,
        }

    def _create_tektronix_synthetic(self) -> WaveformTrace:
        """Create synthetic Tektronix capture."""
        sample_rate = 10e9  # 10 GSa/s
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
        sample_rate = 100e6  # 100 MSa/s
        duration = 20e-3  # 20 ms
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

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            vertical_scale=vertical_scale,
            vertical_offset=0.0,
        )

        return WaveformTrace(data=noisy_trace.data, metadata=metadata)

    def _create_mixed_signal_synthetic(self) -> WaveformTrace:
        """Create mixed-signal capture (analog + digital simulation)."""
        sample_rate = 12.5e9  # 12.5 GSa/s
        duration = 2.5e-6  # 2.5 microseconds
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

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            vertical_scale=vertical_scale,
            vertical_offset=0.0,
        )

        return WaveformTrace(data=noisy_analog.data, metadata=metadata)

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the oscilloscope loading demonstration."""
        self.subsection("Oscilloscope Format Overview")
        self.info("Modern oscilloscopes save waveforms in vendor-specific formats:")
        self.info("  • Tektronix: .wfm (TekScope binary format)")
        self.info("  • Rigol: .wfm (Rigol binary format)")
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

        self.subsection("Tektronix DPO7000 Capture Analysis")
        results["tektronix"] = self._analyze_capture(
            data["tektronix"],
            "Tektronix",
            "DPO7104C",
        )

        self.subsection("Rigol DS1054Z Capture Analysis")
        results["rigol"] = self._analyze_capture(
            data["rigol"],
            "Rigol",
            "DS1054Z",
        )

        self.subsection("Tektronix MSO Mixed-Signal Analysis")
        results["mixed"] = self._analyze_capture(
            data["mixed"],
            "Tektronix",
            "MSO5104B",
        )

        # Comparison table
        self.subsection("Format Comparison")
        self._display_format_comparison(data)

        # Metadata extraction best practices
        self.subsection("Metadata Extraction Best Practices")
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

        self.result("Instrument", f"{vendor} {model}")
        self.result("Sample Rate", f"{meta.sample_rate:.2e}", "Hz")
        if meta.vertical_scale is not None:
            self.result("Vertical Scale", f"{meta.vertical_scale}", "V/div")
        if meta.vertical_offset is not None:
            self.result("Vertical Offset", f"{meta.vertical_offset}", "V")

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
        self.info("""
Key Metadata Fields to Extract:

1. TIMING INFORMATION
   - sample_rate: Critical for frequency analysis
   - horizontal_scale: Time per division
   - duration: Total capture time

2. VOLTAGE INFORMATION
   - vertical_scale: Volts per division
   - vertical_offset: DC offset applied
   - resolution_bits: ADC resolution

3. INSTRUMENT IDENTIFICATION
   - instrument_vendor: "Tektronix", "Rigol", etc.
   - instrument_model: Specific model
   - channel: Channel number
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate oscilloscope loading results."""
        suite = ValidationSuite()

        # Validate Tektronix capture
        if "tektronix" in results:
            tek = results["tektronix"]
            suite.check_approximately(
                tek["sample_rate"],
                10e9,
                tolerance=0.01,
                name="Tektronix sample rate",
            )
            # 0.8V peak sine = 0.566V RMS
            suite.check_approximately(
                tek["vrms"],
                0.566,
                tolerance=0.15,
                name="Tektronix RMS voltage",
            )

        # Validate Rigol capture
        if "rigol" in results:
            rigol = results["rigol"]
            suite.check_approximately(
                rigol["sample_rate"],
                100e6,
                tolerance=0.01,
                name="Rigol sample rate",
            )
            # Square wave should have higher RMS than sine
            suite.check_true(rigol["vrms"] > 1.0, "Rigol square wave RMS > 1.0V")

        # Validate mixed-signal capture
        if "mixed" in results:
            mixed = results["mixed"]
            suite.check_approximately(
                mixed["sample_rate"],
                12.5e9,
                tolerance=0.01,
                name="MSO sample rate",
            )

        if suite.all_passed():
            self.success("All oscilloscope captures validated!")
            self.info("""
Next steps for real oscilloscope files:
  from oscura.loaders import load
  trace = load("TEK00001.wfm")  # Tektronix
  trace = load("DS1054Z_001.wfm")  # Rigol
            """)
        else:
            self.error("Some oscilloscope validations failed!")

        return suite.all_passed()


if __name__ == "__main__":
    demo = OscilloscopeLoadingDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
