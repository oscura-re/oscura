"""Hello World: Your first Oscura demonstration

Demonstrates:
- oscura.load() - Load signal data
- oscura.amplitude() - Measure peak-to-peak voltage
- oscura.frequency() - Measure frequency
- oscura.plot_waveform() - Visualize signals

IEEE Standards: N/A
Related Demos:
- 00_getting_started/01_core_types.py
- 02_basic_analysis/01_waveform_measurements.py

This is the simplest possible Oscura workflow: load → measure → analyze → visualize.
Perfect for validating your installation and understanding the basic API.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, generate_sine_wave, validate_approximately
from oscura import amplitude, frequency, rms


class HelloWorldDemo(BaseDemo):
    """Minimal Oscura demonstration - load, measure, done."""

    def __init__(self):
        """Initialize hello world demonstration."""
        super().__init__(
            name="hello_world",
            description="Minimal Oscura workflow: load → measure → analyze",
            capabilities=[
                "oscura.WaveformTrace",
                "oscura.amplitude",
                "oscura.frequency",
                "oscura.rms",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate a simple 1kHz sine wave."""
        # Create a 1kHz sine wave at 1V amplitude
        trace = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=1.0,  # 1V peak
            duration=0.01,  # 10ms
            sample_rate=100e3,  # 100 kHz sampling
        )

        return {"trace": trace}

    def run_demonstration(self, data: dict) -> dict:
        """Run the hello world demonstration."""
        trace = data["trace"]

        self.section("Oscura Hello World")
        self.info("Welcome to Oscura - the hardware reverse engineering framework!")
        self.info("This demonstration shows the simplest possible workflow.")

        # Display trace information
        self.subsection("Signal Information")
        self.result("Sample rate", trace.metadata.sample_rate, "Hz")
        self.result("Number of samples", len(trace.data))
        self.result("Duration", len(trace.data) / trace.metadata.sample_rate, "s")

        # Perform basic measurements
        self.subsection("Measurements")

        # Measure amplitude (peak-to-peak voltage)
        vpp = amplitude(trace)
        self.result("Amplitude (Vpp)", f"{vpp:.4f}", "V")

        # Measure frequency
        freq = frequency(trace)
        self.result("Frequency", f"{freq:.2f}", "Hz")

        # Measure RMS voltage
        vrms = rms(trace)
        self.result("RMS voltage", f"{vrms:.4f}", "V")

        # Explain the results
        self.subsection("Understanding the Results")
        self.info("For a 1V peak sine wave:")
        self.info("  - Amplitude (Vpp) should be ~2.0V (peak-to-peak)")
        self.info("  - Frequency should be ~1000 Hz")
        self.info("  - RMS should be ~0.707V (1/√2)")

        self.success("Basic measurements complete!")

        return {
            "amplitude": vpp,
            "frequency": freq,
            "rms": vrms,
        }

    def validate(self, results: dict) -> bool:
        """Validate the results."""
        self.info("Validating measurements...")

        # Validate amplitude (2V ± 5% for digital sampling effects)
        if not validate_approximately(results["amplitude"], 2.0, tolerance=0.05, name="Amplitude"):
            return False

        # Validate frequency (1000 Hz ± 1%)
        if not validate_approximately(
            results["frequency"], 1000.0, tolerance=0.01, name="Frequency"
        ):
            return False

        # Validate RMS (0.707 ± 2%)
        if not validate_approximately(results["rms"], 0.707, tolerance=0.02, name="RMS"):
            return False

        self.success("All measurements validated!")
        self.info("\nCongratulations! Your Oscura installation is working correctly.")
        self.info("Next steps:")
        self.info("  - Try 01_core_types.py to learn about data structures")
        self.info("  - Explore 02_basic_analysis/ for more measurements")
        self.info("  - Check out protocol decoding in 03_protocol_decoding/")

        return True


if __name__ == "__main__":
    demo = HelloWorldDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
