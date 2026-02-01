"""Hello World: Your first Oscura demonstration.

This is the simplest possible Oscura workflow: generate → measure → analyze.
Perfect for validating your installation and understanding the basic API.

Demonstrates:
- oscura.amplitude() - Measure peak-to-peak voltage
- oscura.frequency() - Measure frequency
- oscura.rms() - Measure RMS voltage
- Basic waveform generation

IEEE Standards: N/A

Related Demos:
- 00_getting_started/01_core_types.py
- 02_basic_analysis/01_waveform_basics.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demos.common import BaseDemo, ValidationSuite, run_demo_main
from demos.common.data_generation import generate_sine_wave
from oscura import amplitude, frequency, rms


class HelloWorldDemo(BaseDemo):
    """Minimal Oscura demonstration - generate, measure, validate."""

    name = "Hello World"
    description = "Minimal Oscura workflow: generate → measure → analyze"
    category = "getting_started"
    capabilities: ClassVar[list[str]] = [
        "oscura.WaveformTrace",
        "oscura.amplitude",
        "oscura.frequency",
        "oscura.rms",
    ]
    related_demos: ClassVar[list[str]] = [
        "00_getting_started/01_core_types.py",
        "02_basic_analysis/01_waveform_basics.py",
    ]

    def generate_data(self) -> None:
        """Generate a simple 1kHz sine wave."""
        # Create a 1kHz sine wave at 1V amplitude
        self.trace = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=1.0,  # 1V peak
            duration=0.01,  # 10ms
            sample_rate=100e3,  # 100 kHz sampling
        )

    def run_analysis(self) -> None:
        """Run the hello world demonstration."""
        from demos.common.formatting import print_info, print_result, print_subheader

        print_info("Welcome to Oscura - the hardware reverse engineering framework!")
        print_info("This demonstration shows the simplest possible workflow.")

        # Display trace information
        print_subheader("Signal Information")
        print_result("Sample rate", self.trace.metadata.sample_rate, "Hz")
        print_result("Number of samples", len(self.trace.data))
        duration = len(self.trace.data) / self.trace.metadata.sample_rate
        print_result("Duration", duration, "s")

        # Perform basic measurements
        print_subheader("Measurements")

        # Measure amplitude (peak-to-peak voltage)
        vpp = amplitude(self.trace)
        print_result("Amplitude (Vpp)", vpp, "V")

        # Measure frequency
        freq = frequency(self.trace)
        print_result("Frequency", freq, "Hz")

        # Measure RMS voltage
        vrms = rms(self.trace)
        print_result("RMS voltage", vrms, "V")

        # Explain the results
        print_subheader("Understanding the Results")
        print_info("For a 1V peak sine wave:")
        print_info("  - Amplitude (Vpp) should be ~2.0V (peak-to-peak)")
        print_info("  - Frequency should be ~1000 Hz")
        print_info("  - RMS should be ~0.707V (1/√2)")

        # Store results
        self.results["amplitude"] = vpp
        self.results["frequency"] = freq
        self.results["rms"] = vrms

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate the results."""
        # Validate amplitude (2V ± 5% for digital sampling effects)
        suite.check_close("Amplitude", self.results["amplitude"], 2.0, rtol=0.05, atol=0.0)

        # Validate frequency (1000 Hz ± 1%)
        suite.check_close("Frequency", self.results["frequency"], 1000.0, rtol=0.01, atol=0.0)

        # Validate RMS (0.707 ± 2%)
        suite.check_close("RMS", self.results["rms"], 0.707, rtol=0.02, atol=0.0)

        if suite.all_passed():
            from demos.common.formatting import print_info, print_success

            print_success("All measurements validated!")
            print_info("\nCongratulations! Your Oscura installation is working correctly.")
            print_info("Next steps:")
            print_info("  - Try 01_core_types.py to learn about data structures")
            print_info("  - Explore 02_basic_analysis/ for more measurements")
            print_info("  - Check out protocol decoding in 03_protocol_decoding/")


if __name__ == "__main__":
    sys.exit(run_demo_main(HelloWorldDemo))
