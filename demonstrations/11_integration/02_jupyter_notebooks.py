"""Jupyter Notebooks: Interactive analysis integration

Demonstrates:
- Jupyter magic commands (%oscura, %%analyze)
- Rich HTML display for traces and measurements
- Inline waveform and spectrum visualization
- IPython widgets for interactive parameters
- Notebook-friendly output formats

IEEE Standards: N/A
Related Demos:
- 00_getting_started/01_core_types.py
- 02_basic_analysis/01_waveform_measurements.py
- 04_advanced_analysis/02_spectral_analysis.py

This demonstrates how to use Oscura in Jupyter notebooks with rich
displays, magic commands, and interactive widgets for exploration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    generate_complex_signal,
    generate_sine_wave,
    validate_exists,
)


class JupyterNotebooksDemo(BaseDemo):
    """Demonstrates Jupyter notebook integration."""

    def __init__(self):
        """Initialize Jupyter notebooks demonstration."""
        super().__init__(
            name="jupyter_notebooks",
            description="Interactive analysis with Jupyter notebooks",
            capabilities=[
                "oscura.jupyter.magic.OscuraMagics",
                "oscura.jupyter.display.TraceDisplay",
                "oscura.jupyter.display.MeasurementDisplay",
                "oscura.jupyter.display.display_trace",
                "oscura.jupyter.display.display_measurements",
                "oscura.jupyter.display.display_spectrum",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate test signals for Jupyter display."""
        # Simple sine wave
        simple_trace = generate_sine_wave(
            frequency=1000.0,
            amplitude=1.0,
            duration=0.01,
            sample_rate=100e3,
        )

        # Complex signal for spectral display
        complex_trace = generate_complex_signal(
            fundamentals=[1000, 3000],
            amplitudes=[1.0, 0.3],
            duration=0.01,
            sample_rate=100e3,
        )

        return {
            "simple_trace": simple_trace,
            "complex_trace": complex_trace,
        }

    def run_demonstration(self, data: dict) -> dict:
        """Demonstrate Jupyter notebook integration."""
        from oscura.analyzers.waveform.measurements import (
            amplitude,
            frequency,
        )
        from oscura.analyzers.waveform.spectral import fft
        from oscura.jupyter.display import (
            MeasurementDisplay,
            TraceDisplay,
            display_measurements,
            display_trace,
        )
        from oscura.jupyter.magic import (
            OscuraMagics,
            get_current_trace,
            set_current_trace,
        )

        simple_trace = data["simple_trace"]
        _complex_trace = data["complex_trace"]  # Available for advanced demos

        self.section("1. IPython Magic Commands")
        self.info("Oscura provides magic commands for notebook convenience")

        # Show available magics
        self.subsection("Available Magics")
        magics_info = """
        Line Magics:
          %oscura load <file>      - Load a trace file
          %oscura measure [names]  - Run measurements
          %oscura info             - Show trace information
          %oscura formats          - List supported formats
          %oscura help             - Show help

        Cell Magics:
          %%analyze               - Multi-line analysis block
        """
        print(magics_info)

        # Demonstrate magic functionality (without actual IPython)
        self.subsection("Magic Command Simulation")
        _magics = OscuraMagics()  # Available for magic commands

        # Set current trace
        set_current_trace(simple_trace, "demo_signal.csv")
        self.info("Trace loaded into magic context")

        # Get current trace
        current = get_current_trace()
        self.info(f"Current trace: {type(current).__name__}")

        self.section("2. Rich Display Integration")
        self.info("Rich HTML displays for interactive exploration")

        # Trace display
        self.subsection("TraceDisplay - Rich Trace Information")
        trace_display = TraceDisplay(simple_trace, title="Demo Signal")
        html = trace_display._repr_html_()
        self.info(f"Generated HTML display ({len(html)} characters)")

        # Show simplified version
        self.info("Display includes:")
        self.info("  - Sample count and sample rate")
        self.info("  - Signal duration")
        self.info("  - Min/max/mean/std statistics")
        self.info("  - Channel and source information")

        # Measurement display
        self.subsection("MeasurementDisplay - Rich Results")
        measurements = {
            "frequency": frequency(simple_trace),
            "amplitude": amplitude(simple_trace),
            "rise_time": 2.5e-9,  # Example value
            "fall_time": 2.8e-9,
            "thd": -45.2,
        }

        meas_display = MeasurementDisplay(measurements, title="Signal Measurements")
        html = meas_display._repr_html_()
        self.info(f"Generated HTML display ({len(html)} characters)")

        # Show value formatting
        self.subsection("Automatic Unit Formatting")
        test_values = [
            (1.5e9, "1.5 G"),
            (25.6e6, "25.6 M"),
            (1.234e3, "1.234 k"),
            (0.707, "0.707"),
            (2.5e-3, "2.5 m"),
            (100e-9, "100 n"),
            (5e-12, "5 p"),
        ]

        for value, _expected in test_values:
            formatted = meas_display._format_value(value)
            self.result(f"{value:.2e}", formatted)

        self.section("3. Display Functions")
        self.info("Convenience functions for notebook cells")

        # display_trace
        self.subsection("display_trace()")
        self.info("Display trace with rich formatting:")
        display_trace(simple_trace, title="Simple Sine Wave")

        # display_measurements
        self.subsection("display_measurements()")
        self.info("Display measurements with rich formatting:")
        display_measurements(measurements, title="Waveform Measurements")

        # Note: display_spectrum requires matplotlib, demonstrated conceptually
        self.subsection("display_spectrum()")
        self.info("Displays inline spectrum plot")
        freqs, mags = fft(simple_trace)
        self.info(f"  - Frequencies: {len(freqs)} points")
        self.info(f"  - Magnitudes: {len(mags)} points")
        self.info("  - Log scale option")
        self.info("  - Configurable figure size")

        self.section("4. Interactive Widgets")
        self.info("IPython widgets for parameter exploration")

        # Demonstrate widget concepts
        self.subsection("Widget Example: Frequency Slider")
        widget_code = """
        from ipywidgets import interact, FloatSlider
        import oscura as osc

        @interact(freq=FloatSlider(min=100, max=10000, step=100, value=1000))
        def analyze_signal(freq):
            trace = osc.generate_sine_wave(frequency=freq)
            display_trace(trace)
            display_measurements(osc.measure(trace))
        """
        print(widget_code)

        self.subsection("Widget Example: Filter Cutoff")
        widget_code2 = """
        @interact(cutoff=FloatSlider(min=100, max=50000, value=5000))
        def filter_signal(cutoff):
            filtered = osc.low_pass(trace, cutoff)
            plot_comparison(trace, filtered, labels=['Original', 'Filtered'])
        """
        print(widget_code2)

        self.section("5. Notebook-Friendly Output")
        self.info("Format data for optimal notebook display")

        # DataFrame conversion
        self.subsection("Pandas DataFrame Export")
        batch_results = [
            {"file": "signal_1.wfm", "freq": 1000.0, "amp": 2.0},
            {"file": "signal_2.wfm", "freq": 5000.0, "amp": 1.5},
            {"file": "signal_3.wfm", "freq": 10000.0, "amp": 1.8},
        ]

        self.info("Convert batch results to DataFrame:")
        print("\n  import pandas as pd")
        print("  df = pd.DataFrame(batch_results)")
        print("  df\n")

        # Simulate DataFrame display
        for i, result in enumerate(batch_results):
            print(f"  {i}  {result['file']:15s}  {result['freq']:8.1f}  {result['amp']:.1f}")

        # Plotting integration
        self.subsection("Matplotlib Integration")
        self.info("Oscura plotting works seamlessly in notebooks:")
        plot_example = """
        %matplotlib inline
        from oscura import plot_waveform, plot_spectrum

        plot_waveform(trace)        # Inline display
        plot_spectrum(trace)         # Inline display
        """
        print(plot_example)

        self.section("6. Complete Notebook Workflow")
        self.info("Example: End-to-end signal analysis in notebook")

        notebook_workflow = """
        # Cell 1: Setup
        %load_ext oscura
        import oscura as osc
        from oscura.jupyter import display_trace, display_measurements

        # Cell 2: Load Data
        %oscura load capture.wfm
        display_trace(trace)

        # Cell 3: Quick Measurements
        %oscura measure rise_time fall_time frequency

        # Cell 4: Detailed Analysis
        %%analyze
        measurements = {
            'rise_time': osc.rise_time(trace),
            'fall_time': osc.fall_time(trace),
            'frequency': osc.frequency(trace),
            'amplitude': osc.amplitude(trace),
            'thd': osc.thd(trace),
        }
        display_measurements(measurements)

        # Cell 5: Spectral Analysis
        freqs, mags = osc.fft(trace)
        osc.display_spectrum(freqs, mags, title="Signal Spectrum")

        # Cell 6: Export Results
        import pandas as pd
        df = pd.DataFrame([measurements])
        df.to_csv('results.csv', index=False)
        """

        for line in notebook_workflow.split("\n"):
            print(line)

        self.section("7. Best Practices")
        self.info("Recommendations for notebook workflows")

        best_practices = [
            ("Load extensions early", "%load_ext oscura in first cell"),
            ("Use magic commands", "Quick operations with %oscura"),
            ("Rich displays", "Use display_* functions for better formatting"),
            ("Interactive widgets", "Explore parameters with ipywidgets"),
            ("Export to DataFrame", "Use pandas for tabular data"),
            ("Inline plotting", "Set %matplotlib inline for plots"),
            ("Save notebooks", "Include outputs for documentation"),
        ]

        for practice, description in best_practices:
            self.result(practice, description)

        self.success("Jupyter notebook integration demonstrated!")

        return {
            "trace_display": trace_display,
            "meas_display": meas_display,
            "measurements": measurements,
            "batch_results": batch_results,
        }

    def validate(self, results: dict) -> bool:
        """Validate Jupyter integration results."""
        self.info("Validating Jupyter integration...")

        # Check displays created
        if not validate_exists(results.get("trace_display"), "trace_display"):
            return False

        if type(results.get("trace_display")).__name__ != "TraceDisplay":
            return False

        if not validate_exists(results.get("meas_display"), "meas_display"):
            return False

        if type(results.get("meas_display")).__name__ != "MeasurementDisplay":
            return False

        # Check measurements
        if not validate_exists(results.get("measurements"), "measurements"):
            return False

        required_measurements = ["frequency", "amplitude", "rise_time", "fall_time", "thd"]
        for meas in required_measurements:
            if meas not in results["measurements"]:
                self.error(f"Missing measurement: {meas}")
                return False

        # Check batch results
        if not validate_exists(results.get("batch_results"), "batch_results"):
            return False

        self.success("All Jupyter integration tests passed!")
        self.info("\nNext steps:")
        self.info("  - Try oscura in a Jupyter notebook")
        self.info("  - Run: jupyter notebook")
        self.info("  - Use: %load_ext oscura")
        self.info("  - See 03_llm_integration.py for AI-friendly outputs")

        return True


if __name__ == "__main__":
    demo = JupyterNotebooksDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
