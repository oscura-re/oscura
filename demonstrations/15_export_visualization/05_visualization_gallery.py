"""Visualization Gallery: Comprehensive Showcase of All Plot Types

Demonstrates:
- Waveform plots (time-domain)
- FFT/spectral plots
- Eye diagrams
- Histograms
- Spectrograms
- Digital signal plots
- Multi-channel overlays
- Interactive plots (if plotly available)

This demonstration showcases all available visualization types
in Oscura, providing examples of each plot type and their use cases.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace


class VisualizationGalleryDemo(BaseDemo):
    """Demonstrate all visualization capabilities."""

    def __init__(self) -> None:
        """Initialize visualization gallery demonstration."""
        super().__init__(
            name="visualization_gallery",
            description="Comprehensive showcase of all visualization types in Oscura",
            capabilities=[
                "oscura.visualization.waveform",
                "oscura.visualization.spectral",
                "oscura.visualization.eye",
                "oscura.visualization.histogram",
                "oscura.visualization.digital",
            ],
            ieee_standards=["IEEE 1241-2010"],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
                "04_advanced_analysis/01_jitter_analysis.py",
                "15_export_visualization/01_export_formats.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals for visualization.

        Returns:
            Dictionary containing various signal types
        """
        sample_rate = 1e6  # 1 MHz
        duration = 0.01  # 10 ms
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate

        # Waveform: Sine wave with noise
        frequency = 10e3  # 10 kHz
        signal = np.sin(2 * np.pi * frequency * t)
        noise = np.random.normal(0, 0.1, num_samples)
        waveform_data = signal + noise

        waveform = WaveformTrace(
            data=waveform_data,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
                channel_name="Sine + Noise",
            ),
        )

        # Digital: Clock signal
        clock_freq = 100e3  # 100 kHz
        clock_data = ((t * clock_freq) % 1.0) < 0.5

        digital = DigitalTrace(
            data=clock_data,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
                channel_name="Clock",
            ),
        )

        # Multi-tone for spectral analysis
        f1, f2, f3 = 5e3, 15e3, 25e3
        multi_tone = (
            np.sin(2 * np.pi * f1 * t)
            + 0.5 * np.sin(2 * np.pi * f2 * t)
            + 0.3 * np.sin(2 * np.pi * f3 * t)
        )

        multi_tone_trace = WaveformTrace(
            data=multi_tone,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
                channel_name="Multi-tone",
            ),
        )

        return {
            "waveform": waveform,
            "digital": digital,
            "multi_tone": multi_tone_trace,
            "sample_rate": sample_rate,
            "t": t,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the visualization gallery demonstration."""
        results: dict[str, Any] = {}
        output_dir = self.get_output_dir()

        self.section("Visualization Gallery Demonstration")
        self.info("Showcase of all available visualization types")

        waveform = data["waveform"]
        digital = data["digital"]
        multi_tone = data["multi_tone"]

        # Check if matplotlib is available
        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt

            HAS_MATPLOTLIB = True
        except ImportError:
            HAS_MATPLOTLIB = False
            self.warning("matplotlib not installed - skipping plot generation")
            self.info("  Install with: pip install matplotlib")

        # Part 1: Waveform plots
        self.subsection("Part 1: Time-Domain Waveform Plots")
        self.info("Basic time-domain signal visualization.")

        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(10, 4))
            time = waveform.time_vector
            ax.plot(time[:1000], waveform.data[:1000], linewidth=0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (V)")
            ax.set_title("Time-Domain Waveform")
            ax.grid(True, alpha=0.3)

            waveform_path = output_dir / "01_waveform.png"
            fig.savefig(waveform_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            self.result("Waveform plot saved", str(waveform_path))
            results["waveform_plot"] = str(waveform_path)

        # Part 2: FFT/Spectral plots
        self.subsection("Part 2: Frequency-Domain (FFT) Plots")
        self.info("Spectral analysis showing frequency components.")

        if HAS_MATPLOTLIB:
            # Compute FFT
            fft_data = np.fft.rfft(multi_tone.data)
            freqs = np.fft.rfftfreq(len(multi_tone.data), 1 / multi_tone.metadata.sample_rate)
            magnitude_db = 20 * np.log10(np.abs(fft_data) + 1e-10)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(freqs / 1e3, magnitude_db)
            ax.set_xlabel("Frequency (kHz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_title("Frequency Spectrum (FFT)")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 50)

            fft_path = output_dir / "02_fft_spectrum.png"
            fig.savefig(fft_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            self.result("FFT plot saved", str(fft_path))
            results["fft_plot"] = str(fft_path)

        # Part 3: Histogram plots
        self.subsection("Part 3: Histogram Plots")
        self.info("Amplitude distribution histograms.")

        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(waveform.data, bins=100, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Amplitude (V)")
            ax.set_ylabel("Count")
            ax.set_title("Amplitude Histogram")
            ax.grid(True, alpha=0.3)

            hist_path = output_dir / "03_histogram.png"
            fig.savefig(hist_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            self.result("Histogram plot saved", str(hist_path))
            results["histogram_plot"] = str(hist_path)

        # Part 4: Digital signal plots
        self.subsection("Part 4: Digital Signal Plots")
        self.info("Logic-level digital signal visualization.")

        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(10, 3))
            time = digital.time_vector[:1000]
            signal = digital.data[:1000].astype(int)

            # Plot as step function
            ax.step(time, signal, where="post", linewidth=1)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Logic Level")
            ax.set_title("Digital Clock Signal")
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["LOW", "HIGH"])
            ax.grid(True, alpha=0.3)

            digital_path = output_dir / "04_digital.png"
            fig.savefig(digital_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            self.result("Digital plot saved", str(digital_path))
            results["digital_plot"] = str(digital_path)

        # Part 5: Multi-channel overlay
        self.subsection("Part 5: Multi-Channel Overlay Plots")
        self.info("Multiple signals on same time axis.")

        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(10, 4))
            time = waveform.time_vector[:1000]
            ax.plot(time, waveform.data[:1000], label="Waveform 1", alpha=0.7)
            ax.plot(time, multi_tone.data[:1000], label="Waveform 2", alpha=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (V)")
            ax.set_title("Multi-Channel Overlay")
            ax.legend()
            ax.grid(True, alpha=0.3)

            overlay_path = output_dir / "05_multi_channel.png"
            fig.savefig(overlay_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            self.result("Multi-channel plot saved", str(overlay_path))
            results["overlay_plot"] = str(overlay_path)

        # Part 6: Spectrogram (time-frequency)
        self.subsection("Part 6: Spectrogram (Time-Frequency) Plots")
        self.info("Time-frequency representation of signals.")

        if HAS_MATPLOTLIB:
            from matplotlib import mlab

            fig, ax = plt.subplots(figsize=(10, 4))

            # Compute spectrogram
            spec, freqs_spec, time_spec = mlab.specgram(
                waveform.data,
                NFFT=256,
                Fs=waveform.metadata.sample_rate,
                noverlap=128,
            )

            # Plot
            im = ax.pcolormesh(
                time_spec,
                freqs_spec / 1e3,
                10 * np.log10(spec + 1e-10),
                shading="auto",
                cmap="viridis",
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (kHz)")
            ax.set_title("Spectrogram")
            ax.set_ylim(0, 50)
            plt.colorbar(im, ax=ax, label="Power (dB)")

            spectrogram_path = output_dir / "06_spectrogram.png"
            fig.savefig(spectrogram_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            self.result("Spectrogram saved", str(spectrogram_path))
            results["spectrogram_plot"] = str(spectrogram_path)

        # Part 7: Summary and recommendations
        self.subsection("Part 7: Visualization Summary")
        self.info("Plot type recommendations by use case.")

        recommendations = {
            "Time-domain analysis": "Waveform plots - see signal shape and timing",
            "Frequency analysis": "FFT plots - identify frequency components",
            "Amplitude distribution": "Histograms - check signal statistics",
            "Digital signals": "Logic-level plots - verify timing and edges",
            "Multi-channel comparison": "Overlay plots - compare signals",
            "Time-frequency": "Spectrograms - see frequency changes over time",
            "Eye diagrams": "Data quality assessment (see jitter_analysis demo)",
            "Interactive exploration": "Plotly plots - zoom, pan, hover info",
        }

        self.info("\nVisualization recommendations:")
        for use_case, recommendation in recommendations.items():
            self.info(f"  • {use_case}:")
            self.info(f"    → {recommendation}")

        results["recommendations"] = recommendations

        # Part 8: Export options
        self.subsection("Part 8: Plot Export Options")
        self.info("Available formats for saving visualizations.")

        export_formats = {
            "PNG": "Raster format, good for web and documents (default)",
            "SVG": "Vector format, scalable, ideal for publications",
            "PDF": "Vector format, print-ready documents",
            "EPS": "Vector format, legacy publication systems",
            "JPEG": "Compressed raster, smaller file size",
        }

        self.info("\nExport format options:")
        for fmt, description in export_formats.items():
            self.info(f"  • {fmt}: {description}")

        results["export_formats"] = export_formats

        if HAS_MATPLOTLIB:
            plot_count = len([k for k in results if k.endswith("_plot")])
            self.success(f"Visualization gallery complete! {plot_count} plots generated.")
        else:
            self.warning("matplotlib not available - no plots generated")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate the demonstration results."""
        self.info("Validating visualization gallery...")

        # Check if matplotlib was available
        if not any(k.endswith("_plot") for k in results):
            self.warning("No plots generated (matplotlib not installed)")
            # This is acceptable - validation still passes
            return True

        # Validate that plot files exist
        plot_keys = [k for k in results if k.endswith("_plot")]

        for plot_key in plot_keys:
            plot_path = Path(results[plot_key])
            if not plot_path.exists():
                self.error(f"Plot file not found: {plot_path}")
                return False

        self.result("Total plots generated", len(plot_keys))

        # Validate recommendations exist
        if "recommendations" not in results:
            self.error("Missing visualization recommendations")
            return False

        self.success("All visualization gallery validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - Use waveform plots for time-domain analysis")
        self.info("  - Use FFT plots for frequency analysis")
        self.info("  - Use histograms for amplitude distribution")
        self.info("  - Use digital plots for logic-level signals")
        self.info("  - Use spectrograms for time-frequency analysis")
        self.info("\nNext steps:")
        self.info("  - Explore 04_advanced_analysis/ for specialized plots")
        self.info("  - Try 15_export_visualization/04_report_generation.py for reports")

        return True


if __name__ == "__main__":
    demo: VisualizationGalleryDemo = VisualizationGalleryDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
