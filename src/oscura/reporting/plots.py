"""Plot generation for comprehensive analysis reports.

This module provides intelligent plot generation for different analysis domains,
using the existing visualization library and returning figures for the OutputManager
to save.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    import matplotlib.pyplot as plt

    # Use non-interactive backend for automated plot generation
    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from oscura.reporting.config import AnalysisConfig, AnalysisDomain
    from oscura.reporting.output import OutputManager


class PlotGenerator:
    """Generates visualization plots from analysis results.

    Intelligently creates appropriate plots for each analysis domain based on
    available data. Uses the existing oscura.visualization module and returns
    matplotlib Figure objects for the OutputManager to save.

    Attributes:
        config: Analysis configuration (optional, for plot settings).

    Requirements:

    Example:
        >>> config = AnalysisConfig(plot_format="png", plot_dpi=150)
        >>> generator = PlotGenerator(config)
        >>> paths = generator.generate_plots(
        ...     AnalysisDomain.SPECTRAL,
        ...     {"fft": {"frequencies": freq, "magnitude_db": mag}},
        ...     output_manager
        ... )
    """

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        """Initialize plot generator.

        Args:
            config: Analysis configuration for plot settings (format, DPI, etc.).
                   If None, uses defaults.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plot generation")

        self.config = config

    def generate_plots(
        self,
        domain: AnalysisDomain,
        results: dict[str, Any],
        output_manager: OutputManager,
    ) -> list[Path]:
        """Generate all appropriate plots for an analysis domain.

        Inspects the results dictionary and generates appropriate visualization
        plots based on the domain and available data. Returns list of saved
        plot paths.

        Args:
            domain: Analysis domain (e.g., SPECTRAL, WAVEFORM, DIGITAL).
            results: Dictionary of analysis results for this domain.
            output_manager: OutputManager instance for saving plots.

        Returns:
            List of paths to saved plot files.

        Example:
            >>> results = {
            ...     "fft": {"frequencies": freq_array, "magnitude_db": mag_array},
            ...     "psd": {"frequencies": freq_array, "psd": psd_array}
            ... }
            >>> paths = generator.generate_plots(
            ...     AnalysisDomain.SPECTRAL,
            ...     results,
            ...     output_manager
            ... )
        """

        plot_format, plot_dpi = self._get_plot_settings()
        saved_paths: list[Path] = []

        saved_paths.extend(
            self._generate_registered_plots(results, domain, output_manager, plot_format, plot_dpi)
        )

        saved_paths.extend(
            self._generate_domain_plots(domain, results, output_manager, plot_format, plot_dpi)
        )

        return saved_paths

    def _get_plot_settings(self) -> tuple[str, int]:
        """Get plot format and DPI from config."""
        plot_format = self.config.plot_format if self.config else "png"
        plot_dpi = self.config.plot_dpi if self.config else 150
        return plot_format, plot_dpi

    def _generate_registered_plots(
        self,
        results: dict[str, Any],
        domain: AnalysisDomain,
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> list[Path]:
        """Generate plots using registered plot functions."""
        saved_paths: list[Path] = []

        for analysis_name, result_data in results.items():
            if not isinstance(result_data, dict):
                continue

            key = (domain, analysis_name)
            if key in PLOT_REGISTRY:
                path = self._try_generate_plot(
                    PLOT_REGISTRY[key],
                    result_data,
                    domain,
                    analysis_name,
                    output_manager,
                    plot_format,
                    plot_dpi,
                )
                if path is not None:
                    saved_paths.append(path)

        return saved_paths

    def _try_generate_plot(
        self,
        plot_func: Callable[[dict[str, Any]], Figure | None],
        result_data: dict[str, Any],
        domain: AnalysisDomain,
        analysis_name: str,
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> Path | None:
        """Attempt to generate and save a single plot."""
        try:
            fig = plot_func(result_data)
            if fig is None:
                return None

            path = output_manager.save_plot(
                domain, analysis_name, fig, format=plot_format, dpi=plot_dpi
            )
            plt.close(fig)
            return path
        except Exception as e:
            logger.warning("Failed to generate %s plot: %s", analysis_name, e)
        return None

    def _generate_domain_plots(
        self,
        domain: AnalysisDomain,
        results: dict[str, Any],
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> list[Path]:
        """Generate domain-specific plots based on analysis domain."""
        from oscura.reporting.config import AnalysisDomain

        domain_generators = {
            AnalysisDomain.SPECTRAL: self._generate_spectral_plots,
            AnalysisDomain.WAVEFORM: self._generate_waveform_plots,
            AnalysisDomain.DIGITAL: self._generate_digital_plots,
            AnalysisDomain.STATISTICS: self._generate_statistics_plots,
            AnalysisDomain.JITTER: self._generate_jitter_plots,
            AnalysisDomain.EYE: self._generate_eye_plots,
            AnalysisDomain.PATTERNS: self._generate_pattern_plots,
            AnalysisDomain.POWER: self._generate_power_plots,
        }

        generator = domain_generators.get(domain)
        if generator is None:
            return []

        try:
            return generator(results, domain, output_manager, plot_format, plot_dpi)
        except Exception as e:
            logger.warning("Error in domain-level plot generation for %s: %s", domain.value, e)
            return []

    def _generate_spectral_plots(
        self,
        results: dict[str, Any],
        domain: AnalysisDomain,
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> list[Path]:
        """Generate spectral analysis plots (FFT, PSD, spectrogram)."""
        paths: list[Path] = []

        # Generate FFT plot
        fft_path = self._try_generate_fft_plot(
            results, domain, output_manager, plot_format, plot_dpi
        )
        if fft_path:
            paths.append(fft_path)

        # Generate PSD plot
        psd_path = self._try_generate_psd_plot(
            results, domain, output_manager, plot_format, plot_dpi
        )
        if psd_path:
            paths.append(psd_path)

        # Generate spectrogram
        spec_path = self._try_generate_spectrogram(
            results, domain, output_manager, plot_format, plot_dpi
        )
        if spec_path:
            paths.append(spec_path)

        return paths

    def _try_generate_fft_plot(
        self,
        results: dict[str, Any],
        domain: AnalysisDomain,
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> Path | None:
        """Try to generate FFT spectrum plot."""
        if "fft" not in results or not isinstance(results["fft"], dict):
            return None

        fft_data = results["fft"]
        if "frequencies" not in fft_data or "magnitude_db" not in fft_data:
            return None

        try:
            fig = self._plot_spectrum(fft_data, title="FFT Magnitude Spectrum")
            path = output_manager.save_plot(
                domain, "fft_spectrum", fig, format=plot_format, dpi=plot_dpi
            )
            plt.close(fig)
            return path
        except Exception:
            return None

    def _try_generate_psd_plot(
        self,
        results: dict[str, Any],
        domain: AnalysisDomain,
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> Path | None:
        """Try to generate PSD plot."""
        if "psd" not in results or not isinstance(results["psd"], dict):
            return None

        psd_data = results["psd"]
        if "frequencies" not in psd_data or "psd" not in psd_data:
            return None

        try:
            fig = self._plot_spectrum(
                psd_data, title="Power Spectral Density", ylabel="PSD (dB/Hz)"
            )
            path = output_manager.save_plot(
                domain, "psd_spectrum", fig, format=plot_format, dpi=plot_dpi
            )
            plt.close(fig)
            return path
        except Exception:
            return None

    def _try_generate_spectrogram(
        self,
        results: dict[str, Any],
        domain: AnalysisDomain,
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> Path | None:
        """Try to generate spectrogram plot."""
        if "spectrogram" not in results or not isinstance(results["spectrogram"], dict):
            return None

        spec_data = results["spectrogram"]
        required_keys = ["times", "frequencies", "Sxx_db"]
        if not all(key in spec_data for key in required_keys):
            return None

        try:
            fig = self._plot_spectrogram(spec_data)
            path = output_manager.save_plot(
                domain, "spectrogram", fig, format=plot_format, dpi=plot_dpi
            )
            plt.close(fig)
            return path
        except Exception:
            return None

    def _generate_waveform_plots(
        self,
        results: dict[str, Any],
        domain: AnalysisDomain,
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> list[Path]:
        """Generate waveform analysis plots (time series, histograms)."""
        paths: list[Path] = []

        # Look for time-series data
        for key in ["amplitude", "voltage", "signal", "data"]:
            if key in results and isinstance(results[key], np.ndarray | list):
                try:
                    fig = self._plot_time_series(
                        {"data": results[key]}, title=f"{key.title()} vs Time"
                    )
                    path = output_manager.save_plot(
                        domain, f"{key}_timeseries", fig, format=plot_format, dpi=plot_dpi
                    )
                    paths.append(path)
                    plt.close(fig)
                    break
                except Exception:
                    pass

        # Histogram of amplitudes
        for key in ["amplitude", "voltage", "data"]:
            if key in results and isinstance(results[key], np.ndarray | list):
                try:
                    fig = self._plot_histogram(
                        {"data": results[key]}, title=f"{key.title()} Distribution"
                    )
                    path = output_manager.save_plot(
                        domain, f"{key}_histogram", fig, format=plot_format, dpi=plot_dpi
                    )
                    paths.append(path)
                    plt.close(fig)
                    break
                except Exception:
                    pass

        return paths

    def _generate_digital_plots(
        self,
        results: dict[str, Any],
        domain: AnalysisDomain,
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> list[Path]:
        """Generate digital signal analysis plots (edges, timing)."""
        paths: list[Path] = []

        # Edge histogram (rise/fall time distribution)
        if "edges" in results and isinstance(results["edges"], dict):
            edges_data = results["edges"]
            if "rise_times" in edges_data:
                try:
                    fig = self._plot_histogram(
                        {"data": edges_data["rise_times"]}, title="Rise Time Distribution"
                    )
                    path = output_manager.save_plot(
                        domain, "rise_time_hist", fig, format=plot_format, dpi=plot_dpi
                    )
                    paths.append(path)
                    plt.close(fig)
                except Exception:
                    pass

        return paths

    def _generate_statistics_plots(
        self,
        results: dict[str, Any],
        domain: AnalysisDomain,
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> list[Path]:
        """Generate statistical analysis plots (distributions, box plots)."""
        paths: list[Path] = []

        # Histogram of distribution
        if "distribution" in results and isinstance(results["distribution"], dict):
            dist_data = results["distribution"]
            if "data" in dist_data:
                try:
                    fig = self._plot_histogram(dist_data, title="Statistical Distribution")
                    path = output_manager.save_plot(
                        domain, "distribution", fig, format=plot_format, dpi=plot_dpi
                    )
                    paths.append(path)
                    plt.close(fig)
                except Exception:
                    pass

        return paths

    def _generate_jitter_plots(
        self,
        results: dict[str, Any],
        domain: AnalysisDomain,
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> list[Path]:
        """Generate jitter analysis plots (TIE histogram, bathtub curve)."""
        paths: list[Path] = []

        # TIE (Time Interval Error) histogram
        if "tie" in results and isinstance(results["tie"], np.ndarray | list):
            try:
                fig = self._plot_histogram(
                    {"data": results["tie"]}, title="Time Interval Error (TIE)"
                )
                path = output_manager.save_plot(
                    domain, "tie_histogram", fig, format=plot_format, dpi=plot_dpi
                )
                paths.append(path)
                plt.close(fig)
            except Exception:
                pass

        return paths

    def _generate_eye_plots(
        self,
        results: dict[str, Any],
        domain: AnalysisDomain,
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> list[Path]:
        """Generate eye diagram plots."""
        # Eye diagrams are typically generated by the analyzer itself
        # This is a placeholder for future enhancements
        return []

    def _generate_pattern_plots(
        self,
        results: dict[str, Any],
        domain: AnalysisDomain,
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> list[Path]:
        """Generate pattern analysis plots (motifs, sequences)."""
        paths: list[Path] = []

        # Pattern occurrence histogram
        if "patterns" in results and isinstance(results["patterns"], dict):
            pattern_data = results["patterns"]
            if "occurrences" in pattern_data:
                try:
                    fig = self._plot_histogram(
                        {"data": pattern_data["occurrences"]}, title="Pattern Occurrences"
                    )
                    path = output_manager.save_plot(
                        domain, "pattern_occurrences", fig, format=plot_format, dpi=plot_dpi
                    )
                    paths.append(path)
                    plt.close(fig)
                except Exception:
                    pass

        return paths

    def _generate_power_plots(
        self,
        results: dict[str, Any],
        domain: AnalysisDomain,
        output_manager: OutputManager,
        plot_format: str,
        plot_dpi: int,
    ) -> list[Path]:
        """Generate power analysis plots (power vs time, efficiency)."""
        paths: list[Path] = []

        # Power time series
        if "power" in results and isinstance(results["power"], np.ndarray | list):
            try:
                fig = self._plot_time_series(
                    {"data": results["power"]}, title="Power vs Time", ylabel="Power (W)"
                )
                path = output_manager.save_plot(
                    domain, "power_timeseries", fig, format=plot_format, dpi=plot_dpi
                )
                paths.append(path)
                plt.close(fig)
            except Exception:
                pass

        return paths

    # ============================================================================
    # Individual plot methods
    # ============================================================================

    def _plot_spectrum(
        self,
        data: dict[str, Any],
        title: str = "Spectrum",
        ylabel: str = "Magnitude (dB)",
    ) -> Figure:
        """Plot frequency spectrum (FFT, PSD, etc.).

        Args:
            data: Dictionary with 'frequencies' and magnitude data.
            title: Plot title.
            ylabel: Y-axis label.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If frequency/magnitude data is missing or empty.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        frequencies = np.asarray(data.get("frequencies", []))
        # Try multiple possible keys for magnitude data
        magnitude = None
        for key in ["magnitude_db", "psd", "magnitude", "power_db"]:
            if key in data:
                magnitude = np.asarray(data[key])
                break

        if magnitude is None or len(frequencies) == 0 or len(magnitude) == 0:
            plt.close(fig)
            raise ValueError("Missing or empty frequency/magnitude data")

        # Auto-select frequency unit
        max_freq = frequencies[-1] if len(frequencies) > 0 else 1.0
        if max_freq >= 1e9:
            freq_unit = "GHz"
            freq_scale = 1e9
        elif max_freq >= 1e6:
            freq_unit = "MHz"
            freq_scale = 1e6
        elif max_freq >= 1e3:
            freq_unit = "kHz"
            freq_scale = 1e3
        else:
            freq_unit = "Hz"
            freq_scale = 1.0

        ax.plot(frequencies / freq_scale, magnitude, linewidth=0.8)
        ax.set_xlabel(f"Frequency ({freq_unit})")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xscale("log")

        fig.tight_layout()
        return fig

    def _plot_histogram(
        self,
        data: dict[str, Any],
        title: str = "Histogram",
        xlabel: str = "Value",
    ) -> Figure:
        """Plot histogram of data distribution.

        Args:
            data: Dictionary with 'data' array.
            title: Plot title.
            xlabel: X-axis label.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If data array is empty or contains no finite values.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        values = np.asarray(data.get("data", []))
        if len(values) == 0:
            plt.close(fig)
            raise ValueError("Empty data array for histogram")

        # Remove NaN/Inf values
        values = values[np.isfinite(values)]
        if len(values) == 0:
            plt.close(fig)
            raise ValueError("No finite values for histogram")

        # Auto-select number of bins (Sturges' rule with limits)
        n_bins = min(50, max(10, int(np.ceil(np.log2(len(values)) + 1))))

        ax.hist(values, bins=n_bins, alpha=0.7, edgecolor="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        return fig

    def _plot_time_series(
        self,
        data: dict[str, Any],
        title: str = "Time Series",
        ylabel: str = "Amplitude",
    ) -> Figure:
        """Plot time-domain data.

        Args:
            data: Dictionary with 'data' and optionally 'time' arrays.
            title: Plot title.
            ylabel: Y-axis label.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If data array is empty.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        values = np.asarray(data.get("data", []))
        if len(values) == 0:
            plt.close(fig)
            raise ValueError("Empty data array for time series")

        time = data.get("time", np.arange(len(values)))
        time = np.asarray(time)

        # Auto-select time unit
        max_time = time[-1] if len(time) > 0 else 1.0
        if max_time < 1e-6:
            time_unit = "ns"
            time_scale = 1e9
        elif max_time < 1e-3:
            time_unit = "us"
            time_scale = 1e6
        elif max_time < 1:
            time_unit = "ms"
            time_scale = 1e3
        else:
            time_unit = "s"
            time_scale = 1.0

        ax.plot(time * time_scale, values, linewidth=0.8)
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def _plot_spectrogram(self, data: dict[str, Any]) -> Figure:
        """Plot spectrogram (time-frequency heatmap).

        Args:
            data: Dictionary with 'times', 'frequencies', and 'Sxx_db' arrays.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If spectrogram data is missing or empty.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        times = np.asarray(data.get("times", []))
        frequencies = np.asarray(data.get("frequencies", []))
        Sxx_db = np.asarray(data.get("Sxx_db", []))

        if len(times) == 0 or len(frequencies) == 0 or Sxx_db.size == 0:
            plt.close(fig)
            raise ValueError("Missing spectrogram data")

        # Auto-select units
        max_time = times[-1] if len(times) > 0 else 1.0
        if max_time < 1e-6:
            time_unit = "ns"
            time_scale = 1e9
        elif max_time < 1e-3:
            time_unit = "us"
            time_scale = 1e6
        elif max_time < 1:
            time_unit = "ms"
            time_scale = 1e3
        else:
            time_unit = "s"
            time_scale = 1.0

        max_freq = frequencies[-1] if len(frequencies) > 0 else 1.0
        if max_freq >= 1e9:
            freq_unit = "GHz"
            freq_scale = 1e9
        elif max_freq >= 1e6:
            freq_unit = "MHz"
            freq_scale = 1e6
        elif max_freq >= 1e3:
            freq_unit = "kHz"
            freq_scale = 1e3
        else:
            freq_unit = "Hz"
            freq_scale = 1.0

        # Auto color limits
        valid_db = Sxx_db[np.isfinite(Sxx_db)]
        if len(valid_db) > 0:
            vmax = np.max(valid_db)
            vmin = max(np.min(valid_db), vmax - 80)
        else:
            vmin, vmax = None, None

        pcm = ax.pcolormesh(
            times * time_scale,
            frequencies / freq_scale,
            Sxx_db,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel(f"Frequency ({freq_unit})")
        ax.set_title("Spectrogram")

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Magnitude (dB)")

        fig.tight_layout()
        return fig


# ============================================================================
# Plot Registry
# ============================================================================

# Maps (domain, analysis_name) tuples to plot generation functions
# This allows custom plot functions to be registered for specific analyses
PLOT_REGISTRY: dict[
    tuple[AnalysisDomain, str] | AnalysisDomain, Callable[[dict[str, Any]], Figure]
] = {}


def register_plot(
    domain: AnalysisDomain,
    analysis_name: str | None = None,
) -> Callable[[Callable[[dict[str, Any]], Figure]], Callable[[dict[str, Any]], Figure]]:
    """Decorator to register a custom plot function.

    Args:
        domain: Analysis domain.
        analysis_name: Specific analysis name (optional). If None, registers
                      for entire domain.

    Returns:
        Decorator function.

    Example:
        >>> @register_plot(AnalysisDomain.SPECTRAL, "custom_fft")
        ... def plot_custom_fft(data: dict[str, Any]) -> Figure:
        ...     fig, ax = plt.subplots()
        ...     # Custom plotting code
        ...     return fig
    """

    def decorator(func: Callable[[dict[str, Any]], Figure]) -> Callable[[dict[str, Any]], Figure]:
        if analysis_name:
            PLOT_REGISTRY[(domain, analysis_name)] = func
        else:
            PLOT_REGISTRY[domain] = func
        return func

    return decorator


__all__ = [
    "PLOT_REGISTRY",
    "PlotGenerator",
    "register_plot",
]
