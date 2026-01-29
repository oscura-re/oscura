"""Demo plotting utilities for Oscura.

This module provides convenient visualization helpers for demos,
wrapping oscura.visualization functions with demo-specific defaults
like auto-saving to output directories and consistent styling.

Example:
    from demos.common.plotting import DemoPlotter

    plotter = DemoPlotter(output_dir=Path("./outputs"))

    # Plot waveform and save
    plotter.plot_waveform(trace, title="My Signal", name="waveform")

    # Plot FFT and save
    plotter.plot_fft(trace, title="Signal FFT", name="spectrum")

    # Get list of generated plots
    print(plotter.generated_plots)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Check for matplotlib availability
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for demos
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


class DemoPlotter:
    """Demo visualization helper with auto-save functionality.

    Provides a convenient interface for creating plots in demos,
    automatically saving to an output directory with consistent styling.

    Attributes:
        output_dir: Directory where plots are saved.
        generated_plots: List of paths to generated plot files.
        show_plots: Whether to display plots interactively.
        dpi: Resolution for saved plots.
    """

    def __init__(
        self,
        output_dir: Path | str | None = None,
        *,
        show_plots: bool = False,
        dpi: int = 150,
        style: str = "default",
    ):
        """Initialize demo plotter.

        Args:
            output_dir: Directory to save plots. If None, uses current directory.
            show_plots: Display plots interactively (False for CI/automated runs).
            dpi: Resolution for saved images.
            style: Matplotlib style to use.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show_plots = show_plots
        self.dpi = dpi
        self.style = style
        self.generated_plots: list[Path] = []

        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")

    def _save_figure(self, fig: Figure, name: str) -> Path:
        """Save figure to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = self.output_dir / filename

        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        self.generated_plots.append(filepath)

        if not self.show_plots:
            plt.close(fig)

        return filepath

    def plot_waveform(
        self,
        trace: WaveformTrace,
        *,
        name: str = "waveform",
        title: str | None = None,
        time_unit: str = "auto",
        show_measurements: bool = True,
        **kwargs: Any,
    ) -> Path:
        """Plot time-domain waveform.

        Args:
            trace: Waveform trace to plot.
            name: Filename base for saved plot.
            title: Plot title.
            time_unit: Time unit for x-axis.
            show_measurements: Show measurement annotations.
            **kwargs: Additional arguments to plot_waveform.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization import plot_waveform

        fig = plot_waveform(
            trace,
            time_unit=time_unit,
            title=title or "Waveform",
            show=self.show_plots,
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_fft(
        self,
        trace: WaveformTrace,
        *,
        name: str = "spectrum",
        title: str | None = None,
        freq_unit: str = "auto",
        log_scale: bool = True,
        **kwargs: Any,
    ) -> Path:
        """Plot FFT magnitude spectrum.

        Args:
            trace: Waveform trace to analyze.
            name: Filename base for saved plot.
            title: Plot title.
            freq_unit: Frequency unit for x-axis.
            log_scale: Use log scale for frequency axis.
            **kwargs: Additional arguments to plot_fft.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization import plot_fft

        fig = plot_fft(
            trace,
            freq_unit=freq_unit,
            log_scale=log_scale,
            title=title or "FFT Magnitude Spectrum",
            show=self.show_plots,
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_psd(
        self,
        trace: WaveformTrace,
        *,
        name: str = "psd",
        title: str | None = None,
        freq_unit: str = "auto",
        **kwargs: Any,
    ) -> Path:
        """Plot Power Spectral Density.

        Args:
            trace: Waveform trace to analyze.
            name: Filename base for saved plot.
            title: Plot title.
            freq_unit: Frequency unit for x-axis.
            **kwargs: Additional arguments to plot_psd.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization import plot_psd

        fig = plot_psd(
            trace,
            freq_unit=freq_unit,
            title=title or "Power Spectral Density",
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_spectrogram(
        self,
        trace: WaveformTrace,
        *,
        name: str = "spectrogram",
        title: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Plot spectrogram (time-frequency).

        Args:
            trace: Waveform trace to analyze.
            name: Filename base for saved plot.
            title: Plot title.
            **kwargs: Additional arguments to plot_spectrogram.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization import plot_spectrogram

        fig = plot_spectrogram(
            trace,
            title=title or "Spectrogram",
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_thd_bars(
        self,
        harmonic_magnitudes: NDArray[np.floating[Any]],
        *,
        name: str = "thd_harmonics",
        title: str | None = None,
        fundamental_freq: float | None = None,
        thd_value: float | None = None,
        **kwargs: Any,
    ) -> Path:
        """Plot THD harmonic bar chart.

        Args:
            harmonic_magnitudes: Array of harmonic magnitudes in dB.
            name: Filename base for saved plot.
            title: Plot title.
            fundamental_freq: Fundamental frequency for labels.
            thd_value: THD value to display.
            **kwargs: Additional arguments to plot_thd_bars.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization import plot_thd_bars

        fig = plot_thd_bars(
            harmonic_magnitudes,
            fundamental_freq=fundamental_freq,
            thd_value=thd_value,
            title=title or "Harmonic Distortion Analysis",
            show=self.show_plots,
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_quality_summary(
        self,
        metrics: dict[str, float],
        *,
        name: str = "quality_summary",
        title: str | None = None,
        show_specs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> Path:
        """Plot signal quality summary.

        Args:
            metrics: Dictionary with SNR, SINAD, THD, ENOB, SFDR values.
            name: Filename base for saved plot.
            title: Plot title.
            show_specs: Specification values for pass/fail indication.
            **kwargs: Additional arguments.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization import plot_quality_summary

        fig = plot_quality_summary(
            metrics,
            title=title or "Signal Quality Summary",
            show_specs=show_specs,
            show=self.show_plots,
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_eye(
        self,
        data: NDArray[np.floating[Any]],
        *,
        name: str = "eye_diagram",
        title: str | None = None,
        bit_period: float | None = None,
        **kwargs: Any,
    ) -> Path:
        """Plot eye diagram.

        Args:
            data: Signal data for eye diagram.
            name: Filename base for saved plot.
            title: Plot title.
            bit_period: Bit period for folding.
            **kwargs: Additional arguments to plot_eye.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization import plot_eye

        fig = plot_eye(
            data,
            bit_period=bit_period,
            title=title or "Eye Diagram",
            show=self.show_plots,
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_bathtub(
        self,
        ber_curve: NDArray[np.floating[Any]],
        *,
        name: str = "bathtub",
        title: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Plot bathtub curve.

        Args:
            ber_curve: BER curve data.
            name: Filename base for saved plot.
            title: Plot title.
            **kwargs: Additional arguments to plot_bathtub.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization import plot_bathtub

        fig = plot_bathtub(
            ber_curve,
            title=title or "Bathtub Curve",
            show=self.show_plots,
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_timing(
        self,
        signals: dict[str, NDArray[np.floating[Any]]],
        time: NDArray[np.floating[Any]],
        *,
        name: str = "timing_diagram",
        title: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Plot digital timing diagram.

        Args:
            signals: Dictionary of signal name to data.
            time: Time axis.
            name: Filename base for saved plot.
            title: Plot title.
            **kwargs: Additional arguments to plot_timing.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization import plot_timing

        fig = plot_timing(
            signals,
            time,
            title=title or "Timing Diagram",
            show=self.show_plots,
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_protocol_decode(
        self,
        frames: list[Any],
        *,
        name: str = "protocol_decode",
        title: str | None = None,
        protocol: str = "generic",
        **kwargs: Any,
    ) -> Path:
        """Plot protocol decode visualization.

        Args:
            frames: List of decoded frames.
            name: Filename base for saved plot.
            title: Plot title.
            protocol: Protocol type for specialized visualization.
            **kwargs: Additional arguments.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization import plot_protocol_decode

        fig = plot_protocol_decode(
            frames,
            title=title or f"{protocol.upper()} Protocol Decode",
            show=self.show_plots,
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_tdr(
        self,
        impedance: NDArray[np.floating[Any]],
        distance: NDArray[np.floating[Any]],
        *,
        name: str = "tdr_impedance",
        title: str | None = None,
        z0: float = 50.0,
        **kwargs: Any,
    ) -> Path:
        """Plot TDR impedance profile.

        Args:
            impedance: Impedance values in Ohms.
            distance: Distance values in meters.
            name: Filename base for saved plot.
            title: Plot title.
            z0: Reference impedance.
            **kwargs: Additional arguments.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization.signal_integrity import plot_tdr

        fig = plot_tdr(
            impedance,
            distance,
            z0=z0,
            title=title or "TDR Impedance Profile",
            show=self.show_plots,
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_power_profile(
        self,
        voltage: NDArray[np.floating[Any]],
        current: NDArray[np.floating[Any]],
        time: NDArray[np.floating[Any]],
        *,
        name: str = "power_profile",
        title: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Plot power profile (voltage, current, power).

        Args:
            voltage: Voltage waveform.
            current: Current waveform.
            time: Time axis.
            name: Filename base for saved plot.
            title: Plot title.
            **kwargs: Additional arguments.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization.power import plot_power_profile

        fig = plot_power_profile(
            voltage,
            current,
            time,
            title=title or "Power Profile",
            show=self.show_plots,
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_efficiency_curve(
        self,
        load_percentages: NDArray[np.floating[Any]],
        efficiencies: NDArray[np.floating[Any]],
        *,
        name: str = "efficiency",
        title: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Plot efficiency curve.

        Args:
            load_percentages: Load percentages.
            efficiencies: Efficiency values.
            name: Filename base for saved plot.
            title: Plot title.
            **kwargs: Additional arguments.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization.power_extended import plot_efficiency_curve

        fig = plot_efficiency_curve(
            load_percentages,
            efficiencies,
            title=title or "Efficiency vs Load",
            show=self.show_plots,
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_state_machine(
        self,
        transitions: list[Any],
        *,
        name: str = "state_machine",
        title: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Plot inferred state machine.

        Args:
            transitions: List of state transitions.
            name: Filename base for saved plot.
            title: Plot title.
            **kwargs: Additional arguments.

        Returns:
            Path to saved plot file.
        """
        from oscura.visualization import plot_state_machine

        fig = plot_state_machine(
            transitions,
            title=title or "Inferred State Machine",
            show=self.show_plots,
            **kwargs,
        )
        return self._save_figure(fig, name)

    def plot_multi_panel(
        self,
        panels: list[dict[str, Any]],
        *,
        name: str = "multi_panel",
        title: str | None = None,
        ncols: int = 2,
        figsize: tuple[float, float] | None = None,
    ) -> Path:
        """Create multi-panel figure.

        Args:
            panels: List of panel configurations, each with:
                - 'type': Plot type ('waveform', 'fft', 'psd', etc.)
                - 'data': Data for the plot
                - 'title': Panel title
                - Additional kwargs for the specific plot type
            name: Filename base for saved plot.
            title: Overall figure title.
            ncols: Number of columns.
            figsize: Figure size. Auto-calculated if None.

        Returns:
            Path to saved plot file.
        """
        n_panels = len(panels)
        nrows = (n_panels + ncols - 1) // ncols

        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # Flatten axes array for easy iteration
        if n_panels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Hide unused subplots
        for i in range(n_panels, len(axes)):
            axes[i].set_visible(False)

        # Plot each panel
        for i, panel in enumerate(panels):
            ax = axes[i]
            plot_type = panel.get("type", "waveform")
            panel_title = panel.get("title", f"Panel {i + 1}")

            if plot_type == "waveform":
                from oscura.visualization import plot_waveform

                plot_waveform(panel["data"], ax=ax, title=panel_title, show=False)
            elif plot_type == "fft":
                from oscura.visualization import plot_fft

                plot_fft(panel["data"], ax=ax, title=panel_title, show=False)
            elif plot_type == "psd":
                from oscura.visualization import plot_psd

                plot_psd(panel["data"], ax=ax, title=panel_title)
            elif plot_type == "custom":
                # Allow custom plotting function
                plot_func = panel.get("plot_func")
                if plot_func:
                    plot_func(ax, panel.get("data"), **panel.get("kwargs", {}))
                ax.set_title(panel_title)

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold")

        fig.tight_layout()

        return self._save_figure(fig, name)

    def summary(self) -> str:
        """Generate summary of generated plots.

        Returns:
            Summary string listing all generated plot files.
        """
        if not self.generated_plots:
            return "No plots generated."

        lines = [f"Generated {len(self.generated_plots)} plots:"]
        for path in self.generated_plots:
            lines.append(f"  - {path}")
        return "\n".join(lines)


def create_demo_plotter(demo_dir: Path | str, **kwargs: Any) -> DemoPlotter:
    """Create DemoPlotter with standard output directory structure.

    Args:
        demo_dir: Demo directory (usually Path(__file__).parent).
        **kwargs: Additional arguments for DemoPlotter.

    Returns:
        Configured DemoPlotter instance.
    """
    output_dir = Path(demo_dir) / "outputs"
    return DemoPlotter(output_dir=output_dir, **kwargs)
