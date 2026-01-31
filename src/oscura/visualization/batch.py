"""Batch plot generation for comprehensive signal visualization.

This module provides utilities for generating multiple related plots from
signal traces in a single operation, useful for comprehensive analysis reports.

Example:
    >>> from oscura.visualization import batch
    >>> trace = osc.load("signal.wfm")
    >>> plots = batch.generate_all_plots(trace, output_format="base64")
    >>> # Returns: {"waveform": <base64>, "fft": <base64>, ...}
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

# Import trace types for runtime isinstance checks
from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

# Plot configuration constants
PLOT_DPI = 150
FIGURE_SIZE = (10, 6)

# Colorblind-safe palette (Tol Bright)
COLORS = {
    "primary": "#4477AA",  # Blue
    "secondary": "#EE6677",  # Red
    "success": "#228833",  # Green
    "warning": "#CCBB44",  # Yellow
    "danger": "#CC78BC",  # Purple
    "gray": "#949494",  # Gray
}


def fig_to_base64(fig: Figure, *, dpi: int = PLOT_DPI) -> str:
    """Convert matplotlib figure to base64-encoded PNG.

    Args:
        fig: Matplotlib figure object.
        dpi: Resolution in dots per inch.

    Returns:
        Base64-encoded PNG image string with data URI prefix.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> b64_str = fig_to_base64(fig)
        >>> assert b64_str.startswith("data:image/png;base64,")
    """
    buf = BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def plot_waveform(
    trace: WaveformTrace | DigitalTrace,
    *,
    title: str = "Time-Domain Waveform",
    sample_limit: int = 10000,
) -> str:
    """Generate time-domain waveform plot.

    Args:
        trace: WaveformTrace or DigitalTrace to plot.
        title: Plot title.
        sample_limit: Maximum samples to plot (downsamples if exceeded).

    Returns:
        Base64-encoded PNG image string.

    Example:
        >>> trace = osc.load("signal.wfm")
        >>> plot_data = plot_waveform(trace, title="My Signal")
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Prepare data with downsampling if needed
    data = trace.data
    if len(data) > sample_limit:
        step = len(data) // sample_limit
        data = data[::step]
        time = np.arange(len(data)) * step / trace.metadata.sample_rate
    else:
        time = np.arange(len(data)) / trace.metadata.sample_rate

    # Plot waveform
    ax.plot(time * 1000, data, color=COLORS["primary"], linewidth=0.8, alpha=0.9)

    # Styling
    ax.set_xlabel("Time (ms)", fontsize=11, fontweight="bold")
    is_bool = isinstance(data[0], (bool, np.bool_)) if len(data) > 0 else False
    ylabel = "Logic Level" if is_bool else "Amplitude (V)"
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add mean line for analog signals
    if not is_bool:
        mean_val = float(np.mean(data))
        ax.axhline(
            mean_val,
            color=COLORS["danger"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Mean: {mean_val:.3f} V",
        )
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_fft_spectrum(
    trace: WaveformTrace,
    *,
    title: str = "FFT Magnitude Spectrum",
) -> str:
    """Generate FFT magnitude spectrum plot.

    Args:
        trace: WaveformTrace to analyze.
        title: Plot title.

    Returns:
        Base64-encoded PNG image string.

    Example:
        >>> trace = osc.load("signal.wfm")
        >>> spectrum_plot = plot_fft_spectrum(trace)
    """
    import oscura as osc

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Compute FFT using oscura framework
    fft_result = osc.fft(trace)
    freqs = fft_result[0]
    mags = fft_result[1]

    # Convert to dB
    mags_db = 20 * np.log10(np.abs(mags) + 1e-12)

    # Plot only positive frequencies up to Nyquist
    nyquist_idx = len(freqs) // 2
    freqs_plot = freqs[:nyquist_idx]
    mags_plot = mags_db[:nyquist_idx]

    ax.plot(freqs_plot / 1000, mags_plot, color=COLORS["primary"], linewidth=1.2)

    # Find and mark fundamental frequency
    max_idx = np.argmax(mags_plot[10:]) + 10  # Skip DC component
    fund_freq = freqs_plot[max_idx]
    fund_mag = mags_plot[max_idx]
    ax.plot(
        fund_freq / 1000,
        fund_mag,
        "o",
        color=COLORS["secondary"],
        markersize=10,
        label=f"Fundamental: {fund_freq:.1f} Hz",
    )

    # Styling
    ax.set_xlabel("Frequency (kHz)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Magnitude (dB)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_histogram(
    data: NDArray[Any],
    *,
    title: str = "Amplitude Distribution",
    bins: int = 50,
) -> str:
    """Generate amplitude histogram with statistical overlays.

    Args:
        data: Signal data array.
        title: Plot title.
        bins: Number of histogram bins.

    Returns:
        Base64-encoded PNG image string.

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(1000)
        >>> hist_plot = plot_histogram(data)
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Create histogram
    _n, _bins_edges, _patches = ax.hist(
        data, bins=bins, color=COLORS["primary"], alpha=0.7, edgecolor="black", linewidth=0.5
    )

    # Add statistical overlays
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)

    ax.axvline(
        mean_val, color=COLORS["danger"], linestyle="--", linewidth=2, label=f"Mean: {mean_val:.3f}"
    )
    ax.axvline(
        median_val,
        color=COLORS["success"],
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_val:.3f}",
    )
    ax.axvline(
        mean_val + std_val,
        color=COLORS["gray"],
        linestyle=":",
        linewidth=1.5,
        label=f"±1std: {std_val:.3f}",
        alpha=0.7,
    )
    ax.axvline(mean_val - std_val, color=COLORS["gray"], linestyle=":", linewidth=1.5, alpha=0.7)

    # Styling
    ax.set_xlabel("Amplitude (V)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_spectrogram(
    trace: WaveformTrace,
    *,
    title: str = "Spectrogram",
    nfft: int = 1024,
    noverlap: int | None = None,
) -> str:
    """Generate spectrogram (time-frequency) plot.

    Args:
        trace: WaveformTrace to analyze.
        title: Plot title.
        nfft: FFT window size.
        noverlap: Number of overlapping samples (default: nfft // 2).

    Returns:
        Base64-encoded PNG image string.

    Example:
        >>> trace = osc.load("signal.wfm")
        >>> specgram = plot_spectrogram(trace, nfft=512)
    """
    if noverlap is None:
        noverlap = nfft // 2

    fig, ax = plt.subplots(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 0.8))

    # Generate spectrogram
    sample_rate = trace.metadata.sample_rate
    data = trace.data

    # Use matplotlib's specgram
    _spectrum, _freqs, _t, im = ax.specgram(
        data, Fs=sample_rate, cmap="viridis", NFFT=nfft, noverlap=noverlap
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Power (dB)")
    cbar.set_label("Power (dB)", fontsize=10, fontweight="bold")

    # Styling
    ax.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_logic_analyzer(
    trace: DigitalTrace,
    *,
    title: str = "Logic Analyzer View",
    max_samples: int = 1000,
) -> str:
    """Generate logic analyzer view for digital signals.

    Args:
        trace: DigitalTrace to plot.
        title: Plot title.
        max_samples: Maximum samples to display.

    Returns:
        Base64-encoded PNG image string.

    Example:
        >>> digital_trace = osc.load("digital_signal.wfm")
        >>> logic_plot = plot_logic_analyzer(digital_trace)
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Prepare data with downsampling if needed
    data = trace.data.astype(float)
    if len(data) > max_samples:
        step = len(data) // max_samples
        data = data[::step]
        time = np.arange(len(data)) * step / trace.metadata.sample_rate
    else:
        time = np.arange(len(data)) / trace.metadata.sample_rate

    # Plot as step function (digital signal)
    ax.step(time * 1e6, data, where="post", color=COLORS["primary"], linewidth=1.5)
    ax.fill_between(time * 1e6, 0, data, step="post", alpha=0.3, color=COLORS["primary"])

    # Add grid lines at logic levels
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.axhline(1, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    # Styling
    ax.set_xlabel("Time (μs)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Logic Level", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(-0.2, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LOW", "HIGH"])
    ax.grid(True, alpha=0.3, axis="x", linestyle="--")

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_statistics_summary(
    data: NDArray[Any],
    *,
    title: str = "Statistical Summary",
) -> str:
    """Generate statistical summary with box and violin plots.

    Args:
        data: Signal data array.
        title: Plot title.

    Returns:
        Base64-encoded PNG image string.

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(1000)
        >>> stats_plot = plot_statistics_summary(data)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 0.7))

    # Box plot
    ax1.boxplot(
        [data],
        vert=True,
        patch_artist=True,
        widths=0.5,
        boxprops={"facecolor": COLORS["primary"], "alpha": 0.7},
        medianprops={"color": COLORS["danger"], "linewidth": 2},
        whiskerprops={"color": "black", "linewidth": 1.5},
        capprops={"color": "black", "linewidth": 1.5},
    )

    ax1.set_ylabel("Amplitude (V)", fontsize=11, fontweight="bold")
    ax1.set_title("Box Plot", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax1.set_xticks([])

    # Violin plot
    parts = ax2.violinplot([data], vert=True, widths=0.7, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(COLORS["success"])
        pc.set_alpha(0.7)

    ax2.set_ylabel("Amplitude (V)", fontsize=11, fontweight="bold")
    ax2.set_title("Violin Plot", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax2.set_xticks([])

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig_to_base64(fig)


def generate_all_plots(
    trace: WaveformTrace | DigitalTrace,
    *,
    output_format: str = "base64",
    verbose: bool = True,
) -> dict[str, str]:
    """Generate all applicable plots for a signal trace.

    Automatically detects signal type and generates appropriate plots:
    - Analog signals: waveform, FFT spectrum, histogram, spectrogram, statistics
    - Digital signals: waveform, logic analyzer view

    Args:
        trace: WaveformTrace or DigitalTrace to visualize.
        output_format: Output format ("base64" only currently supported).
        verbose: Print progress messages.

    Returns:
        Dictionary mapping plot names to base64 image strings.

    Raises:
        ValueError: If output_format is not "base64".

    Example:
        >>> trace = osc.load("signal.wfm")
        >>> plots = generate_all_plots(trace)
        >>> len(plots)  # 5 plots for analog signal
        5
    """
    if output_format != "base64":
        raise ValueError(f"Only 'base64' output format supported, got '{output_format}'")

    plots = {}
    is_digital = (
        trace.is_digital if hasattr(trace, "is_digital") else isinstance(trace, DigitalTrace)
    )

    # Always generate waveform plot
    try:
        plots["waveform"] = plot_waveform(trace)
        if verbose:
            print("  ✓ Generated waveform plot")
    except Exception as e:
        if verbose:
            print(f"  ⚠ Waveform plot failed: {e}")

    if not is_digital:
        # Analog signal plots
        if isinstance(trace, WaveformTrace):  # Type narrowing
            try:
                plots["fft"] = plot_fft_spectrum(trace)
                if verbose:
                    print("  ✓ Generated FFT spectrum")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ FFT plot failed: {e}")

            try:
                plots["histogram"] = plot_histogram(trace.data)
                if verbose:
                    print("  ✓ Generated histogram")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Histogram plot failed: {e}")

            try:
                plots["spectrogram"] = plot_spectrogram(trace)
                if verbose:
                    print("  ✓ Generated spectrogram")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Spectrogram plot failed: {e}")

            try:
                plots["statistics"] = plot_statistics_summary(trace.data)
                if verbose:
                    print("  ✓ Generated statistics summary")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Statistics plot failed: {e}")
    else:
        # Digital signal plots
        from oscura.core.types import DigitalTrace as DigitalTraceType

        if isinstance(trace, DigitalTraceType):  # Type narrowing
            try:
                plots["logic"] = plot_logic_analyzer(trace)
                if verbose:
                    print("  ✓ Generated logic analyzer view")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Logic analyzer plot failed: {e}")

    return plots


__all__ = [
    "COLORS",
    "FIGURE_SIZE",
    "PLOT_DPI",
    "fig_to_base64",
    "generate_all_plots",
    "plot_fft_spectrum",
    "plot_histogram",
    "plot_logic_analyzer",
    "plot_spectrogram",
    "plot_statistics_summary",
    "plot_waveform",
]
