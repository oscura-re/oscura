"""Thumbnail rendering for fast signal previews.

This module provides fast preview rendering with reduced detail
for gallery and browser contexts.


Example:
    >>> from oscura.visualization.thumbnails import render_thumbnail
    >>> fig = render_thumbnail(signal, sample_rate, size=(400, 300))

References:
    Aggressive decimation for performance
    Simplified rendering without expensive features
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

try:
    import matplotlib  # noqa: F401
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def render_thumbnail(
    signal: NDArray[np.float64],
    sample_rate: float | None = None,
    *,
    size: tuple[int, int] = (400, 300),
    width: int | None = None,
    height: int | None = None,
    max_samples: int = 1000,
    time_unit: str = "auto",
    title: str | None = None,
    dpi: int = 72,
) -> Figure:
    """Render fast preview thumbnail of signal.

    : Fast preview rendering mode with reduced detail,
    simplified styles, and lower resolution for quick plot generation.

    Target performance: <100ms for typical signals (goal: 50ms)

    Args:
        signal: Input signal array
        sample_rate: Sample rate in Hz. If None, uses 1.0 (sample indices as x-axis).
        size: Thumbnail size in pixels (width, height), default (400, 300)
        width: Width in pixels (alternative to size). If specified, height defaults to 3/4 of width.
        height: Height in pixels (alternative to size).
        max_samples: Maximum samples to plot (default: 1000, aggressive decimation)
        time_unit: Time unit for x-axis ("s", "ms", "us", "ns", "auto")
        title: Optional title
        dpi: DPI for rendering (default: 72)

    Returns:
        Matplotlib Figure object configured for fast rendering

    Raises:
        ValueError: If signal is empty or sample_rate is invalid
        ImportError: If matplotlib is not available

    Example:
        >>> signal = np.sin(2*np.pi*1000*np.arange(0, 0.01, 1/1e6))
        >>> fig = render_thumbnail(signal, 1e6, size=(400, 300))
        >>> fig.savefig("preview.png")
        >>> # Without sample rate
        >>> fig = render_thumbnail(data, width=100, height=50)

    References:
        VIS-018: Thumbnail Mode
        Fixed-count decimation for uniform sampling
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Default sample rate if not provided
    if sample_rate is None:
        sample_rate = 1.0

    if len(signal) == 0:
        raise ValueError("Signal cannot be empty")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if max_samples < 10:
        raise ValueError("max_samples must be >= 10")

    # Handle width/height as alternative to size
    if width is not None:
        h = height if height is not None else int(width * 0.75)
        size = (width, h)
    elif height is not None:
        size = (int(height * 4 / 3), height)

    # Configure matplotlib for fast rendering (no anti-aliasing, etc.)
    with plt.rc_context(
        {
            "path.simplify": True,
            "path.simplify_threshold": 1.0,
            "agg.path.chunksize": 1000,
            "lines.antialiased": False,
            "patch.antialiased": False,
            "text.antialiased": False,
        }
    ):
        # Calculate figure size in inches
        width_inches = size[0] / dpi
        height_inches = size[1] / dpi

        # Create figure with no fancy features
        fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=dpi)

        # Decimate signal to max_samples
        decimated_signal = _decimate_uniform(signal, max_samples)

        # Create time vector for decimated signal
        total_time = len(signal) / sample_rate
        time = np.linspace(0, total_time, len(decimated_signal))

        # Auto-select time unit
        if time_unit == "auto":
            if total_time < 1e-6:
                time_unit = "ns"
            elif total_time < 1e-3:
                time_unit = "us"
            elif total_time < 1:
                time_unit = "ms"
            else:
                time_unit = "s"

        time_multipliers = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
        multiplier = time_multipliers.get(time_unit, 1.0)
        time_scaled = time * multiplier

        # Plot with simplified style
        ax.plot(time_scaled, decimated_signal, "b-", linewidth=0.5, antialiased=False)

        # Minimal labels (no grid, no fancy formatting)
        ax.set_xlabel(f"Time ({time_unit})", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)

        if title:
            ax.set_title(title, fontsize=9)

        # Reduce tick label size
        ax.tick_params(labelsize=7)

        # Tight layout to maximize plot area
        fig.tight_layout(pad=0.5)

    return fig


def _decimate_uniform(signal: NDArray[np.float64], target_samples: int) -> NDArray[np.float64]:
    """Decimate signal to exactly target_samples using uniform stride.

    Args:
        signal: Input signal
        target_samples: Target number of samples

    Returns:
        Decimated signal with exactly target_samples
    """
    if len(signal) <= target_samples:
        return signal

    # Calculate uniform stride
    stride = len(signal) // target_samples

    # Sample at uniform intervals
    indices = np.arange(0, len(signal), stride)[:target_samples]

    decimated: NDArray[np.float64] = signal[indices]
    return decimated


def render_thumbnail_multichannel(
    signals: list[NDArray[np.float64]],
    sample_rate: float,
    *,
    size: tuple[int, int] = (400, 300),
    max_samples: int = 1000,
    time_unit: str = "auto",
    channel_names: list[str] | None = None,
    dpi: int = 72,
) -> Figure:
    """Render fast preview thumbnail of multiple channels.

    : Fast multi-channel preview rendering.

    Args:
        signals: List of signal arrays
        sample_rate: Sample rate in Hz
        size: Thumbnail size in pixels (width, height)
        max_samples: Maximum samples per channel
        time_unit: Time unit for x-axis
        channel_names: Optional channel names
        dpi: DPI for rendering

    Returns:
        Matplotlib Figure object

    Raises:
        ValueError: If inputs are invalid
        ImportError: If matplotlib is not available

    Example:
        >>> signals = [ch1_data, ch2_data, ch3_data]
        >>> fig = render_thumbnail_multichannel(signals, 1e6)

    References:
        VIS-018: Thumbnail Mode
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if len(signals) == 0:
        raise ValueError("Must provide at least one signal")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")

    n_channels = len(signals)

    if channel_names is None:
        channel_names = [f"CH{i + 1}" for i in range(n_channels)]

    # Configure matplotlib for fast rendering
    with plt.rc_context(
        {
            "path.simplify": True,
            "path.simplify_threshold": 1.0,
            "agg.path.chunksize": 1000,
            "lines.antialiased": False,
            "patch.antialiased": False,
            "text.antialiased": False,
        }
    ):
        # Calculate figure size
        width_inches = size[0] / dpi
        height_inches = size[1] / dpi

        fig, axes = plt.subplots(
            n_channels,
            1,
            figsize=(width_inches, height_inches),
            dpi=dpi,
            sharex=True,
        )

        if n_channels == 1:
            axes = [axes]

        # Get time unit from first signal
        if len(signals[0]) > 0:
            total_time = len(signals[0]) / sample_rate
            if time_unit == "auto":
                if total_time < 1e-6:
                    time_unit = "ns"
                elif total_time < 1e-3:
                    time_unit = "us"
                elif total_time < 1:
                    time_unit = "ms"
                else:
                    time_unit = "s"
        else:
            time_unit = "s"

        time_multipliers = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
        multiplier = time_multipliers.get(time_unit, 1.0)

        # Plot each channel
        for i, (sig, name, ax) in enumerate(zip(signals, channel_names, axes, strict=False)):
            if len(sig) == 0:
                continue

            # Decimate signal
            decimated = _decimate_uniform(sig, max_samples)

            # Time vector
            total_time = len(sig) / sample_rate
            time = np.linspace(0, total_time, len(decimated)) * multiplier

            # Plot
            ax.plot(time, decimated, "b-", linewidth=0.5, antialiased=False)

            # Channel label
            ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
            ax.tick_params(labelsize=6)

            # Only x-label on bottom
            if i == n_channels - 1:
                ax.set_xlabel(f"Time ({time_unit})", fontsize=8)

        fig.tight_layout(pad=0.3)

    return fig


__all__ = [
    "render_thumbnail",
    "render_thumbnail_multichannel",
]
