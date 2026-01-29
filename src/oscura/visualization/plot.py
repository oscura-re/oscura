"""Plotting functions namespace.

This module provides a namespace for plot functions to support:
    from oscura.visualization import plot
    plot.plot_trace(trace)

Re-exports main plotting functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace

from oscura.visualization.digital import (
    plot_logic_analyzer,
    plot_timing,
)
from oscura.visualization.eye import (
    plot_bathtub,
    plot_eye,
)
from oscura.visualization.interactive import (
    plot_bode,
    plot_histogram,
    plot_phase,
    plot_waterfall,
)
from oscura.visualization.spectral import (
    plot_fft,
    plot_psd,
    plot_spectrogram,
    plot_spectrum,
)
from oscura.visualization.waveform import (
    plot_multi_channel,
    plot_waveform,
    plot_xy,
)


def plot_trace(trace: WaveformTrace, **kwargs: Any) -> Any:
    """Plot a trace using plot_waveform.

    Convenience alias for plot_waveform.

    Args:
        trace: Waveform trace to plot.
        **kwargs: Additional arguments passed to plot_waveform.

    Returns:
        Matplotlib axes object.
    """
    return plot_waveform(trace, **kwargs)


def add_annotation(text: str, **kwargs: Any) -> None:
    """Add annotation to current plot.

    Placeholder for annotation functionality.

    Args:
        text: Annotation text.
        **kwargs: Additional arguments passed to ax.annotate.
    """
    import matplotlib.pyplot as plt

    ax = plt.gca()
    ax.annotate(text, xy=(0.5, 0.95), xycoords="axes fraction", **kwargs)


__all__ = [
    "add_annotation",
    "plot_bathtub",
    "plot_bode",
    "plot_eye",
    "plot_fft",
    "plot_histogram",
    "plot_logic_analyzer",
    "plot_multi_channel",
    "plot_phase",
    "plot_psd",
    "plot_spectrogram",
    "plot_spectrum",
    "plot_timing",
    "plot_trace",
    "plot_waterfall",
    "plot_waveform",
    "plot_xy",
]
