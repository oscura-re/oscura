"""Common plotting functions for demonstrations.

Note: Plotting is optional. Demonstrations should work without matplotlib installed.
"""

from __future__ import annotations

from typing import Any


def plot_waveform(
    time: Any,
    voltage: Any,
    title: str = "Waveform",
    xlabel: str = "Time (s)",
    ylabel: str = "Voltage (V)",
    save_path: str | None = None,
) -> None:
    """Plot a waveform.

    Args:
        time: Time array
        voltage: Voltage array
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(time, voltage)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_spectrum(
    frequency: Any,
    magnitude: Any,
    title: str = "Frequency Spectrum",
    xlabel: str = "Frequency (Hz)",
    ylabel: str = "Magnitude (dB)",
    save_path: str | None = None,
) -> None:
    """Plot a frequency spectrum.

    Args:
        frequency: Frequency array
        magnitude: Magnitude array (in dB)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(frequency, magnitude)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison(
    x: Any,
    y1: Any,
    y2: Any,
    label1: str = "Signal 1",
    label2: str = "Signal 2",
    title: str = "Comparison",
    xlabel: str = "X",
    ylabel: str = "Y",
    save_path: str | None = None,
) -> None:
    """Plot two signals for comparison.

    Args:
        x: X-axis data
        y1: First signal
        y2: Second signal
        label1: Label for first signal
        label2: Label for second signal
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(x, y1, label=label1, alpha=0.7)
    plt.plot(x, y2, label=label2, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()
