"""Oscura Visualize Command - Interactive Waveform Viewer.

Provides CLI for launching interactive waveform visualization.


Example:
    $ oscura visualize signal.wfm
    $ oscura visualize capture.wfm --protocol uart
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from oscura.core.types import WaveformTrace

logger = logging.getLogger("oscura.cli.visualize")


@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--protocol",
    type=str,
    default=None,
    help="Protocol for overlay annotations.",
)
@click.option(
    "--save",
    type=click.Path(),
    default=None,
    help="Save plot to file instead of showing.",
)
@click.pass_context
def visualize(
    ctx: click.Context,
    file: str,
    protocol: str | None,
    save: str | None,
) -> None:
    """Launch interactive waveform viewer.

    Opens an interactive plot window for waveform analysis with zoom,
    pan, and measurement tools.

    Args:
        ctx: Click context object.
        file: Path to waveform file.
        protocol: Optional protocol for annotations.
        save: Save plot to file instead of displaying.

    Examples:

        \b
        # Interactive viewer
        $ oscura visualize signal.wfm

        \b
        # With protocol overlay
        $ oscura visualize capture.wfm --protocol uart

        \b
        # Save to file
        $ oscura visualize data.wfm --save plot.png
    """
    verbose = ctx.obj.get("verbose", 0)

    if verbose:
        logger.info(f"Visualizing: {file}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        from oscura.core.types import IQTrace, WaveformTrace
        from oscura.loaders import load

        # Load file
        trace = load(file)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot waveform (handle IQTrace separately)
        if isinstance(trace, IQTrace):
            # Plot I/Q components
            time_axis = np.arange(len(trace.i_data)) / trace.metadata.sample_rate
            ax.plot(time_axis * 1e3, trace.i_data, linewidth=0.5, label="I")
            ax.plot(time_axis * 1e3, trace.q_data, linewidth=0.5, label="Q")
            ax.legend()
        else:
            # Plot regular waveform or digital trace
            time_axis = np.arange(len(trace.data)) / trace.metadata.sample_rate
            ax.plot(time_axis * 1e3, trace.data, linewidth=0.5)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f"Waveform: {Path(file).name}")
        ax.grid(True, alpha=0.3)

        # Add protocol overlay if requested (only for non-IQ traces)
        if protocol and isinstance(trace, WaveformTrace):
            _add_protocol_overlay(ax, trace, protocol)

        # Save or show
        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")
            click.echo(f"Saved to: {save}")
        else:
            plt.tight_layout()
            plt.show()

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        if verbose > 1:
            raise
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _add_protocol_overlay(ax: Axes, trace: WaveformTrace, protocol: str) -> None:
    """Add protocol annotations to plot.

    Args:
        ax: Matplotlib axis.
        trace: Waveform trace.
        protocol: Protocol name.
    """
    import numpy as np

    # Convert to digital
    threshold = (np.max(trace.data) + np.min(trace.data)) / 2
    digital = trace.data > threshold

    # Find edges
    edges = np.where(np.diff(digital.astype(int)) != 0)[0]

    # Mark edges
    time_axis = np.arange(len(trace.data)) / trace.metadata.sample_rate
    for edge in edges[:100]:  # Limit to first 100 edges
        ax.axvline(time_axis[edge] * 1e3, color="red", alpha=0.3, linewidth=0.5)

    ax.text(
        0.98,
        0.98,
        f"Protocol: {protocol.upper()}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
