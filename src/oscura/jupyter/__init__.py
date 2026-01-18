"""Jupyter and IPython integration for Oscura.

This package provides IPython magic commands, rich display integration,
and Jupyter notebook-specific features.

  - IPython magic commands (%oscura, %%analyze)
  - Rich HTML display for results
  - Inline plot rendering
  - Progress bars (tqdm integration)
"""

from oscura.jupyter.display import (
    MeasurementDisplay,
    TraceDisplay,
    display_measurements,
    display_spectrum,
    display_trace,
)
from oscura.jupyter.magic import (
    OscuraMagics,
    load_ipython_extension,
)

__all__ = [
    "MeasurementDisplay",
    "OscuraMagics",
    "TraceDisplay",
    "display_measurements",
    "display_spectrum",
    "display_trace",
    "load_ipython_extension",
]
