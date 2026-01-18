"""Exporters namespace module.

This module provides a namespace for export functions to support:
    from oscura.exporters import exporters
    exporters.csv(trace, "output.csv")

Re-exports main export functions with short names.
"""

from oscura.exporters.csv import (
    export_csv as csv,
)
from oscura.exporters.hdf5 import (
    export_hdf5 as hdf5,
)
from oscura.exporters.html_export import (
    export_html as html,
)
from oscura.exporters.json_export import (
    export_json as json,
)
from oscura.exporters.markdown_export import (
    export_markdown as markdown,
)
from oscura.exporters.matlab_export import (
    export_mat as mat,
)
from oscura.exporters.npz_export import (
    export_npz as npz,
)
from oscura.exporters.spice_export import (
    export_pwl as pwl,
)

__all__ = [
    "csv",
    "hdf5",
    "html",
    "json",
    "markdown",
    "mat",
    "npz",
    "pwl",
]
