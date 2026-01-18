"""Oscura Command-Line Interface.

This module provides command-line tools for signal analysis workflows.


Example:
    $ oscura --help
    $ oscura characterize signal.wfm
    $ oscura decode uart.wfm --protocol uart
"""

from oscura.cli.main import cli

__all__ = ["cli"]
