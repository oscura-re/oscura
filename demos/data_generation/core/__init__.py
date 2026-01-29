"""Core signal generation components.

This module contains the base classes and main builder for signal generation.
"""

from demos.data_generation.core.base import BaseSignalGenerator
from demos.data_generation.core.builder import SignalBuilder

__all__ = [
    "BaseSignalGenerator",
    "SignalBuilder",
]
