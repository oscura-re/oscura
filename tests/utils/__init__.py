"""Test utilities for Oscura test suite.

This module provides centralized utilities for:
- Mocking optional dependencies
- Test data factories
- Custom assertions
- Synthetic data generation
- Fixture helpers
"""

from tests.utils.assertions import (
    assert_packet_valid,
    assert_signals_equal,
    assert_within_tolerance,
)
from tests.utils.factories import PacketFactory, SignalFactory, WaveformFactory
from tests.utils.mocking import mock_optional_module, mock_rigol_wfm

__all__ = [
    "PacketFactory",
    "SignalFactory",
    "WaveformFactory",
    "assert_packet_valid",
    "assert_signals_equal",
    "assert_within_tolerance",
    "mock_optional_module",
    "mock_rigol_wfm",
]
