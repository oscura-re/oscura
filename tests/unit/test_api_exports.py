"""Tests for TopLevelExports."""

from __future__ import annotations

import pytest

import oscura as osc
from oscura.utils.builders import SignalBuilder

pytestmark = pytest.mark.unit


class TestTopLevelExports:
    """Test that all new APIs are accessible at top level."""

    def test_signal_builder_accessible(self) -> None:
        """Test SignalBuilder is accessible via tk namespace."""
        assert hasattr(osc, "SignalBuilder")
        builder = osc.SignalBuilder()
        assert isinstance(builder, SignalBuilder)

    def test_convenience_functions_accessible(self) -> None:
        """Test convenience functions are accessible."""
        assert hasattr(osc, "quick_spectral")
        assert hasattr(osc, "auto_decode")
        assert hasattr(osc, "smart_filter")

    def test_workflow_accessible(self) -> None:
        """Test reverse_engineer_signal is accessible."""
        assert hasattr(osc, "reverse_engineer_signal")
        assert hasattr(osc.workflows, "reverse_engineer_signal")

    def test_discovery_functions_accessible(self) -> None:
        """Test discovery functions are accessible."""
        assert hasattr(osc, "characterize_signal")
        assert hasattr(osc, "find_anomalies")
        assert hasattr(osc, "assess_data_quality")
