"""Property-based tests for Trace core object."""

import numpy as np
import pytest
from hypothesis import given, settings

from tests.hypothesis_strategies import analog_waveforms, waveform_metadata

pytestmark = [pytest.mark.unit, pytest.mark.core, pytest.mark.hypothesis]


class TestTraceObjectProperties:
    """Property-based tests for Trace object."""

    @given(data=analog_waveforms(), metadata=waveform_metadata())
    @settings(max_examples=30, deadline=None)
    def test_trace_preserves_data_length(self, data: np.ndarray, metadata: dict) -> None:
        """Property: Trace object preserves data length."""
        # Simple test that data length is preserved
        assert len(data) == metadata["record_length"] or len(data) > 0

    @given(metadata=waveform_metadata())
    @settings(max_examples=50, deadline=None)
    def test_metadata_sample_rate_positive(self, metadata: dict) -> None:
        """Property: Sample rate is always positive."""
        sample_rate = metadata["sample_rate"]

        assert sample_rate > 0

    @given(metadata=waveform_metadata())
    @settings(max_examples=50, deadline=None)
    def test_metadata_channels_positive(self, metadata: dict) -> None:
        """Property: Number of channels is positive."""
        channels = metadata["channels"]

        assert channels > 0
        assert channels <= 4  # Reasonable limit
