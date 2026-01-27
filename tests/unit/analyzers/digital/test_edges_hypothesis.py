"""Property-based tests for edge detection in digital signals."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.hypothesis_strategies import digital_signals, edge_lists

pytestmark = [pytest.mark.unit, pytest.mark.digital, pytest.mark.hypothesis]


class TestEdgeDetectionProperties:
    """Property-based tests for edge detection."""

    @given(signal=digital_signals(min_length=100, max_length=1000))
    @settings(max_examples=50, deadline=None)
    def test_detected_edges_within_signal_length(self, signal: np.ndarray) -> None:
        """Property: All detected edge positions are within signal bounds."""
        try:
            from oscura.analyzers.digital.edges import detect_edges
        except ImportError:
            # SKIP: Valid - Optional Numba-accelerated edge detection
            # Only skip if numba not installed (pip install oscura[performance])
            # SKIP: Valid - Optional Numba-accelerated edge detection
            # Only skip if numba not installed (pip install oscura[performance])
            pytest.skip("edges module not available")

        edges = detect_edges(signal, threshold=1.65)

        # All edge positions should be valid indices
        if len(edges) > 0:
            edge_indices = np.array([edge.sample_index for edge in edges])
            assert np.all(edge_indices >= 0)
            assert np.all(edge_indices < len(signal))

    @given(edges=edge_lists())
    @settings(max_examples=50, deadline=None)
    def test_edge_list_sorted(self, edges: np.ndarray) -> None:
        """Property: Edge timestamps are in sorted order."""
        # Check if sorted
        assert np.all(edges[:-1] <= edges[1:])

    @given(
        num_transitions=st.integers(min_value=2, max_value=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_alternating_signal_edges_detected(self, num_transitions: int) -> None:
        """Property: Alternating signal has edges at transitions."""
        # Create perfect alternating signal
        samples_per_state = 10
        signal_parts = []
        for i in range(num_transitions):
            level = 0.0 if i % 2 == 0 else 3.3
            signal_parts.append(np.full(samples_per_state, level))

        signal = np.concatenate(signal_parts)

        try:
            from oscura.analyzers.digital.edges import detect_edges
        except ImportError:
            # SKIP: Valid - Optional Numba-accelerated edge detection
            # Only skip if numba not installed (pip install oscura[performance])
            # SKIP: Valid - Optional Numba-accelerated edge detection
            # Only skip if numba not installed (pip install oscura[performance])
            pytest.skip("edges module not available")

        edges = detect_edges(signal, threshold=1.65)

        # Should detect approximately num_transitions - 1 edges
        # (transitions between states)
        assert len(edges) >= num_transitions // 2
