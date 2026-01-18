"""Property-based tests for binary data loading."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = [pytest.mark.unit, pytest.mark.loader, pytest.mark.hypothesis]


class TestBinaryLoadingProperties:
    """Property-based tests for binary data loading."""

    @given(
        data_length=st.integers(min_value=1, max_value=10000),
        dtype=st.sampled_from([np.uint8, np.uint16, np.uint32, np.int8, np.int16]),
    )
    @settings(max_examples=50, deadline=None)
    def test_binary_data_roundtrip(self, data_length: int, dtype: type) -> None:
        """Property: Binary data survives save/load roundtrip."""
        rng = np.random.default_rng(42)
        original_data = rng.integers(0, 256, data_length, dtype=np.uint8).astype(dtype)

        # Convert to bytes
        data_bytes = original_data.tobytes()

        # Load back
        loaded_data = np.frombuffer(data_bytes, dtype=dtype)

        assert np.array_equal(original_data, loaded_data)

    @given(
        data_length=st.integers(min_value=100, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_byte_order_preserved(self, data_length: int) -> None:
        """Property: Byte order is preserved in binary data."""
        rng = np.random.default_rng(42)
        original_data = rng.integers(0, 256, data_length, dtype=np.uint8)

        # Convert to bytes and back
        data_bytes = original_data.tobytes()
        loaded_data = np.frombuffer(data_bytes, dtype=np.uint8)

        # Order should be preserved
        assert np.array_equal(original_data, loaded_data)
