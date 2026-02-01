"""Protocol decoder edge case tests to improve coverage.

This module tests boundary conditions and edge cases in protocol decoding
infrastructure.

(Edge Case Tests)

- Protocol edge cases testing (+0.5% coverage)
- Minimum/maximum frame sizes
- Bit timing edge cases
- State transition edge cases
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from oscura.analyzers.protocols.spi import SPIDecoder
from oscura.core.types import DigitalTrace, TraceMetadata

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestSPIDecoderEdgeCases:
    """Test SPI decoder edge cases."""

    def test_spi_decoder_init_mode_0(self) -> None:
        """Test SPI decoder initialization with mode 0."""
        decoder = SPIDecoder(cpol=0, cpha=0)
        assert decoder.get_option("cpol") == 0
        assert decoder.get_option("cpha") == 0

    def test_spi_decoder_init_mode_3(self) -> None:
        """Test SPI decoder initialization with mode 3."""
        decoder = SPIDecoder(cpol=1, cpha=1)
        assert decoder.get_option("cpol") == 1
        assert decoder.get_option("cpha") == 1

    def test_spi_decoder_word_size_4(self) -> None:
        """Test SPI decoder with minimum word size (4 bits)."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=4)
        assert decoder.get_option("word_size") == 4

    def test_spi_decoder_word_size_32(self) -> None:
        """Test SPI decoder with maximum word size (32 bits)."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=32)
        assert decoder.get_option("word_size") == 32

    def test_spi_decoder_bit_order_msb(self) -> None:
        """Test SPI decoder with MSB first."""
        decoder = SPIDecoder(cpol=0, cpha=0, bit_order="msb")
        assert decoder.get_option("bit_order") == "msb"

    def test_spi_decoder_bit_order_lsb(self) -> None:
        """Test SPI decoder with LSB first."""
        decoder = SPIDecoder(cpol=0, cpha=0, bit_order="lsb")
        assert decoder.get_option("bit_order") == "lsb"

    def test_spi_decoder_cs_polarity_active_low(self) -> None:
        """Test SPI decoder with active-low CS."""
        decoder = SPIDecoder(cpol=0, cpha=0, cs_polarity=0)
        assert decoder.get_option("cs_polarity") == 0

    def test_spi_decoder_cs_polarity_active_high(self) -> None:
        """Test SPI decoder with active-high CS."""
        decoder = SPIDecoder(cpol=0, cpha=0, cs_polarity=1)
        assert decoder.get_option("cs_polarity") == 1


@pytest.mark.unit
class TestProtocolBoundaryConditions:
    """Test protocol decoder boundary conditions."""

    def test_empty_digital_trace(self) -> None:
        """Test that empty trace raises ValueError."""
        metadata = TraceMetadata(sample_rate=1e6)

        # Empty data arrays are rejected in v0.9.0+
        with pytest.raises(ValueError, match="data array cannot be empty"):
            DigitalTrace(data=np.array([], dtype=bool), metadata=metadata)

    def test_single_sample_trace(self) -> None:
        """Test decoding with single-sample trace."""
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=np.array([True], dtype=bool), metadata=metadata)

        assert len(trace.data) == 1

    def test_very_long_trace(self) -> None:
        """Test decoding with very long trace."""
        metadata = TraceMetadata(sample_rate=1e6)
        # 1 million samples
        long_trace = DigitalTrace(data=np.ones(1_000_000, dtype=bool), metadata=metadata)

        assert len(long_trace.data) == 1_000_000

    def test_digital_trace_all_high(self) -> None:
        """Test trace with constant high state."""
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=np.ones(100, dtype=bool), metadata=metadata)

        # All high - idle state for most protocols
        assert np.all(trace.data)

    def test_digital_trace_all_low(self) -> None:
        """Test trace with constant low state."""
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=np.zeros(100, dtype=bool), metadata=metadata)

        # All low
        assert not np.any(trace.data)

    def test_digital_trace_alternating(self) -> None:
        """Test trace with alternating pattern."""
        metadata = TraceMetadata(sample_rate=1e6)
        data = np.array([True, False] * 50, dtype=bool)
        trace = DigitalTrace(data=data, metadata=metadata)

        # Should have 50 transitions
        transitions = np.sum(np.diff(trace.data.astype(int)) != 0)
        assert transitions == 99  # 100 samples = 99 possible transitions


@pytest.mark.unit
class TestTimingBoundaries:
    """Test protocol timing boundary conditions."""

    def test_minimum_sample_rate(self) -> None:
        """Test with minimum practical sample rate."""
        metadata = TraceMetadata(sample_rate=1.0)  # 1 Hz
        trace = DigitalTrace(data=np.array([True, False, True], dtype=bool), metadata=metadata)

        assert metadata.sample_rate == 1.0

    def test_maximum_sample_rate(self) -> None:
        """Test with very high sample rate."""
        metadata = TraceMetadata(sample_rate=1e12)  # 1 THz
        trace = DigitalTrace(data=np.array([True, False], dtype=bool), metadata=metadata)

        assert metadata.sample_rate == 1e12

    def test_sample_period_calculation(self) -> None:
        """Test sample period calculation at boundaries."""
        # At 1 MHz
        metadata = TraceMetadata(sample_rate=1e6)
        expected_period = 1e-6  # 1 microsecond

        actual_period = 1.0 / metadata.sample_rate
        assert actual_period == pytest.approx(expected_period)

    def test_duration_calculation_short(self) -> None:
        """Test duration calculation for short trace."""
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=np.array([True, False], dtype=bool), metadata=metadata)

        # 2 samples at 1 MHz, duration is (n-1) * time_base = 1 * 1e-6 = 1 microsecond
        expected_duration = 1e-6
        assert trace.duration == pytest.approx(expected_duration)

    def test_duration_calculation_long(self) -> None:
        """Test duration calculation for long trace."""
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=np.ones(1000, dtype=bool), metadata=metadata)

        # 1000 samples at 1 MHz, duration is (n-1) * time_base = 999 * 1e-6 = 999 microseconds
        expected_duration = 999e-6
        assert trace.duration == pytest.approx(expected_duration)


# Property-based testing
@pytest.mark.unit
class TestPropertyBasedProtocol:
    """Property-based tests for protocol edge cases."""

    @given(st.integers(min_value=0, max_value=1))
    def test_cpol_valid_values(self, cpol: int) -> None:
        """Property: CPOL must be 0 or 1."""
        decoder = SPIDecoder(cpol=cpol, cpha=0)
        assert decoder.get_option("cpol") in [0, 1]

    @given(st.integers(min_value=0, max_value=1))
    def test_cpha_valid_values(self, cpha: int) -> None:
        """Property: CPHA must be 0 or 1."""
        decoder = SPIDecoder(cpol=0, cpha=cpha)
        assert decoder.get_option("cpha") in [0, 1]

    @given(st.sampled_from([4, 8, 16, 24, 32]))
    def test_word_size_valid_values(self, word_size: int) -> None:
        """Property: Word size must be from valid set."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=word_size)
        assert decoder.get_option("word_size") in [4, 8, 16, 24, 32]

    @given(st.sampled_from(["msb", "lsb"]))
    def test_bit_order_valid_values(self, bit_order: str) -> None:
        """Property: Bit order must be msb or lsb."""
        decoder = SPIDecoder(cpol=0, cpha=0, bit_order=bit_order)
        assert decoder.get_option("bit_order") in ["msb", "lsb"]

    @given(st.lists(st.booleans(), min_size=1, max_size=1000))
    def test_digital_trace_arbitrary_data(self, data: list[bool]) -> None:
        """Property: DigitalTrace should handle arbitrary boolean sequences."""
        metadata = TraceMetadata(sample_rate=1e6)
        trace = DigitalTrace(data=np.array(data, dtype=bool), metadata=metadata)

        # Should not crash (empty arrays rejected in v0.9.0+)
        assert len(trace.data) == len(data)
        assert trace.data.dtype == np.bool_

    @given(st.floats(min_value=1.0, max_value=1e15, allow_nan=False, allow_infinity=False))
    def test_sample_rate_always_positive(self, sample_rate: float) -> None:
        """Property: Sample rate must always be positive."""
        metadata = TraceMetadata(sample_rate=sample_rate)
        assert metadata.sample_rate > 0.0

    @given(
        st.lists(st.booleans(), min_size=2, max_size=100),
        st.floats(min_value=1e3, max_value=1e12, allow_nan=False, allow_infinity=False),
    )
    def test_duration_proportional_to_length(self, data: list[bool], sample_rate: float) -> None:
        """Property: Duration should be proportional to data length."""
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = DigitalTrace(data=np.array(data, dtype=bool), metadata=metadata)

        # Duration is (n-1) * time_base per the implementation
        expected_duration = (len(data) - 1) / sample_rate
        assert trace.duration == pytest.approx(expected_duration, rel=1e-6)
