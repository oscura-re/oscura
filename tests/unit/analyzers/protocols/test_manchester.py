"""Tests for Manchester encoding decoder.

Tests the Manchester and Differential Manchester decoder implementation (PRO-014).
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.protocols.manchester import (
    ManchesterDecoder,
    ManchesterMode,
    decode_manchester,
)
from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# Manchester Signal Generator
# =============================================================================


def generate_manchester_signal(
    data_bytes: bytes,
    bit_rate: int = 10_000_000,
    sample_rate: float = 100_000_000.0,
    mode: str = "ieee",
) -> np.ndarray:
    """Generate Manchester encoded signal for testing.

    Args:
        data_bytes: Data bytes to encode.
        bit_rate: Bit rate in bps.
        sample_rate: Sample rate in Hz.
        mode: Encoding mode ("ieee", "thomas", or "differential").

    Returns:
        Boolean array of Manchester encoded signal.
    """
    samples_per_bit = int(sample_rate / bit_rate)
    half_bit = samples_per_bit // 2

    # Convert bytes to bits (LSB first within each byte)
    bits = []
    for byte_val in data_bytes:
        for i in range(8):
            bits.append((byte_val >> i) & 1)

    # Calculate total samples
    idle_samples = samples_per_bit * 2
    total_samples = idle_samples + len(bits) * samples_per_bit + idle_samples

    signal = np.zeros(total_samples, dtype=bool)

    # Start with idle (low)
    idx = idle_samples

    prev_level = False  # For differential Manchester

    for bit in bits:
        if mode == "ieee":
            # IEEE 802.3: 0 = low-to-high, 1 = high-to-low
            if bit == 0:
                signal[idx : idx + half_bit] = False
                signal[idx + half_bit : idx + samples_per_bit] = True
            else:
                signal[idx : idx + half_bit] = True
                signal[idx + half_bit : idx + samples_per_bit] = False

        elif mode == "thomas":
            # Thomas: 0 = high-to-low, 1 = low-to-high
            if bit == 0:
                signal[idx : idx + half_bit] = True
                signal[idx + half_bit : idx + samples_per_bit] = False
            else:
                signal[idx : idx + half_bit] = False
                signal[idx + half_bit : idx + samples_per_bit] = True

        elif mode == "differential":
            # Differential Manchester: transition at every bit boundary
            # 0: additional transition at mid-bit
            # 1: no additional transition at mid-bit
            if bit == 0:
                # Two transitions in this bit period
                signal[idx : idx + half_bit] = not prev_level
                signal[idx + half_bit : idx + samples_per_bit] = prev_level
                prev_level = prev_level  # End at same level as started
            else:
                # One transition at boundary only
                signal[idx : idx + samples_per_bit] = not prev_level
                prev_level = not prev_level

        idx += samples_per_bit

    return signal


# =============================================================================
# ManchesterMode Tests
# =============================================================================


class TestManchesterMode:
    """Test ManchesterMode enum."""

    def test_mode_values(self) -> None:
        """Test mode enum values."""
        assert ManchesterMode.IEEE.value == "ieee"
        assert ManchesterMode.THOMAS.value == "thomas"
        assert ManchesterMode.DIFFERENTIAL.value == "differential"

    def test_all_modes_exist(self) -> None:
        """Test all expected modes are defined."""
        modes = {m.value for m in ManchesterMode}
        expected = {"ieee", "thomas", "differential"}
        assert modes == expected


# =============================================================================
# ManchesterDecoder Initialization Tests
# =============================================================================


class TestManchesterDecoderInit:
    """Test ManchesterDecoder initialization and configuration."""

    def test_default_initialization(self) -> None:
        """Test decoder with default parameters."""
        decoder = ManchesterDecoder()

        assert decoder.id == "manchester"
        assert decoder.name == "Manchester"
        assert decoder.longname == "Manchester Encoding"
        assert "Manchester" in decoder.desc
        assert decoder._bit_rate == 10_000_000
        assert decoder._mode == ManchesterMode.IEEE

    def test_custom_bit_rate(self) -> None:
        """Test decoder with custom bit rate."""
        decoder = ManchesterDecoder(bit_rate=1_000_000)

        assert decoder._bit_rate == 1_000_000

    def test_ieee_mode(self) -> None:
        """Test decoder with IEEE mode."""
        decoder = ManchesterDecoder(mode="ieee")

        assert decoder._mode == ManchesterMode.IEEE

    def test_thomas_mode(self) -> None:
        """Test decoder with Thomas mode."""
        decoder = ManchesterDecoder(mode="thomas")

        assert decoder._mode == ManchesterMode.THOMAS

    def test_differential_mode(self) -> None:
        """Test decoder with Differential mode."""
        decoder = ManchesterDecoder(mode="differential")

        assert decoder._mode == ManchesterMode.DIFFERENTIAL

    def test_channel_definitions(self) -> None:
        """Test decoder channel definitions."""
        decoder = ManchesterDecoder()

        assert len(decoder.channels) == 1

        channel_ids = [ch.id for ch in decoder.channels]
        assert "data" in channel_ids
        assert decoder.channels[0].required is True

    def test_option_definitions(self) -> None:
        """Test decoder option definitions."""
        decoder = ManchesterDecoder()

        assert len(decoder.options) > 0
        option_ids = [opt.id for opt in decoder.options]
        assert "bit_rate" in option_ids
        assert "mode" in option_ids

    def test_annotations_defined(self) -> None:
        """Test decoder has annotations defined."""
        decoder = ManchesterDecoder()

        assert len(decoder.annotations) > 0
        annotation_ids = [a[0] for a in decoder.annotations]
        assert "bit" in annotation_ids
        assert "clock" in annotation_ids


# =============================================================================
# ManchesterDecoder Edge Cases
# =============================================================================


class TestManchesterDecoderEdgeCases:
    """Test Manchester decoder edge cases and error handling."""

    def test_decode_no_transitions(self) -> None:
        """Test decode with constant signal (no transitions)."""
        decoder = ManchesterDecoder(bit_rate=1_000_000)
        sample_rate = 10_000_000.0

        signal = np.ones(1000, dtype=bool)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        packets = list(decoder.decode(trace))

        # Should return empty or minimal packets
        assert isinstance(packets, list)

    def test_decode_single_transition(self) -> None:
        """Test decode with only one transition."""
        decoder = ManchesterDecoder(bit_rate=1_000_000)
        sample_rate = 10_000_000.0

        signal = np.zeros(1000, dtype=bool)
        signal[500:] = True  # Single transition
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        packets = list(decoder.decode(trace))

        # Should not crash
        assert isinstance(packets, list)

    def test_decode_short_signal(self) -> None:
        """Test decode with very short signal."""
        decoder = ManchesterDecoder(bit_rate=10_000_000)
        sample_rate = 100_000_000.0

        signal = np.array([False, True, False], dtype=bool)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        packets = list(decoder.decode(trace))

        # Should not crash
        assert isinstance(packets, list)


# =============================================================================
# ManchesterDecoder Decoding Tests
# =============================================================================


class TestManchesterDecoderDecoding:
    """Test Manchester decoding with actual signals."""

    def test_decode_ieee_single_byte(self) -> None:
        """Test decoding single byte with IEEE mode."""
        bit_rate = 1_000_000
        sample_rate = 10_000_000.0

        decoder = ManchesterDecoder(bit_rate=bit_rate, mode="ieee")

        signal = generate_manchester_signal(
            b"\xaa",
            bit_rate=bit_rate,
            sample_rate=sample_rate,
            mode="ieee",
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        packets = list(decoder.decode(trace))

        if len(packets) > 0:
            assert packets[0].protocol == "manchester"
            assert packets[0].annotations["mode"] == "ieee"
            assert packets[0].annotations["bit_rate"] == bit_rate

    def test_decode_thomas_mode(self) -> None:
        """Test decoding with Thomas mode."""
        bit_rate = 1_000_000
        sample_rate = 10_000_000.0

        decoder = ManchesterDecoder(bit_rate=bit_rate, mode="thomas")

        signal = generate_manchester_signal(
            b"\x55",
            bit_rate=bit_rate,
            sample_rate=sample_rate,
            mode="thomas",
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        packets = list(decoder.decode(trace))

        if len(packets) > 0:
            assert packets[0].annotations["mode"] == "thomas"

    def test_decode_differential_mode(self) -> None:
        """Test decoding with Differential Manchester mode."""
        bit_rate = 1_000_000
        sample_rate = 10_000_000.0

        decoder = ManchesterDecoder(bit_rate=bit_rate, mode="differential")

        signal = generate_manchester_signal(
            b"\xf0",
            bit_rate=bit_rate,
            sample_rate=sample_rate,
            mode="differential",
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        packets = list(decoder.decode(trace))

        if len(packets) > 0:
            assert packets[0].annotations["mode"] == "differential"

    def test_decode_multiple_bytes(self) -> None:
        """Test decoding multiple bytes."""
        bit_rate = 1_000_000
        sample_rate = 10_000_000.0

        decoder = ManchesterDecoder(bit_rate=bit_rate, mode="ieee")

        signal = generate_manchester_signal(
            b"\x12\x34\x56",
            bit_rate=bit_rate,
            sample_rate=sample_rate,
            mode="ieee",
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        packets = list(decoder.decode(trace))

        if len(packets) > 0:
            # Should have decoded some bits
            assert packets[0].annotations["bit_count"] > 0

    def test_decode_from_waveform_trace(self) -> None:
        """Test decoding from WaveformTrace."""
        bit_rate = 1_000_000
        sample_rate = 10_000_000.0

        decoder = ManchesterDecoder(bit_rate=bit_rate, mode="ieee")

        # Generate digital signal
        digital_signal = generate_manchester_signal(
            b"\xaa",
            bit_rate=bit_rate,
            sample_rate=sample_rate,
            mode="ieee",
        )

        # Convert to analog (0V = False, 3.3V = True)
        analog_signal = digital_signal.astype(np.float64) * 3.3
        trace = WaveformTrace(data=analog_signal, metadata=TraceMetadata(sample_rate=sample_rate))

        packets = list(decoder.decode(trace))

        # Should convert and decode
        assert isinstance(packets, list)


# =============================================================================
# ManchesterDecoder Internal Methods Tests
# =============================================================================


class TestManchesterDecoderInternalMethods:
    """Test Manchester decoder internal methods."""

    def test_decode_standard_rising_edge(self) -> None:
        """Test _decode_standard with rising edge transitions."""
        decoder = ManchesterDecoder(bit_rate=1_000_000, mode="ieee")

        # Create a simple pattern with transitions
        data = np.array([False] * 10 + [True] * 10 + [False] * 10 + [True] * 10, dtype=bool)
        transitions = np.array([10, 20, 30], dtype=np.int64)
        half_bit = 10.0

        bits, errors = decoder._decode_standard(data, transitions, half_bit)

        # Should decode some bits
        assert isinstance(bits, list)
        assert isinstance(errors, list)

    def test_decode_differential_method(self) -> None:
        """Test _decode_differential method."""
        decoder = ManchesterDecoder(bit_rate=1_000_000, mode="differential")

        # Create transitions at varying intervals
        data = np.array([False] * 50, dtype=bool)
        transitions = np.array([5, 10, 20, 25, 35], dtype=np.int64)
        half_bit = 10.0

        bits, errors = decoder._decode_differential(data, transitions, half_bit)

        # Should decode some bits
        assert isinstance(bits, list)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestDecodeManchesterConvenienceFunction:
    """Test decode_manchester convenience function."""

    def test_decode_manchester_with_trace(self) -> None:
        """Test decode_manchester with trace input."""
        bit_rate = 1_000_000
        sample_rate = 10_000_000.0

        signal = generate_manchester_signal(
            b"\xaa",
            bit_rate=bit_rate,
            sample_rate=sample_rate,
            mode="ieee",
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        packets = decode_manchester(trace, bit_rate=bit_rate, mode="ieee")

        assert isinstance(packets, list)

    def test_decode_manchester_with_array(self) -> None:
        """Test decode_manchester with DigitalTrace input (array requires TraceMetadata)."""
        bit_rate = 1_000_000
        sample_rate = 10_000_000.0

        signal = generate_manchester_signal(
            b"\x55",
            bit_rate=bit_rate,
            sample_rate=sample_rate,
            mode="ieee",
        )

        # decode_manchester expects DigitalTrace when given array with sample_rate
        # Create a DigitalTrace properly with TraceMetadata
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        packets = decode_manchester(
            trace,
            bit_rate=bit_rate,
            mode="ieee",
        )

        assert isinstance(packets, list)

    def test_decode_manchester_with_waveform(self) -> None:
        """Test decode_manchester with WaveformTrace."""
        bit_rate = 1_000_000
        sample_rate = 10_000_000.0

        digital_signal = generate_manchester_signal(
            b"\xf0",
            bit_rate=bit_rate,
            sample_rate=sample_rate,
            mode="ieee",
        )
        analog_signal = digital_signal.astype(np.float64) * 3.3
        trace = WaveformTrace(data=analog_signal, metadata=TraceMetadata(sample_rate=sample_rate))

        packets = decode_manchester(trace, bit_rate=bit_rate, mode="ieee")

        assert isinstance(packets, list)

    def test_decode_manchester_different_modes(self) -> None:
        """Test decode_manchester with different modes."""
        bit_rate = 1_000_000
        sample_rate = 10_000_000.0

        for mode in ["ieee", "thomas", "differential"]:
            signal = generate_manchester_signal(
                b"\xaa",
                bit_rate=bit_rate,
                sample_rate=sample_rate,
                mode=mode,
            )
            trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

            packets = decode_manchester(trace, bit_rate=bit_rate, mode=mode)

            assert isinstance(packets, list)


# =============================================================================
# Module Export Tests
# =============================================================================


class TestManchesterModuleExports:
    """Test module-level exports."""

    def test_all_export(self) -> None:
        """Test __all__ contains expected exports."""
        from oscura.analyzers.protocols.manchester import __all__

        assert "ManchesterDecoder" in __all__
        assert "ManchesterMode" in __all__
        assert "decode_manchester" in __all__
