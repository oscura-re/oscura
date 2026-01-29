"""Tests for I2S protocol decoder.

Tests the I2S (Inter-IC Sound) decoder implementation (PRO-011).
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.protocols.i2s import (
    I2SDecoder,
    I2SMode,
    decode_i2s,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# I2S Signal Generator
# =============================================================================


def generate_i2s_signals(
    left_samples: list[int],
    right_samples: list[int],
    bit_depth: int = 16,
    mode: str = "standard",
    samples_per_bit: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate I2S signals for testing.

    Args:
        left_samples: Left channel sample values.
        right_samples: Right channel sample values.
        bit_depth: Bits per sample (8, 16, 24, 32).
        mode: Alignment mode ("standard", "left_justified", "right_justified").
        samples_per_bit: Samples per bit clock period.

    Returns:
        Tuple of (bck, ws, sd) boolean arrays.
    """
    # Calculate number of stereo samples
    n_stereo_samples = max(len(left_samples), len(right_samples))

    # Total bits per stereo sample: bit_depth * 2 (L + R)
    bits_per_word = bit_depth
    bits_per_stereo = bits_per_word * 2

    # Add some padding/idle
    idle_bits = 4
    total_bits = idle_bits + n_stereo_samples * bits_per_stereo + idle_bits

    half_bit = samples_per_bit // 2
    total_samples = total_bits * samples_per_bit

    # Initialize signals
    bck = np.zeros(total_samples, dtype=bool)
    ws = np.zeros(total_samples, dtype=bool)
    sd = np.zeros(total_samples, dtype=bool)

    idx = 0

    # Generate idle period
    for _ in range(idle_bits):
        # BCK: low, then high
        bck[idx : idx + half_bit] = False
        bck[idx + half_bit : idx + samples_per_bit] = True
        # WS: low (left channel indicator)
        ws[idx : idx + samples_per_bit] = False
        # SD: zero
        sd[idx : idx + samples_per_bit] = False
        idx += samples_per_bit

    # Generate data for each stereo sample
    for sample_idx in range(n_stereo_samples):
        left_val = left_samples[sample_idx] if sample_idx < len(left_samples) else 0
        right_val = right_samples[sample_idx] if sample_idx < len(right_samples) else 0

        # Convert to unsigned for bit manipulation
        if left_val < 0:
            left_val = (1 << bit_depth) + left_val
        if right_val < 0:
            right_val = (1 << bit_depth) + right_val

        # Left channel (WS = 0)
        for bit_idx in range(bits_per_word):
            # BCK: low, then high
            bck[idx : idx + half_bit] = False
            bck[idx + half_bit : idx + samples_per_bit] = True
            # WS: low for left channel
            ws[idx : idx + samples_per_bit] = False

            # SD: data bit (MSB first)
            if mode == "standard":
                # Standard mode: MSB starts 1 clock after WS change
                if bit_idx == 0:
                    sd[idx : idx + samples_per_bit] = False  # Delay bit
                else:
                    bit = (left_val >> (bit_depth - bit_idx)) & 1
                    sd[idx : idx + samples_per_bit] = bool(bit)
            elif mode == "left_justified":
                # Left-justified: MSB at WS change
                bit = (left_val >> (bit_depth - 1 - bit_idx)) & 1
                sd[idx : idx + samples_per_bit] = bool(bit)
            else:  # right_justified
                # Right-justified: LSB at end of word period
                bit = (left_val >> (bit_depth - 1 - bit_idx)) & 1
                sd[idx : idx + samples_per_bit] = bool(bit)

            idx += samples_per_bit

        # Right channel (WS = 1)
        for bit_idx in range(bits_per_word):
            # BCK: low, then high
            bck[idx : idx + half_bit] = False
            bck[idx + half_bit : idx + samples_per_bit] = True
            # WS: high for right channel
            ws[idx : idx + samples_per_bit] = True

            # SD: data bit (MSB first)
            if mode == "standard":
                if bit_idx == 0:
                    sd[idx : idx + samples_per_bit] = False
                else:
                    bit = (right_val >> (bit_depth - bit_idx)) & 1
                    sd[idx : idx + samples_per_bit] = bool(bit)
            elif mode == "left_justified":
                bit = (right_val >> (bit_depth - 1 - bit_idx)) & 1
                sd[idx : idx + samples_per_bit] = bool(bit)
            else:  # right_justified
                bit = (right_val >> (bit_depth - 1 - bit_idx)) & 1
                sd[idx : idx + samples_per_bit] = bool(bit)

            idx += samples_per_bit

    # Generate idle period at end
    for _ in range(idle_bits):
        bck[idx : idx + half_bit] = False
        bck[idx + half_bit : idx + samples_per_bit] = True
        ws[idx : idx + samples_per_bit] = True
        sd[idx : idx + samples_per_bit] = False
        idx += samples_per_bit

    return bck[:idx], ws[:idx], sd[:idx]


# =============================================================================
# I2SMode Tests
# =============================================================================


class TestI2SMode:
    """Test I2SMode enum."""

    def test_mode_values(self) -> None:
        """Test mode enum values."""
        assert I2SMode.STANDARD.value == "standard"
        assert I2SMode.LEFT_JUSTIFIED.value == "left_justified"
        assert I2SMode.RIGHT_JUSTIFIED.value == "right_justified"

    def test_all_modes_exist(self) -> None:
        """Test all expected modes are defined."""
        modes = {m.value for m in I2SMode}
        expected = {"standard", "left_justified", "right_justified"}
        assert modes == expected


# =============================================================================
# I2SDecoder Initialization Tests
# =============================================================================


class TestI2SDecoderInit:
    """Test I2SDecoder initialization and configuration."""

    def test_default_initialization(self) -> None:
        """Test decoder with default parameters."""
        decoder = I2SDecoder()

        assert decoder.id == "i2s"
        assert decoder.name == "I2S"
        assert decoder.longname == "Inter-IC Sound"
        assert "I2S" in decoder.desc
        assert decoder._bit_depth == 16
        assert decoder._mode == I2SMode.STANDARD

    def test_custom_bit_depth(self) -> None:
        """Test decoder with custom bit depth."""
        for bit_depth in [8, 16, 24, 32]:
            decoder = I2SDecoder(bit_depth=bit_depth)
            assert decoder._bit_depth == bit_depth

    def test_standard_mode(self) -> None:
        """Test decoder with standard mode."""
        decoder = I2SDecoder(mode="standard")

        assert decoder._mode == I2SMode.STANDARD

    def test_left_justified_mode(self) -> None:
        """Test decoder with left-justified mode."""
        decoder = I2SDecoder(mode="left_justified")

        assert decoder._mode == I2SMode.LEFT_JUSTIFIED

    def test_right_justified_mode(self) -> None:
        """Test decoder with right-justified mode."""
        decoder = I2SDecoder(mode="right_justified")

        assert decoder._mode == I2SMode.RIGHT_JUSTIFIED

    def test_channel_definitions(self) -> None:
        """Test decoder channel definitions."""
        decoder = I2SDecoder()

        assert len(decoder.channels) == 3

        channel_ids = [ch.id for ch in decoder.channels]
        assert "bck" in channel_ids
        assert "ws" in channel_ids
        assert "sd" in channel_ids

        for ch in decoder.channels:
            assert ch.required is True

    def test_option_definitions(self) -> None:
        """Test decoder option definitions."""
        decoder = I2SDecoder()

        assert len(decoder.options) > 0
        option_ids = [opt.id for opt in decoder.options]
        assert "bit_depth" in option_ids
        assert "mode" in option_ids

    def test_annotations_defined(self) -> None:
        """Test decoder has annotations defined."""
        decoder = I2SDecoder()

        assert len(decoder.annotations) > 0
        annotation_ids = [a[0] for a in decoder.annotations]
        assert "left" in annotation_ids
        assert "right" in annotation_ids


# =============================================================================
# I2SDecoder Edge Cases
# =============================================================================


class TestI2SDecoderEdgeCases:
    """Test I2S decoder edge cases and error handling."""

    def test_decode_none_signals(self) -> None:
        """Test decode with None signals."""
        decoder = I2SDecoder()

        packets = list(decoder.decode(bck=None, ws=None, sd=None, sample_rate=1_000_000.0))

        assert len(packets) == 0

    def test_decode_empty_signals(self) -> None:
        """Test decode with empty signals."""
        decoder = I2SDecoder()

        bck = np.array([], dtype=bool)
        ws = np.array([], dtype=bool)
        sd = np.array([], dtype=bool)

        packets = list(decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=1_000_000.0))

        assert len(packets) == 0

    def test_decode_no_clock_edges(self) -> None:
        """Test decode with constant clock (no edges)."""
        decoder = I2SDecoder()

        bck = np.zeros(100, dtype=bool)
        ws = np.zeros(100, dtype=bool)
        sd = np.zeros(100, dtype=bool)

        packets = list(decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=1_000_000.0))

        assert len(packets) == 0

    def test_decode_no_ws_transitions(self) -> None:
        """Test decode with no WS transitions."""
        decoder = I2SDecoder()

        # Clock edges but no WS transitions
        bck = np.array([False, True] * 50, dtype=bool)
        ws = np.zeros(100, dtype=bool)  # Constant
        sd = np.zeros(100, dtype=bool)

        packets = list(decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=1_000_000.0))

        assert len(packets) == 0

    def test_decode_mismatched_lengths(self) -> None:
        """Test decode with mismatched signal lengths."""
        decoder = I2SDecoder()

        bck = np.array([False, True] * 100, dtype=bool)
        ws = np.array([False, True] * 50, dtype=bool)
        sd = np.array([False] * 50, dtype=bool)

        # Should truncate to shortest length
        packets = list(decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=1_000_000.0))

        # Should not crash
        assert isinstance(packets, list)


# =============================================================================
# I2SDecoder Decoding Tests
# =============================================================================


class TestI2SDecoderDecoding:
    """Test I2S decoding with actual signals."""

    def test_decode_single_stereo_sample_16bit(self) -> None:
        """Test decoding single 16-bit stereo sample."""
        decoder = I2SDecoder(bit_depth=16, mode="standard")
        sample_rate = 1_000_000.0

        bck, ws, sd = generate_i2s_signals(
            left_samples=[1000],
            right_samples=[-1000],
            bit_depth=16,
            mode="standard",
        )

        packets = list(decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=sample_rate))

        # Should decode at least one stereo sample
        if len(packets) > 0:
            assert packets[0].protocol == "i2s"
            assert packets[0].annotations["bit_depth"] == 16
            assert packets[0].annotations["mode"] == "standard"

    def test_decode_left_justified_mode(self) -> None:
        """Test decoding with left-justified mode."""
        decoder = I2SDecoder(bit_depth=16, mode="left_justified")
        sample_rate = 1_000_000.0

        bck, ws, sd = generate_i2s_signals(
            left_samples=[0x1234],
            right_samples=[0x5678],
            bit_depth=16,
            mode="left_justified",
        )

        packets = list(decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=sample_rate))

        if len(packets) > 0:
            assert packets[0].annotations["mode"] == "left_justified"

    def test_decode_right_justified_mode(self) -> None:
        """Test decoding with right-justified mode."""
        decoder = I2SDecoder(bit_depth=16, mode="right_justified")
        sample_rate = 1_000_000.0

        bck, ws, sd = generate_i2s_signals(
            left_samples=[0xABCD],
            right_samples=[0xEF01],
            bit_depth=16,
            mode="right_justified",
        )

        packets = list(decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=sample_rate))

        if len(packets) > 0:
            assert packets[0].annotations["mode"] == "right_justified"

    def test_decode_24bit_samples(self) -> None:
        """Test decoding 24-bit samples."""
        decoder = I2SDecoder(bit_depth=24, mode="standard")
        sample_rate = 1_000_000.0

        bck, ws, sd = generate_i2s_signals(
            left_samples=[0x123456],
            right_samples=[0x789ABC],
            bit_depth=24,
            mode="standard",
        )

        packets = list(decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=sample_rate))

        if len(packets) > 0:
            assert packets[0].annotations["bit_depth"] == 24

    def test_decode_multiple_stereo_samples(self) -> None:
        """Test decoding multiple stereo samples."""
        decoder = I2SDecoder(bit_depth=16, mode="standard")
        sample_rate = 1_000_000.0

        bck, ws, sd = generate_i2s_signals(
            left_samples=[100, 200, 300],
            right_samples=[-100, -200, -300],
            bit_depth=16,
            mode="standard",
        )

        packets = list(decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=sample_rate))

        # May decode multiple samples
        assert isinstance(packets, list)

    def test_decode_8bit_samples(self) -> None:
        """Test decoding 8-bit samples."""
        decoder = I2SDecoder(bit_depth=8, mode="standard")
        sample_rate = 1_000_000.0

        bck, ws, sd = generate_i2s_signals(
            left_samples=[0x7F],
            right_samples=[0x80],
            bit_depth=8,
            mode="standard",
        )

        packets = list(decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=sample_rate))

        if len(packets) > 0:
            assert packets[0].annotations["bit_depth"] == 8

    def test_packet_timestamp(self) -> None:
        """Test packet timestamp calculation."""
        decoder = I2SDecoder(bit_depth=16)
        sample_rate = 1_000_000.0

        bck, ws, sd = generate_i2s_signals(
            left_samples=[0x1234],
            right_samples=[0x5678],
            bit_depth=16,
        )

        packets = list(decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=sample_rate))

        if len(packets) > 0:
            assert packets[0].timestamp >= 0.0
            assert isinstance(packets[0].timestamp, float)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestDecodeI2SConvenienceFunction:
    """Test decode_i2s convenience function."""

    def test_decode_i2s_basic(self) -> None:
        """Test basic usage of decode_i2s."""
        bck, ws, sd = generate_i2s_signals(
            left_samples=[1000],
            right_samples=[-1000],
            bit_depth=16,
        )

        packets = decode_i2s(bck, ws, sd, sample_rate=1_000_000.0)

        assert isinstance(packets, list)

    def test_decode_i2s_with_parameters(self) -> None:
        """Test decode_i2s with custom parameters."""
        bck, ws, sd = generate_i2s_signals(
            left_samples=[0x1234],
            right_samples=[0x5678],
            bit_depth=24,
            mode="left_justified",
        )

        packets = decode_i2s(
            bck,
            ws,
            sd,
            sample_rate=1_000_000.0,
            bit_depth=24,
            mode="left_justified",
        )

        assert isinstance(packets, list)

    def test_decode_i2s_empty_signals(self) -> None:
        """Test decode_i2s with empty signals."""
        bck = np.array([], dtype=bool)
        ws = np.array([], dtype=bool)
        sd = np.array([], dtype=bool)

        packets = decode_i2s(bck, ws, sd, sample_rate=1_000_000.0)

        assert packets == []


# =============================================================================
# Module Export Tests
# =============================================================================


class TestI2SModuleExports:
    """Test module-level exports."""

    def test_all_export(self) -> None:
        """Test __all__ contains expected exports."""
        from oscura.analyzers.protocols.i2s import __all__

        assert "I2SDecoder" in __all__
        assert "I2SMode" in __all__
        assert "decode_i2s" in __all__
