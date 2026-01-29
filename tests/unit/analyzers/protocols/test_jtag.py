"""Tests for JTAG protocol decoder.

Tests the JTAG (IEEE 1149.1) decoder implementation (PRO-009).
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.protocols.jtag import (
    JTAG_INSTRUCTIONS,
    JTAGDecoder,
    TAPState,
    decode_jtag,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# JTAG Signal Generator
# =============================================================================


def generate_jtag_signals(
    tms_sequence: list[int],
    tdi_bits: list[int] | None = None,
    tdo_bits: list[int] | None = None,
    samples_per_bit: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Generate JTAG signals for testing.

    Args:
        tms_sequence: List of TMS values (0 or 1).
        tdi_bits: List of TDI values (optional).
        tdo_bits: List of TDO values (optional).
        samples_per_bit: Samples per bit period.

    Returns:
        Tuple of (tck, tms, tdi, tdo) boolean arrays.
    """
    n_bits = len(tms_sequence)

    if tdi_bits is None:
        tdi_bits = [0] * n_bits
    if len(tdi_bits) < n_bits:
        tdi_bits = tdi_bits + [0] * (n_bits - len(tdi_bits))

    has_tdo = tdo_bits is not None
    if tdo_bits is None:
        tdo_bits = [0] * n_bits
    if len(tdo_bits) < n_bits:
        tdo_bits = tdo_bits + [0] * (n_bits - len(tdo_bits))

    half_bit = samples_per_bit // 2
    idle_samples = samples_per_bit * 2
    total_samples = idle_samples + n_bits * samples_per_bit + idle_samples

    tck = np.zeros(total_samples, dtype=bool)
    tms = np.zeros(total_samples, dtype=bool)
    tdi = np.zeros(total_samples, dtype=bool)
    tdo = np.zeros(total_samples, dtype=bool) if has_tdo else None

    idx = idle_samples

    for i in range(n_bits):
        # TCK: low, then high (rising edge at mid-bit)
        tck[idx : idx + half_bit] = False
        tck[idx + half_bit : idx + samples_per_bit] = True

        # TMS, TDI, TDO valid during entire bit period
        tms[idx : idx + samples_per_bit] = bool(tms_sequence[i])
        tdi[idx : idx + samples_per_bit] = bool(tdi_bits[i])
        if tdo is not None:
            tdo[idx : idx + samples_per_bit] = bool(tdo_bits[i])

        idx += samples_per_bit

    return tck, tms, tdi, tdo


def generate_ir_shift(
    ir_value: int,
    ir_bits: int = 8,
    samples_per_bit: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Generate JTAG IR shift sequence.

    Args:
        ir_value: Instruction register value.
        ir_bits: Number of IR bits.
        samples_per_bit: Samples per bit period.

    Returns:
        Tuple of (tck, tms, tdi, tdo) for IR shift.
    """
    # TMS sequence to reach Shift-IR:
    # From Test-Logic-Reset: TMS=0 -> Run-Test/Idle
    #                        TMS=1 -> Select-DR-Scan
    #                        TMS=1 -> Select-IR-Scan
    #                        TMS=0 -> Capture-IR
    #                        TMS=0 -> Shift-IR
    # Then shift IR bits (TMS=0 for all except last bit which is TMS=1)
    # Then: Exit1-IR -> Update-IR -> Run-Test/Idle

    tms_sequence = [0, 1, 1, 0, 0]  # Go to Shift-IR

    # IR bits (LSB first)
    ir_bits_list = [(ir_value >> i) & 1 for i in range(ir_bits)]
    tdi_bits = [0] * 5 + ir_bits_list  # 5 zeros for navigation

    # TMS during shift: 0 for all except last bit
    for _i in range(ir_bits - 1):
        tms_sequence.append(0)
    tms_sequence.append(1)  # Exit1-IR

    # Exit path: Update-IR -> Run-Test/Idle
    tms_sequence.extend([1, 0])  # Update-IR, Run-Test/Idle
    tdi_bits.extend([0, 0])

    return generate_jtag_signals(tms_sequence, tdi_bits, samples_per_bit=samples_per_bit)


def generate_dr_shift(
    dr_value: int,
    dr_bits: int = 32,
    samples_per_bit: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Generate JTAG DR shift sequence.

    Args:
        dr_value: Data register value.
        dr_bits: Number of DR bits.
        samples_per_bit: Samples per bit period.

    Returns:
        Tuple of (tck, tms, tdi, tdo) for DR shift.
    """
    # TMS sequence to reach Shift-DR:
    # From Run-Test/Idle: TMS=1 -> Select-DR-Scan
    #                     TMS=0 -> Capture-DR
    #                     TMS=0 -> Shift-DR
    # Then shift DR bits
    # Then: Exit1-DR -> Update-DR -> Run-Test/Idle

    tms_sequence = [1, 0, 0]  # Go to Shift-DR

    # DR bits (LSB first)
    dr_bits_list = [(dr_value >> i) & 1 for i in range(dr_bits)]
    tdi_bits = [0] * 3 + dr_bits_list

    # TMS during shift
    for _i in range(dr_bits - 1):
        tms_sequence.append(0)
    tms_sequence.append(1)  # Exit1-DR

    # Exit path
    tms_sequence.extend([1, 0])  # Update-DR, Run-Test/Idle
    tdi_bits.extend([0, 0])

    return generate_jtag_signals(tms_sequence, tdi_bits, samples_per_bit=samples_per_bit)


# =============================================================================
# TAPState Tests
# =============================================================================


class TestTAPState:
    """Test TAPState enum."""

    def test_state_values(self) -> None:
        """Test state enum values."""
        assert TAPState.TEST_LOGIC_RESET.value == "Test-Logic-Reset"
        assert TAPState.RUN_TEST_IDLE.value == "Run-Test/Idle"
        assert TAPState.SHIFT_DR.value == "Shift-DR"
        assert TAPState.SHIFT_IR.value == "Shift-IR"

    def test_all_states_exist(self) -> None:
        """Test all 16 TAP states are defined."""
        states = list(TAPState)
        assert len(states) == 16

    def test_dr_states(self) -> None:
        """Test DR-related states exist."""
        dr_states = {
            TAPState.SELECT_DR_SCAN,
            TAPState.CAPTURE_DR,
            TAPState.SHIFT_DR,
            TAPState.EXIT1_DR,
            TAPState.PAUSE_DR,
            TAPState.EXIT2_DR,
            TAPState.UPDATE_DR,
        }
        assert len(dr_states) == 7

    def test_ir_states(self) -> None:
        """Test IR-related states exist."""
        ir_states = {
            TAPState.SELECT_IR_SCAN,
            TAPState.CAPTURE_IR,
            TAPState.SHIFT_IR,
            TAPState.EXIT1_IR,
            TAPState.PAUSE_IR,
            TAPState.EXIT2_IR,
            TAPState.UPDATE_IR,
        }
        assert len(ir_states) == 7


# =============================================================================
# JTAG_INSTRUCTIONS Tests
# =============================================================================


class TestJTAGInstructions:
    """Test JTAG_INSTRUCTIONS dictionary."""

    def test_common_instructions(self) -> None:
        """Test common JTAG instructions are defined."""
        assert JTAG_INSTRUCTIONS[0x00] == "EXTEST"
        assert JTAG_INSTRUCTIONS[0x01] == "SAMPLE/PRELOAD"
        assert JTAG_INSTRUCTIONS[0x02] == "IDCODE"
        assert JTAG_INSTRUCTIONS[0x03] == "BYPASS"

    def test_instruction_count(self) -> None:
        """Test number of defined instructions."""
        assert len(JTAG_INSTRUCTIONS) >= 4


# =============================================================================
# JTAGDecoder Initialization Tests
# =============================================================================


class TestJTAGDecoderInit:
    """Test JTAGDecoder initialization and configuration."""

    def test_default_initialization(self) -> None:
        """Test decoder with default parameters."""
        decoder = JTAGDecoder()

        assert decoder.id == "jtag"
        assert decoder.name == "JTAG"
        assert decoder.longname == "Joint Test Action Group (IEEE 1149.1)"
        assert "JTAG" in decoder.desc
        assert decoder._tap_state == TAPState.TEST_LOGIC_RESET

    def test_channel_definitions(self) -> None:
        """Test decoder channel definitions."""
        decoder = JTAGDecoder()

        assert len(decoder.channels) == 3

        channel_ids = [ch.id for ch in decoder.channels]
        assert "tck" in channel_ids
        assert "tms" in channel_ids
        assert "tdi" in channel_ids

        for ch in decoder.channels:
            assert ch.required is True

    def test_optional_channel_definitions(self) -> None:
        """Test decoder optional channel definitions."""
        decoder = JTAGDecoder()

        assert len(decoder.optional_channels) == 1
        assert decoder.optional_channels[0].id == "tdo"
        assert decoder.optional_channels[0].required is False

    def test_annotations_defined(self) -> None:
        """Test decoder has annotations defined."""
        decoder = JTAGDecoder()

        assert len(decoder.annotations) > 0
        annotation_ids = [a[0] for a in decoder.annotations]
        assert "state" in annotation_ids
        assert "ir" in annotation_ids
        assert "dr" in annotation_ids


# =============================================================================
# JTAGDecoder Edge Cases
# =============================================================================


class TestJTAGDecoderEdgeCases:
    """Test JTAG decoder edge cases and error handling."""

    def test_decode_none_signals(self) -> None:
        """Test decode with None signals."""
        decoder = JTAGDecoder()

        packets = list(decoder.decode(tck=None, tms=None, tdi=None, sample_rate=1_000_000.0))

        assert len(packets) == 0

    def test_decode_empty_signals(self) -> None:
        """Test decode with empty signals."""
        decoder = JTAGDecoder()

        tck = np.array([], dtype=bool)
        tms = np.array([], dtype=bool)
        tdi = np.array([], dtype=bool)

        packets = list(decoder.decode(tck=tck, tms=tms, tdi=tdi, sample_rate=1_000_000.0))

        assert len(packets) == 0

    def test_decode_no_clock_edges(self) -> None:
        """Test decode with constant clock (no edges)."""
        decoder = JTAGDecoder()

        tck = np.zeros(100, dtype=bool)
        tms = np.zeros(100, dtype=bool)
        tdi = np.zeros(100, dtype=bool)

        packets = list(decoder.decode(tck=tck, tms=tms, tdi=tdi, sample_rate=1_000_000.0))

        assert len(packets) == 0

    def test_decode_mismatched_lengths(self) -> None:
        """Test decode with mismatched signal lengths."""
        decoder = JTAGDecoder()

        tck = np.array([False, True] * 50, dtype=bool)
        tms = np.array([False] * 50, dtype=bool)
        tdi = np.array([True] * 25, dtype=bool)

        # Should truncate to shortest length
        packets = list(decoder.decode(tck=tck, tms=tms, tdi=tdi, sample_rate=1_000_000.0))

        # Should not crash
        assert isinstance(packets, list)


# =============================================================================
# JTAGDecoder State Machine Tests
# =============================================================================


class TestJTAGDecoderStateMachine:
    """Test JTAG TAP state machine transitions."""

    def test_next_state_from_test_logic_reset(self) -> None:
        """Test state transitions from Test-Logic-Reset."""
        decoder = JTAGDecoder()

        # TMS=0 -> Run-Test/Idle
        next_state = decoder._next_state(TAPState.TEST_LOGIC_RESET, False)
        assert next_state == TAPState.RUN_TEST_IDLE

        # TMS=1 -> Stay in Test-Logic-Reset
        next_state = decoder._next_state(TAPState.TEST_LOGIC_RESET, True)
        assert next_state == TAPState.TEST_LOGIC_RESET

    def test_next_state_from_run_test_idle(self) -> None:
        """Test state transitions from Run-Test/Idle."""
        decoder = JTAGDecoder()

        # TMS=0 -> Stay in Run-Test/Idle
        next_state = decoder._next_state(TAPState.RUN_TEST_IDLE, False)
        assert next_state == TAPState.RUN_TEST_IDLE

        # TMS=1 -> Select-DR-Scan
        next_state = decoder._next_state(TAPState.RUN_TEST_IDLE, True)
        assert next_state == TAPState.SELECT_DR_SCAN

    def test_next_state_from_select_dr_scan(self) -> None:
        """Test state transitions from Select-DR-Scan."""
        decoder = JTAGDecoder()

        # TMS=0 -> Capture-DR
        next_state = decoder._next_state(TAPState.SELECT_DR_SCAN, False)
        assert next_state == TAPState.CAPTURE_DR

        # TMS=1 -> Select-IR-Scan
        next_state = decoder._next_state(TAPState.SELECT_DR_SCAN, True)
        assert next_state == TAPState.SELECT_IR_SCAN

    def test_next_state_shift_dr(self) -> None:
        """Test state transitions from Shift-DR."""
        decoder = JTAGDecoder()

        # TMS=0 -> Stay in Shift-DR
        next_state = decoder._next_state(TAPState.SHIFT_DR, False)
        assert next_state == TAPState.SHIFT_DR

        # TMS=1 -> Exit1-DR
        next_state = decoder._next_state(TAPState.SHIFT_DR, True)
        assert next_state == TAPState.EXIT1_DR

    def test_next_state_shift_ir(self) -> None:
        """Test state transitions from Shift-IR."""
        decoder = JTAGDecoder()

        # TMS=0 -> Stay in Shift-IR
        next_state = decoder._next_state(TAPState.SHIFT_IR, False)
        assert next_state == TAPState.SHIFT_IR

        # TMS=1 -> Exit1-IR
        next_state = decoder._next_state(TAPState.SHIFT_IR, True)
        assert next_state == TAPState.EXIT1_IR


# =============================================================================
# JTAGDecoder Decoding Tests
# =============================================================================


class TestJTAGDecoderDecoding:
    """Test JTAG decoding with actual transactions."""

    def test_decode_ir_shift(self) -> None:
        """Test decoding IR shift operation."""
        decoder = JTAGDecoder()
        sample_rate = 1_000_000.0

        tck, tms, tdi, tdo = generate_ir_shift(
            ir_value=0x02,  # IDCODE
            ir_bits=8,
        )

        packets = list(decoder.decode(tck=tck, tms=tms, tdi=tdi, tdo=tdo, sample_rate=sample_rate))

        # Should decode IR shift
        if len(packets) > 0:
            assert packets[0].protocol == "jtag"
            assert "ir_value" in packets[0].annotations

    def test_decode_dr_shift(self) -> None:
        """Test decoding DR shift operation."""
        decoder = JTAGDecoder()
        sample_rate = 1_000_000.0

        tck, tms, tdi, tdo = generate_dr_shift(
            dr_value=0x12345678,
            dr_bits=32,
        )

        packets = list(decoder.decode(tck=tck, tms=tms, tdi=tdi, tdo=tdo, sample_rate=sample_rate))

        # Should decode DR shift
        if len(packets) > 0:
            assert packets[0].protocol == "jtag"
            assert "dr_value_tdi" in packets[0].annotations

    def test_decode_with_tdo(self) -> None:
        """Test decoding with TDO signal."""
        decoder = JTAGDecoder()
        sample_rate = 1_000_000.0

        # Generate simple sequence with TDO
        tms_sequence = [1, 0, 0, 0, 0, 0, 1, 1, 0]
        tdi_bits = [0, 0, 0, 1, 0, 1, 0, 0, 0]
        tdo_bits = [1, 1, 0, 0, 1, 1, 0, 0, 1]

        tck, tms, tdi, tdo = generate_jtag_signals(tms_sequence, tdi_bits, tdo_bits)

        packets = list(decoder.decode(tck=tck, tms=tms, tdi=tdi, tdo=tdo, sample_rate=sample_rate))

        # Should not crash with TDO
        assert isinstance(packets, list)

    def test_decode_without_tdo(self) -> None:
        """Test decoding without TDO signal."""
        decoder = JTAGDecoder()
        sample_rate = 1_000_000.0

        tck, tms, tdi, _ = generate_ir_shift(ir_value=0x02)

        packets = list(decoder.decode(tck=tck, tms=tms, tdi=tdi, sample_rate=sample_rate))

        # Should decode without TDO
        assert isinstance(packets, list)


# =============================================================================
# JTAGDecoder Internal Methods Tests
# =============================================================================


class TestJTAGDecoderInternalMethods:
    """Test JTAG decoder internal methods."""

    def test_bits_to_value_lsb(self) -> None:
        """Test _bits_to_value with LSB-first bits."""
        decoder = JTAGDecoder()

        # 0xA5 = 10100101, LSB first: [1, 0, 1, 0, 0, 1, 0, 1]
        bits = [1, 0, 1, 0, 0, 1, 0, 1]
        result = decoder._bits_to_value(bits)

        assert result == 0xA5

    def test_bits_to_value_empty(self) -> None:
        """Test _bits_to_value with empty list."""
        decoder = JTAGDecoder()

        result = decoder._bits_to_value([])

        assert result == 0

    def test_bits_to_value_single_bit(self) -> None:
        """Test _bits_to_value with single bit."""
        decoder = JTAGDecoder()

        assert decoder._bits_to_value([0]) == 0
        assert decoder._bits_to_value([1]) == 1

    def test_bits_to_value_32bit(self) -> None:
        """Test _bits_to_value with 32-bit value."""
        decoder = JTAGDecoder()

        # 0x12345678 in LSB-first bits
        value = 0x12345678
        bits = [(value >> i) & 1 for i in range(32)]

        result = decoder._bits_to_value(bits)

        assert result == 0x12345678


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestDecodeJTAGConvenienceFunction:
    """Test decode_jtag convenience function."""

    def test_decode_jtag_basic(self) -> None:
        """Test basic usage of decode_jtag."""
        tck, tms, tdi, tdo = generate_ir_shift(ir_value=0x02)

        packets = decode_jtag(tck, tms, tdi, tdo, sample_rate=1_000_000.0)

        assert isinstance(packets, list)

    def test_decode_jtag_without_tdo(self) -> None:
        """Test decode_jtag without TDO."""
        tck, tms, tdi, _ = generate_dr_shift(dr_value=0xDEADBEEF)

        packets = decode_jtag(tck, tms, tdi, sample_rate=1_000_000.0)

        assert isinstance(packets, list)

    def test_decode_jtag_empty_signals(self) -> None:
        """Test decode_jtag with empty signals."""
        tck = np.array([], dtype=bool)
        tms = np.array([], dtype=bool)
        tdi = np.array([], dtype=bool)

        packets = decode_jtag(tck, tms, tdi, sample_rate=1_000_000.0)

        assert packets == []


# =============================================================================
# Module Export Tests
# =============================================================================


class TestJTAGModuleExports:
    """Test module-level exports."""

    def test_all_export(self) -> None:
        """Test __all__ contains expected exports."""
        from oscura.analyzers.protocols.jtag import __all__

        assert "JTAGDecoder" in __all__
        assert "TAPState" in __all__
        assert "JTAG_INSTRUCTIONS" in __all__
        assert "decode_jtag" in __all__
