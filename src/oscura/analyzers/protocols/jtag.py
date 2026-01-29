"""JTAG protocol decoder.

This module provides IEEE 1149.1 JTAG/Boundary-Scan protocol decoding
with TAP state machine tracking and IR/DR data extraction.


Example:
    >>> from oscura.analyzers.protocols.jtag import JTAGDecoder
    >>> decoder = JTAGDecoder()
    >>> for packet in decoder.decode(tck=tck, tms=tms, tdi=tdi, tdo=tdo):
    ...     print(f"State: {packet.annotations['tap_state']}")

References:
    IEEE 1149.1-2013 Standard Test Access Port and Boundary-Scan Architecture
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from oscura.analyzers.protocols.base import (
    AnnotationLevel,
    ChannelDef,
    SyncDecoder,
)
from oscura.core.types import DigitalTrace, ProtocolPacket

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


class TAPState(Enum):
    """JTAG TAP Controller states."""

    TEST_LOGIC_RESET = "Test-Logic-Reset"
    RUN_TEST_IDLE = "Run-Test/Idle"
    SELECT_DR_SCAN = "Select-DR-Scan"
    CAPTURE_DR = "Capture-DR"
    SHIFT_DR = "Shift-DR"
    EXIT1_DR = "Exit1-DR"
    PAUSE_DR = "Pause-DR"
    EXIT2_DR = "Exit2-DR"
    UPDATE_DR = "Update-DR"
    SELECT_IR_SCAN = "Select-IR-Scan"
    CAPTURE_IR = "Capture-IR"
    SHIFT_IR = "Shift-IR"
    EXIT1_IR = "Exit1-IR"
    PAUSE_IR = "Pause-IR"
    EXIT2_IR = "Exit2-IR"
    UPDATE_IR = "Update-IR"


# Standard JTAG instructions
JTAG_INSTRUCTIONS = {
    0x00: "EXTEST",
    0x01: "SAMPLE/PRELOAD",
    0x02: "IDCODE",
    0x03: "BYPASS",
    0x04: "INTEST",
    0x05: "RUNBIST",
    0x06: "CLAMP",
    0x07: "HIGHZ",
}


class JTAGDecoder(SyncDecoder):
    """JTAG protocol decoder.

    Decodes JTAG bus transactions including TAP state machine transitions,
    IR/DR shift operations, and standard instruction identification.

    Attributes:
        id: "jtag"
        name: "JTAG"
        channels: [tck, tms, tdi] (required), [tdo] (optional)

    Example:
        >>> decoder = JTAGDecoder()
        >>> for packet in decoder.decode(tck=tck, tms=tms, tdi=tdi, sample_rate=1e6):
        ...     print(f"IR: {packet.annotations.get('ir_value', 'N/A')}")
    """

    id = "jtag"
    name = "JTAG"
    longname = "Joint Test Action Group (IEEE 1149.1)"
    desc = "JTAG/Boundary-Scan protocol decoder"

    channels = [
        ChannelDef("tck", "TCK", "Test Clock", required=True),
        ChannelDef("tms", "TMS", "Test Mode Select", required=True),
        ChannelDef("tdi", "TDI", "Test Data In", required=True),
    ]

    optional_channels = [
        ChannelDef("tdo", "TDO", "Test Data Out", required=False),
    ]

    options = []

    annotations = [
        ("state", "TAP state"),
        ("ir", "Instruction register"),
        ("dr", "Data register"),
        ("instruction", "Decoded instruction"),
    ]

    def __init__(self) -> None:
        """Initialize JTAG decoder."""
        super().__init__()
        self._tap_state = TAPState.TEST_LOGIC_RESET
        self._shift_bits_tdi: list[int] = []
        self._shift_bits_tdo: list[int] = []

    def decode(  # type: ignore[override]
        self,
        trace: DigitalTrace | None = None,
        *,
        tck: NDArray[np.bool_] | None = None,
        tms: NDArray[np.bool_] | None = None,
        tdi: NDArray[np.bool_] | None = None,
        tdo: NDArray[np.bool_] | None = None,
        sample_rate: float = 1.0,
    ) -> Iterator[ProtocolPacket]:
        """Decode JTAG transactions.

        Args:
            trace: Optional primary trace.
            tck: Test Clock signal.
            tms: Test Mode Select signal.
            tdi: Test Data In signal.
            tdo: Test Data Out signal (optional).
            sample_rate: Sample rate in Hz.

        Yields:
            Decoded JTAG operations as ProtocolPacket objects.

        Example:
            >>> decoder = JTAGDecoder()
            >>> for pkt in decoder.decode(tck=tck, tms=tms, tdi=tdi, sample_rate=10e6):
            ...     print(f"State: {pkt.annotations['tap_state']}")
        """
        if tck is None or tms is None or tdi is None:
            return

        tck, tms, tdi, tdo = self._align_signals(tck, tms, tdi, tdo)
        rising_edges = np.where(~tck[:-1] & tck[1:])[0] + 1

        if len(rising_edges) == 0:
            return

        state_start_idx = 0
        trans_num = 0

        for edge_idx in rising_edges:
            tms_val = bool(tms[edge_idx])
            new_state = self._next_state(self._tap_state, tms_val)

            # Handle data shifting in SHIFT states
            if self._tap_state in (TAPState.SHIFT_IR, TAPState.SHIFT_DR):
                self._shift_data_bit(edge_idx, tdi, tdo)

            # Handle state transitions and emit packets
            if new_state != self._tap_state:
                packet = self._emit_state_packet(state_start_idx, edge_idx, sample_rate, trans_num)
                if packet is not None:
                    yield packet
                    trans_num += 1

                # Reset shift buffers on state change
                if new_state not in [TAPState.SHIFT_IR, TAPState.SHIFT_DR]:
                    self._shift_bits_tdi = []
                    self._shift_bits_tdo = []

                self._tap_state = new_state
                state_start_idx = edge_idx

    def _align_signals(
        self,
        tck: NDArray[np.bool_],
        tms: NDArray[np.bool_],
        tdi: NDArray[np.bool_],
        tdo: NDArray[np.bool_] | None,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_] | None]:
        """Align all signals to the same length.

        Args:
            tck: Test Clock signal.
            tms: Test Mode Select signal.
            tdi: Test Data In signal.
            tdo: Test Data Out signal (optional).

        Returns:
            Tuple of aligned signals (tck, tms, tdi, tdo).
        """
        n_samples = min(len(tck), len(tms), len(tdi))
        if tdo is not None:
            n_samples = min(n_samples, len(tdo))

        return (
            tck[:n_samples],
            tms[:n_samples],
            tdi[:n_samples],
            tdo[:n_samples] if tdo is not None else None,
        )

    def _shift_data_bit(
        self,
        edge_idx: int,
        tdi: NDArray[np.bool_],
        tdo: NDArray[np.bool_] | None,
    ) -> None:
        """Shift a single bit of data on TDI/TDO.

        Args:
            edge_idx: Edge index where shift occurs.
            tdi: Test Data In signal.
            tdo: Test Data Out signal (optional).
        """
        tdi_bit = 1 if tdi[edge_idx] else 0
        self._shift_bits_tdi.append(tdi_bit)

        if tdo is not None:
            tdo_bit = 1 if tdo[edge_idx] else 0
            self._shift_bits_tdo.append(tdo_bit)

    def _emit_state_packet(
        self,
        state_start_idx: int,
        edge_idx: int,
        sample_rate: float,
        trans_num: int,
    ) -> ProtocolPacket | None:
        """Emit packet on state change if we have shifted data.

        Args:
            state_start_idx: Start index of current state.
            edge_idx: Current edge index.
            sample_rate: Sample rate in Hz.
            trans_num: Transaction number.

        Returns:
            ProtocolPacket if data was shifted, None otherwise.
        """
        if self._tap_state == TAPState.SHIFT_IR and len(self._shift_bits_tdi) > 0:
            return self._create_ir_packet(state_start_idx, edge_idx, sample_rate, trans_num)
        elif self._tap_state == TAPState.SHIFT_DR and len(self._shift_bits_tdi) > 0:
            return self._create_dr_packet(state_start_idx, edge_idx, sample_rate, trans_num)
        return None

    def _create_ir_packet(
        self,
        state_start_idx: int,
        edge_idx: int,
        sample_rate: float,
        trans_num: int,
    ) -> ProtocolPacket:
        """Create IR shift packet.

        Args:
            state_start_idx: Start index of shift state.
            edge_idx: End index of shift state.
            sample_rate: Sample rate in Hz.
            trans_num: Transaction number.

        Returns:
            ProtocolPacket for IR shift.
        """
        ir_value = self._bits_to_value(self._shift_bits_tdi)
        start_time = state_start_idx / sample_rate
        end_time = edge_idx / sample_rate
        instruction_name = JTAG_INSTRUCTIONS.get(ir_value, "UNKNOWN")

        self.put_annotation(
            start_time,
            end_time,
            AnnotationLevel.FIELDS,
            f"IR: 0x{ir_value:02X} ({instruction_name})",
        )

        annotations = {
            "transaction_num": trans_num,
            "tap_state": self._tap_state.value,
            "ir_value": ir_value,
            "ir_bits": len(self._shift_bits_tdi),
            "instruction": instruction_name,
        }

        return ProtocolPacket(
            timestamp=start_time,
            protocol="jtag",
            data=bytes([ir_value]),
            annotations=annotations,
            errors=[],
        )

    def _create_dr_packet(
        self,
        state_start_idx: int,
        edge_idx: int,
        sample_rate: float,
        trans_num: int,
    ) -> ProtocolPacket:
        """Create DR shift packet.

        Args:
            state_start_idx: Start index of shift state.
            edge_idx: End index of shift state.
            sample_rate: Sample rate in Hz.
            trans_num: Transaction number.

        Returns:
            ProtocolPacket for DR shift.
        """
        dr_value_tdi = self._bits_to_value(self._shift_bits_tdi)
        start_time = state_start_idx / sample_rate
        end_time = edge_idx / sample_rate

        byte_count = (len(self._shift_bits_tdi) + 7) // 8
        dr_bytes = dr_value_tdi.to_bytes(byte_count, "little")

        self.put_annotation(
            start_time,
            end_time,
            AnnotationLevel.FIELDS,
            f"DR: 0x{dr_value_tdi:X} ({len(self._shift_bits_tdi)} bits)",
        )

        annotations = {
            "transaction_num": trans_num,
            "tap_state": self._tap_state.value,
            "dr_value_tdi": dr_value_tdi,
            "dr_bits": len(self._shift_bits_tdi),
        }

        if len(self._shift_bits_tdo) > 0:
            dr_value_tdo = self._bits_to_value(self._shift_bits_tdo)
            annotations["dr_value_tdo"] = dr_value_tdo

        return ProtocolPacket(
            timestamp=start_time,
            protocol="jtag",
            data=dr_bytes,
            annotations=annotations,
            errors=[],
        )

    def _next_state(self, current: TAPState, tms: bool) -> TAPState:
        """Compute next TAP state based on TMS value.

        Args:
            current: Current TAP state.
            tms: TMS signal value.

        Returns:
            Next TAP state.
        """
        # TAP state machine (IEEE 1149.1 Figure 6-1)
        transitions = {
            TAPState.TEST_LOGIC_RESET: {
                False: TAPState.RUN_TEST_IDLE,
                True: TAPState.TEST_LOGIC_RESET,
            },
            TAPState.RUN_TEST_IDLE: {
                False: TAPState.RUN_TEST_IDLE,
                True: TAPState.SELECT_DR_SCAN,
            },
            TAPState.SELECT_DR_SCAN: {
                False: TAPState.CAPTURE_DR,
                True: TAPState.SELECT_IR_SCAN,
            },
            TAPState.CAPTURE_DR: {
                False: TAPState.SHIFT_DR,
                True: TAPState.EXIT1_DR,
            },
            TAPState.SHIFT_DR: {
                False: TAPState.SHIFT_DR,
                True: TAPState.EXIT1_DR,
            },
            TAPState.EXIT1_DR: {
                False: TAPState.PAUSE_DR,
                True: TAPState.UPDATE_DR,
            },
            TAPState.PAUSE_DR: {
                False: TAPState.PAUSE_DR,
                True: TAPState.EXIT2_DR,
            },
            TAPState.EXIT2_DR: {
                False: TAPState.SHIFT_DR,
                True: TAPState.UPDATE_DR,
            },
            TAPState.UPDATE_DR: {
                False: TAPState.RUN_TEST_IDLE,
                True: TAPState.SELECT_DR_SCAN,
            },
            TAPState.SELECT_IR_SCAN: {
                False: TAPState.CAPTURE_IR,
                True: TAPState.TEST_LOGIC_RESET,
            },
            TAPState.CAPTURE_IR: {
                False: TAPState.SHIFT_IR,
                True: TAPState.EXIT1_IR,
            },
            TAPState.SHIFT_IR: {
                False: TAPState.SHIFT_IR,
                True: TAPState.EXIT1_IR,
            },
            TAPState.EXIT1_IR: {
                False: TAPState.PAUSE_IR,
                True: TAPState.UPDATE_IR,
            },
            TAPState.PAUSE_IR: {
                False: TAPState.PAUSE_IR,
                True: TAPState.EXIT2_IR,
            },
            TAPState.EXIT2_IR: {
                False: TAPState.SHIFT_IR,
                True: TAPState.UPDATE_IR,
            },
            TAPState.UPDATE_IR: {
                False: TAPState.RUN_TEST_IDLE,
                True: TAPState.SELECT_DR_SCAN,
            },
        }

        return transitions[current][tms]

    def _bits_to_value(self, bits: list[int]) -> int:
        """Convert bit list to integer (LSB first).

        Args:
            bits: List of bit values (0 or 1).

        Returns:
            Integer value.
        """
        value = 0
        for i, bit in enumerate(bits):
            value |= bit << i
        return value


def decode_jtag(
    tck: NDArray[np.bool_],
    tms: NDArray[np.bool_],
    tdi: NDArray[np.bool_],
    tdo: NDArray[np.bool_] | None = None,
    sample_rate: float = 1.0,
) -> list[ProtocolPacket]:
    """Convenience function to decode JTAG transactions.

    Args:
        tck: Test Clock signal.
        tms: Test Mode Select signal.
        tdi: Test Data In signal.
        tdo: Test Data Out signal (optional).
        sample_rate: Sample rate in Hz.

    Returns:
        List of decoded JTAG transactions.

    Example:
        >>> packets = decode_jtag(tck, tms, tdi, tdo, sample_rate=10e6)
        >>> for pkt in packets:
        ...     print(f"IR: {pkt.annotations.get('ir_value', 'N/A')}")
    """
    decoder = JTAGDecoder()
    return list(decoder.decode(tck=tck, tms=tms, tdi=tdi, tdo=tdo, sample_rate=sample_rate))


__all__ = ["JTAG_INSTRUCTIONS", "JTAGDecoder", "TAPState", "decode_jtag"]
