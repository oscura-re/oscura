"""Tests for parallel bus protocol decoders."""

import numpy as np
import pytest

from oscura.analyzers.protocols.parallel_bus import (
    GPIBMessageType,
    ISACycleType,
    decode_centronics,
    decode_gpib,
    decode_isa_bus,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


class TestGPIBDecoder:
    """Test IEEE-488 (GPIB) protocol decoder."""

    def _create_gpib_signals(self, data_bytes: list[int], atn_states: list[bool]):
        """Create GPIB test signals."""
        num_samples = len(data_bytes) * 100
        sample_rate = 1e6

        # Create DIO lines
        dio_lines = []
        for bit in range(8):
            line = np.zeros(num_samples, dtype=bool)
            for i, byte in enumerate(data_bytes):
                start_idx = i * 100 + 50
                line[start_idx : start_idx + 20] = bool(byte & (1 << bit))
            dio_lines.append(line)

        # Create control signals
        dav = np.ones(num_samples, dtype=bool)
        for i in range(len(data_bytes)):
            start_idx = i * 100 + 50
            dav[start_idx : start_idx + 20] = False  # Assert DAV

        nrfd = np.ones(num_samples, dtype=bool)
        ndac = np.ones(num_samples, dtype=bool)
        eoi = np.ones(num_samples, dtype=bool)

        # Set ATN based on states
        atn = np.ones(num_samples, dtype=bool)
        for i, atn_state in enumerate(atn_states):
            start_idx = i * 100
            atn[start_idx : start_idx + 100] = not atn_state  # Active low

        return dio_lines, dav, nrfd, ndac, eoi, atn, sample_rate

    def test_decode_data_bytes(self):
        """Test decoding data bytes."""
        data_bytes = [0x41, 0x42, 0x43]  # 'ABC'
        atn_states = [False, False, False]  # Data mode

        dio, dav, nrfd, ndac, eoi, atn, sr = self._create_gpib_signals(data_bytes, atn_states)
        frames = decode_gpib(dio, dav, nrfd, ndac, eoi, atn, sr)

        assert len(frames) == 3
        assert frames[0].message_type == GPIBMessageType.DATA
        assert frames[0].data == 0x41
        assert frames[1].data == 0x42
        assert frames[2].data == 0x43

    def test_decode_listen_address(self):
        """Test decoding listen address."""
        data_bytes = [0x20]  # Listen address 0
        atn_states = [True]  # Command mode

        dio, dav, nrfd, ndac, eoi, atn, sr = self._create_gpib_signals(data_bytes, atn_states)
        frames = decode_gpib(dio, dav, nrfd, ndac, eoi, atn, sr)

        assert len(frames) == 1
        assert frames[0].message_type == GPIBMessageType.LISTEN_ADDRESS
        assert "Listen address 0" in frames[0].description

    def test_decode_talk_address(self):
        """Test decoding talk address."""
        data_bytes = [0x40]  # Talk address 0
        atn_states = [True]

        dio, dav, nrfd, ndac, eoi, atn, sr = self._create_gpib_signals(data_bytes, atn_states)
        frames = decode_gpib(dio, dav, nrfd, ndac, eoi, atn, sr)

        assert len(frames) == 1
        assert frames[0].message_type == GPIBMessageType.TALK_ADDRESS

    def test_decode_universal_command(self):
        """Test decoding universal command."""
        data_bytes = [0x11]  # DCL (Device Clear)
        atn_states = [True]

        dio, dav, nrfd, ndac, eoi, atn, sr = self._create_gpib_signals(data_bytes, atn_states)
        frames = decode_gpib(dio, dav, nrfd, ndac, eoi, atn, sr)

        assert len(frames) == 1
        assert frames[0].message_type == GPIBMessageType.UNIVERSAL_COMMAND
        assert "DCL" in frames[0].description

    def test_wrong_number_of_dio_lines(self):
        """Test error with wrong number of DIO lines."""
        with pytest.raises(ValueError):
            decode_gpib(
                [np.zeros(100, dtype=bool)] * 7,  # Only 7 lines
                np.zeros(100, dtype=bool),
                np.zeros(100, dtype=bool),
                np.zeros(100, dtype=bool),
                np.zeros(100, dtype=bool),
                np.zeros(100, dtype=bool),
            )


class TestCentronicsDecoder:
    """Test Centronics parallel printer decoder."""

    def _create_centronics_signals(self, data_bytes: list[int]):
        """Create Centronics test signals."""
        num_samples = len(data_bytes) * 100
        sample_rate = 1e6

        # Create data lines
        data_lines = []
        for bit in range(8):
            line = np.zeros(num_samples, dtype=bool)
            for i, byte in enumerate(data_bytes):
                start_idx = i * 100 + 50
                line[start_idx : start_idx + 20] = bool(byte & (1 << bit))
            data_lines.append(line)

        # Create strobe (active low pulse)
        strobe = np.ones(num_samples, dtype=bool)
        for i in range(len(data_bytes)):
            start_idx = i * 100 + 50
            strobe[start_idx : start_idx + 10] = False

        # Create busy and ack
        busy = np.zeros(num_samples, dtype=bool)
        ack = np.ones(num_samples, dtype=bool)

        return data_lines, strobe, busy, ack, sample_rate

    def test_decode_ascii_text(self):
        """Test decoding ASCII text."""
        data_bytes = [0x48, 0x65, 0x6C, 0x6C, 0x6F]  # 'Hello'
        data, strobe, busy, ack, sr = self._create_centronics_signals(data_bytes)
        frames = decode_centronics(data, strobe, busy, ack, sr)

        assert len(frames) == 5
        assert frames[0].character == "H"
        assert frames[1].character == "e"
        assert frames[2].character == "l"
        assert frames[3].character == "l"
        assert frames[4].character == "o"

    def test_decode_control_characters(self):
        """Test decoding control characters."""
        data_bytes = [0x0D, 0x0A]  # CR, LF
        data, strobe, busy, ack, sr = self._create_centronics_signals(data_bytes)
        frames = decode_centronics(data, strobe, busy, ack, sr)

        assert len(frames) == 2
        assert frames[0].character is None  # Not printable
        assert frames[1].character is None

    def test_control_signals(self):
        """Test control signal capture."""
        data_bytes = [0x41]
        data, strobe, busy, ack, sr = self._create_centronics_signals(data_bytes)
        frames = decode_centronics(data, strobe, busy, ack, sr)

        assert len(frames) == 1
        assert "busy" in frames[0].control
        assert "ack" in frames[0].control

    def test_wrong_number_of_data_lines(self):
        """Test error with wrong number of data lines."""
        with pytest.raises(ValueError):
            decode_centronics(
                [np.zeros(100, dtype=bool)] * 7,  # Only 7 lines
                np.zeros(100, dtype=bool),
                np.zeros(100, dtype=bool),
                np.zeros(100, dtype=bool),
            )


class TestISABusDecoder:
    """Test ISA bus protocol decoder."""

    def _create_isa_signals(self, transactions: list[tuple[str, int, int | None]]):
        """Create ISA bus test signals.

        Args:
            transactions: List of (type, address, data) tuples.
        """
        num_samples = len(transactions) * 200
        sample_rate = 1e6

        # Create address lines (20-bit)
        address_lines = []
        for bit in range(20):
            line = np.zeros(num_samples, dtype=bool)
            for i, (_, addr, _) in enumerate(transactions):
                start_idx = i * 200 + 50
                line[start_idx : start_idx + 50] = bool(addr & (1 << bit))
            address_lines.append(line)

        # Create data lines (8-bit)
        data_lines = []
        for bit in range(8):
            line = np.zeros(num_samples, dtype=bool)
            for i, (_, _, data) in enumerate(transactions):
                if data is not None:
                    start_idx = i * 200 + 100
                    line[start_idx : start_idx + 50] = bool(data & (1 << bit))
            data_lines.append(line)

        # Create control signals
        ior = np.ones(num_samples, dtype=bool)
        iow = np.ones(num_samples, dtype=bool)
        memr = np.ones(num_samples, dtype=bool)
        memw = np.ones(num_samples, dtype=bool)
        ale = np.zeros(num_samples, dtype=bool)

        for i, (txn_type, _, _) in enumerate(transactions):
            start_idx = i * 200 + 50
            ale[start_idx : start_idx + 10] = True  # ALE pulse

            if txn_type == "io_read":
                ior[start_idx + 50 : start_idx + 100] = False
            elif txn_type == "io_write":
                iow[start_idx + 50 : start_idx + 100] = False
            elif txn_type == "mem_read":
                memr[start_idx + 50 : start_idx + 100] = False
            elif txn_type == "mem_write":
                memw[start_idx + 50 : start_idx + 100] = False

        return address_lines, data_lines, ior, iow, memr, memw, ale, sample_rate

    def test_decode_io_read(self):
        """Test decoding I/O read transaction."""
        transactions = [("io_read", 0x3F8, 0x55)]  # Read from COM1
        addr, data, ior, iow, memr, memw, ale, sr = self._create_isa_signals(transactions)
        frames = decode_isa_bus(addr, data, ior, iow, memr, memw, ale, sr)

        assert len(frames) >= 1
        assert frames[0].cycle_type == ISACycleType.IO_READ
        assert frames[0].address == 0x3F8
        assert frames[0].data == 0x55

    def test_decode_io_write(self):
        """Test decoding I/O write transaction."""
        transactions = [("io_write", 0x3F8, 0xAA)]
        addr, data, ior, iow, memr, memw, ale, sr = self._create_isa_signals(transactions)
        frames = decode_isa_bus(addr, data, ior, iow, memr, memw, ale, sr)

        assert len(frames) >= 1
        assert frames[0].cycle_type == ISACycleType.IO_WRITE
        assert frames[0].data == 0xAA

    def test_decode_memory_read(self):
        """Test decoding memory read transaction."""
        transactions = [("mem_read", 0x40000, 0x12)]
        addr, data, ior, iow, memr, memw, ale, sr = self._create_isa_signals(transactions)
        frames = decode_isa_bus(addr, data, ior, iow, memr, memw, ale, sr)

        assert len(frames) >= 1
        assert frames[0].cycle_type == ISACycleType.MEMORY_READ
        assert frames[0].address == 0x40000

    def test_decode_memory_write(self):
        """Test decoding memory write transaction."""
        transactions = [("mem_write", 0x40000, 0x34)]
        addr, data, ior, iow, memr, memw, ale, sr = self._create_isa_signals(transactions)
        frames = decode_isa_bus(addr, data, ior, iow, memr, memw, ale, sr)

        assert len(frames) >= 1
        assert frames[0].cycle_type == ISACycleType.MEMORY_WRITE
        assert frames[0].data == 0x34

    def test_multiple_transactions(self):
        """Test decoding multiple transactions."""
        transactions = [
            ("io_read", 0x3F8, 0x55),
            ("io_write", 0x3F8, 0xAA),
            ("mem_read", 0x40000, 0x12),
        ]
        addr, data, ior, iow, memr, memw, ale, sr = self._create_isa_signals(transactions)
        frames = decode_isa_bus(addr, data, ior, iow, memr, memw, ale, sr)

        assert len(frames) >= 3

    def test_insufficient_address_lines(self):
        """Test error with insufficient address lines."""
        with pytest.raises(ValueError):
            decode_isa_bus(
                [np.zeros(100, dtype=bool)] * 15,  # Only 15 address lines
                [np.zeros(100, dtype=bool)] * 8,
                np.zeros(100, dtype=bool),
                np.zeros(100, dtype=bool),
                np.zeros(100, dtype=bool),
                np.zeros(100, dtype=bool),
                np.zeros(100, dtype=bool),
            )
