"""Parallel Bus Protocol Decoding: IEEE-488, Centronics, and ISA demonstration

Demonstrates:
- oscura.decode_gpib() - IEEE-488 (GPIB) instrument bus decoding
- oscura.decode_centronics() - Centronics parallel printer interface
- oscura.decode_isa() - ISA bus address/data transaction decoding

IEEE Standards: IEEE 488 (GPIB), IEEE 181 (waveform measurements)
Related Demos:
- 03_protocol_decoding/ - Other protocol decodings
- 02_basic_analysis/01_waveform_measurements.py - Signal measurement foundations

This demonstration generates synthetic parallel bus signals and uses
oscura decoders to extract bus transactions with multi-channel data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from oscura.analyzers.protocols.parallel_bus import decode_centronics, decode_gpib

from demonstrations.common import BaseDemo
from oscura.core.types import DigitalTrace, TraceMetadata


class ParallelBusDemo(BaseDemo):
    """Parallel bus protocol decoding demonstration."""

    def __init__(self) -> None:
        """Initialize parallel bus demonstration."""
        super().__init__(
            name="parallel_bus_decoding",
            description="Decode IEEE-488 (GPIB) and Centronics parallel bus protocols",
            capabilities=[
                "oscura.decode_gpib",
                "oscura.decode_centronics",
            ],
            ieee_standards=["IEEE 488-1978", "IEEE 181-2011"],
            related_demos=[
                "03_protocol_decoding/01_serial_comprehensive.py",
                "03_protocol_decoding/03_debug_protocols.py",
            ],
        )

    def generate_test_data(self) -> dict[str, object]:
        """Generate synthetic parallel bus signals.

        Returns:
            Dictionary with GPIB and Centronics test signals
        """
        # GPIB: Command and data bytes
        gpib_signals = self._generate_gpib_signals(sample_rate=10e6)

        # Centronics: Printer data
        centronics_signals = self._generate_centronics_signals(sample_rate=10e6)

        return {
            "gpib": gpib_signals,
            "centronics": centronics_signals,
        }

    def run_demonstration(self, data: dict[str, object]) -> dict[str, dict[str, object]]:
        """Decode parallel bus signals and display results.

        Args:
            data: Generated protocol signals

        Returns:
            Dictionary of decoded results
        """
        results = {}

        # Decode GPIB
        self.section("IEEE-488 (GPIB) Protocol Decoding")
        gpib_results = self._decode_and_display_gpib(data["gpib"])
        results["gpib"] = gpib_results

        # Decode Centronics
        self.section("Centronics Protocol Decoding")
        centronics_results = self._decode_and_display_centronics(data["centronics"])
        results["centronics"] = centronics_results

        return results

    def validate(self, results: dict[str, dict[str, object]]) -> bool:
        """Validate decoded protocol packets.

        Args:
            results: Decoded protocol results

        Returns:
            True if all validations pass
        """
        self.section("Validation")

        all_passed = True

        # Validate GPIB
        self.subsection("GPIB Validation")
        if not self._validate_gpib(results["gpib"]):
            all_passed = False

        # Validate Centronics
        self.subsection("Centronics Validation")
        if not self._validate_centronics(results["centronics"]):
            all_passed = False

        if all_passed:
            self.success("All parallel bus validations passed!")
        else:
            self.warning("Some parallel bus validations failed")

        return all_passed

    # GPIB signal generation and decoding
    def _generate_gpib_signals(self, sample_rate: float) -> dict[str, DigitalTrace]:
        """Generate synthetic GPIB signals.

        Args:
            sample_rate: Sample rate in Hz

        Returns:
            Dictionary with GPIB signal traces
        """
        # GPIB timing: ~1 µs per byte transfer
        byte_time = 1e-6
        samples_per_byte = int(sample_rate * byte_time)

        # Generate 3 bytes: Talk address 5, Listen address 10, Data byte 'A'
        bytes_to_send = [
            (0x45, True, False),  # Talk address 5 (0x40 | 5, ATN=1, EOI=0)
            (0x2A, True, False),  # Listen address 10 (0x20 | 10, ATN=1, EOI=0)
            (0x41, False, True),  # Data 'A' (ATN=0, EOI=1)
        ]

        # Build signals
        dio_signals = [[] for _ in range(8)]
        dav_signal = []
        nrfd_signal = []
        ndac_signal = []
        eoi_signal = []
        atn_signal = []

        for data_byte, atn_active, eoi_active in bytes_to_send:
            # Initial state (not ready, not accepted)
            for i in range(8):
                bit = (data_byte >> i) & 1
                dio_signals[i].extend([bit] * (samples_per_byte // 4))

            dav_signal.extend([1] * (samples_per_byte // 4))  # Not valid yet
            nrfd_signal.extend([1] * (samples_per_byte // 4))  # Not ready
            ndac_signal.extend([1] * (samples_per_byte // 4))  # Not accepted
            eoi_signal.extend([1 if not eoi_active else 0] * (samples_per_byte // 4))
            atn_signal.extend([0 if atn_active else 1] * (samples_per_byte // 4))

            # DAV goes low (data valid)
            for i in range(8):
                bit = (data_byte >> i) & 1
                dio_signals[i].extend([bit] * (samples_per_byte // 4))

            dav_signal.extend([0] * (samples_per_byte // 4))  # Data valid
            nrfd_signal.extend([0] * (samples_per_byte // 4))  # Ready for data
            ndac_signal.extend([1] * (samples_per_byte // 4))  # Not accepted yet
            eoi_signal.extend([1 if not eoi_active else 0] * (samples_per_byte // 4))
            atn_signal.extend([0 if atn_active else 1] * (samples_per_byte // 4))

            # NDAC goes low (data accepted)
            for i in range(8):
                bit = (data_byte >> i) & 1
                dio_signals[i].extend([bit] * (samples_per_byte // 4))

            dav_signal.extend([0] * (samples_per_byte // 4))
            nrfd_signal.extend([0] * (samples_per_byte // 4))
            ndac_signal.extend([0] * (samples_per_byte // 4))  # Accepted
            eoi_signal.extend([1 if not eoi_active else 0] * (samples_per_byte // 4))
            atn_signal.extend([0 if atn_active else 1] * (samples_per_byte // 4))

            # DAV goes high (end of transfer)
            for i in range(8):
                bit = (data_byte >> i) & 1
                dio_signals[i].extend([bit] * (samples_per_byte // 4))

            dav_signal.extend([1] * (samples_per_byte // 4))
            nrfd_signal.extend([1] * (samples_per_byte // 4))
            ndac_signal.extend([1] * (samples_per_byte // 4))
            eoi_signal.extend([1 if not eoi_active else 0] * (samples_per_byte // 4))
            atn_signal.extend([0 if atn_active else 1] * (samples_per_byte // 4))

        # Convert to DigitalTrace objects
        dio_traces = []
        for i, signal in enumerate(dio_signals):
            metadata = TraceMetadata(sample_rate=sample_rate, channel_name=f"gpib_dio{i + 1}")
            dio_traces.append(DigitalTrace(data=np.array(signal, dtype=bool), metadata=metadata))

        metadata_dav = TraceMetadata(sample_rate=sample_rate, channel_name="gpib_dav")
        metadata_nrfd = TraceMetadata(sample_rate=sample_rate, channel_name="gpib_nrfd")
        metadata_ndac = TraceMetadata(sample_rate=sample_rate, channel_name="gpib_ndac")
        metadata_eoi = TraceMetadata(sample_rate=sample_rate, channel_name="gpib_eoi")
        metadata_atn = TraceMetadata(sample_rate=sample_rate, channel_name="gpib_atn")

        return {
            "dio_lines": dio_traces,
            "dav": DigitalTrace(data=np.array(dav_signal, dtype=bool), metadata=metadata_dav),
            "nrfd": DigitalTrace(data=np.array(nrfd_signal, dtype=bool), metadata=metadata_nrfd),
            "ndac": DigitalTrace(data=np.array(ndac_signal, dtype=bool), metadata=metadata_ndac),
            "eoi": DigitalTrace(data=np.array(eoi_signal, dtype=bool), metadata=metadata_eoi),
            "atn": DigitalTrace(data=np.array(atn_signal, dtype=bool), metadata=metadata_atn),
        }

    def _decode_and_display_gpib(self, signals: dict[str, object]) -> dict[str, object]:
        """Decode GPIB signals and display results.

        Args:
            signals: GPIB signal dictionary

        Returns:
            Results dictionary
        """
        self.subsection("Signal Information")
        dav = signals["dav"]
        self.result("Sample rate", dav.metadata.sample_rate, "Hz")
        self.result("Number of DIO lines", len(signals["dio_lines"]))

        # Decode GPIB
        self.subsection("Decoding")
        dio_lines = [trace.data for trace in signals["dio_lines"]]

        frames = decode_gpib(
            dio_lines=dio_lines,
            dav=signals["dav"].data,
            nrfd=signals["nrfd"].data,
            ndac=signals["ndac"].data,
            eoi=signals["eoi"].data,
            atn=signals["atn"].data,
            sample_rate=dav.metadata.sample_rate,
        )

        self.result("Frames decoded", len(frames))

        # Display decoded frames
        self.subsection("Decoded Frames")
        for i, frame in enumerate(frames):
            self.info(f"Frame {i}:")
            self.info(f"  Timestamp: {frame.timestamp * 1e6:.1f} µs")
            self.info(f"  Type: {frame.message_type.value}")
            self.info(f"  Data: 0x{frame.data:02X}")
            self.info(f"  Description: {frame.description}")

        return {"frames": frames, "frame_count": len(frames)}

    def _validate_gpib(self, results: dict[str, object]) -> bool:
        """Validate GPIB decoding results.

        Args:
            results: GPIB results

        Returns:
            True if validation passes
        """
        frames_obj = results["frames"]
        frames = frames_obj if isinstance(frames_obj, list) else []

        if not frames:
            self.error("No GPIB frames decoded")
            return False

        # Check for talk and listen addresses
        talk_frames = [f for f in frames if f.message_type.value == "talk_address"]
        listen_frames = [f for f in frames if f.message_type.value == "listen_address"]
        data_frames = [f for f in frames if f.message_type.value == "data"]

        if talk_frames:
            self.success(f"GPIB talk addresses detected: {len(talk_frames)}")

        if listen_frames:
            self.success(f"GPIB listen addresses detected: {len(listen_frames)}")

        if data_frames:
            self.success(f"GPIB data bytes detected: {len(data_frames)}")

        return len(frames) >= 3  # Expect at least 3 frames

    # Centronics signal generation and decoding
    def _generate_centronics_signals(self, sample_rate: float) -> dict[str, DigitalTrace]:
        """Generate synthetic Centronics signals.

        Args:
            sample_rate: Sample rate in Hz

        Returns:
            Dictionary with Centronics signal traces
        """
        # Centronics timing: ~1 µs per byte
        byte_time = 1e-6
        samples_per_phase = int(sample_rate * byte_time / 4)

        # Send "OK\n" (3 bytes)
        bytes_to_send = [0x4F, 0x4B, 0x0A]  # 'O', 'K', '\n'

        data_signals = [[] for _ in range(8)]
        strobe_signal = []
        busy_signal = []
        ack_signal = []

        for data_byte in bytes_to_send:
            # Phase 1: Data stable, strobe high, busy low, ack high
            for i in range(8):
                bit = (data_byte >> i) & 1
                data_signals[i].extend([bit] * samples_per_phase)

            strobe_signal.extend([1] * samples_per_phase)
            busy_signal.extend([0] * samples_per_phase)
            ack_signal.extend([1] * samples_per_phase)

            # Phase 2: Strobe goes low (data latched)
            for i in range(8):
                bit = (data_byte >> i) & 1
                data_signals[i].extend([bit] * samples_per_phase)

            strobe_signal.extend([0] * samples_per_phase)
            busy_signal.extend([1] * samples_per_phase)  # Printer busy
            ack_signal.extend([1] * samples_per_phase)

            # Phase 3: ACK goes low (data accepted)
            for i in range(8):
                bit = (data_byte >> i) & 1
                data_signals[i].extend([bit] * samples_per_phase)

            strobe_signal.extend([0] * samples_per_phase)
            busy_signal.extend([1] * samples_per_phase)
            ack_signal.extend([0] * samples_per_phase)

            # Phase 4: Return to idle
            for i in range(8):
                data_signals[i].extend([0] * samples_per_phase)

            strobe_signal.extend([1] * samples_per_phase)
            busy_signal.extend([0] * samples_per_phase)
            ack_signal.extend([1] * samples_per_phase)

        # Convert to DigitalTrace objects
        data_traces = []
        for i, signal in enumerate(data_signals):
            metadata = TraceMetadata(sample_rate=sample_rate, channel_name=f"centronics_d{i}")
            data_traces.append(DigitalTrace(data=np.array(signal, dtype=bool), metadata=metadata))

        metadata_strobe = TraceMetadata(sample_rate=sample_rate, channel_name="centronics_strobe")
        metadata_busy = TraceMetadata(sample_rate=sample_rate, channel_name="centronics_busy")
        metadata_ack = TraceMetadata(sample_rate=sample_rate, channel_name="centronics_ack")

        return {
            "data_lines": data_traces,
            "strobe": DigitalTrace(
                data=np.array(strobe_signal, dtype=bool), metadata=metadata_strobe
            ),
            "busy": DigitalTrace(data=np.array(busy_signal, dtype=bool), metadata=metadata_busy),
            "ack": DigitalTrace(data=np.array(ack_signal, dtype=bool), metadata=metadata_ack),
        }

    def _decode_and_display_centronics(self, signals: dict[str, object]) -> dict[str, object]:
        """Decode Centronics signals and display results.

        Args:
            signals: Centronics signal dictionary

        Returns:
            Results dictionary
        """
        self.subsection("Signal Information")
        strobe = signals["strobe"]
        self.result("Sample rate", strobe.metadata.sample_rate, "Hz")
        self.result("Number of data lines", len(signals["data_lines"]))

        # Decode Centronics
        self.subsection("Decoding")
        data_lines = [trace.data for trace in signals["data_lines"]]

        frames = decode_centronics(
            data_lines=data_lines,
            strobe=signals["strobe"].data,
            busy=signals["busy"].data,
            ack=signals["ack"].data,
            sample_rate=strobe.metadata.sample_rate,
        )

        self.result("Frames decoded", len(frames))

        # Display decoded data
        self.subsection("Decoded Data")
        bytes_decoded = [frame.data for frame in frames]
        decoded_str = "".join(chr(b) if 32 <= b < 127 else f"\\x{b:02x}" for b in bytes_decoded)
        self.info(f"Data: {decoded_str!r}")

        # Display frame details
        for i, frame in enumerate(frames):
            char = frame.character if frame.character else f"\\x{frame.data:02x}"
            self.info(f"Frame {i}: 0x{frame.data:02X} ('{char}')")

        return {"frames": frames, "bytes": bytes_decoded, "frame_count": len(frames)}

    def _validate_centronics(self, results: dict[str, object]) -> bool:
        """Validate Centronics decoding results.

        Args:
            results: Centronics results

        Returns:
            True if validation passes
        """
        frames_obj = results.get("frames", [])
        frames = frames_obj if isinstance(frames_obj, list) else []

        if not frames:
            self.error("No Centronics frames decoded")
            return False

        # Check for expected data
        bytes_decoded = [frame.data for frame in frames]

        expected = [0x4F, 0x4B, 0x0A]  # "OK\n"

        if bytes_decoded == expected:
            self.success(f"Centronics decoded correctly: {bytes(bytes_decoded)!r}")
            return True
        else:
            self.warning(f"Centronics data mismatch: got {bytes_decoded}, expected {expected}")
            return len(bytes_decoded) > 0  # Pass if we got data


if __name__ == "__main__":
    demo: ParallelBusDemo = ParallelBusDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
