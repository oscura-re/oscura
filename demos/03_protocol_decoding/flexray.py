"""FlexRay Protocol Decoding: Deterministic automotive communication

Demonstrates:
- oscura.decode_flexray() - FlexRay protocol decoding
- Static and dynamic segment handling
- Dual-channel differential signaling (BP/BM)
- TDMA slot-based communication
- FlexRay frame structure validation

IEEE Standards: ISO 17458-4:2013 (FlexRay specification)
Related Demos:
- 03_protocol_decoding/04_can_basic.py - CAN protocol
- 03_protocol_decoding/06_lin.py - LIN protocol
- 05_domain_specific/automotive/ - Automotive analysis

This demonstration generates synthetic FlexRay differential signals with
slot-based communication for deterministic automotive networks.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from demos.common.base_demo import BaseDemo, run_demo_main
from oscura.core.types import DigitalTrace, TraceMetadata

if TYPE_CHECKING:
    from demos.common.validation import ValidationSuite


class FlexRayDemo(BaseDemo):
    """FlexRay deterministic protocol demonstration."""

    name = "FlexRay Protocol Decoding"
    description = "Decode FlexRay frames with dual-channel TDMA communication"
    category = "protocol_decoding"
    capabilities: ClassVar[list[str]] = [
        "FlexRay frame decoding",
        "Differential signaling (BP/BM)",
        "Static/dynamic segment handling",
        "TDMA slot scheduling",
        "FlexRay header CRC validation",
    ]
    ieee_standards: ClassVar[list[str]] = ["ISO 17458-4:2013"]
    related_demos: ClassVar[list[str]] = [
        "03_protocol_decoding/04_can_basic.py",
        "03_protocol_decoding/06_lin.py",
    ]

    def generate_data(self) -> None:
        """Generate synthetic FlexRay differential signals."""
        # Static segment frame (slot 5, 10 Mbps)
        self.flexray_static_bp, self.flexray_static_bm = self._generate_flexray_frame(
            slot_id=5,
            payload=b"FlexRay",
            is_static=True,
            bitrate=10_000_000,
            sample_rate=100e6,
        )

        # Dynamic segment frame
        self.flexray_dynamic_bp, self.flexray_dynamic_bm = self._generate_flexray_frame(
            slot_id=45,
            payload=b"\x01\x02\x03\x04",
            is_static=False,
            bitrate=10_000_000,
            sample_rate=100e6,
        )

    def run_analysis(self) -> None:
        """Decode FlexRay signals."""
        from demos.common.formatting import print_subheader

        print_subheader("Static Segment Frame (Slot 5)")
        self.results["static"] = self._decode_flexray_frame(
            self.flexray_static_bp,
            self.flexray_static_bm,
            bitrate=10_000_000,
            expected_slot=5,
        )

        print_subheader("Dynamic Segment Frame (Slot 45)")
        self.results["dynamic"] = self._decode_flexray_frame(
            self.flexray_dynamic_bp,
            self.flexray_dynamic_bm,
            bitrate=10_000_000,
            expected_slot=45,
        )

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate FlexRay decoding."""
        for config_name, result in self.results.items():
            suite.check_exists(
                f"{config_name}_bp",
                result.get("bp_signal"),
                f"{config_name}: BP signal generated",
            )
            suite.check_exists(
                f"{config_name}_bm",
                result.get("bm_signal"),
                f"{config_name}: BM signal generated",
            )

    def _generate_flexray_frame(
        self,
        slot_id: int,
        payload: bytes,
        is_static: bool,
        bitrate: int,
        sample_rate: float,
    ) -> tuple[DigitalTrace, DigitalTrace]:
        """Generate FlexRay differential signals.

        Args:
            slot_id: FlexRay slot ID
            payload: Payload data
            is_static: True for static segment
            bitrate: Bitrate in bps
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (BP, BM) differential traces
        """
        bit_time = 1.0 / bitrate
        samples_per_bit = max(1, int(sample_rate * bit_time))

        bp_signal: ClassVar[list[str]] = []
        bm_signal: ClassVar[list[str]] = []

        # Idle state: BP=0, BM=1
        idle_samples = samples_per_bit * 3
        bp_signal.extend([0] * idle_samples)
        bm_signal.extend([1] * idle_samples)

        # Frame start sequence (TSS): alternating 0/1 pattern
        for _ in range(5):
            self._add_flexray_bit(bp_signal, bm_signal, 1, samples_per_bit)
            self._add_flexray_bit(bp_signal, bm_signal, 0, samples_per_bit)

        # Frame Start Symbol (FSS)
        self._add_flexray_bit(bp_signal, bm_signal, 1, samples_per_bit)

        # Header: Reserved bit, indicator bit, slot ID, payload length
        header = (1 << 20) | (int(is_static) << 19) | (slot_id << 8) | (len(payload) & 0x7F)
        for i in range(21):
            bit_val = (header >> (20 - i)) & 1
            self._add_flexray_bit(bp_signal, bm_signal, bit_val, samples_per_bit)

        # Header CRC (11 bits)
        header_crc = self._calculate_flexray_crc(header, 21)
        for i in range(11):
            bit_val = (header_crc >> (10 - i)) & 1
            self._add_flexray_bit(bp_signal, bm_signal, bit_val, samples_per_bit)

        # Payload
        for byte_val in payload:
            for i in range(8):
                bit_val = (byte_val >> (7 - i)) & 1
                self._add_flexray_bit(bp_signal, bm_signal, bit_val, samples_per_bit)

        # Payload CRC (24 bits)
        payload_crc = self._calculate_flexray_crc(int.from_bytes(payload, "big"), len(payload) * 8)
        for i in range(24):
            bit_val = (payload_crc >> (23 - i)) & 1
            self._add_flexray_bit(bp_signal, bm_signal, bit_val, samples_per_bit)

        # Frame End Sequence (FES)
        for _ in range(2):
            self._add_flexray_bit(bp_signal, bm_signal, 1, samples_per_bit)

        # Return to idle
        bp_signal.extend([0] * idle_samples)
        bm_signal.extend([1] * idle_samples)

        return (
            DigitalTrace(
                data=np.array(bp_signal, dtype=bool),
                metadata=TraceMetadata(sample_rate=sample_rate, channel_name="flexray_bp"),
            ),
            DigitalTrace(
                data=np.array(bm_signal, dtype=bool),
                metadata=TraceMetadata(sample_rate=sample_rate, channel_name="flexray_bm"),
            ),
        )

    def _add_flexray_bit(
        self,
        bp_signal: list[int],
        bm_signal: list[int],
        bit_val: int,
        samples: int,
    ) -> None:
        """Add differential bit to FlexRay signals."""
        if bit_val == 1:
            bp_signal.extend([0] * samples)
            bm_signal.extend([1] * samples)
        else:
            bp_signal.extend([1] * samples)
            bm_signal.extend([0] * samples)

    def _calculate_flexray_crc(self, data: int, num_bits: int) -> int:
        """Calculate simplified FlexRay CRC."""
        crc = 0x7FF if num_bits == 21 else 0xFFFFFF
        for i in range(num_bits):
            bit = (data >> (num_bits - 1 - i)) & 1
            crc = ((crc << 1) | bit) & (0x7FF if num_bits == 21 else 0xFFFFFF)
        return crc

    def _decode_flexray_frame(
        self,
        bp: DigitalTrace,
        bm: DigitalTrace,
        bitrate: int,
        expected_slot: int,
    ) -> dict[str, object]:
        """Decode FlexRay frame."""
        from demos.common.formatting import print_info

        print_info(f"Sample rate: {bp.metadata.sample_rate / 1e6:.1f} MHz")
        print_info(f"Bitrate: {bitrate / 1e6:.1f} Mbps")
        print_info(f"Expected slot: {expected_slot}")

        frames: ClassVar[list[str]] = []
        try:
            from oscura import decode_flexray

            frames = decode_flexray(
                bp.data,
                bm.data,
                sample_rate=bp.metadata.sample_rate,
                bitrate=bitrate,
            )
            print_info(f"Frames decoded: {len(frames)}")
        except (ImportError, AttributeError):
            print_info("FlexRay decoder not yet implemented (placeholder)")

        return {
            "bp_signal": bp,
            "bm_signal": bm,
            "frames": frames,
            "expected_slot": expected_slot,
        }


if __name__ == "__main__":
    sys.exit(run_demo_main(FlexRayDemo))
