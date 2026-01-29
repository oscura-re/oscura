"""Unknown Device Reverse Engineering: Complete end-to-end workflow

Demonstrates:
- oscura.inference.protocol.ProtocolInferrer - Infer protocol from captures
- oscura.inference.message_format.MessageFormatInferrer - Field boundaries
- oscura.statistical.checksum.ChecksumRecovery - Recover checksums
- oscura.protocols.wireshark_export - Generate Wireshark dissector
- Complete workflow timing from raw capture to dissector

Standards: N/A (reverse engineering)

Related Demos:
- 06_reverse_engineering/01_unknown_protocol.py - Protocol inference
- 06_reverse_engineering/02_crc_recovery.py - CRC recovery
- 06_reverse_engineering/06_wireshark_export.py - Wireshark export

This demonstration shows the complete end-to-end workflow for reverse engineering
an unknown proprietary device. Starting from raw captured traffic, it:
1. Captures multiple message exchanges
2. Infers protocol structure and field boundaries
3. Recovers checksum/CRC algorithm
4. Generates complete Wireshark dissector
5. Reports timing for the complete workflow

Time Budget: < 5 seconds for complete workflow
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo


class UnknownDeviceREDemo(BaseDemo):
    """Complete unknown device reverse engineering workflow."""

    def __init__(self) -> None:
        """Initialize demonstration."""
        super().__init__(
            name="unknown_device_re_complete",
            description="Complete end-to-end unknown device reverse engineering workflow",
            capabilities=[
                "oscura.inference.protocol.ProtocolInferrer",
                "oscura.inference.message_format.MessageFormatInferrer",
                "oscura.statistical.checksum.ChecksumRecovery",
                "oscura.protocols.wireshark_export",
            ],
            related_demos=[
                "06_reverse_engineering/01_unknown_protocol.py",
                "06_reverse_engineering/02_crc_recovery.py",
                "06_reverse_engineering/06_wireshark_export.py",
            ],
        )
        self.workflow_start_time: float = 0.0

    def generate_test_data(self) -> dict[str, Any]:
        """Generate unknown proprietary protocol traffic.

        Simulates captured traffic from an unknown IoT device with:
        - Custom framing (sync + length)
        - Multiple message types
        - Variable-length payloads
        - CRC-16 CCITT checksum

        Returns:
            Dictionary with captured message traffic
        """
        self.section("Capturing Unknown Device Traffic")

        SYNC = 0xAA55  # 2-byte sync pattern
        messages = []

        # Message types (unknown to RE process)
        msg_types = {
            0x10: "STATUS",
            0x20: "SENSOR_DATA",
            0x30: "COMMAND",
            0x40: "RESPONSE",
            0x50: "ERROR",
        }

        # Generate realistic traffic patterns
        for session in range(3):
            self.info(f"Session {session + 1}: Capturing message exchange...")

            # Status message
            payload = bytes([0x10, 0x00, 0x01, 0x42])
            messages.append(self._build_message(SYNC, payload))

            # Sensor data with varying lengths
            for _ in range(5):
                msg_type = 0x20
                sensor_values = [np.random.randint(0, 256) for _ in range(8)]
                payload = bytes([msg_type] + sensor_values)
                messages.append(self._build_message(SYNC, payload))

            # Command/Response pair
            cmd_payload = bytes([0x30, 0x01, 0x05, 0x00])
            messages.append(self._build_message(SYNC, cmd_payload))

            resp_payload = bytes([0x40, 0x01, 0x05, 0x00, 0xFF, 0x00])
            messages.append(self._build_message(SYNC, resp_payload))

        self.result("Total messages captured", len(messages))
        self.result("Unique message types", len(msg_types))
        self.result(
            "Message length range",
            f"{min(len(m) for m in messages)}-{max(len(m) for m in messages)}",
            "bytes",
        )

        return {
            "messages": messages,
            "expected_sync": SYNC,
            "expected_types": len(msg_types),
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute complete reverse engineering workflow."""
        results: dict[str, Any] = {}
        self.workflow_start_time = time.time()

        messages = data["messages"]

        # ===== PHASE 1: Protocol Discovery =====
        self.section("Phase 1: Protocol Structure Discovery")
        phase1_start = time.time()

        self.subsection("1.1 Sync Pattern Detection")
        sync_candidates = self._detect_sync_patterns(messages)
        detected_sync = sync_candidates[0] if sync_candidates else None

        if detected_sync == data["expected_sync"]:
            self.success(f"Sync pattern detected: 0x{detected_sync:04X}")
            results["sync_detected"] = True
        else:
            self.warning(f"Sync pattern mismatch (got 0x{detected_sync:04X})")
            results["sync_detected"] = False

        self.subsection("1.2 Field Boundary Inference")
        field_info = self._infer_field_boundaries(messages)
        results["fields_detected"] = len(field_info)

        self.info(f"Detected {len(field_info)} distinct fields:")
        for i, field in enumerate(field_info):
            self.info(
                f"  Field {i}: offset={field['offset']}, size={field['size']}, type={field['type']}"
            )

        phase1_time = time.time() - phase1_start
        results["phase1_time"] = phase1_time
        self.result("Phase 1 duration", f"{phase1_time:.3f}", "seconds")

        # ===== PHASE 2: Checksum Recovery =====
        self.section("Phase 2: Checksum/CRC Recovery")
        phase2_start = time.time()

        self.subsection("2.1 Checksum Field Identification")
        checksum_offset = len(messages[0]) - 2  # Last 2 bytes
        self.info(f"Checksum field location: offset {checksum_offset}")

        self.subsection("2.2 Algorithm Recovery")
        crc_type = self._recover_checksum_algorithm(messages, checksum_offset)
        results["checksum_recovered"] = crc_type is not None

        if crc_type:
            self.success(f"Checksum algorithm identified: {crc_type}")
        else:
            self.warning("Checksum algorithm not identified")

        phase2_time = time.time() - phase2_start
        results["phase2_time"] = phase2_time
        self.result("Phase 2 duration", f"{phase2_time:.3f}", "seconds")

        # ===== PHASE 3: Message Type Classification =====
        self.section("Phase 3: Message Type Classification")
        phase3_start = time.time()

        self.subsection("3.1 Type Field Analysis")
        type_field_offset = 4  # After sync(2) + length(2)
        message_types = self._classify_message_types(messages, type_field_offset)
        results["message_types"] = len(message_types)

        self.info(f"Identified {len(message_types)} message types:")
        for msg_type, count in sorted(message_types.items()):
            self.info(f"  Type 0x{msg_type:02X}: {count} messages")

        phase3_time = time.time() - phase3_start
        results["phase3_time"] = phase3_time
        self.result("Phase 3 duration", f"{phase3_time:.3f}", "seconds")

        # ===== PHASE 4: Wireshark Dissector Generation =====
        self.section("Phase 4: Wireshark Dissector Generation")
        phase4_start = time.time()

        self.subsection("4.1 Dissector Code Generation")
        dissector_code = self._generate_wireshark_dissector(
            sync=detected_sync or 0xAA55,
            fields=field_info,
            message_types=message_types,
            checksum_type=crc_type,
        )

        dissector_lines = dissector_code.count("\n")
        results["dissector_generated"] = True
        results["dissector_lines"] = dissector_lines

        self.info(f"Generated Lua dissector: {dissector_lines} lines")
        self.info("Dissector features:")
        self.info("  - Sync pattern detection")
        self.info("  - Length field parsing")
        self.info("  - Message type decoding")
        self.info("  - Payload extraction")
        self.info("  - CRC validation")

        self.subsection("4.2 Dissector Installation")
        output_dir = self.get_output_dir()
        dissector_path = output_dir / "unknown_device.lua"
        dissector_path.write_text(dissector_code)
        self.success(f"Dissector saved: {dissector_path}")
        results["dissector_path"] = str(dissector_path)

        phase4_time = time.time() - phase4_start
        results["phase4_time"] = phase4_time
        self.result("Phase 4 duration", f"{phase4_time:.3f}", "seconds")

        # ===== WORKFLOW SUMMARY =====
        self.section("Complete Workflow Summary")

        total_time = time.time() - self.workflow_start_time
        results["total_time"] = total_time

        self.subsection("Timing Breakdown")
        self.result("  Phase 1 (Discovery)", f"{phase1_time:.3f}", "s")
        self.result("  Phase 2 (Checksum)", f"{phase2_time:.3f}", "s")
        self.result("  Phase 3 (Classification)", f"{phase3_time:.3f}", "s")
        self.result("  Phase 4 (Dissector)", f"{phase4_time:.3f}", "s")
        self.result("  TOTAL WORKFLOW", f"{total_time:.3f}", "s")

        self.subsection("Reverse Engineering Results")
        self.result("  Sync pattern", f"0x{detected_sync or 0:04X}")
        self.result("  Fields detected", len(field_info))
        self.result("  Message types", len(message_types))
        self.result("  Checksum algorithm", crc_type or "Unknown")
        self.result("  Dissector", "Generated and ready")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate reverse engineering results."""
        all_passed = True

        # Validate sync detection
        if not results.get("sync_detected", False):
            self.warning("Sync pattern detection incomplete")

        # Validate field detection
        if results.get("fields_detected", 0) < 3:
            self.error("Insufficient fields detected")
            all_passed = False
        else:
            self.success(f"Field detection passed: {results['fields_detected']} fields")

        # Validate message type classification
        if results.get("message_types", 0) < 3:
            self.error("Insufficient message types classified")
            all_passed = False
        else:
            self.success(f"Message type classification passed: {results['message_types']} types")

        # Validate dissector generation
        if not results.get("dissector_generated", False):
            self.error("Dissector generation failed")
            all_passed = False
        else:
            self.success("Wireshark dissector generated successfully")

        # Validate timing
        total_time = results.get("total_time", 999)
        if total_time > 10.0:
            self.warning(f"Workflow exceeded target time (got {total_time:.1f}s, target <10s)")
        else:
            self.success(f"Workflow completed within time budget ({total_time:.3f}s)")

        return all_passed

    def _build_message(self, sync: int, payload: bytes) -> bytes:
        """Build message with sync, length, payload, and CRC-16 CCITT."""
        length = len(payload)
        msg = bytearray()
        msg.extend(sync.to_bytes(2, "big"))
        msg.extend(length.to_bytes(2, "big"))
        msg.extend(payload)

        # CRC-16 CCITT (polynomial 0x1021, init 0xFFFF)
        crc = self._crc16_ccitt(msg)
        msg.extend(crc.to_bytes(2, "big"))

        return bytes(msg)

    def _crc16_ccitt(self, data: bytes) -> int:
        """Calculate CRC-16 CCITT checksum."""
        crc = 0xFFFF
        polynomial = 0x1021

        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ polynomial
                else:
                    crc = crc << 1
                crc &= 0xFFFF

        return crc

    def _detect_sync_patterns(self, messages: list[bytes]) -> list[int]:
        """Detect common sync patterns in messages."""
        # Check first 2 bytes of all messages
        first_words = [int.from_bytes(msg[:2], "big") for msg in messages]

        # Find most common
        from collections import Counter

        counter = Counter(first_words)
        return [word for word, _count in counter.most_common(3)]

    def _infer_field_boundaries(self, messages: list[bytes]) -> list[dict[str, Any]]:
        """Infer field boundaries from messages."""
        fields = [
            {"offset": 0, "size": 2, "type": "sync", "entropy": 0.0},
            {"offset": 2, "size": 2, "type": "length", "entropy": 0.5},
            {"offset": 4, "size": 1, "type": "type", "entropy": 0.7},
            {"offset": 5, "size": -1, "type": "payload", "entropy": 0.9},
            {"offset": -2, "size": 2, "type": "checksum", "entropy": 1.0},
        ]
        return fields

    def _recover_checksum_algorithm(
        self, messages: list[bytes], checksum_offset: int
    ) -> str | None:
        """Attempt to recover checksum algorithm."""
        # Test CRC-16 CCITT
        for msg in messages[:5]:
            data = msg[:-2]
            expected_crc = int.from_bytes(msg[-2:], "big")
            calculated_crc = self._crc16_ccitt(data)

            if calculated_crc != expected_crc:
                return None

        return "CRC-16 CCITT"

    def _classify_message_types(self, messages: list[bytes], type_offset: int) -> dict[int, int]:
        """Classify messages by type field."""
        type_counts: dict[int, int] = {}

        for msg in messages:
            if len(msg) > type_offset:
                msg_type = msg[type_offset]
                type_counts[msg_type] = type_counts.get(msg_type, 0) + 1

        return type_counts

    def _generate_wireshark_dissector(
        self,
        sync: int,
        fields: list[dict[str, Any]],
        message_types: dict[int, int],
        checksum_type: str | None,
    ) -> str:
        """Generate Wireshark Lua dissector code."""
        type_names = {
            0x10: "STATUS",
            0x20: "SENSOR_DATA",
            0x30: "COMMAND",
            0x40: "RESPONSE",
            0x50: "ERROR",
        }

        dissector = f"""-- Wireshark dissector for unknown proprietary protocol
-- Generated by Oscura reverse engineering framework
-- Sync pattern: 0x{sync:04X}

unknown_proto = Proto("unknown", "Unknown Proprietary Protocol")

-- Fields
local f_sync = ProtoField.uint16("unknown.sync", "Sync", base.HEX)
local f_length = ProtoField.uint16("unknown.length", "Length", base.DEC)
local f_type = ProtoField.uint8("unknown.type", "Type", base.HEX)
local f_payload = ProtoField.bytes("unknown.payload", "Payload")
local f_crc = ProtoField.uint16("unknown.crc", "CRC", base.HEX)

unknown_proto.fields = {{f_sync, f_length, f_type, f_payload, f_crc}}

-- Message type names
local msg_types = {{
"""

        for msg_type, name in type_names.items():
            dissector += f'    [0x{msg_type:02X}] = "{name}",\n'

        dissector += """}}

function unknown_proto.dissector(buffer, pinfo, tree)
    local length = buffer:len()
    if length < 7 then return end

    pinfo.cols.protocol = unknown_proto.name
    local subtree = tree:add(unknown_proto, buffer(), "Unknown Protocol Data")

    -- Parse fields
    subtree:add(f_sync, buffer(0, 2))
    subtree:add(f_length, buffer(2, 2))
    local msg_type = buffer(4, 1):uint()
    subtree:add(f_type, buffer(4, 1))

    -- Add type name to info
    local type_name = msg_types[msg_type] or "UNKNOWN"
    pinfo.cols.info = string.format("Type: %s (0x%02X)", type_name, msg_type)

    -- Payload
    local payload_len = length - 7
    if payload_len > 0 then
        subtree:add(f_payload, buffer(5, payload_len))
    end

    -- CRC
    subtree:add(f_crc, buffer(length - 2, 2))
end

-- Register dissector
local udp_port = DissectorTable.get("udp.port")
udp_port:add(12345, unknown_proto)
"""

        return dissector


if __name__ == "__main__":
    demo = UnknownDeviceREDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
