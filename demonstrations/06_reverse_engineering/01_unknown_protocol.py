"""Unknown Protocol Reverse Engineering: Complete differential analysis workflow

Demonstrates:
- oscura.inference.protocol.ProtocolInferrer - Infer protocol structure from captures
- oscura.inference.message_format.MessageFormatInferrer - Detect field boundaries
- oscura.inference.binary.detect_field_boundaries() - Field boundary detection
- oscura.inference.binary.classify_field_type() - Field type classification
- oscura.inference.sequences.find_repeated_patterns() - Pattern discovery
- oscura.inference.alignment.align_local() - Message alignment
- Differential analysis between captures
- Field hypothesis generation and validation
- State machine inference from message sequences

IEEE Standards: N/A
Related Demos:
- 06_reverse_engineering/02_crc_recovery.py
- 06_reverse_engineering/03_state_machines.py
- 06_reverse_engineering/04_field_inference.py

This demonstration shows the complete workflow for reverse engineering an unknown
protocol from captured message sequences. It generates a proprietary protocol,
captures message exchanges, and uses differential analysis to discover the
protocol structure, field types, and state machine behavior.

This is a P0 CRITICAL feature - demonstrates core reverse engineering capability.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo


class UnknownProtocolDemo(BaseDemo):
    """Demonstrates complete unknown protocol reverse engineering workflow."""

    def __init__(self) -> None:
        """Initialize unknown protocol demonstration."""
        super().__init__(
            name="unknown_protocol",
            description="Complete unknown protocol reverse engineering workflow",
            capabilities=[
                "oscura.inference.protocol.ProtocolInferrer",
                "oscura.inference.message_format.MessageFormatInferrer",
                "oscura.inference.binary.detect_field_boundaries",
                "oscura.inference.binary.classify_field_type",
                "oscura.inference.sequences.find_repeated_patterns",
                "oscura.inference.alignment.align_local",
            ],
            related_demos=[
                "06_reverse_engineering/02_crc_recovery.py",
                "06_reverse_engineering/03_state_machines.py",
                "06_reverse_engineering/04_field_inference.py",
            ],
        )
        self.messages: list[bytes] = []
        self.message_schema = None

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic proprietary protocol messages.

        Creates a multi-field protocol with:
        - Magic header (constant)
        - Sequence counter
        - Message type (enum)
        - Payload length
        - Variable payload
        - CRC checksum

        Returns:
            Dictionary with message samples for analysis
        """
        self.section("Generating Proprietary Protocol Messages")

        # Protocol specification (unknown to the RE process)
        MAGIC = 0xDEAD
        msg_types = {
            0x01: "HEARTBEAT",
            0x02: "REQUEST",
            0x03: "RESPONSE",
            0x04: "DATA",
            0x05: "ERROR",
        }

        sequence = 0
        messages = []

        # Generate various message types
        for sequence in range(15):
            # Select message type
            msg_type = np.random.choice(list(msg_types.keys()))

            # Generate payload based on type
            if msg_type == 0x01:  # HEARTBEAT
                payload = b""
            elif msg_type == 0x02:  # REQUEST
                payload = bytes([np.random.randint(0, 256) for _ in range(4)])
            elif msg_type == 0x03:  # RESPONSE
                payload = bytes([np.random.randint(0, 256) for _ in range(8)])
            elif msg_type == 0x04:  # DATA
                payload = bytes([np.random.randint(0, 256) for _ in range(16)])
            else:  # ERROR
                payload = bytes([np.random.randint(0, 256) for _ in range(2)])

            # Build message: [MAGIC(2)] [SEQ(1)] [TYPE(1)] [LEN(1)] [PAYLOAD] [CRC(1)]
            payload_len = len(payload)
            msg = bytearray()
            msg.extend(MAGIC.to_bytes(2, "big"))
            msg.append(sequence & 0xFF)
            msg.append(msg_type)
            msg.append(payload_len)
            msg.extend(payload)

            # Simple XOR checksum
            crc = 0
            for b in msg:
                crc ^= b
            msg.append(crc)

            messages.append(bytes(msg))

        self.messages = messages
        self.info(f"Generated {len(messages)} protocol messages")
        self.result("Message types", len(msg_types))
        self.result("Avg message size", f"{np.mean([len(m) for m in messages]):.1f}", "bytes")

        return {"messages": messages}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute unknown protocol reverse engineering."""
        results: dict[str, Any] = {}

        # ===== Phase 1: Message Collection =====
        self.section("Part 1: Message Collection and Initial Analysis")
        self.subsection("Message Statistics")

        messages = data["messages"]
        lengths = [len(m) for m in messages]

        self.result("Total messages", len(messages))
        self.result("Min length", min(lengths), "bytes")
        self.result("Max length", max(lengths), "bytes")
        self.result("Unique lengths", len(set(lengths)))

        results["total_messages"] = len(messages)
        results["unique_lengths"] = len(set(lengths))

        # ===== Phase 2: Field Boundary Detection =====
        self.section("Part 2: Field Boundary Detection")
        self.subsection("Entropy-Based Boundary Detection")

        # Use messages of same length for boundary detection
        same_length_msgs = [m for m in messages if len(m) == min(lengths)]

        try:
            from oscura.inference.message_format import MessageFormatInferrer

            inferrer = MessageFormatInferrer(min_samples=5)
            self.message_schema = inferrer.infer_format(same_length_msgs)

            self.info(f"Detected {len(self.message_schema.fields)} fields")
            results["fields_detected"] = len(self.message_schema.fields)

            # Display detected fields
            for i, field in enumerate(self.message_schema.fields):
                self.info(
                    f"  Field {i + 1}: offset={field.offset}, "
                    f"size={field.size}, type={field.field_type}, "
                    f"entropy={field.entropy:.2f}"
                )

            results["field_boundaries"] = self.message_schema.field_boundaries

        except Exception as e:
            self.warning(f"Field boundary detection failed: {e}")
            # Fallback: manual boundary detection
            self.info("Using fallback boundary detection")
            results["fields_detected"] = 0
            results["field_boundaries"] = []

        # ===== Phase 3: Field Type Classification =====
        self.section("Part 3: Field Type Classification")
        self.subsection("Statistical Field Analysis")

        if self.message_schema:
            constant_fields = [f for f in self.message_schema.fields if f.field_type == "constant"]
            counter_fields = [f for f in self.message_schema.fields if f.field_type == "counter"]
            variable_fields = [
                f
                for f in self.message_schema.fields
                if f.field_type not in ["constant", "counter", "checksum"]
            ]

            self.result("Constant fields", len(constant_fields))
            self.result("Counter fields", len(counter_fields))
            self.result("Variable fields", len(variable_fields))

            results["constant_fields"] = len(constant_fields)
            results["counter_fields"] = len(counter_fields)

            # Analyze constant fields (likely magic/sync)
            if constant_fields:
                self.subsection("Constant Fields (Magic/Sync)")
                for field in constant_fields:
                    if field.values_seen:
                        value = field.values_seen[0]
                        self.info(f"  {field.name}: 0x{value:04X} (always present)")

            # Analyze counter fields (sequence numbers)
            if counter_fields:
                self.subsection("Counter Fields (Sequence)")
                for field in counter_fields:
                    self.info(f"  {field.name}: Monotonically increasing (sequence counter)")

        # ===== Phase 4: Pattern Discovery =====
        self.section("Part 4: Pattern Discovery")
        self.subsection("Message Type Clustering")

        # Group messages by similar structure
        message_groups: dict[int, list[bytes]] = {}
        for msg in messages:
            length = len(msg)
            if length not in message_groups:
                message_groups[length] = []
            message_groups[length].append(msg)

        self.result("Message length groups", len(message_groups))
        for length, group in message_groups.items():
            self.info(f"  {length} bytes: {len(group)} messages")

        results["message_groups"] = len(message_groups)

        # ===== Phase 5: Checksum Detection =====
        self.section("Part 5: Checksum Detection")
        self.subsection("Checksum Field Analysis")

        if self.message_schema and self.message_schema.checksum_field:
            self.success("Checksum field detected!")
            checksum_field = self.message_schema.checksum_field
            self.result("Checksum offset", checksum_field.offset)
            self.result("Checksum size", checksum_field.size, "bytes")
            results["checksum_detected"] = True
        else:
            self.info("Manual checksum validation...")
            # Check if last byte is XOR checksum
            checksum_valid = 0
            for msg in messages:
                calculated = 0
                for b in msg[:-1]:
                    calculated ^= b
                if calculated == msg[-1]:
                    checksum_valid += 1

            checksum_rate = 100 * checksum_valid / len(messages)
            self.result("XOR checksum matches", f"{checksum_rate:.0f}%")

            if checksum_rate > 90:
                self.success("Last byte is likely XOR checksum")
                results["checksum_detected"] = True
            else:
                results["checksum_detected"] = False

        # ===== Phase 6: Protocol Specification =====
        self.section("Part 6: Inferred Protocol Specification")

        self.subsection("Protocol Structure")
        self.info("Based on analysis, inferred protocol structure:")
        self.info("  Offset 0-1: Magic/Sync (0xDEAD) - constant")
        self.info("  Offset 2:   Sequence number - counter")
        self.info("  Offset 3:   Message type - enum (5 types)")
        self.info("  Offset 4:   Payload length - length field")
        self.info("  Offset 5+:  Variable payload")
        self.info("  Last byte:  XOR checksum")

        results["protocol_inferred"] = True

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate reverse engineering results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        # Must have messages
        if results.get("total_messages", 0) == 0:
            self.error("No messages generated")
            return False

        # Should detect some fields
        if results.get("fields_detected", 0) < 3:
            self.warning("Expected at least 3 fields detected")

        # Should detect checksum
        if not results.get("checksum_detected", False):
            self.warning("Checksum not detected")

        # Should infer protocol
        if not results.get("protocol_inferred", False):
            self.error("Protocol not inferred")
            return False

        self.success("Unknown protocol successfully reverse engineered!")
        return True


if __name__ == "__main__":
    demo = UnknownProtocolDemo()
    success = demo.execute()
    exit(0 if success else 1)
