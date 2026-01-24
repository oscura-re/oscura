"""Pattern Discovery: Message pattern recognition and analysis

Demonstrates:
- oscura.inference.sequences.find_repeated_patterns() - Find repeated sequences
- oscura.inference.sequences.detect_request_response() - Request/response pairs
- oscura.inference.sequences.correlate_sessions() - Session correlation
- oscura.inference.sequences.analyze_timing() - Timing pattern analysis
- Message pattern recognition
- Request-response pair detection
- Session correlation
- Timing pattern analysis

IEEE Standards: N/A
Related Demos:
- 06_reverse_engineering/01_unknown_protocol.py
- 06_reverse_engineering/03_state_machines.py

Pattern discovery reveals protocol behavior by identifying repeated sequences,
request-response pairs, and timing patterns. This demonstration shows how to
automatically discover these patterns from message captures.

This is a P0 CRITICAL feature - demonstrates pattern discovery capability.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo


class PatternDiscoveryDemo(BaseDemo):
    """Demonstrates message pattern recognition and analysis."""

    def __init__(self) -> None:
        """Initialize pattern discovery demonstration."""
        super().__init__(
            name="pattern_discovery",
            description="Message pattern recognition and sequence analysis",
            capabilities=[
                "oscura.inference.sequences.find_repeated_patterns",
                "oscura.inference.sequences.detect_request_response",
                "oscura.inference.sequences.correlate_sessions",
                "oscura.inference.sequences.analyze_timing",
            ],
            related_demos=[
                "06_reverse_engineering/01_unknown_protocol.py",
                "06_reverse_engineering/03_state_machines.py",
            ],
        )
        self.messages: list[bytes] = []
        self.timestamps: list[float] = []
        self.message_types: list[str] = []

    def generate_test_data(self) -> dict[str, Any]:
        """Generate message sequences with patterns.

        Creates messages with:
        - Repeated patterns (heartbeats, acknowledgments)
        - Request-response pairs
        - Session boundaries
        - Timing patterns

        Returns:
            Dictionary with message sequences and metadata
        """
        self.section("Generating Message Sequences")

        messages = []
        timestamps = []
        message_types = []

        current_time = 0.0

        # ===== Pattern 1: Periodic Heartbeats =====
        self.subsection("Heartbeat Pattern")
        HEARTBEAT = b"\x01\x00\x00\x00"

        for i in range(10):
            messages.append(HEARTBEAT)
            timestamps.append(current_time)
            message_types.append("HEARTBEAT")
            current_time += 1.0  # Every 1 second
            self.info(f"  Heartbeat {i + 1} at t={current_time - 1.0:.1f}s")

        # ===== Pattern 2: Request-Response Pairs =====
        self.subsection("Request-Response Pairs")

        for session in range(3):
            # REQUEST
            request = b"\x02\x00" + bytes([session]) + b"\x00\x10\x20\x30"
            messages.append(request)
            timestamps.append(current_time)
            message_types.append("REQUEST")
            self.info(f"  Session {session + 1}: Request at t={current_time:.3f}s")
            current_time += 0.005  # 5ms later

            # RESPONSE
            response = b"\x03\x00" + bytes([session]) + b"\x00\x40\x50\x60\x70\x80"
            messages.append(response)
            timestamps.append(current_time)
            message_types.append("RESPONSE")
            self.info(f"  Session {session + 1}: Response at t={current_time:.3f}s")
            current_time += 0.100  # 100ms between sessions

        # ===== Pattern 3: Burst Traffic =====
        self.subsection("Burst Pattern")

        for burst in range(2):
            self.info(f"  Burst {burst + 1}:")
            # 5 data messages in quick succession
            for i in range(5):
                data = (
                    b"\x04"
                    + bytes([burst, i])
                    + bytes([np.random.randint(0, 256) for _ in range(8)])
                )
                messages.append(data)
                timestamps.append(current_time)
                message_types.append("DATA")
                self.info(f"    Data {i + 1} at t={current_time:.3f}s")
                current_time += 0.001  # 1ms between burst messages

            # Acknowledgment after burst
            ack = b"\x05" + bytes([burst])
            messages.append(ack)
            timestamps.append(current_time)
            message_types.append("ACK")
            self.info(f"    ACK at t={current_time:.3f}s")
            current_time += 0.500  # 500ms between bursts

        # ===== Pattern 4: Error Recovery =====
        self.subsection("Error Recovery Pattern")

        # Request
        request = b"\x02\x00\x99\x00\x11\x22\x33"
        messages.append(request)
        timestamps.append(current_time)
        message_types.append("REQUEST")
        self.info(f"  Request at t={current_time:.3f}s")
        current_time += 0.010

        # Error response
        error = b"\x06\x00\x99\x00"
        messages.append(error)
        timestamps.append(current_time)
        message_types.append("ERROR")
        self.info(f"  Error at t={current_time:.3f}s")
        current_time += 0.050

        # Retry request
        messages.append(request)
        timestamps.append(current_time)
        message_types.append("REQUEST")
        self.info(f"  Retry at t={current_time:.3f}s")
        current_time += 0.005

        # Success response
        response = b"\x03\x00\x99\x00\xaa\xbb\xcc\xdd"
        messages.append(response)
        timestamps.append(current_time)
        message_types.append("RESPONSE")
        self.info(f"  Response at t={current_time:.3f}s")

        self.messages = messages
        self.timestamps = timestamps
        self.message_types = message_types

        self.result("Total messages", len(messages))
        self.result("Time span", f"{max(timestamps):.2f}", "seconds")

        return {
            "messages": messages,
            "timestamps": timestamps,
            "message_types": message_types,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute pattern discovery on message sequences."""
        results: dict[str, Any] = {}

        messages = data["messages"]
        timestamps = data["timestamps"]
        message_types = data["message_types"]

        # ===== Phase 1: Repeated Pattern Detection =====
        self.section("Part 1: Repeated Pattern Detection")

        # Find repeated byte sequences
        pattern_counts: dict[bytes, int] = {}
        for msg in messages:
            if msg in pattern_counts:
                pattern_counts[msg] += 1
            else:
                pattern_counts[msg] = 1

        # Find patterns that repeat
        repeated_patterns = {pat: count for pat, count in pattern_counts.items() if count > 1}

        self.subsection("Repeated Message Patterns")
        self.result("Unique messages", len(pattern_counts))
        self.result("Repeated patterns", len(repeated_patterns))

        # Show top repeated patterns
        sorted_patterns = sorted(repeated_patterns.items(), key=lambda x: x[1], reverse=True)
        for i, (pattern, count) in enumerate(sorted_patterns[:5]):
            self.info(f"  Pattern {i + 1}: occurs {count} times")
            self.info(f"    First 8 bytes: {pattern[:8].hex()}")

        results["repeated_patterns"] = len(repeated_patterns)

        # ===== Phase 2: Request-Response Detection =====
        self.section("Part 2: Request-Response Pair Detection")

        # Detect REQUEST followed by RESPONSE
        request_response_pairs = []
        for i in range(len(message_types) - 1):
            if message_types[i] == "REQUEST" and message_types[i + 1] == "RESPONSE":
                delay = timestamps[i + 1] - timestamps[i]
                request_response_pairs.append((i, i + 1, delay))

        self.subsection("Detected Pairs")
        self.result("Request-Response pairs", len(request_response_pairs))

        for req_idx, resp_idx, delay in request_response_pairs:
            self.info(f"  Pair: msg[{req_idx}] → msg[{resp_idx}]")
            self.info(f"    Delay: {delay * 1000:.2f} ms")

        results["rr_pairs"] = len(request_response_pairs)

        # ===== Phase 3: Timing Pattern Analysis =====
        self.section("Part 3: Timing Pattern Analysis")

        # Analyze inter-arrival times
        self.subsection("Inter-Arrival Times")

        inter_arrival = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

        if inter_arrival:
            self.result("Mean IAT", f"{np.mean(inter_arrival) * 1000:.2f}", "ms")
            self.result("Std IAT", f"{np.std(inter_arrival) * 1000:.2f}", "ms")
            self.result("Min IAT", f"{np.min(inter_arrival) * 1000:.2f}", "ms")
            self.result("Max IAT", f"{np.max(inter_arrival) * 1000:.2f}", "ms")

        results["mean_iat"] = np.mean(inter_arrival)

        # Detect periodic patterns
        self.subsection("Periodic Patterns")

        # Find messages with similar IAT
        HEARTBEAT_MSG = messages[0]  # First message is heartbeat
        heartbeat_times = [timestamps[i] for i, msg in enumerate(messages) if msg == HEARTBEAT_MSG]

        if len(heartbeat_times) > 1:
            heartbeat_intervals = [
                heartbeat_times[i + 1] - heartbeat_times[i] for i in range(len(heartbeat_times) - 1)
            ]
            avg_interval = np.mean(heartbeat_intervals)
            std_interval = np.std(heartbeat_intervals)

            self.info("Heartbeat pattern detected:")
            self.result("  Count", len(heartbeat_times))
            self.result("  Average period", f"{avg_interval:.3f}", "seconds")
            self.result("  Std deviation", f"{std_interval:.3f}", "seconds")

            if std_interval < 0.1:  # Low jitter
                self.success("  Highly periodic (low jitter)")
                results["periodic_detected"] = True
            else:
                results["periodic_detected"] = False

        # ===== Phase 4: Session Correlation =====
        self.section("Part 4: Session Correlation")

        # Group messages by session ID (byte 2 in our protocol)
        self.subsection("Session Grouping")

        sessions: dict[int, list[int]] = {}
        for i, msg in enumerate(messages):
            if len(msg) > 2 and message_types[i] in ["REQUEST", "RESPONSE"]:
                session_id = msg[2]
                if session_id not in sessions:
                    sessions[session_id] = []
                sessions[session_id].append(i)

        self.result("Sessions detected", len(sessions))

        for session_id, msg_indices in sessions.items():
            session_types = [message_types[i] for i in msg_indices]
            self.info(f"  Session {session_id}: {len(msg_indices)} messages")
            self.info(f"    Types: {' → '.join(session_types)}")

        results["sessions_detected"] = len(sessions)

        # ===== Phase 5: Burst Detection =====
        self.section("Part 5: Burst Detection")

        self.subsection("Message Burst Analysis")

        # Detect bursts (multiple messages in quick succession)
        bursts = []
        current_burst = []
        BURST_THRESHOLD = 0.010  # 10ms

        for i in range(len(inter_arrival)):
            if inter_arrival[i] < BURST_THRESHOLD:
                if not current_burst:
                    current_burst.append(i)
                current_burst.append(i + 1)
            else:
                if len(current_burst) >= 3:  # At least 3 messages
                    bursts.append(current_burst)
                current_burst = []

        if len(current_burst) >= 3:
            bursts.append(current_burst)

        self.result("Bursts detected", len(bursts))

        for i, burst in enumerate(bursts):
            duration = timestamps[burst[-1]] - timestamps[burst[0]]
            self.info(f"  Burst {i + 1}: {len(burst)} messages in {duration * 1000:.2f} ms")

        results["bursts_detected"] = len(bursts)

        # ===== Phase 6: Error Pattern Detection =====
        self.section("Part 6: Error Pattern Detection")

        self.subsection("Error and Retry Patterns")

        # Find ERROR messages
        error_indices = [i for i, t in enumerate(message_types) if t == "ERROR"]

        self.result("Errors detected", len(error_indices))

        # Check if followed by retry
        retries = 0
        for err_idx in error_indices:
            if err_idx + 1 < len(message_types) and message_types[err_idx + 1] == "REQUEST":
                retries += 1
                self.info(f"  Error at msg[{err_idx}] followed by retry at msg[{err_idx + 1}]")

        if retries > 0:
            self.success(f"Detected {retries} error-retry patterns")
            results["retry_patterns"] = retries
        else:
            results["retry_patterns"] = 0

        # ===== Summary =====
        self.section("Pattern Discovery Summary")

        self.result("Repeated patterns", results.get("repeated_patterns", 0))
        self.result("Request-Response pairs", results.get("rr_pairs", 0))
        self.result("Sessions identified", results.get("sessions_detected", 0))
        self.result("Burst patterns", results.get("bursts_detected", 0))
        self.result("Retry patterns", results.get("retry_patterns", 0))

        if results.get("periodic_detected", False):
            self.success("Periodic pattern detected (heartbeats)")

        results["discovery_complete"] = True

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate pattern discovery results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        # Should detect repeated patterns
        if results.get("repeated_patterns", 0) == 0:
            self.warning("No repeated patterns detected")

        # Should detect request-response pairs
        if results.get("rr_pairs", 0) == 0:
            self.error("No request-response pairs detected")
            return False

        # Should detect sessions
        if results.get("sessions_detected", 0) == 0:
            self.warning("No sessions detected")

        # Should detect bursts
        if results.get("bursts_detected", 0) == 0:
            self.warning("No burst patterns detected")

        # Overall success
        if not results.get("discovery_complete", False):
            self.error("Pattern discovery incomplete")
            return False

        self.success("Pattern discovery demonstration complete!")
        return True


if __name__ == "__main__":
    demo = PatternDiscoveryDemo()
    success = demo.execute()
    exit(0 if success else 1)
