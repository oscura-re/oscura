"""State Machine Learning: Extract protocol state machines from traces

Demonstrates:
- oscura.inference.state_machine.StateMachineInferrer - Learn state machines
- oscura.inference.state_machine.RPNI algorithm - Passive automaton learning
- oscura.inference.state_machine.PrefixTreeAcceptor - Build PTA
- oscura.inference.state_machine.accepts() - Test sequence acceptance
- oscura.inference.state_machine.to_dot() - Export to DOT format
- State extraction from protocol interactions
- Transition diagram generation
- Sequence validation

IEEE Standards: N/A
Related Demos:
- 06_reverse_engineering/01_unknown_protocol.py
- 06_reverse_engineering/05_pattern_discovery.py

State machines are fundamental to protocol behavior. This demonstration shows
how to learn protocol state machines from observed message sequences using
the RPNI (Regular Positive and Negative Inference) algorithm.

This is a P0 CRITICAL feature - demonstrates state machine inference capability.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo


class StateMachineLearningDemo(BaseDemo):
    """Demonstrates state machine learning from protocol traces."""

    def __init__(self) -> None:
        """Initialize state machine learning demonstration."""
        super().__init__(
            name="state_machine_learning",
            description="Learn protocol state machines from message sequences",
            capabilities=[
                "oscura.inference.state_machine.StateMachineInferrer",
                "oscura.inference.state_machine.RPNI",
                "oscura.inference.state_machine.PrefixTreeAcceptor",
                "oscura.inference.state_machine.accepts",
                "oscura.inference.state_machine.to_dot",
            ],
            related_demos=[
                "06_reverse_engineering/01_unknown_protocol.py",
                "06_reverse_engineering/05_pattern_discovery.py",
            ],
        )
        self.positive_traces: list[list[str]] = []
        self.negative_traces: list[list[str]] = []
        self.learned_automaton = None

    def generate_test_data(self) -> dict[str, Any]:
        """Generate protocol traces from a known state machine.

        Creates traces representing a simple authentication protocol:
        - States: IDLE, AUTHENTICATED, ACTIVE
        - Transitions: CONNECT, LOGIN, LOGOUT, REQUEST, RESPONSE

        Returns:
            Dictionary with positive and negative trace samples
        """
        self.section("Generating Protocol Traces")

        # ===== Positive Samples (Valid Sequences) =====
        self.subsection("Valid Protocol Sequences")

        self.positive_traces = [
            # Basic authentication flow
            ["CONNECT", "LOGIN", "LOGOUT"],
            ["CONNECT", "LOGIN", "REQUEST", "RESPONSE", "LOGOUT"],
            # Multiple requests
            ["CONNECT", "LOGIN", "REQUEST", "RESPONSE", "REQUEST", "RESPONSE", "LOGOUT"],
            # Simple disconnect
            ["CONNECT", "LOGOUT"],
            # Multiple sessions
            ["CONNECT", "LOGIN", "LOGOUT", "CONNECT", "LOGIN", "LOGOUT"],
            # Extended session
            [
                "CONNECT",
                "LOGIN",
                "REQUEST",
                "RESPONSE",
                "REQUEST",
                "RESPONSE",
                "REQUEST",
                "RESPONSE",
                "LOGOUT",
            ],
            # Quick session
            ["CONNECT", "LOGIN", "REQUEST", "RESPONSE", "LOGOUT"],
            # Idle disconnect
            ["CONNECT", "LOGIN", "LOGOUT"],
        ]

        for i, trace in enumerate(self.positive_traces):
            trace_str = " → ".join(trace[:5])
            if len(trace) > 5:
                trace_str += " → ..."
            self.info(f"  Trace {i + 1}: {trace_str}")

        self.result("Valid traces", len(self.positive_traces))

        # ===== Negative Samples (Invalid Sequences) =====
        self.subsection("Invalid Protocol Sequences")

        self.negative_traces = [
            # LOGIN without CONNECT
            ["LOGIN", "REQUEST"],
            # REQUEST without LOGIN
            ["CONNECT", "REQUEST"],
            # Double CONNECT
            ["CONNECT", "CONNECT", "LOGIN"],
            # LOGOUT without CONNECT
            ["LOGOUT"],
            # RESPONSE without REQUEST
            ["CONNECT", "LOGIN", "RESPONSE"],
            # REQUEST after LOGOUT
            ["CONNECT", "LOGIN", "LOGOUT", "REQUEST"],
            # Double LOGIN
            ["CONNECT", "LOGIN", "LOGIN"],
        ]

        for i, trace in enumerate(self.negative_traces):
            trace_str = " → ".join(trace)
            self.info(f"  Trace {i + 1}: {trace_str}")

        self.result("Invalid traces", len(self.negative_traces))

        return {
            "positive_traces": self.positive_traces,
            "negative_traces": self.negative_traces,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute state machine learning using RPNI algorithm."""
        results: dict[str, Any] = {}

        try:
            from oscura.inference.state_machine import StateMachineInferrer
        except ImportError:
            self.error("State machine module not available")
            return results

        # ===== Phase 1: RPNI Learning =====
        self.section("Part 1: RPNI State Machine Learning")

        inferrer = StateMachineInferrer()

        self.info("Running RPNI algorithm...")
        self.info("  1. Building Prefix Tree Acceptor from positive samples")
        self.info("  2. Merging states while maintaining consistency")
        self.info("  3. Validating against negative samples")

        try:
            self.learned_automaton = inferrer.infer(
                positive_samples=data["positive_traces"],
                negative_samples=data["negative_traces"],
            )
        except Exception as e:
            self.error(f"Learning failed: {e}")
            results["learned"] = False
            return results

        if self.learned_automaton is None:
            self.error("Failed to learn automaton")
            results["learned"] = False
            return results

        self.success("Successfully learned state machine!")
        results["learned"] = True

        # ===== Phase 2: Automaton Analysis =====
        self.section("Part 2: Learned Automaton Structure")

        self.subsection("Statistics")
        self.result("States", len(self.learned_automaton.states))
        self.result("Transitions", len(self.learned_automaton.transitions))
        self.result("Alphabet size", len(self.learned_automaton.alphabet))
        self.result("Accepting states", len(self.learned_automaton.accepting_states))

        results["num_states"] = len(self.learned_automaton.states)
        results["num_transitions"] = len(self.learned_automaton.transitions)
        results["alphabet_size"] = len(self.learned_automaton.alphabet)

        # Display states
        self.subsection("States")
        for state in self.learned_automaton.states:
            flags = []
            if state.is_initial:
                flags.append("INITIAL")
            if state.is_accepting:
                flags.append("ACCEPTING")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            self.info(f"  {state.name}{flag_str}")

        # Display transitions
        self.subsection("Transitions")
        for trans in self.learned_automaton.transitions:
            src = next(s for s in self.learned_automaton.states if s.id == trans.source)
            tgt = next(s for s in self.learned_automaton.states if s.id == trans.target)
            self.info(f"  {src.name} --[{trans.symbol}]--> {tgt.name}")

        # ===== Phase 3: Validation =====
        self.section("Part 3: Validation Against Training Data")

        # Validate positive samples
        self.subsection("Positive Sample Validation")
        positive_accepted = 0
        for i, trace in enumerate(data["positive_traces"]):
            accepted = self.learned_automaton.accepts(trace)
            if accepted:
                positive_accepted += 1
                status = "✓ ACCEPT"
            else:
                status = "✗ REJECT"

            trace_str = " → ".join(trace[:4])
            if len(trace) > 4:
                trace_str += " → ..."
            self.info(f"  [{status}] Trace {i + 1}: {trace_str}")

        positive_rate = positive_accepted / len(data["positive_traces"]) * 100
        self.result("Acceptance rate", f"{positive_rate:.0f}%")

        results["positive_accepted"] = positive_accepted
        results["positive_total"] = len(data["positive_traces"])

        # Validate negative samples
        self.subsection("Negative Sample Validation")
        negative_rejected = 0
        for i, trace in enumerate(data["negative_traces"]):
            accepted = self.learned_automaton.accepts(trace)
            correct = not accepted
            if correct:
                negative_rejected += 1
                status = "✓ REJECT"
            else:
                status = "✗ ACCEPT (wrong!)"

            trace_str = " → ".join(trace)
            self.info(f"  [{status}] Trace {i + 1}: {trace_str}")

        negative_rate = negative_rejected / len(data["negative_traces"]) * 100
        self.result("Rejection rate", f"{negative_rate:.0f}%")

        results["negative_rejected"] = negative_rejected
        results["negative_total"] = len(data["negative_traces"])

        # ===== Phase 4: Novel Sequence Testing =====
        self.section("Part 4: Testing Novel Sequences")

        test_sequences = [
            (["CONNECT", "LOGIN", "REQUEST", "RESPONSE", "REQUEST", "RESPONSE", "LOGOUT"], True),
            (["CONNECT", "LOGOUT", "CONNECT", "LOGIN", "LOGOUT"], True),
            (["LOGIN", "LOGOUT"], False),  # Missing CONNECT
            (["CONNECT", "REQUEST", "RESPONSE"], False),  # No LOGIN
        ]

        novel_correct = 0
        for seq, expected_accept in test_sequences:
            actual = self.learned_automaton.accepts(seq)
            correct = actual == expected_accept
            if correct:
                novel_correct += 1

            exp_str = "ACCEPT" if expected_accept else "REJECT"
            act_str = "ACCEPT" if actual else "REJECT"
            status = "✓" if correct else "✗"

            trace_str = " → ".join(seq[:4])
            if len(seq) > 4:
                trace_str += " → ..."
            self.info(f"  [{status}] {trace_str}")
            self.info(f"      Expected: {exp_str}, Got: {act_str}")

        results["novel_correct"] = novel_correct
        results["novel_total"] = len(test_sequences)

        # ===== Phase 5: Export =====
        self.section("Part 5: Export to DOT Format")

        try:
            dot_output = self.learned_automaton.to_dot()
            output_dir = self.get_output_dir()
            dot_file = output_dir / "state_machine.dot"
            dot_file.write_text(dot_output)

            self.success(f"DOT file saved: {dot_file}")
            self.info("Visualize with: dot -Tpng state_machine.dot -o state_machine.png")

            results["dot_exported"] = True

            # Show preview
            self.subsection("DOT Format Preview")
            lines = dot_output.split("\n")
            for line in lines[:15]:
                self.info(f"  {line}")
            if len(lines) > 15:
                self.info(f"  ... ({len(lines) - 15} more lines)")

        except Exception as e:
            self.warning(f"DOT export failed: {e}")
            results["dot_exported"] = False

        # ===== Summary =====
        self.section("Learning Summary")

        total_correct = positive_accepted + negative_rejected + novel_correct
        total_tests = (
            len(data["positive_traces"]) + len(data["negative_traces"]) + len(test_sequences)
        )
        accuracy = total_correct / total_tests * 100

        self.result("Overall accuracy", f"{accuracy:.0f}%")
        self.result("States learned", len(self.learned_automaton.states))
        self.result("Transitions learned", len(self.learned_automaton.transitions))

        results["overall_accuracy"] = accuracy

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate state machine learning results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        # Must have learned automaton
        if not results.get("learned", False):
            self.error("State machine not learned")
            return False

        # Must have reasonable structure
        if results.get("num_states", 0) < 2:
            self.error("Expected at least 2 states")
            return False

        if results.get("num_transitions", 0) == 0:
            self.error("No transitions learned")
            return False

        # Check positive acceptance rate
        pos_accepted = results.get("positive_accepted", 0)
        pos_total = results.get("positive_total", 1)
        pos_rate = pos_accepted / pos_total

        if pos_rate < 0.8:
            self.warning(f"Positive acceptance rate {pos_rate * 100:.0f}% below 80%")

        # Check negative rejection rate
        neg_rejected = results.get("negative_rejected", 0)
        neg_total = results.get("negative_total", 1)
        neg_rate = neg_rejected / neg_total

        if neg_rate < 0.7:
            self.warning(f"Negative rejection rate {neg_rate * 100:.0f}% below 70%")

        self.success("State machine learning demonstration complete!")
        return True


if __name__ == "__main__":
    demo = StateMachineLearningDemo()
    success = demo.execute()
    exit(0 if success else 1)
