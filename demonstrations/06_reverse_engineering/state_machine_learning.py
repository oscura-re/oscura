#!/usr/bin/env python3
"""State Machine Learning Demonstration.

This demo showcases Oscura's state machine inference capabilities using
the RPNI (Regular Positive and Negative Inference) algorithm to learn
protocol state machines from observed message sequences.

**Features Demonstrated**:
- RPNI algorithm for passive automaton learning
- Prefix Tree Acceptor (PTA) construction
- State merging optimization
- Negative sample validation
- DOT format export for visualization
- Sequence acceptance testing

**RPNI Algorithm Overview**:
1. Build Prefix Tree Acceptor from positive samples
2. Order states in a canonical manner
3. Try to merge state pairs while maintaining consistency
4. Reject merges that would accept negative samples
5. Converge to minimal consistent DFA

**Use Cases**:
- Protocol reverse engineering
- Behavior modeling from traces
- Anomaly detection (sequence not accepted = anomaly)
- Specification extraction

Usage:
    python state_machine_learning.py
    python state_machine_learning.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, ValidationSuite, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import GREEN, RED, RESET, print_subheader

# Oscura imports
from oscura.inference.state_machine import (
    StateMachineInferrer,
)


class StateMachineLearningDemo(BaseDemo):
    """State Machine Learning Demonstration.

    This demo creates synthetic protocol traces and uses the RPNI algorithm
    to infer the underlying state machine, demonstrating Oscura's
    protocol inference capabilities.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="State Machine Learning Demo",
            description="Demonstrates RPNI algorithm for state machine inference",
            **kwargs,
        )

        # Test data storage
        self.positive_traces = []
        self.negative_traces = []
        self.learned_automaton = None

    def generate_test_data(self) -> dict:
        """Generate or load synthetic protocol traces for learning.

        Creates traces from a known state machine representing a simple
        request-response protocol:

        IDLE -> [REQUEST] -> WAITING -> [RESPONSE] -> PROCESSING -> [DONE] -> IDLE
                                 |                         |
                                 +-- [TIMEOUT] --> IDLE    +-- [ERROR] --> IDLE

        States: IDLE, WAITING, PROCESSING
        Transitions: REQUEST, RESPONSE, DONE, TIMEOUT, ERROR

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data files if they exist
        3. Generate synthetic protocol traces
        """
        # Try loading data from file
        data_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            data_file_to_load = self.data_file
            print_info(f"Loading protocol traces from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("state_machine_learning.npz"):
            data_file_to_load = default_file
            print_info(f"Loading protocol traces from default file: {default_file.name}")

        # Load from file if found
        if data_file_to_load:
            try:
                data = np.load(data_file_to_load, allow_pickle=True)
                self.positive_traces = data["positive_traces"].tolist()
                self.negative_traces = data["negative_traces"].tolist()

                print_result("Loaded from file", data_file_to_load.name)
                print_result("Positive traces", len(self.positive_traces))
                print_result("Negative traces", len(self.negative_traces))

                # Print summary
                print_subheader("Positive Samples (Valid Sequences)")
                for i, trace in enumerate(self.positive_traces):
                    print_info(f"  Trace {i + 1}: {' -> '.join(trace)}")

                print_subheader("Negative Samples (Invalid Sequences)")
                for i, trace in enumerate(self.negative_traces):
                    print_info(f"  Trace {i + 1}: {' -> '.join(trace)}")

                return
            except Exception as e:
                print_info(f"Failed to load from file: {e}, falling back to synthetic")
                data_file_to_load = None

        # Generate synthetic data if not loaded
        print_info("Generating synthetic protocol traces...")

        # ===== Define the "ground truth" protocol behavior =====
        # Valid sequences (positive samples)
        print_subheader("Positive Samples (Valid Sequences)")

        self.positive_traces = [
            # Complete successful transactions
            ["REQUEST", "RESPONSE", "DONE"],
            ["REQUEST", "RESPONSE", "DONE"],
            ["REQUEST", "RESPONSE", "ERROR"],  # Error during processing
            ["REQUEST", "TIMEOUT"],  # Timeout while waiting
            # Multiple transactions
            ["REQUEST", "RESPONSE", "DONE", "REQUEST", "RESPONSE", "DONE"],
            ["REQUEST", "TIMEOUT", "REQUEST", "RESPONSE", "DONE"],
            ["REQUEST", "RESPONSE", "DONE", "REQUEST", "TIMEOUT"],
            # Variations
            ["REQUEST", "RESPONSE", "DONE", "REQUEST", "RESPONSE", "ERROR"],
        ]

        for i, trace in enumerate(self.positive_traces):
            print_info(f"  Trace {i + 1}: {' -> '.join(trace)}")

        print_result("Positive traces", len(self.positive_traces))

        # Invalid sequences (negative samples) - these should NOT be accepted
        print_subheader("Negative Samples (Invalid Sequences)")

        self.negative_traces = [
            # Missing REQUEST at start
            ["RESPONSE", "DONE"],
            # DONE without RESPONSE
            ["REQUEST", "DONE"],
            # Double REQUEST
            ["REQUEST", "REQUEST", "RESPONSE"],
            # RESPONSE after TIMEOUT
            ["REQUEST", "TIMEOUT", "RESPONSE"],
            # Double RESPONSE
            ["REQUEST", "RESPONSE", "RESPONSE"],
            # ERROR without RESPONSE
            ["REQUEST", "ERROR"],
        ]

        for i, trace in enumerate(self.negative_traces):
            print_info(f"  Trace {i + 1}: {' -> '.join(trace)}")

        print_result("Negative traces", len(self.negative_traces))

        return {}

    def run_demonstration(self, data: dict) -> dict:
        """Learn state machine from traces using RPNI algorithm."""
        print_subheader("RPNI Learning")

        # Create inferrer
        inferrer = StateMachineInferrer()

        # Learn automaton from positive and negative samples
        print_info("Running RPNI algorithm...")
        self.learned_automaton = inferrer.infer(
            positive_samples=self.positive_traces,
            negative_samples=self.negative_traces,
        )

        if self.learned_automaton is None:
            print_info(f"{RED}Failed to learn automaton{RESET}")
            self.results["learned"] = False
            return

        self.results["learned"] = True
        print_info(f"{GREEN}Successfully learned automaton{RESET}")

        # Automaton statistics
        print_subheader("Learned Automaton Statistics")
        print_result("States", len(self.learned_automaton.states))
        print_result("Transitions", len(self.learned_automaton.transitions))
        print_result("Alphabet", len(self.learned_automaton.alphabet))
        print_result("Accepting states", len(self.learned_automaton.accepting_states))

        self.results["num_states"] = len(self.learned_automaton.states)
        self.results["num_transitions"] = len(self.learned_automaton.transitions)
        self.results["alphabet_size"] = len(self.learned_automaton.alphabet)

        # Show states
        print_subheader("States")
        for state in self.learned_automaton.states:
            flags = []
            if state.is_initial:
                flags.append("INITIAL")
            if state.is_accepting:
                flags.append("ACCEPTING")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print_info(f"  {state.name}{flag_str}")

        # Show transitions
        print_subheader("Transitions")
        for trans in self.learned_automaton.transitions:
            src = next(s for s in self.learned_automaton.states if s.id == trans.source)
            tgt = next(s for s in self.learned_automaton.states if s.id == trans.target)
            print_info(f"  {src.name} --[{trans.symbol}]--> {tgt.name}")

        # Validate against positive samples
        print_subheader("Validation Against Positive Samples")
        positive_accepted = 0
        for trace in self.positive_traces:
            accepted = self.learned_automaton.accepts(trace)
            status = f"{GREEN}ACCEPT{RESET}" if accepted else f"{RED}REJECT{RESET}"
            if accepted:
                positive_accepted += 1
            trace_str = " -> ".join(trace[:5])
            if len(trace) > 5:
                trace_str += " ..."
            print_info(f"  [{status}] {trace_str}")

        self.results["positive_accepted"] = positive_accepted
        self.results["positive_total"] = len(self.positive_traces)
        positive_rate = positive_accepted / len(self.positive_traces) * 100
        print_result("Positive acceptance rate", f"{positive_rate:.0f}%")

        # Validate against negative samples
        print_subheader("Validation Against Negative Samples")
        negative_rejected = 0
        for trace in self.negative_traces:
            accepted = self.learned_automaton.accepts(trace)
            # For negative samples, we WANT rejection
            correct = not accepted
            status = f"{GREEN}REJECT{RESET}" if correct else f"{RED}ACCEPT (wrong!){RESET}"
            if correct:
                negative_rejected += 1
            trace_str = " -> ".join(trace)
            print_info(f"  [{status}] {trace_str}")

        self.results["negative_rejected"] = negative_rejected
        self.results["negative_total"] = len(self.negative_traces)
        negative_rate = negative_rejected / len(self.negative_traces) * 100
        print_result("Negative rejection rate", f"{negative_rate:.0f}%")

        # Test novel sequences
        print_subheader("Testing Novel Sequences")
        test_sequences = [
            (
                [
                    "REQUEST",
                    "RESPONSE",
                    "DONE",
                    "REQUEST",
                    "RESPONSE",
                    "DONE",
                    "REQUEST",
                    "TIMEOUT",
                ],
                True,
            ),
            (["REQUEST", "RESPONSE", "ERROR", "REQUEST", "RESPONSE", "DONE"], True),
            (["DONE"], False),
            (["REQUEST", "REQUEST"], False),
        ]

        novel_correct = 0
        for seq, expected_accept in test_sequences:
            actual = self.learned_automaton.accepts(seq)
            correct = actual == expected_accept
            if correct:
                novel_correct += 1

            exp_str = "ACCEPT" if expected_accept else "REJECT"
            act_str = "ACCEPT" if actual else "REJECT"
            status = f"{GREEN}CORRECT{RESET}" if correct else f"{RED}WRONG{RESET}"

            trace_str = " -> ".join(seq[:4])
            if len(seq) > 4:
                trace_str += " ..."
            print_info(f"  {trace_str}")
            print_info(f"    Expected: {exp_str}, Got: {act_str} [{status}]")

        self.results["novel_correct"] = novel_correct
        self.results["novel_total"] = len(test_sequences)

        # Export DOT format
        print_subheader("DOT Export")
        dot_output = self.learned_automaton.to_dot()
        dot_file = self.get_output_dir() / "learned_automaton.dot"
        dot_file.write_text(dot_output)
        print_info(f"DOT file saved to: {dot_file}")
        print_info("Visualize with: dot -Tpng learned_automaton.dot -o automaton.png")

        self.results["dot_file"] = str(dot_file)
        self.results["dot_exported"] = True

        # Show DOT preview
        print_info("DOT format preview:")
        for line in dot_output.split("\n")[:10]:
            print_info(f"  {line}")
        if len(dot_output.split("\n")) > 10:
            print_info("  ...")

        # Summary
        print_subheader("Learning Summary")
        total_correct = positive_accepted + negative_rejected + novel_correct
        total_tests = len(self.positive_traces) + len(self.negative_traces) + len(test_sequences)
        accuracy = total_correct / total_tests * 100
        print_result("Overall accuracy", f"{accuracy:.0f}%")
        self.results["overall_accuracy"] = accuracy

        return self.results

    def validate(self, results: dict) -> bool:
        """Validate state machine learning results."""
        suite = ValidationSuite()

        # Check that automaton was learned
        learned = results.get("learned", False)
        suite.add_check(
            "Automaton learned", learned, "Learning succeeded" if learned else "Learning failed"
        )

        if not learned:
            suite.report()
            return suite.all_passed()

        # Check automaton structure
        num_states = results.get("num_states", 0)
        suite.add_check("States discovered", num_states >= 1, f"Got {num_states} states")

        num_transitions = results.get("num_transitions", 0)
        suite.add_check(
            "Transitions discovered", num_transitions > 0, f"Got {num_transitions} transitions"
        )

        # Check positive sample acceptance
        pos_accepted = results.get("positive_accepted", 0)
        pos_total = results.get("positive_total", 1)
        pos_rate = (pos_accepted / pos_total) * 100 if pos_total > 0 else 0
        suite.add_check("Positive samples accepted", pos_rate > 50, f"{pos_rate:.1f}% accepted")

        # Check negative sample rejection
        neg_rejected = results.get("negative_rejected", 0)
        neg_total = results.get("negative_total", 1)
        neg_rate = (neg_rejected / neg_total) * 100 if neg_total > 0 else 0
        suite.add_check("Negative samples rejected", neg_rate > 50, f"{neg_rate:.1f}% rejected")

        # Check DOT export
        dot_generated = results.get("dot_file", "") != ""
        suite.add_check(
            "DOT file generated",
            dot_generated,
            "DOT export succeeded" if dot_generated else "No DOT export",
        )

        # Check overall accuracy
        overall_accuracy = (pos_rate + neg_rate) / 2
        suite.add_check("Overall accuracy", overall_accuracy >= 70.0, f"{overall_accuracy:.1f}%")

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(StateMachineLearningDemo))
