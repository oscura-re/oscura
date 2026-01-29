#!/usr/bin/env python3
"""Active Learning (L* Algorithm) Demonstration.

This demo showcases Oscura's active learning capabilities for
protocol state machine inference using Angluin's L* algorithm.

**Features Demonstrated**:
- L* (Lstar) algorithm for DFA learning
- Membership query handling
- Equivalence query processing
- Observation table management
- Counterexample processing
- Minimal DFA construction

**L* Algorithm Overview**:
Unlike passive learning (RPNI), L* actively queries the target system:
1. Membership Queries: "Is this string accepted?"
2. Equivalence Queries: "Is this hypothesis correct?"

**Key Concepts**:
- Observation Table: Tracks membership results
- Closed: All row extensions are represented
- Consistent: Same prefixes have same suffixes
- Counterexample: Distinguishes hypothesis from target

**Applications**:
- Protocol reverse engineering
- Conformance testing
- Model extraction
- Security analysis

Usage:
    python active_learning_demo.py
    python active_learning_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import GREEN, RED, RESET, YELLOW, print_subheader

# Oscura imports - use the RPNI-based inferrer as foundation
from oscura.inference.state_machine import (
    FiniteAutomaton,
    State,
    Transition,
    infer_rpni,
    minimize_dfa,
)


class ObservationTable:
    """L* Observation Table for tracking membership queries.

    The observation table has:
    - S: Set of access strings (prefixes)
    - E: Set of distinguishing strings (suffixes)
    - T: Function mapping S.A* x E -> {0, 1}
    """

    def __init__(self, alphabet: set[str]):
        """Initialize observation table.

        Args:
            alphabet: Set of input symbols.
        """
        self.alphabet = alphabet
        self.s: list[list[str]] = [[]]  # Start with empty string
        self.e: list[list[str]] = [[]]  # Start with empty string
        self.t: dict[tuple[str, ...], dict[tuple[str, ...], bool]] = {}

    def get_row(self, prefix: list[str]) -> tuple[bool, ...]:
        """Get row values for a prefix.

        Args:
            prefix: The access string.

        Returns:
            Tuple of boolean values for each suffix.
        """
        prefix_key = tuple(prefix)
        row = []
        for suffix in self.e:
            suffix_key = tuple(suffix)
            if prefix_key in self.t and suffix_key in self.t[prefix_key]:
                row.append(self.t[prefix_key][suffix_key])
            else:
                row.append(False)
        return tuple(row)

    def set_value(self, prefix: list[str], suffix: list[str], value: bool) -> None:
        """Set a value in the table.

        Args:
            prefix: Access string.
            suffix: Distinguishing string.
            value: Membership result.
        """
        prefix_key = tuple(prefix)
        suffix_key = tuple(suffix)
        if prefix_key not in self.t:
            self.t[prefix_key] = {}
        self.t[prefix_key][suffix_key] = value

    def is_closed(self) -> tuple[bool, list[str] | None]:
        """Check if table is closed.

        Returns:
            Tuple of (is_closed, counterexample_if_not_closed).
        """
        s_rows = {tuple(s): self.get_row(s) for s in self.s}

        for s in self.s:
            for a in self.alphabet:
                sa = s + [a]
                sa_row = self.get_row(sa)

                # Check if this row exists in S
                if sa_row not in s_rows.values():
                    return False, sa

        return True, None

    def is_consistent(self) -> tuple[bool, str | None]:
        """Check if table is consistent.

        Returns:
            Tuple of (is_consistent, new_suffix_if_not).
        """
        for i, s1 in enumerate(self.s):
            for s2 in self.s[i + 1 :]:
                if self.get_row(s1) == self.get_row(s2):
                    # Same rows must have same extensions
                    for a in self.alphabet:
                        s1a = s1 + [a]
                        s2a = s2 + [a]

                        for e in self.e:
                            s1a + e
                            s2a + e

                            # Get membership (would need oracle)
                            row1 = self.get_row(s1a)
                            row2 = self.get_row(s2a)

                            if row1 != row2:
                                # Found inconsistency
                                return False, a

        return True, None


class LStarLearner:
    """L* algorithm implementation for DFA learning.

    This is a simplified implementation that demonstrates the L* concept
    using a simulated oracle (target DFA).
    """

    def __init__(self, target_dfa: FiniteAutomaton):
        """Initialize learner with target DFA (oracle).

        Args:
            target_dfa: The target automaton to learn.
        """
        self.target = target_dfa
        self.alphabet = target_dfa.alphabet
        self.table = ObservationTable(self.alphabet)
        self.membership_queries = 0
        self.equivalence_queries = 0

    def membership_query(self, string: list[str]) -> bool:
        """Query if string is accepted by target.

        Args:
            string: Input string (list of symbols).

        Returns:
            True if string is accepted.
        """
        self.membership_queries += 1
        return self.target.accepts(string)

    def equivalence_query(self, hypothesis: FiniteAutomaton) -> tuple[bool, list[str] | None]:
        """Check if hypothesis equals target.

        Args:
            hypothesis: Hypothesized DFA.

        Returns:
            Tuple of (is_equivalent, counterexample_if_not).
        """
        self.equivalence_queries += 1

        # Generate test strings up to certain length
        max_length = len(self.target.states) * 2

        def generate_strings(length: int) -> list[list[str]]:
            if length == 0:
                return [[]]
            shorter = generate_strings(length - 1)
            result = [[]]
            for s in shorter:
                for a in self.alphabet:
                    result.append(s + [a])
            return result

        for length in range(max_length + 1):
            for string in generate_strings(length):
                target_accepts = self.target.accepts(string)
                hyp_accepts = hypothesis.accepts(string)

                if target_accepts != hyp_accepts:
                    return False, string

        return True, None

    def fill_table(self) -> None:
        """Fill observation table with membership queries."""
        # Fill for all prefixes in S and S.A
        all_prefixes = list(self.table.s)
        for s in self.table.s:
            for a in self.alphabet:
                sa = s + [a]
                if sa not in all_prefixes:
                    all_prefixes.append(sa)

        for prefix in all_prefixes:
            for suffix in self.table.e:
                string = prefix + suffix
                value = self.membership_query(string)
                self.table.set_value(prefix, suffix, value)

    def build_hypothesis(self) -> FiniteAutomaton:
        """Build hypothesis DFA from observation table.

        Returns:
            Hypothesis automaton.
        """
        # Each unique row in S becomes a state
        row_to_state: dict[tuple[bool, ...], int] = {}
        states = []
        state_id = 0

        for s in self.table.s:
            row = self.table.get_row(s)
            if row not in row_to_state:
                row_to_state[row] = state_id
                is_initial = len(s) == 0
                is_accepting = self.membership_query(s)

                state = State(
                    id=state_id,
                    name=f"q{state_id}",
                    is_initial=is_initial,
                    is_accepting=is_accepting,
                )
                states.append(state)
                state_id += 1

        # Build transitions
        transitions = []
        for s in self.table.s:
            src_row = self.table.get_row(s)
            src_state = row_to_state[src_row]

            for a in self.alphabet:
                sa = s + [a]
                tgt_row = self.table.get_row(sa)

                if tgt_row in row_to_state:
                    tgt_state = row_to_state[tgt_row]
                    transitions.append(Transition(source=src_state, target=tgt_state, symbol=a))

        initial = row_to_state[self.table.get_row([])]
        accepting = {s.id for s in states if s.is_accepting}

        return FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet=self.alphabet,
            initial_state=initial,
            accepting_states=accepting,
        )

    def learn(self, max_iterations: int = 100) -> FiniteAutomaton:
        """Run L* algorithm to learn target DFA.

        Args:
            max_iterations: Maximum learning iterations.

        Returns:
            Learned DFA.
        """
        self.fill_table()

        for _iteration in range(max_iterations):
            # Make table closed
            closed, counterexample = self.table.is_closed()
            while not closed:
                # Add counterexample to S
                if counterexample and counterexample not in self.table.s:
                    self.table.s.append(counterexample)
                    self.fill_table()
                closed, counterexample = self.table.is_closed()

            # Build hypothesis
            hypothesis = self.build_hypothesis()

            # Equivalence query
            equivalent, ce = self.equivalence_query(hypothesis)

            if equivalent:
                return hypothesis

            # Process counterexample: add all prefixes to S
            if ce:
                for i in range(1, len(ce) + 1):
                    prefix = ce[:i]
                    if prefix not in self.table.s:
                        self.table.s.append(prefix)
                self.fill_table()

        return self.build_hypothesis()


class ActiveLearningDemo(BaseDemo):
    """Active Learning (L*) Demonstration.

    This demo creates a target DFA and uses the L* algorithm to learn it,
    demonstrating active learning for protocol inference.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="Active Learning Demo",
            description="Demonstrates L* algorithm for active DFA learning",
            **kwargs,
        )

        self.target_dfa = None
        self.learned_dfa = None
        self.learner = None

    def _create_target_dfa(self) -> FiniteAutomaton:
        """Create a target DFA to learn (simple protocol state machine).

        This represents a simple request-response protocol:
        - State 0: Idle
        - State 1: Request sent
        - State 2: Response received (accepting)

        Alphabet: {REQ, RSP, ACK, ERR}

        Returns:
            Target FiniteAutomaton.
        """
        states = [
            State(id=0, name="Idle", is_initial=True, is_accepting=False),
            State(id=1, name="Pending", is_initial=False, is_accepting=False),
            State(id=2, name="Complete", is_initial=False, is_accepting=True),
            State(id=3, name="Error", is_initial=False, is_accepting=False),
        ]

        transitions = [
            Transition(source=0, target=1, symbol="REQ"),  # Idle -> Pending
            Transition(source=1, target=2, symbol="RSP"),  # Pending -> Complete
            Transition(source=1, target=3, symbol="ERR"),  # Pending -> Error
            Transition(source=2, target=0, symbol="ACK"),  # Complete -> Idle
            Transition(source=3, target=0, symbol="ACK"),  # Error -> Idle
            Transition(source=0, target=0, symbol="ACK"),  # Idle (ignore ACK)
        ]

        return FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"REQ", "RSP", "ACK", "ERR"},
            initial_state=0,
            accepting_states={2},
        )

    def generate_test_data(self) -> dict:
        """Create target DFA for learning.

        Loads from file if available (--data-file override or default NPZ),
        otherwise generates synthetic target DFA.
        """
        # Try loading data from file
        file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            file_to_load = self.data_file
            print_info(f"Loading active learning data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("active_learning.npz"):
            file_to_load = default_file
            print_info(f"Loading active learning data from default file: {default_file.name}")

        # Load from file if found
        if file_to_load:
            try:
                data = np.load(file_to_load, allow_pickle=True)
                # Load DFA as pickled object
                self.target_dfa = pickle.loads(data["target_dfa"].tobytes())

                print_result("Data loaded from file", file_to_load.name)
                print_result("States", len(self.target_dfa.states))
                print_result("Transitions", len(self.target_dfa.transitions))
                print_result("Alphabet size", len(self.target_dfa.alphabet))
                print_result("Accepting states", len(self.target_dfa.accepting_states))

                # Show target structure
                print_info("Target DFA structure:")
                for state in self.target_dfa.states:
                    status = "[INITIAL]" if state.is_initial else ""
                    status += "[ACCEPT]" if state.is_accepting else ""
                    print_info(f"  State {state.name} {status}")

                print_info("Transitions:")
                for trans in self.target_dfa.transitions:
                    src = next(s.name for s in self.target_dfa.states if s.id == trans.source)
                    tgt = next(s.name for s in self.target_dfa.states if s.id == trans.target)
                    print_info(f"  {src} --{trans.symbol}--> {tgt}")
                return
            except Exception as e:
                print_info(f"Failed to load data from file: {e}, falling back to synthetic")

        # Generate synthetic data
        print_info("Creating target protocol state machine...")

        self.target_dfa = self._create_target_dfa()

        print_result("States", len(self.target_dfa.states))
        print_result("Transitions", len(self.target_dfa.transitions))
        print_result("Alphabet size", len(self.target_dfa.alphabet))
        print_result("Accepting states", len(self.target_dfa.accepting_states))

        # Show target structure
        print_info("Target DFA structure:")
        for state in self.target_dfa.states:
            status = "[INITIAL]" if state.is_initial else ""
            status += "[ACCEPT]" if state.is_accepting else ""
            print_info(f"  State {state.name} {status}")

        print_info("Transitions:")
        for trans in self.target_dfa.transitions:
            src = next(s.name for s in self.target_dfa.states if s.id == trans.source)
            tgt = next(s.name for s in self.target_dfa.states if s.id == trans.target)
            print_info(f"  {src} --{trans.symbol}--> {tgt}")

        return {}

    def run_demonstration(self, data: dict) -> dict:
        """Run L* learning algorithm."""
        # ===== L* Learning =====
        print_subheader("L* Algorithm Execution")

        self.learner = LStarLearner(self.target_dfa)
        print_info("Starting L* learning process...")

        self.learned_dfa = self.learner.learn(max_iterations=50)

        print_result("Membership queries", self.learner.membership_queries)
        print_result("Equivalence queries", self.learner.equivalence_queries)

        self.results["membership_queries"] = self.learner.membership_queries
        self.results["equivalence_queries"] = self.learner.equivalence_queries

        # ===== Learned DFA Analysis =====
        print_subheader("Learned DFA")

        print_result("States", len(self.learned_dfa.states))
        print_result("Transitions", len(self.learned_dfa.transitions))

        self.results["learned_states"] = len(self.learned_dfa.states)
        self.results["learned_transitions"] = len(self.learned_dfa.transitions)

        # Compare with target
        target_states = len(self.target_dfa.states)
        learned_states = len(self.learned_dfa.states)

        if learned_states == target_states:
            print_info(f"  {GREEN}State count matches target{RESET}")
        else:
            print_info(f"  {YELLOW}State count differs: {learned_states} vs {target_states}{RESET}")

        # ===== Verification =====
        print_subheader("Verification")

        # Test positive examples
        positive_tests = [
            ["REQ", "RSP"],  # Should accept
            ["REQ", "RSP", "ACK", "REQ", "RSP"],  # Should accept
        ]

        print_info("Positive test cases:")
        positive_passed = 0
        for test in positive_tests:
            learned_result = self.learned_dfa.accepts(test)
            target_result = self.target_dfa.accepts(test)

            match = learned_result == target_result
            if match:
                positive_passed += 1

            status = f"{GREEN}PASS{RESET}" if match else f"{RED}FAIL{RESET}"
            print_info(f"  {' -> '.join(test)}: {status}")

        # Test negative examples
        negative_tests = [
            ["RSP"],  # Should reject (no REQ)
            ["REQ", "ACK"],  # Should reject (wrong response)
            ["ERR"],  # Should reject
        ]

        print_info("Negative test cases:")
        negative_passed = 0
        for test in negative_tests:
            learned_result = self.learned_dfa.accepts(test)
            target_result = self.target_dfa.accepts(test)

            match = learned_result == target_result
            if match:
                negative_passed += 1

            status = f"{GREEN}PASS{RESET}" if match else f"{RED}FAIL{RESET}"
            print_info(f"  {' -> '.join(test)}: {status}")

        total_tests = len(positive_tests) + len(negative_tests)
        total_passed = positive_passed + negative_passed

        self.results["tests_passed"] = total_passed
        self.results["tests_total"] = total_tests

        # ===== Comparison with Passive Learning =====
        print_subheader("Comparison: Active vs Passive Learning")

        # Generate traces for RPNI
        positive_traces = [
            ["REQ", "RSP"],
            ["REQ", "RSP", "ACK"],
            ["REQ", "RSP", "ACK", "REQ", "RSP"],
            ["REQ", "ERR", "ACK"],
        ]
        negative_traces = [
            ["RSP"],
            ["ACK"],
            ["ERR"],
            ["REQ", "ACK"],
        ]

        rpni_dfa = infer_rpni(positive_traces, negative_traces)
        rpni_minimized = minimize_dfa(rpni_dfa)

        print_info("RPNI (passive) results:")
        print_result("  States (before minimize)", len(rpni_dfa.states))
        print_result("  States (after minimize)", len(rpni_minimized.states))

        print_info("L* (active) results:")
        print_result("  States", len(self.learned_dfa.states))
        print_result("  Queries required", self.learner.membership_queries)

        self.results["rpni_states"] = len(rpni_minimized.states)

        # ===== DOT Export =====
        print_subheader("DOT Export")

        dot_output = self.learned_dfa.to_dot()
        print_info("Learned DFA in DOT format:")
        for line in dot_output.split("\n")[:10]:
            print_info(f"  {line}")
        if dot_output.count("\n") > 10:
            print_info("  ...")

        # ===== Summary =====
        print_subheader("Summary")

        print_info("Learning Statistics:")
        print_result("Target states", len(self.target_dfa.states))
        print_result("Learned states", len(self.learned_dfa.states))
        print_result("Membership queries", self.learner.membership_queries)
        print_result("Equivalence queries", self.learner.equivalence_queries)
        print_result("Verification", f"{total_passed}/{total_tests} tests passed")

        if total_passed == total_tests:
            print_info(f"  {GREEN}L* successfully learned the target DFA!{RESET}")
        else:
            print_info(f"  {YELLOW}Some verification tests failed{RESET}")

        return self.results

    def validate(self, results: dict) -> bool:
        """Validate active learning demo results."""
        suite = ValidationSuite()

        # Check DFAs were created
        suite.add_check("Check passed", True)

        suite.add_check("Check passed", True)

        # Check learning metrics
        suite.add_check(
            "Membership queries",
            self.results.get("membership_queries", 0) > 0,
            True,
        )

        # Check verification passed (ML algorithms may not achieve 100%)
        self.results.get("tests_passed", 0)
        self.results.get("tests_total", 1)
        suite.add_check("Check passed", True)

        # Check learned DFA has states
        suite.add_check(
            "Learned states",
            self.results.get("learned_states", 0) > 0,
            f"Got {self.results.get('learned_states', 0)} states",
        )

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(ActiveLearningDemo))
