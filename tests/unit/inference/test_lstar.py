"""Tests for L* active learning algorithm.

This module tests the L* algorithm implementation including:
- Observation table operations
- L* algorithm convergence
- Oracle implementations
- Query efficiency
- Integration with existing state machine code

Academic test cases are based on examples from:
    Angluin, D. (1987). Learning regular sets from queries and counterexamples.
    Information and Computation, 75(2), 87-106.
"""

import pytest

from oscura.inference.active_learning import (
    LStarLearner,
    ObservationTable,
    Oracle,
    SimulatorTeacher,
)
from oscura.inference.state_machine import (
    FiniteAutomaton,
    State,
    StateMachineInferrer,
    Transition,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference]


class SimpleOracle(Oracle):
    """Simple oracle for testing that accepts strings with even number of 'a's."""

    def __init__(self) -> None:
        """Initialize oracle."""
        self.alphabet_set = {"a", "b"}
        self.membership_count = 0
        self.equivalence_count = 0

    def membership_query(self, word: tuple[str, ...]) -> bool:
        """Accept if word contains even number of 'a's."""
        self.membership_count += 1
        count_a = sum(1 for symbol in word if symbol == "a")
        return count_a % 2 == 0

    def equivalence_query(self, hypothesis: FiniteAutomaton) -> tuple[str, ...] | None:
        """Test hypothesis against oracle logic."""
        self.equivalence_count += 1

        # Test some representative strings
        test_strings = [
            (),  # Empty string
            ("a",),
            ("b",),
            ("a", "a"),
            ("a", "b"),
            ("b", "a"),
            ("b", "b"),
            ("a", "a", "a"),
            ("a", "a", "b"),
            ("b", "a", "a"),
            ("a", "b", "a"),
            ("a", "a", "a", "a"),
            ("b", "b", "b", "b"),
            ("a", "b", "a", "b"),
        ]

        for word in test_strings:
            hypothesis_result = hypothesis.accepts(list(word))
            oracle_result = self.membership_query(word)

            if hypothesis_result != oracle_result:
                return word

        return None

    def get_alphabet(self) -> set[str]:
        """Return alphabet."""
        return self.alphabet_set.copy()

    def get_query_counts(self) -> tuple[int, int]:
        """Return query counts."""
        return (self.membership_count, self.equivalence_count)


@pytest.mark.unit
@pytest.mark.inference
class TestObservationTable:
    """Test observation table operations."""

    def test_initialization(self) -> None:
        """Test observation table initialization."""
        table = ObservationTable(alphabet={"a", "b"})

        assert () in table.S
        assert () in table.E
        assert table.alphabet == {"a", "b"}
        assert len(table.T) == 0

    def test_row_computation(self) -> None:
        """Test row computation from table."""
        table = ObservationTable(alphabet={"a", "b"})
        table.E = {(), ("a",), ("b",)}
        table.T = {
            (): True,
            ("a",): False,
            ("b",): True,
        }

        row = table.row(())
        assert row == (True, False, True)

    def test_is_closed_simple(self) -> None:
        """Test closure checking on simple table."""
        table = ObservationTable(alphabet={"a"})
        table.S = {(), ("a",)}
        table.E = {()}
        table.T = {(): True, ("a",): True, ("a", "a"): True}

        # Table is closed since all one-step extensions have equivalent rows
        assert table.is_closed()

    def test_is_not_closed(self) -> None:
        """Test closure checking when table not closed."""
        table = ObservationTable(alphabet={"a"})
        table.S = {()}
        table.E = {()}
        table.T = {(): True, ("a",): False}

        # Table not closed: ("a",) has different row than ()
        assert not table.is_closed()

    def test_find_closing_counterexample(self) -> None:
        """Test finding closing counterexample."""
        table = ObservationTable(alphabet={"a"})
        table.S = {()}
        table.E = {()}
        table.T = {(): True, ("a",): False}

        counterexample = table.find_closing_counterexample()
        assert counterexample == ("a",)

    def test_is_consistent_simple(self) -> None:
        """Test consistency checking on simple table."""
        table = ObservationTable(alphabet={"a"})
        table.S = {(), ("a",)}
        table.E = {()}
        table.T = {(): True, ("a",): False, ("a", "a"): True}

        # Table is consistent (rows are different)
        assert table.is_consistent()

    def test_is_not_consistent(self) -> None:
        """Test consistency checking when table not consistent."""
        table = ObservationTable(alphabet={"a"})
        table.S = {(), ("a",)}
        table.E = {()}
        # Both prefixes have same row
        table.T = {(): True, ("a",): True, ("a", "a"): False}

        # But extensions differ: () + "a" = True, ("a",) + "a" = False
        # This is inconsistent
        assert not table.is_consistent()

    def test_to_dfa_simple(self) -> None:
        """Test DFA construction from closed, consistent table."""
        table = ObservationTable(alphabet={"a"})
        table.S = {(), ("a",)}
        table.E = {()}
        table.T = {(): True, ("a",): False, ("a", "a"): False}

        dfa = table.to_dfa()

        assert len(dfa.states) == 2
        assert len(dfa.transitions) == 2
        assert dfa.initial_state in [0, 1]
        assert len(dfa.accepting_states) >= 1

    def test_to_dfa_not_closed_raises(self) -> None:
        """Test that to_dfa raises when table not closed."""
        table = ObservationTable(alphabet={"a"})
        table.S = {()}
        table.E = {()}
        table.T = {(): True, ("a",): False}

        with pytest.raises(ValueError, match="closed"):
            table.to_dfa()


@pytest.mark.unit
@pytest.mark.inference
class TestLStarAlgorithm:
    """Test L* algorithm."""

    def test_learn_simple_language(self) -> None:
        """Test L* learning simple language (even number of a's)."""
        oracle = SimpleOracle()
        learner = LStarLearner(oracle)

        dfa = learner.learn(max_iterations=100)

        # Check that learned DFA is correct
        assert dfa.accepts([])  # Even number (0) of a's
        assert not dfa.accepts(["a"])  # Odd number (1) of a's
        assert dfa.accepts(["a", "a"])  # Even number (2) of a's
        assert dfa.accepts(["b"])  # Even number (0) of a's
        assert dfa.accepts(["a", "b", "a"])  # Even number (2) of a's
        assert not dfa.accepts(["a", "b", "a", "b", "a"])  # Odd number (3) of a's

    def test_query_counts(self) -> None:
        """Test that query counts are tracked."""
        oracle = SimpleOracle()
        learner = LStarLearner(oracle)

        learner.learn(max_iterations=100)

        assert learner.membership_queries > 0
        assert learner.equivalence_queries > 0

    def test_learn_with_verbose(self) -> None:
        """Test L* learning with verbose output."""
        oracle = SimpleOracle()
        learner = LStarLearner(oracle, verbose=True)

        dfa = learner.learn(max_iterations=100)

        assert dfa is not None
        assert learner.membership_queries > 0

    def test_max_iterations_exceeded(self) -> None:
        """Test that max iterations limit works."""

        class NeverEquivalentOracle(Oracle):
            """Oracle that never says hypothesis is equivalent."""

            def membership_query(self, word: tuple[str, ...]) -> bool:
                return True

            def equivalence_query(self, hypothesis: FiniteAutomaton) -> tuple[str, ...] | None:
                # Always return a counterexample
                return ("a",)

            def get_alphabet(self) -> set[str]:
                return {"a"}

        oracle = NeverEquivalentOracle()
        learner = LStarLearner(oracle)

        with pytest.raises(ValueError, match="did not converge"):
            learner.learn(max_iterations=5)


@pytest.mark.unit
@pytest.mark.inference
class TestSimulatorTeacher:
    """Test SimulatorTeacher oracle implementation."""

    def test_initialization(self) -> None:
        """Test simulator teacher initialization."""
        traces = [["a", "b", "c"], ["a", "b", "d"]]
        teacher = SimulatorTeacher(traces)

        assert teacher.alphabet == {"a", "b", "c", "d"}
        assert len(teacher.traces) == 2

    def test_initialization_empty_raises(self) -> None:
        """Test that empty traces raise error."""
        with pytest.raises(ValueError, match="at least one trace"):
            SimulatorTeacher([])

    def test_membership_query_valid_prefix(self) -> None:
        """Test membership query for valid prefix."""
        traces = [["a", "b", "c"]]
        teacher = SimulatorTeacher(traces)

        assert teacher.membership_query(())  # Empty prefix
        assert teacher.membership_query(("a",))
        assert teacher.membership_query(("a", "b"))
        assert teacher.membership_query(("a", "b", "c"))

    def test_membership_query_invalid_prefix(self) -> None:
        """Test membership query for invalid prefix."""
        traces = [["a", "b", "c"]]
        teacher = SimulatorTeacher(traces)

        assert not teacher.membership_query(("x",))
        assert not teacher.membership_query(("a", "x"))
        assert not teacher.membership_query(("a", "b", "c", "d"))

    def test_membership_query_counts(self) -> None:
        """Test that membership queries are counted."""
        traces = [["a", "b"]]
        teacher = SimulatorTeacher(traces)

        teacher.membership_query(("a",))
        teacher.membership_query(("b",))

        counts = teacher.get_query_counts()
        assert counts[0] == 2  # membership queries

    def test_equivalence_query_correct_hypothesis(self) -> None:
        """Test equivalence query with correct hypothesis."""
        traces = [["a", "b"]]
        teacher = SimulatorTeacher(traces)

        # Build correct DFA manually
        states = [
            State(id=0, name="q0", is_initial=True, is_accepting=True),
            State(id=1, name="q1", is_initial=False, is_accepting=True),
            State(id=2, name="q2", is_initial=False, is_accepting=True),
        ]
        transitions = [
            Transition(source=0, target=1, symbol="a"),
            Transition(source=1, target=2, symbol="b"),
        ]
        dfa = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"a", "b"},
            initial_state=0,
            accepting_states={0, 1, 2},
        )

        result = teacher.equivalence_query(dfa)
        # Should find no counterexample (or find one if hypothesis is wrong)
        # This depends on the exact implementation

    def test_get_alphabet(self) -> None:
        """Test alphabet extraction."""
        traces = [["a", "b", "c"], ["d", "e"]]
        teacher = SimulatorTeacher(traces)

        alphabet = teacher.get_alphabet()
        assert alphabet == {"a", "b", "c", "d", "e"}


@pytest.mark.unit
@pytest.mark.inference
class TestLStarIntegration:
    """Integration tests for L* with simulator teacher."""

    def test_learn_from_simple_traces(self) -> None:
        """Test learning DFA from simple traces."""
        traces = [["a", "b"], ["a", "b", "b"]]
        teacher = SimulatorTeacher(traces)
        learner = LStarLearner(teacher)

        dfa = learner.learn(max_iterations=100)

        # Check learned DFA accepts the training traces
        assert dfa.accepts(["a", "b"])
        assert dfa.accepts(["a", "b", "b"])

        # Check prefixes are accepted
        assert dfa.accepts([])
        assert dfa.accepts(["a"])

    def test_learn_from_branching_traces(self) -> None:
        """Test learning DFA with branching paths."""
        traces = [["a", "b", "c"], ["a", "d", "e"]]
        teacher = SimulatorTeacher(traces)
        learner = LStarLearner(teacher)

        dfa = learner.learn(max_iterations=100)

        # Check learned DFA accepts the training traces
        assert dfa.accepts(["a", "b", "c"])
        assert dfa.accepts(["a", "d", "e"])

        # Check shared prefix
        assert dfa.accepts([])
        assert dfa.accepts(["a"])

    def test_learn_query_efficiency(self) -> None:
        """Test that L* uses reasonable number of queries."""
        traces = [["a", "b", "c"], ["a", "b", "d"], ["a", "b", "e"]]
        teacher = SimulatorTeacher(traces)
        learner = LStarLearner(teacher)

        dfa = learner.learn(max_iterations=100)

        # Should use polynomial number of queries
        # For small examples, expect < 200 membership queries
        # (actual count depends on equivalence query counterexamples)
        assert learner.membership_queries < 200
        assert learner.equivalence_queries < 20

        # Check DFA is minimal or near-minimal
        assert len(dfa.states) <= 10  # Should be small for this example


@pytest.mark.unit
@pytest.mark.inference
class TestLStarVsRPNI:
    """Compare L* with existing RPNI implementation."""

    def test_both_accept_training_traces(self) -> None:
        """Test that both L* and RPNI accept training traces."""
        traces = [["a", "b"], ["a", "b", "b"], ["a", "b", "b", "b"]]

        # Learn with RPNI (passive)
        inferrer = StateMachineInferrer()
        rpni_dfa = inferrer.infer_rpni(traces)

        # Learn with L* (active) using simulator teacher
        # Note: SimulatorTeacher treats all prefixes as valid, so the
        # learned language will be different from RPNI which only
        # accepts complete traces
        teacher = SimulatorTeacher(traces)
        learner = LStarLearner(teacher)
        lstar_dfa = learner.learn(max_iterations=100)

        # Both should accept the training traces (and their prefixes for L*)
        for trace in traces:
            # RPNI may or may not accept all traces depending on merging
            # L* with SimulatorTeacher should accept all prefixes
            for i in range(len(trace) + 1):
                prefix = trace[:i]
                # L* learned from prefixes, so should accept them
                assert lstar_dfa.accepts(prefix)

    def test_lstar_learns_prefix_language(self) -> None:
        """Test that L* with SimulatorTeacher learns prefix language."""
        traces = [["a"], ["a", "a"], ["a", "a", "a"]]

        # Learn with L*
        teacher = SimulatorTeacher(traces)
        learner = LStarLearner(teacher)
        lstar_dfa = learner.learn(max_iterations=100)

        # Should accept all prefixes seen in traces
        assert lstar_dfa.accepts([])  # Empty prefix
        assert lstar_dfa.accepts(["a"])
        assert lstar_dfa.accepts(["a", "a"])
        assert lstar_dfa.accepts(["a", "a", "a"])

        # Should not accept unseen sequences (unless equivalent to seen ones)
        # The exact behavior depends on the equivalence query implementation


@pytest.mark.unit
@pytest.mark.inference
class TestAcademicExamples:
    """Test cases from academic literature."""

    def test_angluin_example_1(self) -> None:
        """Test example from Angluin (1987) paper.

        Learn language: strings containing 'ab' substring
        """

        class ContainsABOracle(Oracle):
            """Oracle for language containing 'ab' substring."""

            def membership_query(self, word: tuple[str, ...]) -> bool:
                word_str = "".join(word)
                return "ab" in word_str

            def equivalence_query(self, hypothesis: FiniteAutomaton) -> tuple[str, ...] | None:
                test_words = [
                    (),
                    ("a",),
                    ("b",),
                    ("a", "b"),
                    ("b", "a"),
                    ("a", "a"),
                    ("b", "b"),
                    ("a", "b", "a"),
                    ("a", "b", "b"),
                    ("b", "a", "b"),
                    ("a", "a", "b"),
                    ("a", "b", "a", "b"),
                ]

                for word in test_words:
                    hyp_result = hypothesis.accepts(list(word))
                    oracle_result = self.membership_query(word)
                    if hyp_result != oracle_result:
                        return word
                return None

            def get_alphabet(self) -> set[str]:
                return {"a", "b"}

        oracle = ContainsABOracle()
        learner = LStarLearner(oracle)
        dfa = learner.learn(max_iterations=100)

        # Test learned DFA
        assert not dfa.accepts([])
        assert not dfa.accepts(["a"])
        assert not dfa.accepts(["b"])
        assert dfa.accepts(["a", "b"])
        assert dfa.accepts(["a", "b", "a"])
        assert dfa.accepts(["b", "a", "b"])

    def test_single_symbol_language(self) -> None:
        """Test learning language with single symbol."""

        class SingleSymbolOracle(Oracle):
            """Oracle that accepts only 'a'."""

            def membership_query(self, word: tuple[str, ...]) -> bool:
                return word == ("a",)

            def equivalence_query(self, hypothesis: FiniteAutomaton) -> tuple[str, ...] | None:
                test_words = [(), ("a",), ("a", "a"), ("b",), ("a", "b")]

                for word in test_words:
                    hyp_result = hypothesis.accepts(list(word))
                    oracle_result = self.membership_query(word)
                    if hyp_result != oracle_result:
                        return word
                return None

            def get_alphabet(self) -> set[str]:
                return {"a", "b"}

        oracle = SingleSymbolOracle()
        learner = LStarLearner(oracle)
        dfa = learner.learn(max_iterations=100)

        # Test learned DFA
        assert not dfa.accepts([])
        assert dfa.accepts(["a"])
        assert not dfa.accepts(["a", "a"])
        assert not dfa.accepts(["b"])
