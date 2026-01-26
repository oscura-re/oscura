"""Property-based tests for state machine inference (RPNI algorithm).

This module tests state machine learning using Hypothesis for property-based testing.
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from oscura.inference.state_machine import (
    FiniteAutomaton,
    State,
    StateMachineInferrer,
    Transition,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference, pytest.mark.hypothesis]


class TestFiniteAutomatonProperties:
    """Property-based tests for finite automaton."""

    @given(
        num_states=st.integers(min_value=2, max_value=10),
        num_accepting=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50, deadline=None)
    def test_automaton_creation_valid(self, num_states: int, num_accepting: int) -> None:
        """Property: Automaton can be created with valid parameters."""
        # Limit accepting states to available states
        assume(num_accepting <= num_states)

        states = [State(id=i, name=f"s{i}") for i in range(num_states)]
        states[0].is_initial = True

        accepting_set = set(range(num_accepting))
        for i in accepting_set:
            states[i].is_accepting = True

        automaton = FiniteAutomaton(
            states=states,
            transitions=[],
            alphabet={"a", "b"},
            initial_state=0,
            accepting_states=accepting_set,
        )

        assert len(automaton.states) == num_states
        assert automaton.initial_state == 0
        assert automaton.accepting_states == accepting_set

    @given(sequence_length=st.integers(min_value=1, max_value=20))
    @settings(max_examples=50, deadline=None)
    def test_accepts_empty_or_any_sequence(self, sequence_length: int) -> None:
        """Property: Simple automaton behaves consistently."""
        # Create simple automaton that accepts all sequences
        states = [State(id=0, name="s0", is_initial=True, is_accepting=True)]
        transitions = [
            Transition(source=0, target=0, symbol="a"),
            Transition(source=0, target=0, symbol="b"),
        ]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"a", "b"},
            initial_state=0,
            accepting_states={0},
        )

        # Should accept any sequence
        test_sequence = ["a", "b"] * sequence_length
        result = automaton.accepts(test_sequence)

        # For this automaton, should accept
        assert result is True

    def test_dot_format_contains_states(self) -> None:
        """Property: DOT format contains all states."""
        states = [
            State(id=0, name="s0", is_initial=True),
            State(id=1, name="s1", is_accepting=True),
        ]
        transitions = [Transition(source=0, target=1, symbol="a")]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"a"},
            initial_state=0,
            accepting_states={1},
        )

        dot_str = automaton.to_dot()

        # Check format
        assert "digraph" in dot_str
        assert "s0" in dot_str
        assert "s1" in dot_str
        assert "s0 -> s1" in dot_str

    @given(
        num_transitions=st.integers(min_value=1, max_value=20),
        count_per_transition=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_transition_counts_preserved(
        self, num_transitions: int, count_per_transition: int
    ) -> None:
        """Property: Transition counts are preserved in automaton."""
        states = [State(id=0, name="s0", is_initial=True)]
        transitions = [
            Transition(source=0, target=0, symbol=f"sym{i}", count=count_per_transition)
            for i in range(num_transitions)
        ]

        alphabet = {f"sym{i}" for i in range(num_transitions)}

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet=alphabet,
            initial_state=0,
            accepting_states=set(),
        )

        for trans in automaton.transitions:
            assert trans.count == count_per_transition


class TestStateMachineInferrerProperties:
    """Property-based tests for RPNI learning algorithm."""

    @given(alphabet_size=st.integers(min_value=2, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_learner_produces_valid_automaton(self, alphabet_size: int) -> None:
        """Property: RPNI learner always produces valid automaton."""
        alphabet = [str(i) for i in range(alphabet_size)]

        # Generate simple positive examples
        positive_examples = [
            [alphabet[0]],
            [alphabet[1]],
            [alphabet[0], alphabet[1]],
        ]
        negative_examples = []  # type: ignore[var-annotated]

        learner = StateMachineInferrer()
        automaton = learner.infer_rpni(
            positive_traces=positive_examples, negative_traces=negative_examples
        )

        # Automaton should be valid
        assert automaton is not None
        assert len(automaton.states) > 0
        assert automaton.initial_state >= 0
        assert all(s.id >= 0 for s in automaton.states)

    @given(
        sequence_length=st.integers(min_value=1, max_value=10),
        alphabet_size=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=30, deadline=None)
    def test_learned_automaton_accepts_positive_examples(
        self, sequence_length: int, alphabet_size: int
    ) -> None:
        """Property: Learned automaton accepts all positive examples."""
        alphabet = [str(i) for i in range(alphabet_size)]

        # Generate sequences
        import random

        random.seed(42)
        positive_examples = [
            [random.choice(alphabet) for _ in range(sequence_length)] for _ in range(5)
        ]

        learner = StateMachineInferrer()
        automaton = learner.infer_rpni(positive_traces=positive_examples, negative_traces=[])

        # Should accept all positive examples
        for example in positive_examples:
            assert automaton.accepts(example)

    @given(num_examples=st.integers(min_value=2, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_state_count_bounded_by_examples(self, num_examples: int) -> None:
        """Property: Number of states is bounded by number of examples."""
        alphabet = ["a", "b"]

        # Generate distinct examples
        import random

        random.seed(42)
        positive_examples = [
            [random.choice(alphabet) for _ in range(i + 1)] for i in range(num_examples)
        ]

        learner = StateMachineInferrer()
        automaton = learner.infer_rpni(positive_traces=positive_examples, negative_traces=[])

        # State count should be reasonable (not exponential)
        # After merging, should be much less than examples
        assert len(automaton.states) <= num_examples * 2


class TestStateAndTransitionProperties:
    """Property-based tests for State and Transition data classes."""

    @given(
        state_id=st.integers(min_value=0, max_value=1000),
        is_initial=st.booleans(),
        is_accepting=st.booleans(),
    )
    @settings(max_examples=100, deadline=None)
    def test_state_properties_preserved(
        self, state_id: int, is_initial: bool, is_accepting: bool
    ) -> None:
        """Property: State properties are preserved after creation."""
        state = State(
            id=state_id,
            name=f"state_{state_id}",
            is_initial=is_initial,
            is_accepting=is_accepting,
        )

        assert state.id == state_id
        assert state.name == f"state_{state_id}"
        assert state.is_initial == is_initial
        assert state.is_accepting == is_accepting

    @given(
        source=st.integers(min_value=0, max_value=100),
        target=st.integers(min_value=0, max_value=100),
        count=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=100, deadline=None)
    def test_transition_properties_preserved(self, source: int, target: int, count: int) -> None:
        """Property: Transition properties are preserved after creation."""
        transition = Transition(source=source, target=target, symbol="test", count=count)

        assert transition.source == source
        assert transition.target == target
        assert transition.symbol == "test"
        assert transition.count == count


class TestAutomatonAcceptanceProperties:
    """Property-based tests for sequence acceptance."""

    def test_empty_sequence_acceptance(self) -> None:
        """Property: Empty sequence accepted only if initial state is accepting."""
        # Case 1: Initial state is accepting
        states1 = [State(id=0, name="s0", is_initial=True, is_accepting=True)]
        automaton1 = FiniteAutomaton(
            states=states1,
            transitions=[],
            alphabet={"a"},
            initial_state=0,
            accepting_states={0},
        )
        assert automaton1.accepts([]) is True

        # Case 2: Initial state is not accepting
        states2 = [State(id=0, name="s0", is_initial=True, is_accepting=False)]
        automaton2 = FiniteAutomaton(
            states=states2,
            transitions=[],
            alphabet={"a"},
            initial_state=0,
            accepting_states=set(),
        )
        assert automaton2.accepts([]) is False

    @given(sequence_length=st.integers(min_value=1, max_value=20))
    @settings(max_examples=50, deadline=None)
    def test_missing_transitions_reject_sequence(self, sequence_length: int) -> None:
        """Property: Sequences requiring missing transitions are rejected."""
        # Create automaton with no transitions
        states = [State(id=0, name="s0", is_initial=True, is_accepting=True)]
        automaton = FiniteAutomaton(
            states=states,
            transitions=[],
            alphabet={"a", "b"},
            initial_state=0,
            accepting_states={0},
        )

        # Any non-empty sequence should be rejected
        test_sequence = ["a"] * sequence_length
        result = automaton.accepts(test_sequence)

        assert result is False

    @given(
        symbol1=st.text(min_size=1, max_size=3),
        symbol2=st.text(min_size=1, max_size=3),
    )
    @settings(max_examples=50, deadline=None)
    def test_deterministic_acceptance(self, symbol1: str, symbol2: str) -> None:
        """Property: Acceptance is deterministic for same sequence."""
        # Assume distinct symbols
        assume(symbol1 != symbol2)

        states = [
            State(id=0, name="s0", is_initial=True),
            State(id=1, name="s1", is_accepting=True),
        ]
        transitions = [Transition(source=0, target=1, symbol=symbol1)]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={symbol1, symbol2},
            initial_state=0,
            accepting_states={1},
        )

        # Test same sequence multiple times
        test_sequence = [symbol1]
        result1 = automaton.accepts(test_sequence)
        result2 = automaton.accepts(test_sequence)
        result3 = automaton.accepts(test_sequence)

        # Should always give same result
        assert result1 == result2 == result3
        assert result1 is True  # Should accept


class TestAutomatonExportProperties:
    """Property-based tests for automaton export functionality."""

    @given(num_states=st.integers(min_value=1, max_value=10))
    @settings(max_examples=30, deadline=None)
    def test_dot_export_valid_format(self, num_states: int) -> None:
        """Property: DOT export produces valid digraph."""
        states = [State(id=i, name=f"s{i}", is_initial=(i == 0)) for i in range(num_states)]

        automaton = FiniteAutomaton(
            states=states,
            transitions=[],
            alphabet={"a"},
            initial_state=0,
            accepting_states=set(),
        )

        dot_str = automaton.to_dot()

        # Should be valid DOT format
        assert dot_str.startswith("digraph StateMachine {")
        assert dot_str.endswith("}")
        assert "rankdir=LR" in dot_str

    def test_networkx_export_preserves_structure(self) -> None:
        """Property: NetworkX export preserves automaton structure."""
        pytest.importorskip("networkx")

        states = [
            State(id=0, name="s0", is_initial=True),
            State(id=1, name="s1", is_accepting=True),
        ]
        transitions = [Transition(source=0, target=1, symbol="a")]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"a"},
            initial_state=0,
            accepting_states={1},
        )

        graph = automaton.to_networkx()

        # Check structure preserved
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
        assert graph.has_edge(0, 1)
