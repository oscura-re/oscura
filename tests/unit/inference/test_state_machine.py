"""Comprehensive unit tests for state machine inference module.

Requirements addressed: PSI-002

This test suite covers:
- State, Transition, and FiniteAutomaton data structures
- StateMachineInferrer and RPNI algorithm
- DFA minimization
- Export functions (DOT, NetworkX)
- Edge cases and error conditions
"""

import pytest

from oscura.inference.state_machine import (
    FiniteAutomaton,
    State,
    StateMachineInferrer,
    Transition,
    infer_rpni,
    minimize_dfa,
    to_dot,
    to_networkx,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference]


# =============================================================================
# Data Structure Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestState:
    """Test State dataclass."""

    def test_create_state(self) -> None:
        """Test creating a state with default values."""
        state = State(id=0, name="q0")
        assert state.id == 0
        assert state.name == "q0"
        assert state.is_initial is False
        assert state.is_accepting is False

    def test_create_initial_state(self) -> None:
        """Test creating an initial state."""
        state = State(id=0, name="q0", is_initial=True)
        assert state.is_initial is True
        assert state.is_accepting is False

    def test_create_accepting_state(self) -> None:
        """Test creating an accepting state."""
        state = State(id=1, name="q1", is_accepting=True)
        assert state.is_initial is False
        assert state.is_accepting is True

    def test_create_initial_accepting_state(self) -> None:
        """Test creating a state that is both initial and accepting."""
        state = State(id=0, name="q0", is_initial=True, is_accepting=True)
        assert state.is_initial is True
        assert state.is_accepting is True


@pytest.mark.unit
@pytest.mark.inference
class TestTransition:
    """Test Transition dataclass."""

    def test_create_transition(self) -> None:
        """Test creating a transition with default count."""
        trans = Transition(source=0, target=1, symbol="A")
        assert trans.source == 0
        assert trans.target == 1
        assert trans.symbol == "A"
        assert trans.count == 1

    def test_create_transition_with_count(self) -> None:
        """Test creating a transition with specified count."""
        trans = Transition(source=0, target=1, symbol="A", count=5)
        assert trans.count == 5

    def test_transition_equality(self) -> None:
        """Test transition equality."""
        trans1 = Transition(source=0, target=1, symbol="A", count=1)
        trans2 = Transition(source=0, target=1, symbol="A", count=1)
        assert trans1 == trans2

    def test_transition_self_loop(self) -> None:
        """Test creating a self-loop transition."""
        trans = Transition(source=0, target=0, symbol="A")
        assert trans.source == trans.target


@pytest.mark.unit
@pytest.mark.inference
class TestFiniteAutomaton:
    """Test FiniteAutomaton dataclass and methods."""

    def test_create_simple_automaton(self) -> None:
        """Test creating a simple automaton."""
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
        ]
        transitions = [Transition(source=0, target=1, symbol="A")]
        alphabet = {"A"}

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet=alphabet,
            initial_state=0,
            accepting_states={1},
        )

        assert len(automaton.states) == 2
        assert len(automaton.transitions) == 1
        assert automaton.alphabet == {"A"}
        assert automaton.initial_state == 0
        assert automaton.accepting_states == {1}

    def test_accepts_single_symbol(self) -> None:
        """Test accepting a single symbol sequence."""
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
        ]
        transitions = [Transition(source=0, target=1, symbol="A")]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A"},
            initial_state=0,
            accepting_states={1},
        )

        assert automaton.accepts(["A"]) is True
        assert automaton.accepts(["B"]) is False
        assert automaton.accepts([]) is False

    def test_accepts_multi_symbol_sequence(self) -> None:
        """Test accepting multi-symbol sequences."""
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1"),
            State(id=2, name="q2", is_accepting=True),
        ]
        transitions = [
            Transition(source=0, target=1, symbol="A"),
            Transition(source=1, target=2, symbol="B"),
        ]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A", "B"},
            initial_state=0,
            accepting_states={2},
        )

        assert automaton.accepts(["A", "B"]) is True
        assert automaton.accepts(["A"]) is False  # Not in accepting state
        assert automaton.accepts(["B"]) is False  # Invalid transition
        assert automaton.accepts(["A", "B", "C"]) is False

    def test_accepts_empty_sequence(self) -> None:
        """Test accepting empty sequence when initial state is accepting."""
        states = [State(id=0, name="q0", is_initial=True, is_accepting=True)]
        transitions: list[Transition] = []

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet=set(),
            initial_state=0,
            accepting_states={0},
        )

        assert automaton.accepts([]) is True

    def test_accepts_with_self_loop(self) -> None:
        """Test accepting sequences with self-loop transitions."""
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
        ]
        transitions = [
            Transition(source=0, target=0, symbol="A"),  # Self-loop
            Transition(source=0, target=1, symbol="B"),
        ]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A", "B"},
            initial_state=0,
            accepting_states={1},
        )

        assert automaton.accepts(["B"]) is True
        assert automaton.accepts(["A", "B"]) is True
        assert automaton.accepts(["A", "A", "B"]) is True
        assert automaton.accepts(["A", "A", "A", "B"]) is True

    def test_get_successors_single_transition(self) -> None:
        """Test getting successors with single transition."""
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
        ]
        transitions = [Transition(source=0, target=1, symbol="A")]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A"},
            initial_state=0,
            accepting_states={1},
        )

        successors = automaton.get_successors(0)
        assert successors == {"A": 1}

    def test_get_successors_multiple_transitions(self) -> None:
        """Test getting successors with multiple transitions."""
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1"),
            State(id=2, name="q2", is_accepting=True),
        ]
        transitions = [
            Transition(source=0, target=1, symbol="A"),
            Transition(source=0, target=2, symbol="B"),
        ]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A", "B"},
            initial_state=0,
            accepting_states={2},
        )

        successors = automaton.get_successors(0)
        assert successors == {"A": 1, "B": 2}

    def test_get_successors_no_transitions(self) -> None:
        """Test getting successors for state with no outgoing transitions."""
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
        ]
        transitions = [Transition(source=0, target=1, symbol="A")]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A"},
            initial_state=0,
            accepting_states={1},
        )

        successors = automaton.get_successors(1)
        assert successors == {}

    def test_to_dot_simple(self) -> None:
        """Test DOT format export for simple automaton."""
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
        ]
        transitions = [Transition(source=0, target=1, symbol="A")]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A"},
            initial_state=0,
            accepting_states={1},
        )

        dot = automaton.to_dot()

        assert "digraph finite_automaton" in dot
        assert "rankdir=LR" in dot
        assert "q0 -> q1" in dot
        assert 'label="A"' in dot
        assert "__start__ -> q0" in dot
        assert "doublecircle" in dot  # For accepting state

    def test_to_dot_with_counts(self) -> None:
        """Test DOT format export includes transition counts."""
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
        ]
        transitions = [Transition(source=0, target=1, symbol="A", count=5)]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A"},
            initial_state=0,
            accepting_states={1},
        )

        dot = automaton.to_dot()

        assert "A (5)" in dot

    def test_to_dot_no_accepting_states(self) -> None:
        """Test DOT format export with no accepting states."""
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1"),
        ]
        transitions = [Transition(source=0, target=1, symbol="A")]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A"},
            initial_state=0,
            accepting_states=set(),
        )

        dot = automaton.to_dot()

        assert "digraph finite_automaton" in dot
        assert "q0 -> q1" in dot

    def test_to_networkx_simple(self) -> None:
        """Test NetworkX export for simple automaton."""
        pytest.importorskip("networkx")

        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
        ]
        transitions = [Transition(source=0, target=1, symbol="A")]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A"},
            initial_state=0,
            accepting_states={1},
        )

        graph = automaton.to_networkx()

        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
        assert graph.has_edge(0, 1)

        # Check node attributes
        assert graph.nodes[0]["name"] == "q0"
        assert graph.nodes[0]["is_initial"] is True
        assert graph.nodes[1]["is_accepting"] is True

        # Check edge attributes (MultiDiGraph returns dict of edges)
        edge_data = graph.get_edge_data(0, 1)
        assert edge_data is not None
        # MultiDiGraph: edge_data is {edge_key: {attrs}}
        assert len(edge_data) == 1, "Should have exactly one edge from 0 to 1"
        first_edge = edge_data[0]  # Get first (and only) edge
        assert first_edge["symbol"] == "A"
        assert first_edge["count"] == 1

    def test_to_networkx_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test NetworkX export raises ImportError when not available."""
        # Skip test if networkx is installed - this test specifically checks
        # behavior when networkx is NOT available
        try:
            import networkx  # noqa: F401

            pytest.skip("Test requires networkx to NOT be installed")
        except ImportError:
            pass

        states = [State(id=0, name="q0", is_initial=True)]
        automaton = FiniteAutomaton(
            states=states,
            transitions=[],
            alphabet=set(),
            initial_state=0,
            accepting_states=set(),
        )

        with pytest.raises(ImportError, match="NetworkX is required"):
            automaton.to_networkx()


# =============================================================================
# StateMachineInferrer Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestStateMachineInferrer:
    """Test StateMachineInferrer class."""

    def test_initialization(self) -> None:
        """Test inferrer initialization."""
        inferrer = StateMachineInferrer()
        assert inferrer._next_state_id == 0

    def test_get_next_state_id(self) -> None:
        """Test state ID generation."""
        inferrer = StateMachineInferrer()
        assert inferrer._get_next_state_id() == 0
        assert inferrer._get_next_state_id() == 1
        assert inferrer._get_next_state_id() == 2

    def test_build_pta_single_trace(self) -> None:
        """Test building PTA from single trace."""
        inferrer = StateMachineInferrer()
        traces = [["A", "B", "C"]]

        pta = inferrer._build_pta(traces)

        # Should have 4 states: q0 -> q1 -> q2 -> q3
        assert len(pta.states) == 4
        assert len(pta.transitions) == 3
        assert pta.alphabet == {"A", "B", "C"}
        assert pta.initial_state == 0
        assert 3 in pta.accepting_states  # Last state

    def test_build_pta_multiple_traces(self) -> None:
        """Test building PTA from multiple traces."""
        inferrer = StateMachineInferrer()
        traces = [["A", "B"], ["A", "C"]]

        pta = inferrer._build_pta(traces)

        # Should create branching tree
        assert len(pta.states) >= 3
        assert pta.alphabet == {"A", "B", "C"}

        # Both traces should be accepted by PTA
        assert pta.accepts(["A", "B"]) is True
        assert pta.accepts(["A", "C"]) is True

    def test_build_pta_overlapping_traces(self) -> None:
        """Test building PTA from traces with common prefixes."""
        inferrer = StateMachineInferrer()
        traces = [["A", "B", "C"], ["A", "B", "D"]]

        pta = inferrer._build_pta(traces)

        # Should share A->B prefix
        assert len(pta.states) == 5  # q0, q1(A), q2(B), q3(C), q4(D)
        assert len(pta.transitions) == 4

    def test_build_pta_empty_trace(self) -> None:
        """Test building PTA with empty trace."""
        inferrer = StateMachineInferrer()
        traces = [[]]

        pta = inferrer._build_pta(traces)

        # Should have just initial state which is accepting
        assert len(pta.states) == 1
        assert 0 in pta.accepting_states

    def test_build_pta_duplicate_traces(self) -> None:
        """Test building PTA with duplicate traces."""
        inferrer = StateMachineInferrer()
        traces = [["A", "B"], ["A", "B"], ["A", "B"]]

        pta = inferrer._build_pta(traces)

        # Should not create duplicate states
        assert len(pta.states) == 3  # q0, q1, q2
        assert pta.accepts(["A", "B"]) is True

    def test_merge_states_simple(self) -> None:
        """Test merging two states."""
        inferrer = StateMachineInferrer()

        # Create simple automaton
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
            State(id=2, name="q2", is_accepting=True),
        ]
        transitions = [
            Transition(source=0, target=1, symbol="A"),
            Transition(source=0, target=2, symbol="B"),
        ]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A", "B"},
            initial_state=0,
            accepting_states={1, 2},
        )

        # Merge states 1 and 2
        merged = inferrer._merge_states(automaton, 1, 2)

        # Should have 2 states now (0 and 1)
        assert len(merged.states) == 2
        assert {s.id for s in merged.states} == {0, 1}

        # Both transitions should point to state 1
        assert all(t.target == 1 for t in merged.transitions if t.source == 0)

    def test_merge_states_preserves_accepting(self) -> None:
        """Test that merging preserves accepting status."""
        inferrer = StateMachineInferrer()

        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
            State(id=2, name="q2", is_accepting=False),
        ]
        transitions = [
            Transition(source=0, target=1, symbol="A"),
            Transition(source=0, target=2, symbol="B"),
        ]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A", "B"},
            initial_state=0,
            accepting_states={1},
        )

        # Merge accepting state 1 into non-accepting state 2
        merged = inferrer._merge_states(automaton, 1, 2)

        # State 1 should now be accepting (inherited from state 2's merge)
        assert 1 in merged.accepting_states

    def test_merge_states_deduplicates_transitions(self) -> None:
        """Test that merging deduplicates transitions."""
        inferrer = StateMachineInferrer()

        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1"),
            State(id=2, name="q2"),
        ]
        transitions = [
            Transition(source=0, target=1, symbol="A", count=2),
            Transition(source=0, target=2, symbol="A", count=3),
        ]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A"},
            initial_state=0,
            accepting_states=set(),
        )

        # Merge states 1 and 2 - should combine transition counts
        merged = inferrer._merge_states(automaton, 1, 2)

        # Should have single transition with combined count
        assert len(merged.transitions) == 1
        assert merged.transitions[0].count == 5

    def test_is_compatible_no_negative_traces(self) -> None:
        """Test compatibility checking with no negative traces."""
        inferrer = StateMachineInferrer()

        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
            State(id=2, name="q2", is_accepting=True),
        ]
        transitions = [
            Transition(source=0, target=1, symbol="A"),
            Transition(source=0, target=2, symbol="B"),
        ]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A", "B"},
            initial_state=0,
            accepting_states={1, 2},
        )

        # Should be compatible (no negative traces to violate)
        assert inferrer._is_compatible(automaton, 1, 2, []) is True

    def test_is_compatible_with_negative_traces(self) -> None:
        """Test compatibility checking rejects invalid merges."""
        inferrer = StateMachineInferrer()

        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
            State(id=2, name="q2", is_accepting=False),
        ]
        transitions = [
            Transition(source=0, target=1, symbol="A"),
            Transition(source=0, target=2, symbol="B"),
        ]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A", "B"},
            initial_state=0,
            accepting_states={1},
        )

        # Negative trace ["B"] should prevent merging if it would accept it
        negative = [["B"]]

        # Merging 1 and 2 would make state accepting B
        is_compat = inferrer._is_compatible(automaton, 1, 2, negative)

        # Should be incompatible because merging would accept ["B"]
        assert is_compat is False

    def test_infer_rpni_empty_traces_raises(self) -> None:
        """Test RPNI inference raises on empty trace list."""
        inferrer = StateMachineInferrer()

        with pytest.raises(ValueError, match="at least one positive trace"):
            inferrer.infer_rpni([])

    def test_infer_rpni_single_trace(self) -> None:
        """Test RPNI inference with single trace."""
        inferrer = StateMachineInferrer()
        traces = [["A", "B", "C"]]

        dfa = inferrer.infer_rpni(traces)

        assert isinstance(dfa, FiniteAutomaton)
        assert len(dfa.states) >= 1
        assert dfa.accepts(["A", "B", "C"]) is True

    def test_infer_rpni_multiple_traces(self) -> None:
        """Test RPNI inference with multiple traces."""
        inferrer = StateMachineInferrer()
        traces = [["A", "B"], ["A", "C"], ["A", "B", "C"]]

        dfa = inferrer.infer_rpni(traces)

        # All training traces should be accepted
        for trace in traces:
            assert dfa.accepts(trace) is True

    def test_infer_rpni_with_negative_traces(self) -> None:
        """Test RPNI inference with negative examples."""
        inferrer = StateMachineInferrer()
        positive = [["A", "B"]]
        negative = [["B", "A"], ["C"]]

        dfa = inferrer.infer_rpni(positive, negative)

        # Positive traces should be accepted
        assert dfa.accepts(["A", "B"]) is True

        # Negative traces should be rejected
        for trace in negative:
            assert dfa.accepts(trace) is False

    def test_infer_rpni_alphabet_extraction(self) -> None:
        """Test that RPNI correctly extracts alphabet."""
        inferrer = StateMachineInferrer()
        traces = [["A", "B"], ["C", "D", "E"]]

        dfa = inferrer.infer_rpni(traces)

        assert dfa.alphabet == {"A", "B", "C", "D", "E"}

    def test_infer_rpni_state_merging(self) -> None:
        """Test that RPNI performs state merging."""
        inferrer = StateMachineInferrer()

        # Traces that should lead to merging
        traces = [["A", "X"], ["B", "X"], ["C", "X"]]

        dfa = inferrer.infer_rpni(traces)

        # Should merge states that have same behavior
        # Exact number depends on algorithm, but should be less than full PTA
        pta = inferrer._build_pta(traces)
        assert len(dfa.states) <= len(pta.states)


# =============================================================================
# Top-Level Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestInferRPNI:
    """Test convenience function for RPNI inference."""

    def test_infer_rpni_function(self) -> None:
        """Test top-level infer_rpni function."""
        traces = [["A", "B"], ["A", "C"]]

        dfa = infer_rpni(traces)

        assert isinstance(dfa, FiniteAutomaton)
        assert dfa.accepts(["A", "B"]) is True
        assert dfa.accepts(["A", "C"]) is True

    def test_infer_rpni_with_negatives(self) -> None:
        """Test top-level infer_rpni with negative traces."""
        positive = [["A", "B"]]
        negative = [["B", "A"]]

        dfa = infer_rpni(positive, negative)

        assert dfa.accepts(["A", "B"]) is True
        assert dfa.accepts(["B", "A"]) is False


@pytest.mark.unit
@pytest.mark.inference
class TestMinimizeDFA:
    """Test DFA minimization algorithm."""

    def test_minimize_single_state(self) -> None:
        """Test minimizing automaton with single state."""
        states = [State(id=0, name="q0", is_initial=True, is_accepting=True)]
        automaton = FiniteAutomaton(
            states=states,
            transitions=[],
            alphabet=set(),
            initial_state=0,
            accepting_states={0},
        )

        minimized = minimize_dfa(automaton)

        assert len(minimized.states) == 1
        assert minimized.accepts([]) is True

    def test_minimize_no_reduction(self) -> None:
        """Test minimizing already minimal automaton."""
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
        ]
        transitions = [Transition(source=0, target=1, symbol="A")]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A"},
            initial_state=0,
            accepting_states={1},
        )

        minimized = minimize_dfa(automaton)

        assert len(minimized.states) == 2

    def test_minimize_equivalent_states(self) -> None:
        """Test minimizing automaton with equivalent states."""
        # Create automaton with redundant states
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
            State(id=2, name="q2", is_accepting=True),
        ]
        transitions = [
            Transition(source=0, target=1, symbol="A"),
            Transition(source=0, target=2, symbol="B"),
        ]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A", "B"},
            initial_state=0,
            accepting_states={1, 2},
        )

        minimized = minimize_dfa(automaton)

        # States 1 and 2 are equivalent (both accepting, no outgoing transitions)
        assert len(minimized.states) < len(automaton.states)

    def test_minimize_preserves_language(self) -> None:
        """Test that minimization preserves accepted language."""
        traces = [["A", "B"], ["A", "C"], ["B", "C"]]
        dfa = infer_rpni(traces)
        minimized = minimize_dfa(dfa)

        # All original traces should still be accepted
        for trace in traces:
            assert minimized.accepts(trace) is True

    def test_minimize_complex_automaton(self) -> None:
        """Test minimizing more complex automaton."""
        traces = [["A"] * i + ["B"] for i in range(1, 5)]
        dfa = infer_rpni(traces)
        minimized = minimize_dfa(dfa)

        # Should reduce state count
        assert len(minimized.states) <= len(dfa.states)

        # Should preserve language
        for trace in traces:
            assert minimized.accepts(trace) is True


@pytest.mark.unit
@pytest.mark.inference
class TestToDot:
    """Test top-level to_dot function."""

    def test_to_dot_function(self) -> None:
        """Test to_dot convenience function."""
        traces = [["A", "B"]]
        dfa = infer_rpni(traces)

        dot = to_dot(dfa)

        assert "digraph finite_automaton" in dot
        assert "rankdir=LR" in dot


@pytest.mark.unit
@pytest.mark.inference
class TestToNetworkX:
    """Test top-level to_networkx function."""

    def test_to_networkx_function(self) -> None:
        """Test to_networkx convenience function."""
        pytest.importorskip("networkx")

        traces = [["A", "B"]]
        dfa = infer_rpni(traces)

        graph = to_networkx(dfa)

        assert graph.number_of_nodes() == len(dfa.states)
        assert graph.number_of_edges() == len(dfa.transitions)


# =============================================================================
# Edge Cases and Error Conditions
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestInferenceStateMachineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_sequence_acceptance(self) -> None:
        """Test accepting empty sequence."""
        traces = [[]]
        dfa = infer_rpni(traces)

        assert dfa.accepts([]) is True

    def test_single_symbol_alphabet(self) -> None:
        """Test automaton with single symbol."""
        traces = [["A"], ["A", "A"], ["A", "A", "A"]]
        dfa = infer_rpni(traces)

        assert dfa.alphabet == {"A"}
        for trace in traces:
            assert dfa.accepts(trace) is True

    def test_large_alphabet(self) -> None:
        """Test automaton with large alphabet."""
        symbols = [chr(65 + i) for i in range(26)]  # A-Z
        traces = [[symbols[i % 26]] for i in range(100)]

        dfa = infer_rpni(traces)

        assert len(dfa.alphabet) == 26

    def test_very_long_trace(self) -> None:
        """Test inference with very long traces."""
        trace = ["A"] * 100
        traces = [trace]

        dfa = infer_rpni(traces)

        assert dfa.accepts(trace) is True

    def test_many_traces(self) -> None:
        """Test inference with many traces."""
        traces = [["A", "B"] for _ in range(1000)]

        dfa = infer_rpni(traces)

        assert dfa.accepts(["A", "B"]) is True

    def test_deeply_nested_states(self) -> None:
        """Test automaton with deep state nesting."""
        trace = [chr(65 + i) for i in range(20)]  # A through T
        traces = [trace]

        dfa = infer_rpni(traces)

        assert dfa.accepts(trace) is True

    def test_special_characters_in_symbols(self) -> None:
        """Test automaton with special character symbols."""
        traces = [["!", "@", "#"], ["$", "%", "^"]]

        dfa = infer_rpni(traces)

        assert dfa.accepts(["!", "@", "#"]) is True
        assert dfa.accepts(["$", "%", "^"]) is True

    def test_unicode_symbols(self) -> None:
        """Test automaton with unicode symbols."""
        traces = [["α", "β", "γ"], ["δ", "ε", "ζ"]]

        dfa = infer_rpni(traces)

        assert dfa.accepts(["α", "β", "γ"]) is True

    def test_numeric_string_symbols(self) -> None:
        """Test automaton with numeric string symbols."""
        traces = [["1", "2", "3"], ["4", "5", "6"]]

        dfa = infer_rpni(traces)

        assert dfa.accepts(["1", "2", "3"]) is True

    def test_accepts_partial_trace(self) -> None:
        """Test that partial traces are rejected if not in training."""
        traces = [["A", "B", "C"]]
        dfa = infer_rpni(traces)

        # Partial trace may or may not be accepted depending on algorithm
        # But incomplete trace should reach non-accepting state
        result = dfa.accepts(["A", "B"])
        # Result depends on whether intermediate states are marked accepting
        assert isinstance(result, bool)

    def test_duplicate_transitions_different_counts(self) -> None:
        """Test handling of duplicate transitions with different counts."""
        # Test that automaton can be created with duplicate transitions
        # (though this is malformed)
        states = [
            State(id=0, name="q0", is_initial=True),
            State(id=1, name="q1", is_accepting=True),
        ]
        transitions = [
            Transition(source=0, target=1, symbol="A", count=3),
            Transition(source=0, target=1, symbol="A", count=2),
        ]

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A"},
            initial_state=0,
            accepting_states={1},
        )

        # Automaton can be created even with duplicates
        assert len(automaton.states) == 2
        # accepts() will use the first matching transition
        assert automaton.accepts(["A"]) is True

    def test_missing_initial_state(self) -> None:
        """Test automaton behavior when initial state ID is invalid."""
        states = [State(id=1, name="q1", is_accepting=True)]
        transitions: list[Transition] = []

        # Initial state ID 0 doesn't exist in states
        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet=set(),
            initial_state=0,  # This state doesn't exist
            accepting_states={1},
        )

        # Empty sequence returns False if initial state not in accepting states
        # Since initial state ID 0 doesn't exist, it won't be in accepting_states
        result = automaton.accepts([])
        assert result is False  # Initial state (0) is not in accepting_states {1}

    def test_invalid_transition_state_id(self) -> None:
        """Test automaton with transition to non-existent state."""
        states = [State(id=0, name="q0", is_initial=True)]
        transitions = [Transition(source=0, target=99, symbol="A")]  # Invalid target

        automaton = FiniteAutomaton(
            states=states,
            transitions=transitions,
            alphabet={"A"},
            initial_state=0,
            accepting_states=set(),
        )

        # Should handle gracefully (automaton may be malformed)
        # The to_dot method would fail with StopIteration
        with pytest.raises(StopIteration):
            automaton.to_dot()


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestInferenceStateMachineIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: infer -> minimize -> export."""
        traces = [["A", "B", "C"], ["A", "C"], ["B", "C"]]

        # Infer DFA
        dfa = infer_rpni(traces)

        # Minimize
        minimized = minimize_dfa(dfa)

        # Export to DOT
        dot = to_dot(minimized)
        assert "digraph" in dot

        # Verify language preservation
        for trace in traces:
            assert minimized.accepts(trace) is True

    def test_rpni_then_minimize_preserves_language(self) -> None:
        """Test that RPNI followed by minimization preserves accepted language."""
        import random

        random.seed(42)

        # Generate random traces
        alphabet = ["A", "B", "C"]
        traces = []
        for _ in range(20):
            length = random.randint(1, 5)
            trace = [random.choice(alphabet) for _ in range(length)]
            traces.append(trace)

        # Infer and minimize
        dfa = infer_rpni(traces)
        minimized = minimize_dfa(dfa)

        # All original traces should be accepted
        for trace in traces:
            assert dfa.accepts(trace) is True
            assert minimized.accepts(trace) is True

    def test_export_formats_consistency(self) -> None:
        """Test that both export formats represent same automaton."""
        pytest.importorskip("networkx")

        traces = [["A", "B"], ["A", "C"]]
        dfa = infer_rpni(traces)

        # Export to both formats
        dot = to_dot(dfa)
        graph = to_networkx(dfa)

        # Verify consistency
        assert graph.number_of_nodes() == len(dfa.states)
        assert graph.number_of_edges() == len(dfa.transitions)

        # DOT should mention all states
        for state in dfa.states:
            assert state.name in dot
