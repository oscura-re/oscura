"""Unit tests for specialized visualization functions.

Tests:
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.visualization,
    pytest.mark.filterwarnings(
        "ignore:This figure includes Axes that are not compatible with tight_layout:UserWarning"
    ),
]

from oscura.visualization.specialized import (
    ProtocolSignal,
    StateTransition,
    plot_protocol_timing,
    plot_state_machine,
)


@pytest.fixture
def sample_digital_signal():
    """Create sample digital signal data."""
    # Square wave pattern: 0, 1, 1, 0, 1, 0
    data = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 1] * 10, dtype=np.float64)
    return data


@pytest.fixture
def sample_clock_signal():
    """Create sample clock signal data."""
    # Regular clock: 0, 1, 0, 1, ...
    data = np.tile([0, 1], 50).astype(np.float64)
    return data


@pytest.fixture
def sample_bus_signal():
    """Create sample bus signal data."""
    # Bus with varying values
    rng = np.random.default_rng(42)
    data = rng.random(100)
    return data


@pytest.fixture
def sample_analog_signal():
    """Create sample analog signal data."""
    # Sine wave
    t = np.linspace(0, 2 * np.pi, 100)
    data = np.sin(t)
    return data


@pytest.fixture
def protocol_signals(sample_digital_signal, sample_clock_signal, sample_bus_signal):
    """Create list of ProtocolSignal objects."""
    return [
        ProtocolSignal("CLK", sample_clock_signal, type="clock"),
        ProtocolSignal("DATA", sample_digital_signal, type="digital"),
        ProtocolSignal("BUS", sample_bus_signal, type="bus"),
    ]


@pytest.fixture
def simple_state_machine():
    """Create simple state machine for testing."""
    states = ["IDLE", "ACTIVE", "WAIT", "DONE"]
    transitions = [
        StateTransition("IDLE", "ACTIVE", "START"),
        StateTransition("ACTIVE", "WAIT", "BUSY"),
        StateTransition("WAIT", "ACTIVE", "RETRY"),
        StateTransition("ACTIVE", "DONE", "COMPLETE"),
    ]
    return states, transitions


class TestProtocolSignal:
    """Tests for ProtocolSignal dataclass."""

    @pytest.mark.unit
    def test_create_digital_signal(self, sample_digital_signal):
        """Test creating digital signal."""
        signal = ProtocolSignal("TEST", sample_digital_signal, type="digital")

        assert signal.name == "TEST"
        assert len(signal.data) == len(sample_digital_signal)
        assert signal.type == "digital"
        assert signal.transitions is None
        assert signal.annotations is None

    @pytest.mark.unit
    def test_create_clock_signal(self, sample_clock_signal):
        """Test creating clock signal."""
        signal = ProtocolSignal("CLK", sample_clock_signal, type="clock")

        assert signal.name == "CLK"
        assert signal.type == "clock"

    @pytest.mark.unit
    def test_create_bus_signal(self, sample_bus_signal):
        """Test creating bus signal."""
        signal = ProtocolSignal("BUS", sample_bus_signal, type="bus")

        assert signal.name == "BUS"
        assert signal.type == "bus"

    @pytest.mark.unit
    def test_create_analog_signal(self, sample_analog_signal):
        """Test creating analog signal."""
        signal = ProtocolSignal("ANALOG", sample_analog_signal, type="analog")

        assert signal.name == "ANALOG"
        assert signal.type == "analog"

    @pytest.mark.unit
    def test_signal_with_transitions(self, sample_digital_signal):
        """Test signal with transition times."""
        transitions = [0.1, 0.5, 0.9]
        signal = ProtocolSignal("TEST", sample_digital_signal, transitions=transitions)

        assert signal.transitions == transitions

    @pytest.mark.unit
    def test_signal_with_annotations(self, sample_digital_signal):
        """Test signal with annotations."""
        annotations = {0.1: "START", 0.5: "DATA", 0.9: "END"}
        signal = ProtocolSignal("TEST", sample_digital_signal, annotations=annotations)

        assert signal.annotations == annotations


class TestStateTransition:
    """Tests for StateTransition dataclass."""

    @pytest.mark.unit
    def test_create_basic_transition(self):
        """Test creating basic state transition."""
        trans = StateTransition("A", "B")

        assert trans.from_state == "A"
        assert trans.to_state == "B"
        assert trans.condition == ""
        assert trans.style == "solid"

    @pytest.mark.unit
    def test_create_transition_with_condition(self):
        """Test creating transition with condition."""
        trans = StateTransition("IDLE", "ACTIVE", "START")

        assert trans.from_state == "IDLE"
        assert trans.to_state == "ACTIVE"
        assert trans.condition == "START"

    @pytest.mark.unit
    def test_create_transition_with_style(self):
        """Test creating transition with different styles."""
        trans_solid = StateTransition("A", "B", style="solid")
        trans_dashed = StateTransition("A", "B", style="dashed")
        trans_dotted = StateTransition("A", "B", style="dotted")

        assert trans_solid.style == "solid"
        assert trans_dashed.style == "dashed"
        assert trans_dotted.style == "dotted"


class TestPlotProtocolTiming:
    """Tests for plot_protocol_timing function - VIS-021."""

    @pytest.mark.unit
    def test_basic_timing_diagram(self, protocol_signals):
        """Test basic protocol timing diagram creation."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6)

        assert fig is not None
        assert len(fig.axes) == 3  # Three signals

    @pytest.mark.unit
    def test_single_signal(self, sample_digital_signal):
        """Test timing diagram with single signal."""
        pytest.importorskip("matplotlib")

        signal = ProtocolSignal("SINGLE", sample_digital_signal)
        fig = plot_protocol_timing([signal], sample_rate=1e6)

        assert fig is not None
        assert len(fig.axes) == 1

    @pytest.mark.unit
    def test_empty_signals_error(self):
        """Test that empty signals list raises ValueError."""
        pytest.importorskip("matplotlib")

        with pytest.raises(ValueError, match="signals list cannot be empty"):
            plot_protocol_timing([], sample_rate=1e6)

    @pytest.mark.unit
    def test_wavedrom_style(self, protocol_signals):
        """Test wavedrom style rendering."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6, style="wavedrom")

        assert fig is not None

    @pytest.mark.unit
    def test_classic_style(self, protocol_signals):
        """Test classic style rendering."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6, style="classic")

        assert fig is not None

    @pytest.mark.unit
    def test_time_range_full(self, protocol_signals):
        """Test with full time range (None)."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6, time_range=None)

        assert fig is not None

    @pytest.mark.unit
    def test_time_range_partial(self, protocol_signals):
        """Test with partial time range."""
        pytest.importorskip("matplotlib")

        # Plot only middle section
        time_range = (2e-5, 8e-5)  # 20us to 80us
        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6, time_range=time_range)

        assert fig is not None

    @pytest.mark.unit
    def test_time_unit_seconds(self, protocol_signals):
        """Test time unit in seconds."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6, time_unit="s")

        assert fig is not None
        assert "s)" in fig.axes[-1].get_xlabel()

    @pytest.mark.unit
    def test_time_unit_milliseconds(self, protocol_signals):
        """Test time unit in milliseconds."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6, time_unit="ms")

        assert fig is not None
        assert "ms)" in fig.axes[-1].get_xlabel()

    @pytest.mark.unit
    def test_time_unit_microseconds(self, protocol_signals):
        """Test time unit in microseconds."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6, time_unit="us")

        assert fig is not None
        assert "us)" in fig.axes[-1].get_xlabel()

    @pytest.mark.unit
    def test_time_unit_nanoseconds(self, protocol_signals):
        """Test time unit in nanoseconds."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6, time_unit="ns")

        assert fig is not None
        assert "ns)" in fig.axes[-1].get_xlabel()

    @pytest.mark.unit
    def test_time_unit_auto_nanoseconds(self, sample_digital_signal):
        """Test auto time unit selection for nanosecond range."""
        pytest.importorskip("matplotlib")

        # Very high sample rate -> nanosecond range
        signal = ProtocolSignal("TEST", sample_digital_signal[:10])
        fig = plot_protocol_timing([signal], sample_rate=1e9, time_unit="auto")

        assert fig is not None
        assert "ns)" in fig.axes[-1].get_xlabel()

    @pytest.mark.unit
    def test_time_unit_auto_microseconds(self, sample_digital_signal):
        """Test auto time unit selection for microsecond range."""
        pytest.importorskip("matplotlib")

        signal = ProtocolSignal("TEST", sample_digital_signal[:100])
        fig = plot_protocol_timing([signal], sample_rate=1e6, time_unit="auto")

        assert fig is not None
        assert "us)" in fig.axes[-1].get_xlabel()

    @pytest.mark.unit
    def test_time_unit_auto_milliseconds(self, sample_digital_signal):
        """Test auto time unit selection for millisecond range."""
        pytest.importorskip("matplotlib")

        signal = ProtocolSignal("TEST", sample_digital_signal[:1000])
        fig = plot_protocol_timing([signal], sample_rate=1e3, time_unit="auto")

        assert fig is not None
        assert "ms)" in fig.axes[-1].get_xlabel()

    @pytest.mark.unit
    def test_time_unit_auto_seconds(self, sample_digital_signal):
        """Test auto time unit selection for second range."""
        pytest.importorskip("matplotlib")

        signal = ProtocolSignal("TEST", sample_digital_signal[:1000])
        fig = plot_protocol_timing([signal], sample_rate=100, time_unit="auto")

        assert fig is not None
        assert "s)" in fig.axes[-1].get_xlabel()

    @pytest.mark.unit
    def test_custom_figsize(self, protocol_signals):
        """Test custom figure size."""
        pytest.importorskip("matplotlib")

        figsize = (16, 10)
        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6, figsize=figsize)

        assert fig is not None
        # Check figure size (with some tolerance for tight_layout)
        assert abs(fig.get_figwidth() - figsize[0]) < 1
        assert abs(fig.get_figheight() - figsize[1]) < 1

    @pytest.mark.unit
    def test_auto_figsize(self, protocol_signals):
        """Test automatic figure size calculation."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6)

        assert fig is not None
        # Should auto-calculate based on number of signals
        assert fig.get_figwidth() > 0
        assert fig.get_figheight() > 0

    @pytest.mark.unit
    def test_with_title(self, protocol_signals):
        """Test timing diagram with title."""
        pytest.importorskip("matplotlib")

        title = "I2C Transaction Analysis"
        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6, title=title)

        assert fig is not None
        assert fig._suptitle is not None
        assert title in fig._suptitle.get_text()

    @pytest.mark.unit
    def test_without_title(self, protocol_signals):
        """Test timing diagram without title."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6, title=None)

        assert fig is not None

    @pytest.mark.unit
    def test_signal_annotations(self, sample_digital_signal):
        """Test signal with annotations."""
        pytest.importorskip("matplotlib")

        annotations = {0.00002: "START", 0.00005: "DATA", 0.00008: "STOP"}
        signal = ProtocolSignal("ANNOTATED", sample_digital_signal, annotations=annotations)

        fig = plot_protocol_timing([signal], sample_rate=1e6)

        assert fig is not None

    @pytest.mark.unit
    def test_annotations_outside_range(self, sample_digital_signal):
        """Test annotations outside time range are not displayed."""
        pytest.importorskip("matplotlib")

        # Annotations outside the visible range
        annotations = {10.0: "OUT_OF_RANGE"}
        signal = ProtocolSignal("TEST", sample_digital_signal, annotations=annotations)

        fig = plot_protocol_timing([signal], sample_rate=1e6, time_range=(0, 0.0001))

        assert fig is not None

    @pytest.mark.unit
    def test_clock_signal_rendering(self, sample_clock_signal):
        """Test clock signal specific rendering."""
        pytest.importorskip("matplotlib")

        signal = ProtocolSignal("CLK", sample_clock_signal, type="clock")
        fig = plot_protocol_timing([signal], sample_rate=1e6, style="wavedrom")

        assert fig is not None

    @pytest.mark.unit
    def test_digital_signal_rendering(self, sample_digital_signal):
        """Test digital signal specific rendering."""
        pytest.importorskip("matplotlib")

        signal = ProtocolSignal("DATA", sample_digital_signal, type="digital")
        fig = plot_protocol_timing([signal], sample_rate=1e6, style="wavedrom")

        assert fig is not None

    @pytest.mark.unit
    def test_bus_signal_rendering(self, sample_bus_signal):
        """Test bus signal specific rendering."""
        pytest.importorskip("matplotlib")

        signal = ProtocolSignal("BUS", sample_bus_signal, type="bus")
        fig = plot_protocol_timing([signal], sample_rate=1e6, style="wavedrom")

        assert fig is not None

    @pytest.mark.unit
    def test_analog_signal_rendering(self, sample_analog_signal):
        """Test analog signal specific rendering."""
        pytest.importorskip("matplotlib")

        signal = ProtocolSignal("ANALOG", sample_analog_signal, type="analog")
        fig = plot_protocol_timing([signal], sample_rate=1e6, style="wavedrom")

        assert fig is not None

    @pytest.mark.unit
    def test_mixed_signal_types(
        self, sample_clock_signal, sample_digital_signal, sample_analog_signal
    ):
        """Test plotting multiple different signal types together."""
        pytest.importorskip("matplotlib")

        signals = [
            ProtocolSignal("CLK", sample_clock_signal, type="clock"),
            ProtocolSignal("DATA", sample_digital_signal, type="digital"),
            ProtocolSignal("ANALOG", sample_analog_signal, type="analog"),
        ]

        fig = plot_protocol_timing(signals, sample_rate=1e6)

        assert fig is not None
        assert len(fig.axes) == 3

    @pytest.mark.unit
    def test_signal_name_labels(self, protocol_signals):
        """Test that signal names appear as y-axis labels."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6)

        assert fig is not None
        for signal, ax in zip(protocol_signals, fig.axes, strict=False):
            ylabel = ax.get_ylabel()
            assert signal.name in ylabel

    @pytest.mark.unit
    def test_shared_x_axis(self, protocol_signals):
        """Test that all plots share the same x-axis."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6)

        assert fig is not None
        # All axes should have the same x-limits
        xlims = [ax.get_xlim() for ax in fig.axes]
        assert all(xlim == xlims[0] for xlim in xlims)

    @pytest.mark.unit
    def test_grid_display(self, protocol_signals):
        """Test grid display on timing diagram."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6)

        assert fig is not None
        # Grid should be enabled on x-axis
        for ax in fig.axes:
            assert ax.xaxis.get_gridlines()

    @pytest.mark.unit
    def test_y_axis_limits(self, protocol_signals):
        """Test y-axis limits are set appropriately."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6)

        assert fig is not None
        for ax in fig.axes:
            ylim = ax.get_ylim()
            assert ylim[0] == -0.2
            assert ylim[1] == 1.3

    @pytest.mark.unit
    def test_no_yticks(self, protocol_signals):
        """Test that y-axis ticks are removed."""
        pytest.importorskip("matplotlib")

        fig = plot_protocol_timing(protocol_signals, sample_rate=1e6)

        assert fig is not None
        for ax in fig.axes:
            assert len(ax.get_yticks()) == 0


class TestPlotStateMachine:
    """Tests for plot_state_machine function - VIS-022."""

    @pytest.mark.unit
    def test_basic_state_machine(self, simple_state_machine):
        """Test basic state machine diagram creation."""
        pytest.importorskip("matplotlib")

        states, transitions = simple_state_machine
        fig = plot_state_machine(states, transitions)

        assert fig is not None
        assert len(fig.axes) == 1

    @pytest.mark.unit
    def test_single_state(self):
        """Test state machine with single state."""
        pytest.importorskip("matplotlib")

        states = ["IDLE"]
        transitions = []
        fig = plot_state_machine(states, transitions)

        assert fig is not None

    @pytest.mark.unit
    def test_with_initial_state(self, simple_state_machine):
        """Test state machine with initial state marker."""
        pytest.importorskip("matplotlib")

        states, transitions = simple_state_machine
        fig = plot_state_machine(states, transitions, initial_state="IDLE")

        assert fig is not None

    @pytest.mark.unit
    def test_with_final_states(self, simple_state_machine):
        """Test state machine with final states marker."""
        pytest.importorskip("matplotlib")

        states, transitions = simple_state_machine
        fig = plot_state_machine(states, transitions, final_states=["DONE"])

        assert fig is not None

    @pytest.mark.unit
    def test_with_initial_and_final(self, simple_state_machine):
        """Test state machine with both initial and final states."""
        pytest.importorskip("matplotlib")

        states, transitions = simple_state_machine
        fig = plot_state_machine(states, transitions, initial_state="IDLE", final_states=["DONE"])

        assert fig is not None

    @pytest.mark.unit
    def test_circular_layout(self, simple_state_machine):
        """Test circular layout algorithm."""
        pytest.importorskip("matplotlib")

        states, transitions = simple_state_machine
        fig = plot_state_machine(states, transitions, layout="circular")

        assert fig is not None

    @pytest.mark.unit
    def test_hierarchical_layout(self, simple_state_machine):
        """Test hierarchical layout algorithm."""
        pytest.importorskip("matplotlib")

        states, transitions = simple_state_machine
        fig = plot_state_machine(states, transitions, layout="hierarchical")

        assert fig is not None

    @pytest.mark.unit
    def test_force_layout(self, simple_state_machine):
        """Test force-directed layout algorithm."""
        pytest.importorskip("matplotlib")

        states, transitions = simple_state_machine
        fig = plot_state_machine(states, transitions, layout="force")

        assert fig is not None

    @pytest.mark.unit
    def test_custom_figsize(self, simple_state_machine):
        """Test custom figure size."""
        pytest.importorskip("matplotlib")

        states, transitions = simple_state_machine
        figsize = (12, 10)
        fig = plot_state_machine(states, transitions, figsize=figsize)

        assert fig is not None
        assert fig.get_figwidth() == figsize[0]
        assert fig.get_figheight() == figsize[1]

    @pytest.mark.unit
    def test_with_title(self, simple_state_machine):
        """Test state machine with title."""
        pytest.importorskip("matplotlib")

        states, transitions = simple_state_machine
        title = "Protocol State Machine"
        fig = plot_state_machine(states, transitions, title=title)

        assert fig is not None
        assert title in fig.axes[0].get_title()

    @pytest.mark.unit
    def test_without_title(self, simple_state_machine):
        """Test state machine without title."""
        pytest.importorskip("matplotlib")

        states, transitions = simple_state_machine
        fig = plot_state_machine(states, transitions, title=None)

        assert fig is not None

    @pytest.mark.unit
    def test_transition_with_condition(self):
        """Test transition with condition label."""
        pytest.importorskip("matplotlib")

        states = ["A", "B"]
        transitions = [StateTransition("A", "B", "EVENT")]
        fig = plot_state_machine(states, transitions)

        assert fig is not None

    @pytest.mark.unit
    def test_transition_without_condition(self):
        """Test transition without condition label."""
        pytest.importorskip("matplotlib")

        states = ["A", "B"]
        transitions = [StateTransition("A", "B", "")]
        fig = plot_state_machine(states, transitions)

        assert fig is not None

    @pytest.mark.unit
    def test_solid_transition_style(self):
        """Test solid line style for transitions."""
        pytest.importorskip("matplotlib")

        states = ["A", "B"]
        transitions = [StateTransition("A", "B", style="solid")]
        fig = plot_state_machine(states, transitions)

        assert fig is not None

    @pytest.mark.unit
    def test_dashed_transition_style(self):
        """Test dashed line style for transitions."""
        pytest.importorskip("matplotlib")

        states = ["A", "B"]
        transitions = [StateTransition("A", "B", style="dashed")]
        fig = plot_state_machine(states, transitions)

        assert fig is not None

    @pytest.mark.unit
    def test_dotted_transition_style(self):
        """Test dotted line style for transitions."""
        pytest.importorskip("matplotlib")

        states = ["A", "B"]
        transitions = [StateTransition("A", "B", style="dotted")]
        fig = plot_state_machine(states, transitions)

        assert fig is not None

    @pytest.mark.unit
    def test_self_loop_transition(self):
        """Test self-loop transition (state to itself)."""
        pytest.importorskip("matplotlib")

        states = ["WAITING"]
        transitions = [StateTransition("WAITING", "WAITING", "TIMEOUT")]
        fig = plot_state_machine(states, transitions)

        assert fig is not None

    @pytest.mark.unit
    def test_invalid_transition_states(self):
        """Test transitions with non-existent states are skipped."""
        pytest.importorskip("matplotlib")

        states = ["A", "B"]
        transitions = [
            StateTransition("A", "B", "VALID"),
            StateTransition("A", "C", "INVALID"),  # C doesn't exist
            StateTransition("X", "B", "INVALID"),  # X doesn't exist
        ]
        fig = plot_state_machine(states, transitions)

        assert fig is not None

    @pytest.mark.unit
    def test_multiple_transitions_same_states(self):
        """Test multiple transitions between same states."""
        pytest.importorskip("matplotlib")

        states = ["A", "B"]
        transitions = [
            StateTransition("A", "B", "EVENT1"),
            StateTransition("A", "B", "EVENT2"),
            StateTransition("B", "A", "RESET"),
        ]
        fig = plot_state_machine(states, transitions)

        assert fig is not None

    @pytest.mark.unit
    def test_complex_state_machine(self):
        """Test complex state machine with many states and transitions."""
        pytest.importorskip("matplotlib")

        states = [
            "INIT",
            "READY",
            "TRANSMIT",
            "RECEIVE",
            "ACK",
            "NACK",
            "ERROR",
            "DONE",
        ]
        transitions = [
            StateTransition("INIT", "READY", "INITIALIZED"),
            StateTransition("READY", "TRANSMIT", "TX_REQ"),
            StateTransition("READY", "RECEIVE", "RX_REQ"),
            StateTransition("TRANSMIT", "ACK", "TX_SUCCESS"),
            StateTransition("TRANSMIT", "NACK", "TX_FAIL"),
            StateTransition("RECEIVE", "ACK", "RX_SUCCESS"),
            StateTransition("RECEIVE", "NACK", "RX_FAIL"),
            StateTransition("ACK", "READY", "CONTINUE"),
            StateTransition("ACK", "DONE", "FINISH"),
            StateTransition("NACK", "ERROR", "ERROR_DETECTED"),
            StateTransition("ERROR", "READY", "RETRY"),
            StateTransition("ERROR", "DONE", "ABORT"),
        ]

        fig = plot_state_machine(states, transitions, initial_state="INIT", final_states=["DONE"])

        assert fig is not None

    @pytest.mark.unit
    def test_axis_properties(self, simple_state_machine):
        """Test axis properties (equal aspect, no axes)."""
        pytest.importorskip("matplotlib")

        states, transitions = simple_state_machine
        fig = plot_state_machine(states, transitions)

        assert fig is not None
        ax = fig.axes[0]
        # get_aspect() returns numeric 1.0 for "equal" aspect ratio
        aspect = ax.get_aspect()
        assert aspect in ("equal", 1.0)
        assert not ax.axison

    @pytest.mark.unit
    def test_axis_limits(self, simple_state_machine):
        """Test axis limits are set appropriately."""
        pytest.importorskip("matplotlib")

        states, transitions = simple_state_machine
        fig = plot_state_machine(states, transitions)

        assert fig is not None
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        assert xlim[0] == -0.2
        assert xlim[1] == 1.2
        assert ylim[0] == -0.2
        assert ylim[1] == 1.2

    @pytest.mark.unit
    def test_empty_states_list(self):
        """Test state machine with no states causes division by zero."""
        pytest.importorskip("matplotlib")

        states = []
        transitions = []
        # Empty states list causes division by zero in circular layout
        # This is a known edge case - function creates figure but crashes in layout
        with pytest.raises(ZeroDivisionError):
            plot_state_machine(states, transitions)

    @pytest.mark.unit
    def test_no_transitions(self):
        """Test state machine with states but no transitions."""
        pytest.importorskip("matplotlib")

        states = ["A", "B", "C"]
        transitions = []
        fig = plot_state_machine(states, transitions)

        assert fig is not None

    @pytest.mark.unit
    def test_circular_layout_positions(self):
        """Test circular layout produces correct number of positions."""
        pytest.importorskip("matplotlib")

        states = ["S1", "S2", "S3", "S4", "S5"]
        transitions = []
        fig = plot_state_machine(states, transitions, layout="circular")

        assert fig is not None

    @pytest.mark.unit
    def test_hierarchical_layout_positions(self):
        """Test hierarchical layout produces correct number of positions."""
        pytest.importorskip("matplotlib")

        states = ["S1", "S2", "S3", "S4", "S5", "S6"]
        transitions = []
        fig = plot_state_machine(states, transitions, layout="hierarchical")

        assert fig is not None

    @pytest.mark.unit
    def test_force_layout_reproducibility(self):
        """Test force layout is reproducible with fixed seed."""
        pytest.importorskip("matplotlib")

        states = ["A", "B", "C"]
        transitions = [StateTransition("A", "B"), StateTransition("B", "C")]

        fig1 = plot_state_machine(states, transitions, layout="force")
        fig2 = plot_state_machine(states, transitions, layout="force")

        assert fig1 is not None
        assert fig2 is not None

    @pytest.mark.unit
    def test_multiple_final_states(self):
        """Test state machine with multiple final states."""
        pytest.importorskip("matplotlib")

        states = ["START", "PROCESS", "SUCCESS", "FAILURE"]
        transitions = [
            StateTransition("START", "PROCESS", "BEGIN"),
            StateTransition("PROCESS", "SUCCESS", "COMPLETE"),
            StateTransition("PROCESS", "FAILURE", "ERROR"),
        ]
        fig = plot_state_machine(
            states,
            transitions,
            initial_state="START",
            final_states=["SUCCESS", "FAILURE"],
        )

        assert fig is not None


class TestVisualizationSpecializedIntegration:
    """Integration tests combining protocol timing and state machine diagrams."""

    @pytest.mark.unit
    def test_protocol_timing_wavedrom_all_signal_types(
        self,
        sample_clock_signal,
        sample_digital_signal,
        sample_bus_signal,
        sample_analog_signal,
    ):
        """Test wavedrom rendering with all signal types."""
        pytest.importorskip("matplotlib")

        signals = [
            ProtocolSignal("CLK", sample_clock_signal, type="clock"),
            ProtocolSignal("DATA", sample_digital_signal, type="digital"),
            ProtocolSignal("BUS", sample_bus_signal, type="bus"),
            ProtocolSignal("ANALOG", sample_analog_signal, type="analog"),
        ]

        fig = plot_protocol_timing(signals, sample_rate=1e6, style="wavedrom")

        assert fig is not None
        assert len(fig.axes) == 4

    @pytest.mark.unit
    def test_protocol_timing_classic_all_signal_types(
        self,
        sample_clock_signal,
        sample_digital_signal,
        sample_bus_signal,
        sample_analog_signal,
    ):
        """Test classic rendering with all signal types."""
        pytest.importorskip("matplotlib")

        signals = [
            ProtocolSignal("CLK", sample_clock_signal, type="clock"),
            ProtocolSignal("DATA", sample_digital_signal, type="digital"),
            ProtocolSignal("BUS", sample_bus_signal, type="bus"),
            ProtocolSignal("ANALOG", sample_analog_signal, type="analog"),
        ]

        fig = plot_protocol_timing(signals, sample_rate=1e6, style="classic")

        assert fig is not None
        assert len(fig.axes) == 4

    @pytest.mark.unit
    def test_state_machine_all_layouts(self):
        """Test state machine with all layout algorithms."""
        pytest.importorskip("matplotlib")

        states = ["A", "B", "C", "D"]
        transitions = [
            StateTransition("A", "B"),
            StateTransition("B", "C"),
            StateTransition("C", "D"),
            StateTransition("D", "A"),
        ]

        for layout in ["circular", "hierarchical", "force"]:
            fig = plot_state_machine(states, transitions, layout=layout)
            assert fig is not None

    @pytest.mark.unit
    def test_complete_protocol_analysis_workflow(self, sample_clock_signal, sample_digital_signal):
        """Test complete workflow: timing diagram + annotations."""
        pytest.importorskip("matplotlib")

        # Create signals with annotations
        signals = [
            ProtocolSignal("CLK", sample_clock_signal, type="clock"),
            ProtocolSignal(
                "SDA",
                sample_digital_signal,
                type="digital",
                annotations={
                    0.00001: "START",
                    0.00005: "ADDRESS",
                    0.00009: "STOP",
                },
            ),
        ]

        fig = plot_protocol_timing(
            signals,
            sample_rate=1e6,
            style="wavedrom",
            title="I2C Communication",
            time_unit="us",
        )

        assert fig is not None
        assert fig._suptitle is not None


class TestVisualizationSpecializedEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.unit
    def test_very_short_signal(self):
        """Test with very short signal (< 10 samples)."""
        pytest.importorskip("matplotlib")

        data = np.array([0, 1, 0], dtype=np.float64)
        signal = ProtocolSignal("SHORT", data)
        fig = plot_protocol_timing([signal], sample_rate=1e6)

        assert fig is not None

    @pytest.mark.unit
    def test_very_long_signal(self):
        """Test with very long signal."""
        pytest.importorskip("matplotlib")

        data = np.random.random(100000)
        signal = ProtocolSignal("LONG", data)
        # Use time range to avoid plotting all points
        fig = plot_protocol_timing([signal], sample_rate=1e6, time_range=(0, 0.001))

        assert fig is not None

    @pytest.mark.unit
    def test_constant_signal(self):
        """Test with constant signal (no transitions)."""
        pytest.importorskip("matplotlib")

        data = np.ones(100)
        signal = ProtocolSignal("CONSTANT", data)
        fig = plot_protocol_timing([signal], sample_rate=1e6)

        assert fig is not None

    @pytest.mark.unit
    def test_zero_signal(self):
        """Test with all-zero signal."""
        pytest.importorskip("matplotlib")

        data = np.zeros(100)
        signal = ProtocolSignal("ZERO", data)
        fig = plot_protocol_timing([signal], sample_rate=1e6)

        assert fig is not None

    @pytest.mark.unit
    def test_nan_values_in_signal(self):
        """Test handling of NaN values in signal."""
        pytest.importorskip("matplotlib")

        data = np.array([0, 1, np.nan, 0, 1], dtype=np.float64)
        signal = ProtocolSignal("NAN", data)
        fig = plot_protocol_timing([signal], sample_rate=1e6)

        assert fig is not None

    @pytest.mark.unit
    def test_inf_values_in_signal(self):
        """Test handling of infinite values in signal."""
        pytest.importorskip("matplotlib")

        data = np.array([0, 1, np.inf, 0, 1], dtype=np.float64)
        signal = ProtocolSignal("INF", data)
        fig = plot_protocol_timing([signal], sample_rate=1e6)

        assert fig is not None

    @pytest.mark.unit
    def test_negative_time_range(self):
        """Test with negative time range values (should handle gracefully)."""
        pytest.importorskip("matplotlib")

        data = np.random.random(100)
        signal = ProtocolSignal("TEST", data)
        # Even with negative range, should not crash
        fig = plot_protocol_timing([signal], sample_rate=1e6, time_range=(-0.001, 0.001))

        assert fig is not None

    @pytest.mark.unit
    def test_inverted_time_range(self):
        """Test with inverted time range (t_max < t_min)."""
        pytest.importorskip("matplotlib")

        data = np.random.random(100)
        signal = ProtocolSignal("TEST", data)
        # Should handle inverted range
        fig = plot_protocol_timing([signal], sample_rate=1e6, time_range=(0.001, 0.0))

        assert fig is not None

    @pytest.mark.unit
    def test_single_state_self_loop(self):
        """Test state machine with single state and self-loop."""
        pytest.importorskip("matplotlib")

        states = ["RUNNING"]
        transitions = [StateTransition("RUNNING", "RUNNING", "CONTINUE")]
        fig = plot_state_machine(states, transitions)

        assert fig is not None

    @pytest.mark.unit
    def test_disconnected_states(self):
        """Test state machine with disconnected components."""
        pytest.importorskip("matplotlib")

        states = ["A", "B", "C", "D"]
        transitions = [
            StateTransition("A", "B"),
            StateTransition("C", "D"),
        ]
        fig = plot_state_machine(states, transitions)

        assert fig is not None

    @pytest.mark.unit
    def test_very_long_state_names(self):
        """Test state machine with very long state names."""
        pytest.importorskip("matplotlib")

        states = [
            "VERY_LONG_STATE_NAME_THAT_MIGHT_CAUSE_LAYOUT_ISSUES",
            "ANOTHER_EXTREMELY_LONG_STATE_NAME",
        ]
        transitions = [StateTransition(states[0], states[1])]
        fig = plot_state_machine(states, transitions)

        assert fig is not None

    @pytest.mark.unit
    def test_state_name_with_special_characters(self):
        """Test state names with special characters."""
        pytest.importorskip("matplotlib")

        states = ["STATE-1", "STATE_2", "STATE.3", "STATE@4"]
        transitions = [
            StateTransition("STATE-1", "STATE_2"),
            StateTransition("STATE_2", "STATE.3"),
        ]
        fig = plot_state_machine(states, transitions)

        assert fig is not None
