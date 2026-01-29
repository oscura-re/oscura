"""CAN reverse engineering session.

This module provides the main user-facing API for CAN bus reverse engineering,
centered around the CANSession class which manages message collections and
provides discovery-oriented analysis workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Lazy import for optional dataframe support
try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False

from oscura.automotive.can.analysis import MessageAnalyzer
from oscura.automotive.can.models import (
    CANMessage,
    CANMessageList,
    MessageAnalysis,
)
from oscura.sessions.base import AnalysisSession, ComparisonResult

if TYPE_CHECKING:
    from oscura.automotive.can.message_wrapper import CANMessageWrapper
    from oscura.automotive.can.patterns import (
        MessagePair,
        MessageSequence,
        TemporalCorrelation,
    )
    from oscura.automotive.can.stimulus_response import StimulusResponseReport
    from oscura.inference.state_machine import FiniteAutomaton

__all__ = ["CANSession"]


class CANSession(AnalysisSession):
    """CAN bus reverse engineering session.

    This is the primary API for discovering and analyzing unknown CAN bus
    protocols. It extends AnalysisSession to provide unified interface for
    multi-session workflows with CAN-specific functionality.

    Features:
    - Recording management (add/remove/compare recordings)
    - Message inventory and filtering
    - Per-message statistical analysis
    - Discovery-oriented workflows
    - Hypothesis testing
    - Documentation generation
    - Pattern discovery (pairs, sequences, correlations)
    - State machine inference

    Example - Basic usage:
        >>> from oscura.sessions import CANSession
        >>> from oscura.hardware.acquisition import FileSource
        >>> session = CANSession(name="Vehicle Analysis")
        >>> session.add_recording("baseline", FileSource("idle.blf"))
        >>> inventory = session.inventory()
        >>> print(inventory)

    Example - Discovery workflow:
        >>> session = CANSession(name="Brake Analysis")
        >>> session.add_recording("data", FileSource("capture.blf"))
        >>> # Focus on a specific message
        >>> msg = session.message(0x280)
        >>> analysis = msg.analyze()
        >>> print(analysis.summary())
        >>> # Test hypothesis
        >>> hypothesis = msg.test_hypothesis(
        ...     signal_name="rpm",
        ...     start_byte=2,
        ...     bit_length=16,
        ...     scale=0.25
        ... )

    Example - Compare recordings:
        >>> session = CANSession(name="Brake Analysis")
        >>> session.add_recording("no_brake", FileSource("idle.blf"))
        >>> session.add_recording("brake_pressed", FileSource("brake.blf"))
        >>> result = session.compare("no_brake", "brake_pressed")
        >>> print(f"Changed messages: {result.changed_bytes}")
    """

    def __init__(
        self,
        name: str = "CAN Session",
        auto_crc: bool = True,
        crc_validate: bool = True,
        crc_min_messages: int = 10,
    ):
        """Initialize CAN session.

        Args:
            name: Session name (default: "CAN Session").
            auto_crc: Enable automatic CRC recovery (default: True).
            crc_validate: Validate CRCs on subsequent messages (default: True).
            crc_min_messages: Minimum messages per ID for CRC recovery (default: 10).

        Example:
            >>> from oscura.sessions import CANSession
            >>> from oscura.hardware.acquisition import FileSource
            >>> session = CANSession(name="Vehicle Analysis")
            >>> session.add_recording("baseline", FileSource("idle.blf"))
            >>> session.add_recording("active", FileSource("running.blf"))
            >>> results = session.analyze()
            >>> # CRC parameters automatically recovered during analysis
            >>> print(session.crc_info)
        """
        super().__init__(name=name)
        self._messages = CANMessageList()
        self._analyses_cache: dict[int, MessageAnalysis] = {}
        self.auto_crc = auto_crc
        self.crc_validate = crc_validate
        self.crc_min_messages = crc_min_messages
        self._crc_params: dict[int, Any] = {}  # Map CAN ID to CRCParameters

    def inventory(self) -> pd.DataFrame:
        """Generate message inventory.

        Returns a pandas DataFrame with one row per unique CAN ID, showing:
        - arbitration_id: CAN ID
        - count: Number of messages
        - frequency_hz: Average frequency in Hz
        - period_ms: Average period in milliseconds
        - first_seen: Timestamp of first message
        - last_seen: Timestamp of last message
        - dlc: Data length (bytes)

        Returns:
            DataFrame with message inventory.

        Raises:
            ImportError: If pandas is not installed.
        """
        if not _HAS_PANDAS:
            raise ImportError(
                "Message inventory requires pandas.\n\n"
                "Install with:\n"
                "  pip install oscura[dataframes]    # DataFrame support\n"
                "  pip install oscura[standard]      # Recommended\n"
                "  pip install oscura[all]           # Everything\n"
            )

        unique_ids = sorted(self._messages.unique_ids())

        inventory_data = []
        for arb_id in unique_ids:
            filtered = self._messages.filter_by_id(arb_id)
            timestamps = [msg.timestamp for msg in filtered.messages]

            count = len(filtered)
            first_seen = min(timestamps)
            last_seen = max(timestamps)
            duration = last_seen - first_seen

            if duration > 0 and count > 1:
                frequency_hz = (count - 1) / duration
                period_ms = (duration / (count - 1)) * 1000
            else:
                frequency_hz = 0.0
                period_ms = 0.0

            # Get DLC from first message
            dlc = filtered.messages[0].dlc

            inventory_data.append(
                {
                    "arbitration_id": f"0x{arb_id:03X}",
                    "count": count,
                    "frequency_hz": f"{frequency_hz:.1f}",
                    "period_ms": f"{period_ms:.1f}",
                    "first_seen": f"{first_seen:.6f}",
                    "last_seen": f"{last_seen:.6f}",
                    "dlc": dlc,
                }
            )

        return pd.DataFrame(inventory_data)

    def message(self, arbitration_id: int) -> CANMessageWrapper:
        """Get message wrapper for analysis of a specific CAN ID.

        Args:
            arbitration_id: CAN ID to analyze.

        Returns:
            CANMessageWrapper for this message ID.

        Raises:
            ValueError: If no messages with this ID exist.
        """
        from oscura.automotive.can.message_wrapper import CANMessageWrapper

        filtered = self._messages.filter_by_id(arbitration_id)
        if not filtered.messages:
            raise ValueError(f"No messages found for ID 0x{arbitration_id:03X}")

        return CANMessageWrapper(self, arbitration_id)

    def analyze_message(self, arbitration_id: int, force_refresh: bool = False) -> MessageAnalysis:
        """Analyze a specific message ID.

        Args:
            arbitration_id: CAN ID to analyze.
            force_refresh: Force re-analysis even if cached.

        Returns:
            MessageAnalysis with complete analysis.
        """
        # Check cache
        if not force_refresh and arbitration_id in self._analyses_cache:
            return self._analyses_cache[arbitration_id]

        # Perform analysis
        analysis = MessageAnalyzer.analyze_message_id(self._messages, arbitration_id)

        # Cache result
        self._analyses_cache[arbitration_id] = analysis

        return analysis

    def filter(
        self,
        min_frequency: float | None = None,
        max_frequency: float | None = None,
        arbitration_ids: list[int] | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> CANSession:
        """Filter messages and return new session.

        Args:
            min_frequency: Minimum message frequency in Hz.
            max_frequency: Maximum message frequency in Hz.
            arbitration_ids: List of CAN IDs to include.
            time_range: Tuple of (start_time, end_time) in seconds.

        Returns:
            New CANSession with filtered messages.

        Note:
            This creates a new session with filtered messages from the current
            internal message collection. This method is primarily for legacy
            workflows. For new code, use add_recording() with separate files.

        Example:
            >>> session.filter(min_frequency=10.0, arbitration_ids=[0x123, 0x456])
        """
        # Apply filters sequentially
        filtered_messages = self._filter_by_time_range(time_range)
        filtered_messages = self._filter_by_arbitration_ids(filtered_messages, arbitration_ids)
        filtered_messages = self._filter_by_frequency(
            filtered_messages, min_frequency, max_frequency
        )

        # Create new session with filtered messages
        new_session = CANSession(name=f"{self.name} (filtered)")
        new_session._messages = CANMessageList(messages=filtered_messages)
        return new_session

    def _filter_by_time_range(self, time_range: tuple[float, float] | None) -> list[CANMessage]:
        """Filter messages by timestamp range.

        Args:
            time_range: Optional (start_time, end_time) in seconds.

        Returns:
            Filtered message list.
        """
        if time_range is None:
            return list(self._messages)

        start_time, end_time = time_range
        return [msg for msg in self._messages if start_time <= msg.timestamp <= end_time]

    def _filter_by_arbitration_ids(
        self, messages: list[CANMessage], arbitration_ids: list[int] | None
    ) -> list[CANMessage]:
        """Filter messages by CAN arbitration IDs.

        Args:
            messages: Messages to filter.
            arbitration_ids: Optional list of CAN IDs to include.

        Returns:
            Filtered message list.
        """
        if arbitration_ids is None:
            return messages

        id_set = set(arbitration_ids)
        return [msg for msg in messages if msg.arbitration_id in id_set]

    def _filter_by_frequency(
        self,
        messages: list[CANMessage],
        min_frequency: float | None,
        max_frequency: float | None,
    ) -> list[CANMessage]:
        """Filter messages by transmission frequency.

        Args:
            messages: Messages to filter.
            min_frequency: Optional minimum frequency in Hz.
            max_frequency: Optional maximum frequency in Hz.

        Returns:
            Filtered message list (only IDs with frequency in range).
        """
        if min_frequency is None and max_frequency is None:
            return messages

        # Calculate frequency for each CAN ID
        valid_ids = self._find_ids_in_frequency_range(messages, min_frequency, max_frequency)

        return [msg for msg in messages if msg.arbitration_id in valid_ids]

    def _find_ids_in_frequency_range(
        self,
        messages: list[CANMessage],
        min_frequency: float | None,
        max_frequency: float | None,
    ) -> set[int]:
        """Find CAN IDs with transmission frequency in specified range.

        Args:
            messages: Messages to analyze.
            min_frequency: Optional minimum frequency in Hz.
            max_frequency: Optional maximum frequency in Hz.

        Returns:
            Set of CAN IDs meeting frequency criteria.
        """
        from collections import defaultdict

        # Group messages by ID
        id_messages: dict[int, list[CANMessage]] = defaultdict(list)
        for msg in messages:
            id_messages[msg.arbitration_id].append(msg)

        # Check frequency for each ID
        valid_ids = set()
        for arb_id, msgs in id_messages.items():
            freq = self._calculate_message_frequency(msgs)
            if freq is None:
                continue

            # Check frequency bounds
            if min_frequency is not None and freq < min_frequency:
                continue
            if max_frequency is not None and freq > max_frequency:
                continue

            valid_ids.add(arb_id)

        return valid_ids

    def _calculate_message_frequency(self, messages: list[CANMessage]) -> float | None:
        """Calculate transmission frequency for a list of messages.

        Args:
            messages: Messages with same arbitration ID.

        Returns:
            Frequency in Hz, or None if cannot be calculated.
        """
        if len(messages) <= 1:
            return None

        timestamps = [m.timestamp for m in messages]
        duration = max(timestamps) - min(timestamps)

        if duration <= 0:
            return None

        return (len(messages) - 1) / duration

    def unique_ids(self) -> set[int]:
        """Get set of unique CAN IDs in this session.

        Returns:
            Set of unique arbitration IDs.
        """
        return self._messages.unique_ids()

    def time_range(self) -> tuple[float, float]:
        """Get time range of all messages.

        Returns:
            Tuple of (first_timestamp, last_timestamp).
        """
        return self._messages.time_range()

    def __len__(self) -> int:
        """Return total number of messages."""
        return len(self._messages)

    def analyze(self) -> dict[str, Any]:
        """Perform comprehensive CAN protocol analysis.

        Implements the AnalysisSession abstract method. Analyzes all messages
        in the current session to discover signals, patterns, and protocol structure.

        Returns:
            Dictionary with analysis results:
            - inventory: Message inventory DataFrame
            - num_messages: Total number of messages
            - num_unique_ids: Number of unique CAN IDs
            - time_range: Tuple of (start, end) timestamps
            - message_analyses: Dict mapping CAN ID to MessageAnalysis
            - patterns: Discovered patterns (pairs, sequences, correlations)

        Example:
            >>> from oscura.sessions import CANSession
            >>> from oscura.hardware.acquisition import FileSource
            >>> session = CANSession(name="Analysis")
            >>> session.add_recording("data", FileSource("capture.blf"))
            >>> results = session.analyze()
            >>> print(f"Found {results['num_unique_ids']} unique CAN IDs")
            >>> print(f"Duration: {results['time_range'][1] - results['time_range'][0]:.2f}s")

        Note:
            This is the unified AnalysisSession interface. For CAN-specific
            workflows, use inventory(), message(), and other domain methods.
        """
        # Generate inventory
        inventory = self.inventory()

        # Automatically recover CRC parameters if enabled
        if self.auto_crc:
            self._auto_recover_crc()

        # Analyze all unique IDs
        message_analyses = {}
        for arb_id in self.unique_ids():
            try:
                analysis = self.analyze_message(arb_id)
                message_analyses[arb_id] = analysis
            except Exception:
                # Skip messages that fail analysis
                continue

        # Find patterns (if enough messages)
        patterns: dict[str, Any] = {}
        if len(self._messages) >= 10:
            try:
                patterns["message_pairs"] = self.find_message_pairs(
                    time_window_ms=100, min_occurrence=3
                )
                patterns["temporal_correlations"] = self.find_temporal_correlations(
                    max_delay_ms=100
                )
            except Exception:
                # Pattern analysis is optional
                pass

        return {
            "inventory": inventory,
            "num_messages": len(self._messages),
            "num_unique_ids": len(self.unique_ids()),
            "time_range": self.time_range() if len(self._messages) > 0 else (0.0, 0.0),
            "message_analyses": message_analyses,
            "patterns": patterns,
        }

    def compare(self, name1: str, name2: str) -> ComparisonResult:
        """Compare two CAN recordings (stimulus-response analysis).

        Overrides AnalysisSession.compare() to provide CAN-specific differential
        analysis. Compares two recordings to detect changed messages, byte-level
        differences, and signal variations.

        Args:
            name1: Name of first recording (baseline).
            name2: Name of second recording (stimulus).

        Returns:
            ComparisonResult with CAN-specific differences:
            - changed_bytes: Number of message-bytes that differ
            - changed_regions: List of (message_id, byte_offset, description)
            - similarity_score: Overall similarity (0.0 to 1.0)
            - details: CAN-specific details (changed_ids, byte_changes, etc.)

        Example:
            >>> from oscura.hardware.acquisition import FileSource
            >>> session = CANSession(name="Brake Analysis")
            >>> session.add_recording("no_brake", FileSource("idle.blf"))
            >>> session.add_recording("brake_pressed", FileSource("brake.blf"))
            >>> result = session.compare("no_brake", "brake_pressed")
            >>> print(f"Changed messages: {result.details['changed_message_ids']}")

        Note:
            This uses the unified AnalysisSession interface. For advanced
            CAN-specific comparison, use compare_to() method.
        """
        # Load recordings as CANSession instances
        recording1 = self._recording_to_session(name1)
        recording2 = self._recording_to_session(name2)

        # Use CAN-specific stimulus-response analysis
        from oscura.automotive.can.stimulus_response import StimulusResponseAnalyzer

        analyzer = StimulusResponseAnalyzer()
        report = analyzer.detect_responses(recording1, recording2)

        # Convert to ComparisonResult
        changed_message_ids = report.changed_messages
        total_byte_changes = sum(len(changes) for changes in report.byte_changes.values())

        # Build changed regions list (message_id, byte_offset, description)
        changed_regions = []
        for msg_id, changes in report.byte_changes.items():
            for change in changes:
                changed_regions.append(
                    (
                        msg_id,
                        change.byte_position,
                        f"Magnitude: {change.change_magnitude:.2f}",
                    )
                )

        # Calculate similarity (1.0 = identical, 0.0 = completely different)
        total_unique_ids = len(recording1.unique_ids().union(recording2.unique_ids()))
        if total_unique_ids > 0:
            similarity = 1.0 - (len(changed_message_ids) / total_unique_ids)
        else:
            similarity = 1.0

        return ComparisonResult(
            recording1=name1,
            recording2=name2,
            changed_bytes=total_byte_changes,
            changed_regions=changed_regions,
            similarity_score=similarity,
            details={
                "changed_message_ids": changed_message_ids,
                "byte_changes": report.byte_changes,
                "new_messages": report.new_messages,
                "disappeared_messages": report.disappeared_messages,
                "stimulus_response_report": report,
            },
        )

    def _recording_to_session(self, name: str) -> CANSession:
        """Convert a recording to a CANSession instance.

        Args:
            name: Recording name.

        Returns:
            CANSession loaded from the recording.

        Raises:
            KeyError: If recording not found.
            ValueError: If recording is not a valid CAN log file.
        """
        if name not in self.recordings:
            available = list(self.recordings.keys())
            raise KeyError(f"Recording '{name}' not found. Available: {available}")

        source, _ = self.recordings[name]

        # Get file path from source and load messages
        # FileSource has a 'path' attribute
        if hasattr(source, "path"):
            from oscura.automotive.loaders import load_automotive_log

            file_path = source.path
            messages = load_automotive_log(file_path)

            # Create new session and populate with messages
            session = CANSession(name=name)
            session._messages = messages
            return session
        else:
            raise ValueError(
                f"Recording '{name}' is not from a file source. "
                "Recording-based comparison requires FileSource."
            )

    def compare_to(self, other_session: CANSession) -> StimulusResponseReport:
        """Compare this session to another to detect changes.

        This method is useful for stimulus-response analysis where you compare
        a baseline capture (no user action) against a stimulus capture (with
        user action) to identify which messages and signals respond.

        Args:
            other_session: Session to compare against (treated as stimulus).

        Returns:
            StimulusResponseReport with detected changes.

        Note:
            For comparing recordings within a session, use compare() instead:
            >>> session.compare("baseline", "stimulus")

            This method is for comparing two separate CANSession instances.

        Example:
            >>> # Compare two session instances directly
            >>> baseline = CANSession(name="Baseline")
            >>> stimulus = CANSession(name="Stimulus")
            >>> # ... populate sessions with messages ...
            >>> report = baseline.compare_to(stimulus)
            >>> print(report.summary())
            >>> # Show which messages changed
            >>> for msg_id in report.changed_messages:
            ...     print(f"0x{msg_id:03X} responded")
        """
        from oscura.automotive.can.stimulus_response import (
            StimulusResponseAnalyzer,
        )

        analyzer = StimulusResponseAnalyzer()
        return analyzer.detect_responses(self, other_session)

    def find_message_pairs(
        self,
        time_window_ms: float = 100,
        min_occurrence: int = 3,
    ) -> list[MessagePair]:
        """Find message pairs that frequently occur together.

        Discovers request-response patterns and coordinated message transmissions
        by detecting messages that consistently appear within a short time window.

        Args:
            time_window_ms: Maximum time window in milliseconds.
            min_occurrence: Minimum number of occurrences to report.

        Returns:
            List of MessagePair objects, sorted by occurrence count.

        Example:
            >>> from oscura.sessions import CANSession
            >>> from oscura.hardware.acquisition import FileSource
            >>> session = CANSession(name="Pattern Analysis")
            >>> session.add_recording("data", FileSource("capture.blf"))
            >>> pairs = session.find_message_pairs(time_window_ms=50)
            >>> for pair in pairs[:5]:
            ...     print(pair)
        """
        from oscura.automotive.can.patterns import PatternAnalyzer

        return PatternAnalyzer.find_message_pairs(
            self, time_window_ms=time_window_ms, min_occurrence=min_occurrence
        )

    def find_message_sequences(
        self,
        max_sequence_length: int = 5,
        time_window_ms: float = 500,
        min_support: float = 0.7,
    ) -> list[MessageSequence]:
        """Find message sequences (A → B → C patterns).

        Discovers multi-step control sequences or protocol handshakes by
        mining sequential patterns in the message stream.

        Args:
            max_sequence_length: Maximum length of sequences (2-10).
            time_window_ms: Maximum time window for entire sequence.
            min_support: Minimum support score (0.0-1.0).

        Returns:
            List of MessageSequence objects, sorted by support.

        Example:
            >>> from oscura.sessions import CANSession
            >>> from oscura.hardware.acquisition import FileSource
            >>> session = CANSession(name="Sequence Analysis")
            >>> session.add_recording("data", FileSource("startup.blf"))
            >>> sequences = session.find_message_sequences(
            ...     max_sequence_length=3,
            ...     time_window_ms=1000
            ... )
            >>> for seq in sequences[:5]:
            ...     print(seq)
        """
        from oscura.automotive.can.patterns import PatternAnalyzer

        return PatternAnalyzer.find_message_sequences(
            self,
            max_sequence_length=max_sequence_length,
            time_window_ms=time_window_ms,
            min_support=min_support,
        )

    def find_temporal_correlations(
        self,
        max_delay_ms: float = 100,
    ) -> dict[tuple[int, int], TemporalCorrelation]:
        """Find temporal correlations between messages.

        Analyzes timing relationships to determine which messages consistently
        follow others with predictable delays.

        Args:
            max_delay_ms: Maximum delay to consider for correlations.

        Returns:
            Dictionary mapping (leader_id, follower_id) to correlation info.

        Example:
            >>> from oscura.sessions import CANSession
            >>> from oscura.hardware.acquisition import FileSource
            >>> session = CANSession(name="Correlation Analysis")
            >>> session.add_recording("data", FileSource("capture.blf"))
            >>> correlations = session.find_temporal_correlations(max_delay_ms=50)
            >>> for (leader, follower), corr in correlations.items():
            ...     print(f"0x{leader:03X} → 0x{follower:03X}: {corr.avg_delay_ms:.2f}ms")
        """
        from oscura.automotive.can.patterns import PatternAnalyzer

        return PatternAnalyzer.find_temporal_correlations(self, max_delay_ms=max_delay_ms)

    def learn_state_machine(
        self, trigger_ids: list[int], context_window_ms: float = 500
    ) -> FiniteAutomaton:
        """Learn state machine from message sequences.

        This method integrates Oscura's state machine inference to learn
        protocol state machines from CAN message sequences around trigger messages.

        Args:
            trigger_ids: CAN IDs that trigger sequence extraction.
            context_window_ms: Time window (ms) before trigger to capture sequences.

        Returns:
            Learned finite automaton representing the state machine.

        Example:
            >>> from oscura.sessions import CANSession
            >>> from oscura.hardware.acquisition import FileSource
            >>> session = CANSession(name="State Machine Learning")
            >>> session.add_recording("data", FileSource("ignition_cycles.blf"))
            >>> automaton = session.learn_state_machine(
            ...     trigger_ids=[0x280],
            ...     context_window_ms=500
            ... )
            >>> print(automaton.to_dot())
        """
        from oscura.automotive.can.state_machine import learn_state_machine

        return learn_state_machine(
            session=self, trigger_ids=trigger_ids, context_window_ms=context_window_ms
        )

    def _auto_recover_crc(self) -> None:
        """Automatically recover CRC parameters for each message ID.

        This method attempts to recover CRC parameters for each CAN ID that has
        enough messages. Recovered parameters are stored in self._crc_params.

        Note:
            Only attempts recovery for IDs with >= crc_min_messages.
            Requires confidence > 0.8 to accept recovered parameters.
        """
        import logging

        from oscura.inference.crc_reverse import CRCReverser

        logger = logging.getLogger(__name__)

        for arb_id in self.unique_ids():
            filtered = self._messages.filter_by_id(arb_id)
            if len(filtered) < self.crc_min_messages:
                continue

            try:
                # Prepare message-CRC pairs
                # Heuristic: try last 1-4 bytes as potential CRC
                messages = []
                for msg in filtered.messages:
                    if msg.dlc >= 3:
                        # Try 1-byte CRC at end
                        messages.append((bytes(msg.data[:-1]), bytes(msg.data[-1:])))

                if len(messages) >= 4:
                    reverser = CRCReverser(verbose=False)
                    params = reverser.reverse(messages)

                    if params is not None and params.confidence > 0.8:
                        self._crc_params[arb_id] = params
                        logger.info(
                            f"Auto-recovered CRC for 0x{arb_id:03X}: "
                            f"poly=0x{params.polynomial:04x}, "
                            f"width={params.width}, confidence={params.confidence:.2f}"
                        )

            except Exception:
                # CRC recovery is best-effort
                pass

    def _validate_crc(self, msg: CANMessage) -> bool:
        """Validate CRC for a message if parameters are known.

        Args:
            msg: CAN message to validate.

        Returns:
            True if CRC is valid or no CRC params available, False if invalid.

        Note:
            Logs warning if CRC validation fails.
        """
        if not self.crc_validate or msg.arbitration_id not in self._crc_params:
            return True

        import logging

        from oscura.inference.crc_reverse import verify_crc

        logger = logging.getLogger(__name__)
        params = self._crc_params[msg.arbitration_id]

        try:
            # Extract message data and CRC based on width
            crc_bytes = params.width // 8
            if msg.dlc < crc_bytes:
                return True  # Message too short, skip validation

            data = bytes(msg.data[:-crc_bytes])
            crc = bytes(msg.data[-crc_bytes:])

            if not verify_crc(data, crc, params):
                logger.warning(
                    f"CRC validation failed for 0x{msg.arbitration_id:03X} "
                    f"at timestamp {msg.timestamp:.6f}"
                )
                return False

        except Exception:
            # Validation errors are non-fatal
            pass

        return True

    @property
    def crc_info(self) -> dict[int, dict[str, Any]]:
        """Get recovered CRC information for all message IDs.

        Returns:
            Dictionary mapping CAN ID to CRC parameter dict.

        Example:
            >>> session = CANSession()
            >>> # ... add recordings and analyze ...
            >>> for can_id, params in session.crc_info.items():
            ...     print(f"0x{can_id:03X}: {params['algorithm_name']}")
        """
        result = {}
        for arb_id, params in self._crc_params.items():
            result[arb_id] = {
                "polynomial": f"0x{params.polynomial:04x}",
                "width": params.width,
                "init": f"0x{params.init:04x}",
                "xor_out": f"0x{params.xor_out:04x}",
                "reflect_in": params.reflect_in,
                "reflect_out": params.reflect_out,
                "confidence": params.confidence,
                "algorithm_name": params.algorithm_name,
            }
        return result

    def __repr__(self) -> str:
        """Human-readable representation."""
        num_messages = len(self._messages)
        num_ids = len(self.unique_ids())
        num_recordings = len(self.recordings)

        if num_messages > 0:
            time_start, time_end = self.time_range()
            duration = time_end - time_start
            return (
                f"CANSession(name={self.name!r}, {num_messages} messages, "
                f"{num_ids} unique IDs, duration={duration:.2f}s, "
                f"recordings={num_recordings})"
            )
        else:
            return f"CANSession(name={self.name!r}, recordings={num_recordings})"
