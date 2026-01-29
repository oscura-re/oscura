"""Multi-protocol session correlation framework.

This module provides comprehensive tools for correlating sessions across
multiple protocols (e.g., CAN + Ethernet + Serial) to discover cross-protocol
communication patterns, dependencies, and session flows.

The framework supports:
- Message correlation across different protocols
- Request-response pair detection
- Dependency graph generation
- Session flow extraction
- Cross-protocol visualization
- Correlation analysis export

Example:
    >>> from oscura.correlation import MultiProtocolCorrelator, ProtocolMessage
    >>>
    >>> # Create correlator
    >>> correlator = MultiProtocolCorrelator(
    ...     time_window=0.1,  # Max 100ms between correlated messages
    ...     min_confidence=0.5
    ... )
    >>>
    >>> # Add CAN message
    >>> can_msg = ProtocolMessage(
    ...     protocol="can",
    ...     timestamp=1.234,
    ...     message_id=0x123,
    ...     payload=b"\\x01\\x02\\x03\\x04",
    ...     source="ECU1",
    ...     destination="ECU2"
    ... )
    >>> correlator.add_message(can_msg)
    >>>
    >>> # Add Ethernet message shortly after (likely related)
    >>> eth_msg = ProtocolMessage(
    ...     protocol="ethernet",
    ...     timestamp=1.238,  # 4ms later
    ...     payload=b"\\x01\\x02\\x03\\x04\\x05",  # Similar payload
    ...     source="192.168.1.10",
    ...     destination="192.168.1.20"
    ... )
    >>> correlator.add_message(eth_msg)
    >>>
    >>> # Find all correlations
    >>> correlations = correlator.correlate_all()
    >>> for corr in correlations:
    ...     print(f"{corr.correlation_type}: {corr.confidence:.2f}")
    ...     print(f"  Evidence: {', '.join(corr.evidence)}")
    >>>
    >>> # Build dependency graph
    >>> graph = correlator.build_dependency_graph()
    >>> print(f"Nodes: {graph.number_of_nodes()}")
    >>> print(f"Edges: {graph.number_of_edges()}")
    >>>
    >>> # Extract logical sessions
    >>> sessions = correlator.extract_sessions()
    >>> for session in sessions:
    ...     print(f"Session: {len(session.messages)} messages")
    ...     print(f"Protocols: {', '.join(session.protocols)}")
    ...     print(f"Duration: {session.end_time - session.start_time:.3f}s")

Correlation Detection Methods:
    The framework uses multiple methods to identify related messages:

    1. Timestamp Proximity: Messages within time_window are candidates
    2. Payload Similarity: Partial or full payload matches using Jaccard
    3. ID Matching: Matching message IDs across protocols
    4. Source/Destination: Matching addresses or identifiers

    Correlation types:
    - broadcast: Near-simultaneous messages across protocols
    - request_response: Request followed by response
    - related_payload: Correlated by payload content

References:
    Network protocol analysis
    Session correlation algorithms
    Graph theory for dependency analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Optional dependencies - import at module level for validation
try:
    import matplotlib
except ImportError:
    matplotlib = None  # type: ignore[assignment]

try:
    import networkx
except ImportError:
    networkx = None  # type: ignore[assignment]


@dataclass
class ProtocolMessage:
    """Generic protocol message for multi-protocol correlation.

    Represents a message from any protocol with common fields for correlation.

    Attributes:
        protocol: Protocol name (e.g., "can", "ethernet", "uart", "spi").
        timestamp: Message timestamp in seconds (float for high precision).
        message_id: Message identifier (int for CAN, str for others, None if N/A).
        payload: Message payload as bytes.
        source: Source address/identifier (optional).
        destination: Destination address/identifier (optional).
        metadata: Additional protocol-specific metadata.

    Example:
        >>> msg = ProtocolMessage(
        ...     protocol="can",
        ...     timestamp=1.234567,
        ...     message_id=0x123,
        ...     payload=b"\\x01\\x02\\x03\\x04",
        ...     source="ECU1",
        ...     metadata={"dlc": 4, "extended": False}
        ... )
    """

    protocol: str
    timestamp: float
    message_id: str | int | None = None
    payload: bytes = b""
    source: str | None = None
    destination: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageCorrelation:
    """Correlation between two messages from different protocols.

    Attributes:
        message1: First message (chronologically earlier).
        message2: Second message (chronologically later).
        correlation_type: Type of correlation detected.
        confidence: Correlation confidence score (0.0-1.0).
        time_delta: Time difference in seconds (message2 - message1).
        evidence: List of evidence strings explaining the correlation.

    Correlation types:
        - "broadcast": Near-simultaneous messages (<10ms) across protocols
        - "request_response": Request followed by response
        - "related_payload": Correlated by payload similarity

    Example:
        >>> corr = MessageCorrelation(
        ...     message1=can_msg,
        ...     message2=eth_msg,
        ...     correlation_type="request_response",
        ...     confidence=0.85,
        ...     time_delta=0.004,
        ...     evidence=["Payload similarity: 0.80", "Source-destination match"]
        ... )
    """

    message1: ProtocolMessage
    message2: ProtocolMessage
    correlation_type: str
    confidence: float
    time_delta: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class SessionFlow:
    """Cross-protocol session flow.

    Represents a logical session composed of correlated messages from
    multiple protocols.

    Attributes:
        start_time: Session start time (first message timestamp).
        end_time: Session end time (last message timestamp).
        messages: All messages in the session (chronologically sorted).
        correlations: All correlations within the session.
        protocols: Set of protocols used in the session.

    Example:
        >>> session = SessionFlow(
        ...     start_time=1.234,
        ...     end_time=1.456,
        ...     messages=[can_msg1, eth_msg1, can_msg2],
        ...     correlations=[corr1, corr2],
        ...     protocols={"can", "ethernet"}
        ... )
        >>> duration = session.end_time - session.start_time
        >>> print(f"Session duration: {duration:.3f}s")
    """

    start_time: float
    end_time: float
    messages: list[ProtocolMessage]
    correlations: list[MessageCorrelation]
    protocols: set[str]


class MultiProtocolCorrelator:
    """Multi-protocol session correlator.

    Analyzes messages from multiple protocols to discover correlations,
    dependencies, and logical session flows.

    Attributes:
        time_window: Maximum time between correlated messages (seconds).
        min_confidence: Minimum confidence threshold for correlations.
        messages: List of all added messages.
        correlations: List of discovered correlations.

    Example:
        >>> correlator = MultiProtocolCorrelator(
        ...     time_window=0.1,  # 100ms max time difference
        ...     min_confidence=0.5  # 50% minimum confidence
        ... )
        >>>
        >>> # Add messages from different protocols
        >>> correlator.add_message(can_msg)
        >>> correlator.add_message(ethernet_msg)
        >>> correlator.add_message(uart_msg)
        >>>
        >>> # Find correlations
        >>> correlations = correlator.correlate_all()
        >>> print(f"Found {len(correlations)} correlations")
        >>>
        >>> # Extract sessions
        >>> sessions = correlator.extract_sessions()
        >>> for session in sessions:
        ...     print(f"Session: {len(session.messages)} messages, "
        ...           f"{len(session.protocols)} protocols")
    """

    def __init__(
        self,
        time_window: float = 0.1,
        min_confidence: float = 0.5,
    ) -> None:
        """Initialize multi-protocol correlator.

        Args:
            time_window: Maximum time between correlated messages (seconds).
                Default 0.1 (100ms). Increase for slower protocols.
            min_confidence: Minimum confidence threshold for correlations (0.0-1.0).
                Default 0.5. Higher values reduce false positives.

        Raises:
            ValueError: If time_window <= 0 or min_confidence not in [0, 1].
        """
        if time_window <= 0:
            raise ValueError(f"time_window must be positive, got {time_window}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")

        self.time_window = time_window
        self.min_confidence = min_confidence
        self.messages: list[ProtocolMessage] = []
        self.correlations: list[MessageCorrelation] = []

    def add_message(self, msg: ProtocolMessage) -> None:
        """Add message from any protocol.

        Messages are stored internally and used for correlation analysis.
        Messages do not need to be added in chronological order.

        Args:
            msg: Protocol message to add.

        Example:
            >>> correlator.add_message(ProtocolMessage(
            ...     protocol="can",
            ...     timestamp=1.234,
            ...     message_id=0x123,
            ...     payload=b"\\x01\\x02"
            ... ))
        """
        self.messages.append(msg)

    def correlate_all(self) -> list[MessageCorrelation]:
        """Find all correlations across protocols.

        Uses multiple correlation methods:
        1. Timestamp proximity (within time_window)
        2. Payload similarity (partial or full match)
        3. ID matching (if available)
        4. Source/destination matching

        Returns:
            List of discovered correlations sorted by confidence (descending).
            Only correlations with confidence >= min_confidence are returned.

        Example:
            >>> correlations = correlator.correlate_all()
            >>> for corr in correlations:
            ...     print(f"{corr.message1.protocol} -> {corr.message2.protocol}: "
            ...           f"{corr.confidence:.2f}")
        """
        # Sort messages by timestamp for efficient windowing
        sorted_messages = sorted(self.messages, key=lambda m: m.timestamp)

        correlations = []

        for i, msg1 in enumerate(sorted_messages):
            # Look for correlations within time window
            for j in range(i + 1, len(sorted_messages)):
                msg2 = sorted_messages[j]

                # Check time window
                time_delta = msg2.timestamp - msg1.timestamp
                if time_delta > self.time_window:
                    break  # No point checking further (sorted by time)

                # Calculate correlation confidence
                confidence, evidence = self._calculate_correlation_confidence(msg1, msg2)

                if confidence >= self.min_confidence:
                    # Determine correlation type
                    if time_delta < 0.01 and msg1.protocol != msg2.protocol:
                        corr_type = "broadcast"
                    elif self._is_request_response(msg1, msg2):
                        corr_type = "request_response"
                    else:
                        corr_type = "related_payload"

                    correlations.append(
                        MessageCorrelation(
                            message1=msg1,
                            message2=msg2,
                            correlation_type=corr_type,
                            confidence=confidence,
                            time_delta=time_delta,
                            evidence=evidence,
                        )
                    )

        # Sort by confidence (descending)
        correlations.sort(key=lambda c: c.confidence, reverse=True)

        self.correlations = correlations
        return correlations

    def find_request_response_pairs(
        self,
        request_protocol: str,
        response_protocol: str,
    ) -> list[MessageCorrelation]:
        """Find request-response pairs across specific protocols.

        Filters correlations to only those matching the specified protocol pair
        and classified as request-response type.

        Args:
            request_protocol: Protocol name for requests (e.g., "can").
            response_protocol: Protocol name for responses (e.g., "ethernet").

        Returns:
            List of request-response correlations matching the protocol pair.

        Example:
            >>> # Find CAN requests that trigger Ethernet responses
            >>> pairs = correlator.find_request_response_pairs("can", "ethernet")
            >>> for pair in pairs:
            ...     print(f"CAN {pair.message1.message_id} -> "
            ...           f"Ethernet after {pair.time_delta*1000:.1f}ms")
        """
        if not self.correlations:
            self.correlate_all()

        return [
            corr
            for corr in self.correlations
            if corr.correlation_type == "request_response"
            and corr.message1.protocol == request_protocol
            and corr.message2.protocol == response_protocol
        ]

    def build_dependency_graph(self) -> Any:
        """Build NetworkX graph showing message dependencies.

        Creates a directed graph where:
        - Nodes represent messages (with protocol, timestamp, ID attributes)
        - Edges represent correlations (with confidence, type, time_delta)

        Returns:
            NetworkX DiGraph with message dependencies.

        Raises:
            ImportError: If networkx is not installed.

        Example:
            >>> graph = correlator.build_dependency_graph()
            >>> print(f"Nodes: {graph.number_of_nodes()}")
            >>> print(f"Edges: {graph.number_of_edges()}")
            >>>
            >>> # Find strongly connected components
            >>> import networkx as nx
            >>> components = list(nx.strongly_connected_components(graph))
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "networkx is required for graph analysis. "
                "Install with: pip install 'oscura[analysis]'"
            ) from e

        if not self.correlations:
            self.correlate_all()

        graph = nx.DiGraph()

        # Add nodes (messages)
        for i, msg in enumerate(self.messages):
            graph.add_node(
                i,
                protocol=msg.protocol,
                timestamp=msg.timestamp,
                id=msg.message_id,
                label=f"{msg.protocol}:{msg.message_id}",
            )

        # Add edges (correlations)
        msg_to_idx = {id(msg): i for i, msg in enumerate(self.messages)}

        for corr in self.correlations:
            idx1 = msg_to_idx[id(corr.message1)]
            idx2 = msg_to_idx[id(corr.message2)]

            graph.add_edge(
                idx1,
                idx2,
                weight=corr.confidence,
                type=corr.correlation_type,
                time_delta=corr.time_delta,
            )

        return graph

    def extract_sessions(self) -> list[SessionFlow]:
        """Extract logical sessions from correlated messages.

        A session is defined as a connected component in the dependency graph,
        representing a group of messages that are directly or indirectly correlated.

        Returns:
            List of SessionFlow objects, each representing a logical session.
            Sessions are sorted by start time.

        Example:
            >>> sessions = correlator.extract_sessions()
            >>> for i, session in enumerate(sessions):
            ...     print(f"Session {i+1}:")
            ...     print(f"  Messages: {len(session.messages)}")
            ...     print(f"  Protocols: {', '.join(session.protocols)}")
            ...     print(f"  Duration: {session.end_time - session.start_time:.3f}s")
        """
        if networkx is None:
            raise ImportError("networkx is required for dependency graph building")
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "networkx is required for session extraction. "
                "Install with: pip install 'oscura[analysis]'"
            ) from e

        graph = self.build_dependency_graph()

        # Find connected components (undirected version)
        undirected = graph.to_undirected()
        components = nx.connected_components(undirected)

        sessions = []
        for component in components:
            # Get messages in this component
            session_messages = [self.messages[i] for i in component]
            session_messages.sort(key=lambda m: m.timestamp)

            # Get correlations within this component
            component_set = set(component)
            msg_indices = {id(msg): i for i, msg in enumerate(self.messages)}

            session_corrs = [
                corr
                for corr in self.correlations
                if msg_indices[id(corr.message1)] in component_set
                and msg_indices[id(corr.message2)] in component_set
            ]

            # Extract protocols used
            protocols = {msg.protocol for msg in session_messages}

            sessions.append(
                SessionFlow(
                    start_time=session_messages[0].timestamp,
                    end_time=session_messages[-1].timestamp,
                    messages=session_messages,
                    correlations=session_corrs,
                    protocols=protocols,
                )
            )

        # Sort sessions by start time
        sessions.sort(key=lambda s: s.start_time)

        return sessions

    def _calculate_correlation_confidence(
        self,
        msg1: ProtocolMessage,
        msg2: ProtocolMessage,
    ) -> tuple[float, list[str]]:
        """Calculate correlation confidence and evidence.

        Uses multiple signals to estimate correlation likelihood:
        - Payload similarity (Jaccard coefficient)
        - Message ID matching
        - Source/destination address matching

        Args:
            msg1: First message.
            msg2: Second message.

        Returns:
            Tuple of (confidence, evidence_list).
            Confidence is clamped to [0.0, 1.0].
        """
        confidence = 0.0
        evidence = []

        # Same protocol = no cross-protocol correlation
        if msg1.protocol == msg2.protocol:
            return 0.0, []

        # Check payload similarity (weighted 40%)
        payload_sim = self._payload_similarity(msg1.payload, msg2.payload)
        if payload_sim > 0.5:
            confidence += 0.4 * payload_sim
            evidence.append(f"Payload similarity: {payload_sim:.2f}")

        # Check ID matching (weighted 30%)
        if msg1.message_id is not None and msg2.message_id is not None:
            if msg1.message_id == msg2.message_id:
                confidence += 0.3
                evidence.append(f"Matching IDs: {msg1.message_id}")

        # Check source/destination matching (weighted 15% each)
        if msg1.source is not None and msg2.destination is not None:
            if msg1.source == msg2.destination:
                confidence += 0.15
                evidence.append("Source-destination match")

        if msg1.destination is not None and msg2.source is not None:
            if msg1.destination == msg2.source:
                confidence += 0.15
                evidence.append("Destination-source match")

        return min(confidence, 1.0), evidence

    def _payload_similarity(self, payload1: bytes, payload2: bytes) -> float:
        """Calculate payload similarity using Jaccard coefficient.

        Computes similarity based on:
        1. Exact containment (one payload contains the other)
        2. Byte set intersection (Jaccard similarity)

        Args:
            payload1: First payload.
            payload2: Second payload.

        Returns:
            Similarity score (0.0-1.0).
        """
        if not payload1 or not payload2:
            return 0.0

        # Check if one contains the other (exact match or prefix/suffix)
        if payload1 in payload2 or payload2 in payload1:
            return 1.0

        # Calculate Jaccard similarity of bytes
        set1 = set(payload1)
        set2 = set(payload2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _is_request_response(
        self,
        msg1: ProtocolMessage,
        msg2: ProtocolMessage,
    ) -> bool:
        """Determine if messages form a request-response pair.

        Heuristics:
        - Source/destination swap (request source = response destination)
        - Time delta in typical RPC range (1-100ms)
        - Payload contains request identifier

        Args:
            msg1: Potential request message.
            msg2: Potential response message.

        Returns:
            True if messages likely form request-response pair.
        """
        # Check source/destination swap
        if (
            msg1.source is not None
            and msg2.destination is not None
            and msg1.source == msg2.destination
        ):
            # Also check time delta is reasonable for RPC
            time_delta = msg2.timestamp - msg1.timestamp
            if 0.001 <= time_delta <= 0.1:  # 1-100ms typical RPC time
                return True

        return False

    def visualize_flow(
        self,
        session: SessionFlow,
        output_path: Path,
    ) -> None:
        """Visualize cross-protocol message flow.

        Generates a timeline diagram showing message flows across protocols.
        Uses matplotlib to create a multi-lane diagram with time on the X-axis
        and different protocols on separate Y-lanes.

        Args:
            session: Session flow to visualize.
            output_path: Path to save visualization (PNG/PDF/SVG).

        Raises:
            ImportError: If matplotlib is not installed.

        Example:
            >>> sessions = correlator.extract_sessions()
            >>> correlator.visualize_flow(sessions[0], Path("session_flow.png"))
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for visualization. Install with: pip install matplotlib"
            ) from e

        fig, ax = plt.subplots(figsize=(12, 6))

        # Assign Y positions to protocols
        protocols_list = sorted(session.protocols)
        protocol_to_y = {proto: i for i, proto in enumerate(protocols_list)}

        # Plot messages
        for msg in session.messages:
            y = protocol_to_y[msg.protocol]
            ax.scatter(msg.timestamp, y, s=100, alpha=0.6)
            ax.text(
                msg.timestamp,
                y + 0.1,
                str(msg.message_id) if msg.message_id else "",
                fontsize=8,
                ha="center",
            )

        # Plot correlations
        msg_to_idx = {id(msg): i for i, msg in enumerate(session.messages)}
        for corr in session.correlations:
            idx1 = msg_to_idx.get(id(corr.message1))
            idx2 = msg_to_idx.get(id(corr.message2))
            if idx1 is not None and idx2 is not None:
                msg1 = session.messages[idx1]
                msg2 = session.messages[idx2]
                y1 = protocol_to_y[msg1.protocol]
                y2 = protocol_to_y[msg2.protocol]

                # Draw arrow
                ax.annotate(
                    "",
                    xy=(msg2.timestamp, y2),
                    xytext=(msg1.timestamp, y1),
                    arrowprops={
                        "arrowstyle": "->",
                        "alpha": 0.3,
                        "lw": 1,
                    },
                )

        # Configure axes
        ax.set_yticks(range(len(protocols_list)))
        ax.set_yticklabels(protocols_list)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Protocol")
        ax.set_title(f"Cross-Protocol Message Flow ({len(session.messages)} messages)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def export_analysis(
        self,
        output_path: Path,
        format: str = "json",
    ) -> None:
        """Export correlation analysis to file.

        Exports all messages, correlations, and sessions to specified format.

        Args:
            output_path: Path to save analysis.
            format: Export format ("json" or "csv").

        Raises:
            ValueError: If format is not supported.

        Example:
            >>> correlator.export_analysis(Path("analysis.json"), format="json")
        """
        if matplotlib is None:
            raise ImportError("matplotlib is required for flow visualization")
        if format == "json":
            self._export_json(output_path)
        elif format == "csv":
            self._export_csv(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self, output_path: Path) -> None:
        """Export analysis to JSON format."""
        import json

        # Build export structure
        data = {
            "config": {
                "time_window": self.time_window,
                "min_confidence": self.min_confidence,
            },
            "messages": [
                {
                    "protocol": msg.protocol,
                    "timestamp": msg.timestamp,
                    "message_id": (
                        msg.message_id if isinstance(msg.message_id, (int, str)) else None
                    ),
                    "payload_hex": msg.payload.hex(),
                    "source": msg.source,
                    "destination": msg.destination,
                    "metadata": msg.metadata,
                }
                for msg in self.messages
            ],
            "correlations": [
                {
                    "message1_idx": self.messages.index(corr.message1),
                    "message2_idx": self.messages.index(corr.message2),
                    "correlation_type": corr.correlation_type,
                    "confidence": corr.confidence,
                    "time_delta": corr.time_delta,
                    "evidence": corr.evidence,
                }
                for corr in self.correlations
            ],
        }

        # Write JSON
        output_path.write_text(json.dumps(data, indent=2))

    def _export_csv(self, output_path: Path) -> None:
        """Export correlations to CSV format."""
        import csv

        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Protocol1",
                    "Timestamp1",
                    "ID1",
                    "Protocol2",
                    "Timestamp2",
                    "ID2",
                    "Type",
                    "Confidence",
                    "TimeDelta",
                    "Evidence",
                ]
            )

            for corr in self.correlations:
                writer.writerow(
                    [
                        corr.message1.protocol,
                        corr.message1.timestamp,
                        corr.message1.message_id,
                        corr.message2.protocol,
                        corr.message2.timestamp,
                        corr.message2.message_id,
                        corr.correlation_type,
                        corr.confidence,
                        corr.time_delta,
                        "; ".join(corr.evidence),
                    ]
                )
