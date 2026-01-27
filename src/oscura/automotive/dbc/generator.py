"""DBC file generator wrapper for backward compatibility.

This module provides a backward-compatible wrapper around the comprehensive
DBC generator in oscura.automotive.can.dbc_generator, supporting the legacy
static method API used by existing code.

For new code, import directly from oscura.automotive.can.dbc_generator instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oscura.automotive.can.discovery import DiscoveryDocument
    from oscura.automotive.can.session import CANSession

from oscura.automotive.can.dbc_generator import (
    DBCGenerator as _DBCGeneratorImpl,
)
from oscura.automotive.can.dbc_generator import (
    DBCMessage,
    DBCSignal,
)

__all__ = ["DBCGenerator"]


class DBCGenerator:
    """Backward-compatible DBC generator wrapper.

    This class provides static methods for generating DBC files from
    DiscoveryDocument and CANSession objects, maintaining API compatibility
    with the original implementation.

    For new code, use oscura.automotive.can.dbc_generator.DBCGenerator directly.
    """

    @staticmethod
    def generate(
        discovery: DiscoveryDocument,
        output_path: Path | str,
        min_confidence: float = 0.0,
        include_comments: bool = True,
    ) -> None:
        """Generate DBC file from discovery document.

        Args:
            discovery: DiscoveryDocument with discovered signals.
            output_path: Output DBC file path.
            min_confidence: Minimum confidence threshold for including signals.
            include_comments: Include evidence as comments in DBC.

        Example:
            >>> from oscura.automotive.can.discovery import DiscoveryDocument
            >>> doc = DiscoveryDocument()
            >>> # ... add messages and signals ...
            >>> DBCGenerator.generate(doc, "output.dbc")
        """

        path = Path(output_path)

        # Create instance of new generator
        gen = _DBCGeneratorImpl()

        # Convert discovery document to DBC format
        for msg_id, msg_discovery in sorted(discovery.messages.items()):
            # Filter signals by confidence
            signals = [s for s in msg_discovery.signals if s.confidence >= min_confidence]

            if not signals:
                continue  # Skip messages with no high-confidence signals

            # Convert signals to DBC format
            dbc_signals = []
            for sig in signals:
                # Convert value_type: DBC only supports unsigned/signed, not float
                # Floats are typically represented as unsigned in DBC
                value_type: str = sig.value_type
                if value_type not in ("unsigned", "signed"):
                    value_type = "unsigned"

                dbc_sig = DBCSignal(
                    name=sig.name,
                    start_bit=sig.start_bit,
                    bit_length=sig.length,
                    byte_order=sig.byte_order,
                    value_type=value_type,  # type: ignore[arg-type]
                    factor=sig.scale,
                    offset=sig.offset,
                    min_value=sig.min_value if sig.min_value is not None else 0.0,
                    max_value=sig.max_value if sig.max_value is not None else 0.0,
                    unit=sig.unit,
                    receivers=["Vector__XXX"],
                    comment="; ".join(sig.evidence) if include_comments and sig.evidence else "",
                )
                dbc_signals.append(dbc_sig)

            # Create DBC message
            dbc_msg = DBCMessage(
                message_id=msg_id,
                name=msg_discovery.name,
                dlc=msg_discovery.length,
                sender=msg_discovery.transmitter if msg_discovery.transmitter else "Vector__XXX",
                signals=dbc_signals,
                comment=msg_discovery.comment if msg_discovery.comment else "",
                cycle_time=int(msg_discovery.cycle_time_ms)
                if msg_discovery.cycle_time_ms is not None
                else None,
                send_type="Cyclic",
            )

            gen.add_message(dbc_msg)

        # Generate DBC file
        gen.generate(path)

    @staticmethod
    def generate_from_session(
        session: CANSession,
        output_path: Path | str,
        min_confidence: float = 0.8,  # Reserved for future use
    ) -> None:
        """Generate DBC file from CANSession with documented signals.

        Args:
            session: CANSession with documented signals.
            output_path: Output DBC file path.
            min_confidence: Minimum confidence threshold (reserved for future use).

        Example:
            >>> from oscura.automotive.can.session import CANSession
            >>> session = CANSession.from_file("capture.blf")
            >>> # ... document signals ...
            >>> DBCGenerator.generate_from_session(session, "output.dbc")
        """
        from oscura.automotive.can.discovery import (
            DiscoveryDocument,
            MessageDiscovery,
            SignalDiscovery,
        )

        # Build discovery document from session
        doc = DiscoveryDocument()

        # Get all unique IDs that have documented signals
        for arb_id in session.unique_ids():
            try:
                msg_wrapper = session.message(arb_id)
                documented = msg_wrapper.get_documented_signals()

                if documented:
                    # Create message discovery
                    analysis = session.analyze_message(arb_id)

                    signal_discoveries = []
                    for sig_def in documented.values():
                        sig_disc = SignalDiscovery.from_definition(
                            sig_def,
                            confidence=1.0,
                            evidence=[],
                        )
                        signal_discoveries.append(sig_disc)

                    msg_disc = MessageDiscovery(
                        id=arb_id,
                        name=f"Message_{arb_id:03X}",
                        length=max(
                            msg.dlc for msg in session._messages.filter_by_id(arb_id).messages
                        ),
                        cycle_time_ms=analysis.period_ms,
                        confidence=1.0,
                        signals=signal_discoveries,
                    )

                    doc.add_message(msg_disc)

            except Exception:
                # Skip messages without documented signals
                pass

        # Generate DBC using main method
        DBCGenerator.generate(doc, output_path, min_confidence=0.0)
