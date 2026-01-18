"""DBC file generator from discovery documents.

This module generates standard DBC files from Oscura discovery documents,
enabling export of reverse-engineered protocols to industry-standard format.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oscura.automotive.can.discovery import (
        DiscoveryDocument,
    )
    from oscura.automotive.can.session import CANSession

__all__ = ["DBCGenerator"]


class DBCGenerator:
    """Generate DBC files from discovery documents."""

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
        """
        path = Path(output_path)

        lines = []

        # DBC header
        lines.append('VERSION ""')
        lines.append("")
        lines.append("NS_ :")
        lines.append("")
        lines.append("BS_:")
        lines.append("")
        lines.append("BU_:")  # No nodes defined
        lines.append("")

        # Generate messages
        for msg in sorted(discovery.messages.values(), key=lambda m: m.id):
            # Filter signals by confidence
            signals = [s for s in msg.signals if s.confidence >= min_confidence]

            if not signals:
                continue  # Skip messages with no high-confidence signals

            # Message definition
            # Format: BO_ <ID> <Name>: <DLC> <Transmitter>
            transmitter = msg.transmitter if msg.transmitter else "Vector__XXX"
            lines.append(f"BO_ {msg.id} {msg.name}: {msg.length} {transmitter}")

            # Signals
            for sig in signals:
                # Format: SG_ <Name> : <StartBit>|<Length>@<ByteOrder><ValueType> (<Scale>,<Offset>) [<Min>|<Max>] "<Unit>" <Receiver>
                byte_order = "0" if sig.byte_order == "big_endian" else "1"
                value_type = "+" if sig.value_type == "unsigned" else "-"

                min_val = sig.min_value if sig.min_value is not None else 0
                max_val = sig.max_value if sig.max_value is not None else 0

                signal_line = (
                    f" SG_ {sig.name} : {sig.start_bit}|{sig.length}@{byte_order}{value_type} "
                    f"({sig.scale},{sig.offset}) [{min_val}|{max_val}] "
                    f'"{sig.unit}" Vector__XXX'
                )

                lines.append(signal_line)

                # Add comment if evidence exists
                if include_comments and sig.evidence:
                    evidence_str = "; ".join(sig.evidence)
                    # DBC comments format: CM_ SG_ <ID> <SignalName> "<Comment>";
                    comment_line = f'CM_ SG_ {msg.id} {sig.name} "{evidence_str}";'
                    lines.append("")
                    lines.append(comment_line)

            lines.append("")

        # Write file with latin-1 encoding (DBC standard)
        with open(path, "w", encoding="latin-1") as f:
            f.write("\n".join(lines))

    @staticmethod
    def generate_from_session(
        session: CANSession,
        output_path: Path | str,
        min_confidence: float = 0.8,  # noqa: ARG004 - API compatibility parameter
    ) -> None:
        """Generate DBC file from CANSession with documented signals.

        Args:
            session: CANSession with documented signals.
            output_path: Output DBC file path.
            min_confidence: Minimum confidence threshold (reserved for future use).
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

        # Generate DBC
        DBCGenerator.generate(doc, output_path, min_confidence=0.0)
