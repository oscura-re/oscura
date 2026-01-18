"""Discovery documentation for CAN reverse engineering.

This module provides functionality to save and load CAN reverse engineering
discoveries in the .tkcan format (YAML-based with evidence tracking).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

if TYPE_CHECKING:
    from oscura.automotive.can.models import SignalDefinition

__all__ = ["DiscoveryDocument", "Hypothesis", "MessageDiscovery", "SignalDiscovery"]


@dataclass
class SignalDiscovery:
    """Documented signal discovery.

    Attributes:
        name: Signal name.
        start_bit: Starting bit position.
        length: Signal length in bits.
        byte_order: Byte order.
        value_type: Value type.
        scale: Scaling factor.
        offset: Offset value.
        unit: Physical unit.
        min_value: Observed minimum value.
        max_value: Observed maximum value.
        confidence: Confidence score (0.0-1.0).
        evidence: List of evidence supporting this discovery.
        comment: Additional notes.
    """

    name: str
    start_bit: int
    length: int
    byte_order: Literal["big_endian", "little_endian"] = "big_endian"
    value_type: Literal["unsigned", "signed", "float"] = "unsigned"
    scale: float = 1.0
    offset: float = 0.0
    unit: str = ""
    min_value: float | None = None
    max_value: float | None = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v
            for k, v in asdict(self).items()
            if v is not None and (not isinstance(v, list) or v)
        }

    @classmethod
    def from_definition(
        cls,
        definition: SignalDefinition,
        confidence: float = 1.0,
        evidence: list[str] | None = None,
    ) -> SignalDiscovery:
        """Create from SignalDefinition.

        Args:
            definition: Signal definition.
            confidence: Confidence score.
            evidence: Evidence list.

        Returns:
            SignalDiscovery instance.
        """
        return cls(
            name=definition.name,
            start_bit=definition.start_bit,
            length=definition.length,
            byte_order=definition.byte_order,
            value_type=definition.value_type,
            scale=definition.scale,
            offset=definition.offset,
            unit=definition.unit,
            min_value=definition.min_value,
            max_value=definition.max_value,
            confidence=confidence,
            evidence=evidence or [],
            comment=definition.comment,
        )


@dataclass
class Hypothesis:
    """A hypothesis under testing.

    Attributes:
        message_id: CAN ID this hypothesis applies to.
        signal: Signal name.
        hypothesis: Description of hypothesis.
        status: Status ('testing', 'confirmed', 'rejected').
        test_plan: Test plan or next steps.
        created: Creation timestamp.
        updated: Last update timestamp.
    """

    message_id: int
    signal: str
    hypothesis: str
    status: Literal["testing", "confirmed", "rejected"] = "testing"
    test_plan: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": f"0x{self.message_id:03X}",
            "signal": self.signal,
            "hypothesis": self.hypothesis,
            "status": self.status,
            "test_plan": self.test_plan,
            "created": self.created,
            "updated": self.updated,
        }


@dataclass
class MessageDiscovery:
    """Documented message discovery.

    Attributes:
        id: CAN arbitration ID.
        name: Message name.
        length: Data length (bytes).
        transmitter: Transmitter node (if known).
        cycle_time_ms: Message period in milliseconds.
        confidence: Confidence score (0.0-1.0).
        evidence: List of evidence supporting this discovery.
        signals: List of discovered signals.
        comment: Additional notes.
    """

    id: int
    name: str
    length: int
    transmitter: str | None = None
    cycle_time_ms: float | None = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    signals: list[SignalDiscovery] = field(default_factory=list)
    comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": f"0x{self.id:03X}",
            "name": self.name,
            "length": self.length,
            "transmitter": self.transmitter,
            "cycle_time_ms": self.cycle_time_ms,
            "confidence": self.confidence,
            "evidence": self.evidence if self.evidence else None,
            "signals": [sig.to_dict() for sig in self.signals] if self.signals else None,
            "comment": self.comment if self.comment else None,
        }


@dataclass
class VehicleInfo:
    """Vehicle information.

    Attributes:
        make: Vehicle manufacturer.
        model: Vehicle model.
        year: Model year.
        vin: Vehicle Identification Number.
        notes: Additional notes.
    """

    make: str = "Unknown"
    model: str = "Unknown"
    year: str | None = None
    vin: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v}


class DiscoveryDocument:
    """CAN reverse engineering discovery document.

    This class manages a collection of discovered messages, signals,
    and hypotheses with evidence tracking.
    """

    def __init__(self) -> None:
        """Initialize empty discovery document."""
        self.format_version = "1.0"
        self.vehicle = VehicleInfo()
        self.messages: dict[int, MessageDiscovery] = {}
        self.hypotheses: list[Hypothesis] = []

    def add_message(self, discovery: MessageDiscovery) -> None:
        """Add message discovery.

        Args:
            discovery: MessageDiscovery to add.
        """
        self.messages[discovery.id] = discovery

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Add hypothesis.

        Args:
            hypothesis: Hypothesis to add.
        """
        self.hypotheses.append(hypothesis)

    def save(self, file_path: Path | str) -> None:
        """Save discoveries to .tkcan file.

        Args:
            file_path: Output file path.
        """
        path = Path(file_path)

        # Build YAML structure
        doc = {
            "format_version": self.format_version,
            "vehicle": self.vehicle.to_dict(),
            "messages": [
                msg.to_dict() for msg in sorted(self.messages.values(), key=lambda m: m.id)
            ],
        }

        if self.hypotheses:
            doc["hypotheses"] = [h.to_dict() for h in self.hypotheses]

        # Write YAML
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, file_path: Path | str) -> DiscoveryDocument:
        """Load discoveries from .tkcan file.

        Args:
            file_path: Input file path.

        Returns:
            DiscoveryDocument instance.
        """
        path = Path(file_path)

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        doc = cls()

        # Load metadata
        doc.format_version = data.get("format_version", "1.0")

        # Load vehicle info
        if "vehicle" in data:
            v = data["vehicle"]
            doc.vehicle = VehicleInfo(
                make=v.get("make", "Unknown"),
                model=v.get("model", "Unknown"),
                year=v.get("year"),
                vin=v.get("vin"),
                notes=v.get("notes", ""),
            )

        # Load messages
        for msg_data in data.get("messages", []):
            # Parse ID
            id_str = msg_data["id"]
            if isinstance(id_str, str):
                msg_id = int(id_str, 16) if id_str.startswith("0x") else int(id_str)
            else:
                msg_id = msg_data["id"]

            # Parse signals
            signals = []
            for sig_data in msg_data.get("signals", []):
                sig = SignalDiscovery(
                    name=sig_data["name"],
                    start_bit=sig_data["start_bit"],
                    length=sig_data["length"],
                    byte_order=sig_data.get("byte_order", "big_endian"),
                    value_type=sig_data.get("value_type", "unsigned"),
                    scale=sig_data.get("scale", 1.0),
                    offset=sig_data.get("offset", 0.0),
                    unit=sig_data.get("unit", ""),
                    min_value=sig_data.get("min_value"),
                    max_value=sig_data.get("max_value"),
                    confidence=sig_data.get("confidence", 0.0),
                    evidence=sig_data.get("evidence", []),
                    comment=sig_data.get("comment", ""),
                )
                signals.append(sig)

            # Create message discovery
            msg_discovery = MessageDiscovery(
                id=msg_id,
                name=msg_data["name"],
                length=msg_data["length"],
                transmitter=msg_data.get("transmitter"),
                cycle_time_ms=msg_data.get("cycle_time_ms"),
                confidence=msg_data.get("confidence", 0.0),
                evidence=msg_data.get("evidence", []),
                signals=signals,
                comment=msg_data.get("comment", ""),
            )

            doc.add_message(msg_discovery)

        # Load hypotheses
        for hyp_data in data.get("hypotheses", []):
            # Parse message ID
            id_str = hyp_data["message_id"]
            if isinstance(id_str, str):
                msg_id = int(id_str, 16) if id_str.startswith("0x") else int(id_str)
            else:
                msg_id = hyp_data["message_id"]

            hyp = Hypothesis(
                message_id=msg_id,
                signal=hyp_data["signal"],
                hypothesis=hyp_data["hypothesis"],
                status=hyp_data.get("status", "testing"),
                test_plan=hyp_data.get("test_plan", ""),
                created=hyp_data.get("created", datetime.now().isoformat()),
                updated=hyp_data.get("updated", datetime.now().isoformat()),
            )
            doc.add_hypothesis(hyp)

        return doc

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"DiscoveryDocument("
            f"{len(self.messages)} messages, "
            f"{sum(len(m.signals) for m in self.messages.values())} signals, "
            f"{len(self.hypotheses)} hypotheses)"
        )
