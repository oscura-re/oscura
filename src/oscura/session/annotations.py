"""Trace annotation support.

This module provides annotation capabilities for marking points of interest
in signal traces.


Example:
    >>> layer = AnnotationLayer("Debug Markers")
    >>> layer.add(Annotation(time=1.5e-6, text="Glitch detected"))
    >>> layer.add(Annotation(time_range=(2e-6, 3e-6), text="Data packet"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AnnotationType(Enum):
    """Types of annotations."""

    POINT = "point"  # Single time point
    RANGE = "range"  # Time range
    VERTICAL = "vertical"  # Vertical line
    HORIZONTAL = "horizontal"  # Horizontal line
    REGION = "region"  # 2D region (time + amplitude)
    TEXT = "text"  # Free-floating text


@dataclass
class Annotation:
    """Single annotation on a trace.

    Attributes:
        text: Annotation text/label
        time: Time point (for point annotations)
        time_range: (start, end) time range
        amplitude: Amplitude value (for horizontal lines)
        amplitude_range: (min, max) amplitude range
        annotation_type: Type of annotation
        color: Display color (hex or name)
        style: Line style ('solid', 'dashed', 'dotted')
        visible: Whether annotation is visible
        created_at: Creation timestamp
        metadata: Additional metadata
    """

    text: str
    time: float | None = None
    time_range: tuple[float, float] | None = None
    amplitude: float | None = None
    amplitude_range: tuple[float, float] | None = None
    annotation_type: AnnotationType = AnnotationType.POINT
    color: str = "#FF6B6B"
    style: str = "solid"
    visible: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Infer annotation type from provided parameters."""
        if self.annotation_type == AnnotationType.POINT:
            if self.time_range is not None:
                self.annotation_type = AnnotationType.RANGE
            elif self.amplitude is not None and self.time is None:
                self.annotation_type = AnnotationType.HORIZONTAL
            elif self.amplitude_range is not None and self.time_range is not None:
                self.annotation_type = AnnotationType.REGION  # type: ignore[unreachable]

    @property
    def start_time(self) -> float | None:
        """Get start time for range annotations."""
        if self.time_range:
            return self.time_range[0]
        return self.time

    @property
    def end_time(self) -> float | None:
        """Get end time for range annotations."""
        if self.time_range:
            return self.time_range[1]
        return self.time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "time": self.time,
            "time_range": self.time_range,
            "amplitude": self.amplitude,
            "amplitude_range": self.amplitude_range,
            "annotation_type": self.annotation_type.value,
            "color": self.color,
            "style": self.style,
            "visible": self.visible,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        """Create from dictionary."""
        data = data.copy()
        data["annotation_type"] = AnnotationType(data.get("annotation_type", "point"))
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class AnnotationLayer:
    """Collection of related annotations.

    Attributes:
        name: Layer name
        annotations: List of annotations
        visible: Whether layer is visible
        locked: Whether layer is locked (read-only)
        color: Default color for new annotations
        description: Layer description
    """

    name: str
    annotations: list[Annotation] = field(default_factory=list)
    visible: bool = True
    locked: bool = False
    color: str = "#FF6B6B"
    description: str = ""

    def add(
        self,
        annotation: Annotation | None = None,
        *,
        text: str = "",
        time: float | None = None,
        time_range: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Annotation:
        """Add annotation to layer.

        Args:
            annotation: Pre-built Annotation object.
            text: Annotation text (if not using pre-built).
            time: Time point.
            time_range: Time range.
            **kwargs: Additional Annotation parameters.

        Returns:
            Added annotation.

        Raises:
            ValueError: If layer is locked.
        """
        if self.locked:
            raise ValueError(f"Layer '{self.name}' is locked")

        if annotation is None:
            annotation = Annotation(
                text=text,
                time=time,
                time_range=time_range,
                color=kwargs.pop("color", self.color),
                **kwargs,
            )

        self.annotations.append(annotation)
        return annotation

    def remove(self, annotation: Annotation) -> bool:
        """Remove annotation from layer.

        Args:
            annotation: Annotation to remove.

        Returns:
            True if removed, False if not found.

        Raises:
            ValueError: If layer is locked.
        """
        if self.locked:
            raise ValueError(f"Layer '{self.name}' is locked")

        try:
            self.annotations.remove(annotation)
            return True
        except ValueError:
            return False

    def find_at_time(
        self,
        time: float,
        tolerance: float = 0.0,
    ) -> list[Annotation]:
        """Find annotations at or near a specific time.

        Args:
            time: Time to search.
            tolerance: Time tolerance for matching.

        Returns:
            List of matching annotations.
        """
        matches = []
        for ann in self.annotations:
            if ann.time is not None:
                if abs(ann.time - time) <= tolerance:
                    matches.append(ann)
            elif ann.time_range is not None and (
                ann.time_range[0] - tolerance <= time <= ann.time_range[1] + tolerance
            ):
                matches.append(ann)
        return matches

    def find_in_range(
        self,
        start_time: float,
        end_time: float,
    ) -> list[Annotation]:
        """Find annotations within a time range.

        Args:
            start_time: Range start.
            end_time: Range end.

        Returns:
            List of annotations within range.
        """
        matches = []
        for ann in self.annotations:
            ann_start = ann.start_time
            ann_end = ann.end_time

            if ann_start is not None and (
                start_time <= ann_start <= end_time
                or (ann_end is not None and ann_start <= end_time and ann_end >= start_time)
            ):
                matches.append(ann)

        return matches

    def clear(self) -> int:
        """Remove all annotations.

        Returns:
            Number of annotations removed.

        Raises:
            ValueError: If layer is locked.
        """
        if self.locked:
            raise ValueError(f"Layer '{self.name}' is locked")

        count = len(self.annotations)
        self.annotations.clear()
        return count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "annotations": [a.to_dict() for a in self.annotations],
            "visible": self.visible,
            "locked": self.locked,
            "color": self.color,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationLayer:
        """Create from dictionary."""
        annotations = [Annotation.from_dict(a) for a in data.get("annotations", [])]
        return cls(
            name=data["name"],
            annotations=annotations,
            visible=data.get("visible", True),
            locked=data.get("locked", False),
            color=data.get("color", "#FF6B6B"),
            description=data.get("description", ""),
        )


__all__ = [
    "Annotation",
    "AnnotationLayer",
    "AnnotationType",
]
