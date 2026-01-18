"""Eye diagram analysis module.

This module provides eye diagram generation and metrics for serial
data signal quality analysis.


Example:
    >>> from oscura.analyzers.eye import generate_eye, eye_height, eye_width
    >>> eye = generate_eye(trace, unit_interval=1e-9)
    >>> height = eye_height(eye)
    >>> width = eye_width(eye)

References:
    IEEE 802.3: Ethernet Physical Layer Specifications
    OIF CEI: Common Electrical I/O
"""

from oscura.analyzers.eye.diagram import (
    EyeDiagram,
    generate_eye,
    generate_eye_from_edges,
)
from oscura.analyzers.eye.metrics import (
    EyeMetrics,
    crossing_percentage,
    eye_contour,
    eye_height,
    eye_width,
    measure_eye,
    q_factor,
)

__all__ = [
    # Diagram generation
    "EyeDiagram",
    # Metrics
    "EyeMetrics",
    "crossing_percentage",
    "eye_contour",
    "eye_height",
    "eye_width",
    "generate_eye",
    "generate_eye_from_edges",
    "measure_eye",
    "q_factor",
]
