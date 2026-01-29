"""Pipeline architecture for composable trace transformations.

This package provides pipeline composition, functional operators, and
base classes for building reusable trace processing workflows.

- RE-INT-001: RE Pipeline Integration
"""

from .base import TraceTransformer
from .composition import Composable, compose, curry, make_composable, pipe
from .pipeline import Pipeline

# RE-INT-001: Reverse Engineering Pipeline
from .reverse_engineering import (
    FlowInfo,
    MessageTypeInfo,
    ProtocolCandidate,
    REAnalysisResult,
    REPipeline,
    StageResult,
    analyze,
)

__all__ = [
    "Composable",
    # RE-INT-001: Reverse Engineering Pipeline
    "FlowInfo",
    "MessageTypeInfo",
    # Pipeline
    "Pipeline",
    "ProtocolCandidate",
    "REAnalysisResult",
    "REPipeline",
    "StageResult",
    # Base classes
    "TraceTransformer",
    "analyze",
    # Composition
    "compose",
    "curry",
    "make_composable",
    "pipe",
]
