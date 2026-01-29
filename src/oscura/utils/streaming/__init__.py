"""Streaming APIs for memory-efficient large file processing.

This package provides chunk-by-chunk processing capabilities for huge
waveform files that don't fit in memory, plus real-time streaming APIs.
"""

from .chunked import (
    StreamingAnalyzer,
    chunked_fft,
    chunked_spectrogram,
    load_trace_chunks,
)
from .progressive import (
    ProgressiveAnalyzer,
    StreamingConfig,
    StreamingProgress,
    create_progressive_analyzer,
)
from .realtime import (
    RealtimeAnalyzer,
    RealtimeBuffer,
    RealtimeConfig,
    RealtimeSource,
    RealtimeStream,
    SimulatedSource,
)

__all__ = [
    "ProgressiveAnalyzer",
    "RealtimeAnalyzer",
    "RealtimeBuffer",
    "RealtimeConfig",
    "RealtimeSource",
    "RealtimeStream",
    "SimulatedSource",
    "StreamingAnalyzer",
    "StreamingConfig",
    "StreamingProgress",
    "chunked_fft",
    "chunked_spectrogram",
    "create_progressive_analyzer",
    "load_trace_chunks",
]
