"""Performance optimization utilities for Oscura.

This package provides memory optimization, streaming analysis, parallel processing,
performance profiling, caching, and monitoring capabilities for processing large
signal files efficiently.
"""

from oscura.utils.performance.caching import (
    CacheBackend,
    CacheEntry,
    CacheManager,
    CachePolicy,
    CacheStats,
    cache,
    get_global_cache,
)
from oscura.utils.performance.memory_optimizer import (
    ChunkingConfig,
    ChunkingStrategy,
    MemoryOptimizer,
    MemoryStats,
    StreamProcessor,
)
from oscura.utils.performance.parallel import (
    ParallelConfig,
    ParallelProcessor,
    ParallelResult,
    WorkerStats,
)
from oscura.utils.performance.profiling import (
    FunctionStats,
    PerformanceProfiler,
    ProfilingMode,
    ProfilingResult,
)

__all__ = [
    "CacheBackend",
    "CacheEntry",
    "CacheManager",
    "CachePolicy",
    "CacheStats",
    "ChunkingConfig",
    "ChunkingStrategy",
    "FunctionStats",
    "MemoryOptimizer",
    "MemoryStats",
    "ParallelConfig",
    "ParallelProcessor",
    "ParallelResult",
    "PerformanceProfiler",
    "ProfilingMode",
    "ProfilingResult",
    "StreamProcessor",
    "WorkerStats",
    "cache",
    "get_global_cache",
]
