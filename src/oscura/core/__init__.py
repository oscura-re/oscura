"""Oscura core module.

Provides core data types, configuration, exception handling, and logging infrastructure.
"""

from oscura.core.audit import (
    AuditEntry,
    AuditTrail,
    get_global_audit_trail,
    record_audit,
)
from oscura.core.cache import (
    CacheEntry,
    CacheStats,
    OscuraCache,
    clear_cache,
    get_cache,
    show_cache_stats,
)
from oscura.core.config import (
    DEFAULT_CONFIG,
    get_config_value,
    load_config,
    save_config,
    validate_config,
)
from oscura.core.correlation import (
    CorrelationContext,
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    with_correlation_id,
)
from oscura.core.cross_domain import (
    DOMAIN_AFFINITY,
    CorrelationResult,
    CrossDomainCorrelator,
    CrossDomainInsight,
    correlate_results,
)
from oscura.core.debug import (
    DebugContext,
    DebugLevel,
    debug_context,  # Backward compatibility alias
    disable_debug,
    enable_debug,
    get_debug_level,
    is_debug_enabled,
)
from oscura.core.edge_cases import (
    EmptyTraceError,
    InsufficientSamplesError,
    SignalQualityReport,
    check_signal_quality,
    check_single_sample,
    handle_empty_trace,
    sanitize_signal,
    validate_signal,
)
from oscura.core.exceptions import (
    AnalysisError,
    ConfigurationError,
    ExportError,
    FormatError,
    InsufficientDataError,
    LoaderError,
    OscuraError,
    SampleRateError,
    UnsupportedFormatError,
    ValidationError,
)
from oscura.core.gpu_backend import (
    GPUBackend,
    gpu,
)
from oscura.core.lazy import (
    LazyAnalysisResult,
    LazyComputeStats,
    LazyDict,
    LazyResult,
    get_lazy_stats,
    lazy,
    reset_lazy_stats,
)
from oscura.core.log_query import (
    LogQuery,
    LogRecord,
    query_logs,
)
from oscura.core.logging import (
    ErrorContextCapture,
    configure_logging,
    format_timestamp,
    get_logger,
    log_exception,
    set_log_level,
)
from oscura.core.memoize import (
    array_hash,
    memoize_analysis,
)
from oscura.core.memory_guard import (
    MemoryGuard,
    can_allocate,
    get_memory_usage_mb,
    get_safe_chunk_size,
    safe_array_size,
)
from oscura.core.memory_monitor import (
    MemoryMonitor,
    MemorySnapshot,
    ProgressMonitor,
    ProgressWithMemory,
    monitor_memory,
)
from oscura.core.memory_progress import (
    MemoryLogEntry,
    MemoryLogger,
    create_progress_callback_with_logging,
    enable_memory_logging_from_cli,
    log_memory,
)
from oscura.core.performance import (
    PerformanceCollector,
    PerformanceContext,
    PerformanceRecord,
    clear_performance_data,
    get_performance_records,
    get_performance_summary,
    timed,
)
from oscura.core.progress import (
    CancellationToken,
    CancelledError,
    ProgressCallback,
    ProgressTracker,
    check_memory_available,
    create_progress_tracker,
    create_simple_progress,
    estimate_memory_usage,
    warn_memory_usage,
)
from oscura.core.provenance import (
    MeasurementResultWithProvenance,
    Provenance,
    compute_input_hash,
    create_provenance,
)
from oscura.core.results import (
    AnalysisResult,
    FFTResult,
    FilterResult,
    MeasurementResult,
    WaveletResult,
)
from oscura.core.types import (
    DigitalTrace,
    ProtocolPacket,
    Trace,
    TraceMetadata,
    WaveformTrace,
)

__all__ = [
    "DEFAULT_CONFIG",
    "DOMAIN_AFFINITY",
    "AnalysisError",
    # Results (API-005)
    "AnalysisResult",
    # Audit (LOG-009)
    "AuditEntry",
    "AuditTrail",
    # Cache (MEM-029, MEM-031)
    "CacheEntry",
    "CacheStats",
    "CancellationToken",
    "CancelledError",
    "ConfigurationError",
    "CorrelationContext",
    # Cross-domain correlation (CORE-007)
    "CorrelationResult",
    "CrossDomainCorrelator",
    "CrossDomainInsight",
    "DebugContext",
    "DebugLevel",
    "DigitalTrace",
    # Edge cases (EDGE-001, EDGE-002, EDGE-003)
    "EmptyTraceError",
    "ErrorContextCapture",
    "ExportError",
    "FFTResult",
    "FilterResult",
    "FormatError",
    # GPU Backend (PERF-001 through PERF-004)
    "GPUBackend",
    "InsufficientDataError",
    "InsufficientSamplesError",
    # Lazy evaluation (PERF-002, MEM-010, API-012)
    "LazyAnalysisResult",
    "LazyComputeStats",
    "LazyDict",
    "LazyResult",
    "LoaderError",
    # Log Query (LOG-010)
    "LogQuery",
    "LogRecord",
    "MeasurementResult",
    # Provenance (API-011)
    "MeasurementResultWithProvenance",
    # Memory guards (MEM-010, MEM-015, MEM-020)
    "MemoryGuard",
    # Memory logging (MEM-025)
    "MemoryLogEntry",
    "MemoryLogger",
    # Memory monitoring
    "MemoryMonitor",
    "MemorySnapshot",
    "OscuraCache",
    # Exceptions
    "OscuraError",
    "PerformanceCollector",
    "PerformanceContext",
    "PerformanceRecord",
    # Progress (PROG-001, PROG-002, PROG-003)
    "ProgressCallback",
    "ProgressMonitor",
    "ProgressTracker",
    "ProgressWithMemory",
    "ProtocolPacket",
    "Provenance",
    "SampleRateError",
    "SignalQualityReport",
    "Trace",
    # Types
    "TraceMetadata",
    "UnsupportedFormatError",
    "ValidationError",
    "WaveformTrace",
    "WaveletResult",
    # Memoization (PERF-001)
    "array_hash",
    "can_allocate",
    "check_memory_available",
    "check_signal_quality",
    "check_single_sample",
    "clear_cache",
    "clear_performance_data",
    "compute_input_hash",
    # Logging (LOG-001, LOG-002, LOG-003, LOG-005, LOG-008)
    "configure_logging",
    "correlate_results",
    "create_progress_callback_with_logging",
    "create_progress_tracker",
    "create_provenance",
    "create_simple_progress",
    "debug_context",  # Backward compatibility
    "disable_debug",
    # Debug (LOG-007)
    "enable_debug",
    "enable_memory_logging_from_cli",
    "estimate_memory_usage",
    "format_timestamp",
    "generate_correlation_id",
    "get_cache",
    "get_config_value",
    # Correlation (LOG-004)
    "get_correlation_id",
    "get_debug_level",
    "get_global_audit_trail",
    "get_lazy_stats",
    "get_logger",
    "get_memory_usage_mb",
    "get_performance_records",
    "get_performance_summary",
    "get_safe_chunk_size",
    "gpu",
    "handle_empty_trace",
    "is_debug_enabled",
    "lazy",
    # Config
    "load_config",
    "log_exception",
    "log_memory",
    "memoize_analysis",
    "monitor_memory",
    "query_logs",
    "record_audit",
    "reset_lazy_stats",
    "safe_array_size",
    "sanitize_signal",
    "save_config",
    "set_correlation_id",
    "set_log_level",
    "show_cache_stats",
    # Performance (LOG-006)
    "timed",
    "validate_config",
    "validate_signal",
    "warn_memory_usage",
    "with_correlation_id",
]
