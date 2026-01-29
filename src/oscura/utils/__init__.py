"""Oscura utilities package.

Provides utility functions for memory management, windowing, progressive analysis,
geometry, serial communication, validation, and bitwise operations.
"""

from oscura.utils.bitwise import bits_to_byte, bits_to_value
from oscura.utils.geometry import generate_leader_line
from oscura.utils.memory import (
    MemoryCheck,
    MemoryCheckError,
    MemoryConfig,
    MemoryEstimate,
    MemoryMonitor,
    check_memory_available,
    configure_memory,
    detect_wsl,
    estimate_memory,
    get_available_memory,
    get_max_memory,
    get_memory_config,
    get_memory_info,
    get_memory_pressure,
    get_swap_available,
    get_total_memory,
    require_memory,
    set_max_memory,
    suggest_downsampling,
)
from oscura.utils.memory_advanced import (
    AdaptiveMeasurementSelector,
    BackpressureController,
    CacheEntry,
    CacheInvalidationStrategy,
    DiskCache,
    GCController,
    MemoryLogEntry,
    MemoryLogger,
    MultiChannelMemoryManager,
    QualityMode,
    QualityModeConfig,
    WSLSwapChecker,
    gc_aggressive,
    get_quality_config,
    get_wsl_memory_limits,
)
from oscura.utils.progressive import (
    PreviewResult,
    ROISelection,
    analyze_roi,
    create_preview,
    estimate_optimal_preview_factor,
    progressive_analysis,
    select_roi,
)
from oscura.utils.serial import connect_serial_port
from oscura.utils.validation import validate_protocol_spec

__all__ = [
    # Advanced memory management (MEM-014, 020, 023, 025, 028, 030-033)
    "AdaptiveMeasurementSelector",
    "BackpressureController",
    "CacheEntry",
    "CacheInvalidationStrategy",
    "DiskCache",
    "GCController",
    # Memory management
    "MemoryCheck",
    "MemoryCheckError",
    "MemoryConfig",
    "MemoryEstimate",
    "MemoryLogEntry",
    "MemoryLogger",
    "MemoryMonitor",
    "MultiChannelMemoryManager",
    # Progressive analysis
    "PreviewResult",
    "QualityMode",
    "QualityModeConfig",
    "ROISelection",
    "WSLSwapChecker",
    "analyze_roi",
    # Bitwise operations
    "bits_to_byte",
    "bits_to_value",
    "check_memory_available",
    "configure_memory",
    # Serial communication
    "connect_serial_port",
    "create_preview",
    "detect_wsl",
    "estimate_memory",
    "estimate_optimal_preview_factor",
    "gc_aggressive",
    # Geometry
    "generate_leader_line",
    "get_available_memory",
    "get_max_memory",
    "get_memory_config",
    "get_memory_info",
    "get_memory_pressure",
    "get_quality_config",
    "get_swap_available",
    "get_total_memory",
    "get_wsl_memory_limits",
    "progressive_analysis",
    "require_memory",
    "select_roi",
    "set_max_memory",
    "suggest_downsampling",
    # Validation
    "validate_protocol_spec",
]
