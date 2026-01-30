"""Oscura - Reverse engineer ANY system from captured waveforms (analog or digital).

The open-source toolkit for complete signal reverse engineering across analog and digital domains.

Oscura provides comprehensive tools for:
- Waveform analysis (rise/fall time, frequency, amplitude)
- Digital signal analysis (edge detection, clock recovery)
- Spectral analysis (FFT, PSD, THD, SNR, SINAD, ENOB)
- Protocol decoding (UART, SPI, I2C, CAN, 1-Wire, and more)
- Signal filtering (IIR, FIR, Butterworth, Chebyshev, Bessel, Elliptic)
- Triggering (edge, pattern, pulse width, glitch, runt, window)
- Power analysis (AC/DC, switching, SOA, efficiency, ripple)
- Arithmetic operations (add, subtract, differentiate, integrate)
- Comparison and limit testing (golden waveform, mask testing)
- Component analysis (TDR, impedance, capacitance, inductance)
- Statistical analysis and distribution metrics
- Memory management and large file handling
- Professional report generation
- EMC compliance testing
- Session management
- Data visualization and export
- Signal generation with fluent builders
- One-call convenience functions

Example:
    >>> import oscura as osc
    >>> trace = osc.load("capture.wfm")
    >>> print(f"Rise time: {osc.rise_time(trace):.2e} s")
    >>> freq, mag = osc.fft(trace)
    >>> print(f"THD: {osc.thd(trace):.1f} dB")
    >>> # One-call spectral analysis
    >>> metrics = osc.quick_spectral(trace, fundamental=1000)
    >>> print(f"THD: {metrics.thd_db:.1f} dB, SNR: {metrics.snr_db:.1f} dB")
    >>> # Auto-decode protocol
    >>> result = osc.auto_decode(trace)
    >>> print(f"Protocol: {result.protocol}, Frames: {len(result.frames)}")
    >>> # Generate test signals
    >>> signal = (osc.SignalBuilder(sample_rate=1e6, duration=0.01)
    ...     .add_sine(frequency=1000)
    ...     .add_noise(snr_db=40)
    ...     .build())
    >>> # Reverse engineer unknown signal
    >>> result = osc.workflows.reverse_engineer_signal(trace)
    >>> print(result.protocol_spec)

For more information, see https://github.com/oscura-re/oscura
"""

# Version dynamically imported from package metadata (SSOT: pyproject.toml)
try:
    from importlib.metadata import version

    __version__ = version("oscura")
except Exception:
    # Fallback for development/testing when package not installed
    __version__ = "0.7.0"

__author__ = "Oscura Contributors"

# Core types
# Digital analysis (top-level convenience access)
from oscura.analyzers.digital.extraction import (
    LOGIC_FAMILIES,
    detect_edges,
    to_digital,
)

# Signal quality analysis (QUAL-001, QUAL-002, QUAL-005, QUAL-006, QUAL-007)
from oscura.analyzers.digital.quality import (
    Glitch,
    MaskTestResult,
    NoiseMarginResult,
    PLLRecoveryResult,
    Violation,
    detect_glitches,
    detect_violations,
    noise_margin,
    pll_clock_recovery,
    signal_quality_summary,
)
from oscura.analyzers.power.ac_power import (
    apparent_power,
    power_factor,
    reactive_power,
)

# Power analysis (top-level convenience access)
from oscura.analyzers.power.basic import (
    average_power,
    energy,
    instantaneous_power,
    power_statistics,
)
from oscura.analyzers.power.efficiency import (
    efficiency,
)
from oscura.analyzers.power.ripple import (
    ripple,
    ripple_statistics,
)

# Protocol decoders (top-level convenience access)
from oscura.analyzers.protocols import (
    decode_can,
    decode_can_fd,
    decode_flexray,
    decode_hdlc,
    decode_i2c,
    decode_i2s,
    decode_jtag,
    decode_lin,
    decode_manchester,
    decode_onewire,
    decode_spi,
    decode_swd,
    decode_uart,
    decode_usb,
)

# Statistics (top-level convenience access)
from oscura.analyzers.statistics.basic import (
    basic_stats,
    percentiles,
    quartiles,
)
from oscura.analyzers.statistics.distribution import (
    distribution_metrics,
    histogram,
)

# Waveform measurements (top-level convenience access)
from oscura.analyzers.waveform.measurements import (
    amplitude,
    duty_cycle,
    fall_time,
    frequency,
    mean,
    measure,
    overshoot,
    period,
    preshoot,
    pulse_width,
    rise_time,
    rms,
    undershoot,
)

# Convenience aliases
vpp = amplitude  # Vpp (peak-to-peak voltage) is a common oscilloscope term

# Spectral analysis (top-level convenience access)
from oscura.analyzers.waveform.spectral import (
    clear_fft_cache,
    configure_fft_cache,
    enob,
    fft,
    get_fft_cache_stats,
    psd,
    sfdr,
    sinad,
    snr,
    spectrogram,
    thd,
)

# Automotive - CAN analysis and DBC generation
from oscura.automotive.can.session import CANSession
from oscura.automotive.dbc.generator import DBCGenerator

# Convenience functions (one-call analysis)
from oscura.convenience import (
    DecodeResult,
    SpectralMetrics,
    auto_decode,
    quick_spectral,
    smart_filter,
)

# Audit trail (LOG-009)
from oscura.core.audit import (
    AuditEntry,
    AuditTrail,
    get_global_audit_trail,
    record_audit,
)

# Configuration
from oscura.core.config import (
    DEFAULT_CONFIG,
    SmartDefaults,
    get_config_value,
    load_config,
    save_config,
    validate_config,
)

# Exceptions - import from core.exceptions (the enhanced version)
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

# Expert API - Extensibility (API-006, API-007, API-008, PLUG-008)
from oscura.core.extensibility import (
    AlgorithmRegistry,
    MeasurementDefinition,
    MeasurementRegistry,
    PluginError,
    PluginManager,
    PluginMetadata,
    PluginTemplate,
    PluginType,
    generate_plugin_template,
    get_algorithm,
    get_algorithms,
    get_measurement_registry,
    get_plugin_manager,
    list_measurements,
    list_plugins,
    load_plugin,
    register_algorithm,
    register_measurement,
)
from oscura.core.extensibility import (
    measure as measure_custom,
)

# Logging
from oscura.core.logging import (
    configure_logging,
    get_logger,
    set_log_level,
)

# Performance timing
from oscura.core.performance import timed

# Expert API - Results (API-005)
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

# Discovery module (signal characterization, anomaly detection)
from oscura.discovery import (
    Anomaly,
    DataQuality,
    SignalCharacterization,
    TraceDiff,
    assess_data_quality,
    characterize_signal,
    find_anomalies,
)
from oscura.discovery import (
    compare_traces as discovery_compare_traces,
)
from oscura.discovery import (
    decode_protocol as discovery_decode_protocol,
)

# Data Export - Removed legacy exports
# Use oscura.export.* modules directly for protocol export functionality
# Auto-Inference (INF-001 to INF-009)
from oscura.inference import (
    AnalysisRecommendation,
    assess_signal_quality,
    auto_spectral_config,
    check_measurement_suitability,
    classify_signal,
    detect_logic_family,
    detect_protocol,
    get_optimal_domain_order,
    recommend_analyses,
    suggest_measurements,
)

# Loaders (including multi-channel support)
from oscura.loaders import get_supported_formats, load, load_all_channels

# Reporting
from oscura.reporting.core import (
    Report,
    ReportConfig,
    generate_report,
)
from oscura.reporting.formatting import (
    NumberFormatter,
    format_value,
    format_with_context,
    format_with_units,
)

# Session Management - Use new AnalysisSession API
from oscura.sessions import BlackBoxSession, GenericSession

# Signal builders (top-level convenience access)
from oscura.utils.builders import (
    SignalBuilder,
)

# Comparison and limit testing
from oscura.utils.comparison.compare import (
    compare_traces,
    correlation,
    difference,
    similarity_score,
)
from oscura.utils.comparison.golden import (
    GoldenReference,
    compare_to_golden,
    create_golden,
)
from oscura.utils.comparison.limits import (
    LimitSpec,
    check_limits,
    create_limit_spec,
    margin_analysis,
)
from oscura.utils.comparison.mask import (
    Mask,
    create_mask,
    eye_mask,
    mask_test,
)

# Component analysis
from oscura.utils.component.impedance import (
    discontinuity_analysis,
    extract_impedance,
    impedance_profile,
)
from oscura.utils.component.reactive import (
    extract_parasitics,
    measure_capacitance,
    measure_inductance,
)
from oscura.utils.component.transmission_line import (
    characteristic_impedance,
    propagation_delay,
    transmission_line_analysis,
    velocity_factor,
)

# Filtering (top-level convenience access)
from oscura.utils.filtering.convenience import (
    band_pass,
    band_stop,
    high_pass,
    low_pass,
    median_filter,
    moving_average,
    notch_filter,
    savgol_filter,
)
from oscura.utils.filtering.design import (
    BandPassFilter,
    BandStopFilter,
    HighPassFilter,
    LowPassFilter,
    design_filter,
)

# Math/arithmetic operations (top-level convenience access)
from oscura.utils.math.arithmetic import (
    absolute,
    add,
    differentiate,
    divide,
    integrate,
    invert,
    math_expression,
    multiply,
    offset,
    scale,
    subtract,
)
from oscura.utils.math.interpolation import (
    align_traces,
    downsample,
    interpolate,
    resample,
)

# Memory management
from oscura.utils.memory import (
    MemoryCheckError,
    check_memory_available,
    estimate_memory,
    get_available_memory,
    get_memory_pressure,
    get_total_memory,
)

# Expert API - Pipeline and Composition (API-001, API-002, API-004)
from oscura.utils.pipeline import (
    Composable,
    Pipeline,
    TraceTransformer,
    compose,
    curry,
    make_composable,
    pipe,
)

# Expert API - Streaming (API-003)
from oscura.utils.streaming import (
    StreamingAnalyzer,
    load_trace_chunks,
)

# Triggering (top-level convenience access)
from oscura.utils.triggering import (
    EdgeTrigger,
    PulseWidthTrigger,
    find_falling_edges,
    find_glitches,
    find_pulses,
    find_rising_edges,
    find_runt_pulses,
    find_triggers,
)

# EMC Compliance (EMC-001, EMC-002, EMC-003)
from oscura.validation.compliance import (
    AVAILABLE_MASKS,
    ComplianceReportFormat,
    ComplianceResult,
    ComplianceViolation,
    DetectorType,
    LimitMask,
    check_compliance,
    create_custom_mask,
    generate_compliance_report,
    load_limit_mask,
)

# Visualization (top-level convenience access)
# Import conditionally - these require matplotlib
try:
    from oscura.visualization import (
        plot_fft,
        plot_spectrum,
        plot_waveform,
    )

    _HAS_VISUALIZATION = True
except ImportError:
    # Visualization features require matplotlib
    _HAS_VISUALIZATION = False
    plot_fft = None  # type: ignore[assignment]
    plot_spectrum = None  # type: ignore[assignment]
    plot_waveform = None  # type: ignore[assignment]

# Workflows (WRK-001 to WRK-005 + reverse engineering)
from oscura.workflows import (
    FieldSpec,
    InferredFrame,
    ProtocolSpec,
    ReverseEngineeringResult,
    characterize_buffer,
    debug_protocol,
    emc_compliance_test,
    power_analysis,
    reverse_engineer_signal,
    signal_integrity_audit,
)

__all__ = [
    # EMC Compliance (EMC-001, EMC-002, EMC-003)
    "AVAILABLE_MASKS",
    "DEFAULT_CONFIG",
    "LOGIC_FAMILIES",
    # Expert API - Extensibility (API-006, API-007, API-008)
    "AlgorithmRegistry",
    "AnalysisError",
    # Auto-Inference (INF-009)
    "AnalysisRecommendation",
    # Expert API - Results (API-005)
    "AnalysisResult",
    # Discovery
    "Anomaly",
    # Audit trail (LOG-009)
    "AuditEntry",
    "AuditTrail",
    "BandPassFilter",
    "BandStopFilter",
    # Session Management
    "BlackBoxSession",
    "CANSession",
    "ComplianceReportFormat",
    "ComplianceResult",
    "ComplianceViolation",
    "Composable",
    "ConfigurationError",
    # Automotive
    "DBCGenerator",
    # Discovery
    "DataQuality",
    # Convenience functions
    "DecodeResult",
    "DetectorType",
    "DigitalTrace",
    "EdgeTrigger",
    "ExportError",
    "FFTResult",
    # Reverse engineering workflow
    "FieldSpec",
    "FilterResult",
    "FormatError",
    "GenericSession",
    # Signal quality (QUAL-005)
    "Glitch",
    "GoldenReference",
    "HighPassFilter",
    # Reverse engineering workflow
    "InferredFrame",
    "InsufficientDataError",
    "LimitMask",
    "LimitSpec",
    "LoaderError",
    "LowPassFilter",
    "Mask",
    # Signal quality (QUAL-006)
    "MaskTestResult",
    "MeasurementDefinition",
    "MeasurementRegistry",
    "MeasurementResult",
    "MemoryCheckError",
    # Signal quality (QUAL-001)
    "NoiseMarginResult",
    "NumberFormatter",
    # Exceptions
    "OscuraError",
    # Signal quality (QUAL-007)
    "PLLRecoveryResult",
    # Expert API - Pipeline (API-001, API-002, API-004)
    "Pipeline",
    "PluginError",
    "PluginManager",
    "PluginMetadata",
    "PluginTemplate",
    "PluginType",
    "ProtocolPacket",
    # Reverse engineering workflow
    "ProtocolSpec",
    "PulseWidthTrigger",
    # Reporting
    "Report",
    "ReportConfig",
    # Reverse engineering workflow
    "ReverseEngineeringResult",
    "SampleRateError",
    # Signal builders
    "SignalBuilder",
    # Discovery
    "SignalCharacterization",
    "SmartDefaults",
    # Convenience functions
    "SpectralMetrics",
    "StreamingAnalyzer",
    "Trace",
    # Discovery
    "TraceDiff",
    # Core types
    "TraceMetadata",
    "TraceTransformer",
    "UnsupportedFormatError",
    "ValidationError",
    # Signal quality (QUAL-002)
    "Violation",
    "WaveformTrace",
    "WaveletResult",
    # Version
    "__version__",
    "absolute",
    # Math operations
    "add",
    "align_traces",
    "amplitude",
    "apparent_power",
    # Discovery
    "assess_data_quality",
    "assess_signal_quality",
    # Convenience functions
    "auto_decode",
    "auto_spectral_config",
    "average_power",
    "band_pass",
    "band_stop",
    # Statistics
    "basic_stats",
    "characteristic_impedance",
    # Workflows (WRK-001 to WRK-005)
    "characterize_buffer",
    # Discovery
    "characterize_signal",
    "check_compliance",
    "check_limits",
    "check_measurement_suitability",
    "check_memory_available",
    "classify_signal",
    # Performance optimization
    "clear_fft_cache",
    "compare_to_golden",
    # Comparison
    "compare_traces",
    "compose",
    "configure_fft_cache",
    # Logging
    "configure_logging",
    "correlation",
    "create_custom_mask",
    "create_golden",
    "create_limit_spec",
    "create_mask",
    "curry",
    "debug_protocol",
    # Protocol decoders
    "decode_can",
    "decode_can_fd",
    "decode_flexray",
    "decode_hdlc",
    "decode_i2c",
    "decode_i2s",
    "decode_jtag",
    "decode_lin",
    "decode_manchester",
    "decode_onewire",
    "decode_spi",
    "decode_swd",
    "decode_uart",
    "decode_usb",
    "design_filter",
    "detect_edges",
    # Signal quality (QUAL-005)
    "detect_glitches",
    # Auto-Inference (INF-001 to INF-003)
    "detect_logic_family",
    "detect_protocol",
    # Signal quality (QUAL-002)
    "detect_violations",
    "difference",
    "differentiate",
    "discontinuity_analysis",
    # Discovery aliases
    "discovery_compare_traces",
    "discovery_decode_protocol",
    "distribution_metrics",
    "divide",
    "downsample",
    "duty_cycle",
    "efficiency",
    "emc_compliance_test",
    "energy",
    "enob",
    # Memory management
    "estimate_memory",
    # Export functions - removed legacy exports
    # Use oscura.export.wireshark, kaitai_struct, scapy_layer instead
    # Component analysis
    "extract_impedance",
    "extract_parasitics",
    "eye_mask",
    "fall_time",
    # Spectral analysis
    "fft",
    # Discovery
    "find_anomalies",
    "find_falling_edges",
    "find_glitches",
    "find_pulses",
    "find_rising_edges",
    "find_runt_pulses",
    # Triggering
    "find_triggers",
    "format_value",
    "format_with_context",
    "format_with_units",
    "frequency",
    "generate_compliance_report",
    "generate_plugin_template",
    "generate_report",
    "get_algorithm",
    "get_algorithms",
    "get_available_memory",
    "get_config_value",
    "get_fft_cache_stats",
    "get_global_audit_trail",
    "get_logger",
    "get_measurement_registry",
    "get_memory_pressure",
    # Auto-Inference (INF-009)
    "get_optimal_domain_order",
    "get_plugin_manager",
    "get_supported_formats",
    "get_total_memory",
    "high_pass",
    "histogram",
    "impedance_profile",
    # Power analysis
    "instantaneous_power",
    "integrate",
    "interpolate",
    "invert",
    "list_measurements",
    "list_plugins",
    # Loaders
    "load",
    # Multi-channel loading (Phase 3)
    "load_all_channels",
    # Configuration
    "load_config",
    "load_limit_mask",
    "load_plugin",
    # Session Management - removed legacy load_session
    # Use AnalysisSession API instead
    # Expert API - Streaming (API-003)
    "load_trace_chunks",
    # Filtering
    "low_pass",
    "make_composable",
    "margin_analysis",
    "mask_test",
    "math_expression",
    "mean",
    "measure",
    "measure_capacitance",
    "measure_custom",
    "measure_inductance",
    "median_filter",
    "moving_average",
    "multiply",
    # Signal quality (QUAL-001)
    "noise_margin",
    "notch_filter",
    "offset",
    "overshoot",
    "percentiles",
    "period",
    "pipe",
    # Signal quality (QUAL-007)
    "pll_clock_recovery",
    # Visualization
    "plot_fft",
    "plot_spectrum",
    "plot_waveform",
    "power_analysis",
    "power_factor",
    "power_statistics",
    "preshoot",
    "propagation_delay",
    "psd",
    "pulse_width",
    "quartiles",
    # Convenience functions
    "quick_spectral",
    "reactive_power",
    # Auto-Inference (INF-009)
    "recommend_analyses",
    "record_audit",
    "register_algorithm",
    "register_measurement",
    "resample",
    # Reverse engineering workflow
    "reverse_engineer_signal",
    "ripple",
    "ripple_statistics",
    # Waveform measurements
    "rise_time",
    "rms",
    "save_config",
    "savgol_filter",
    "scale",
    "set_log_level",
    "sfdr",
    "signal_integrity_audit",
    # Signal quality summary
    "signal_quality_summary",
    "similarity_score",
    "sinad",
    # Convenience functions
    "smart_filter",
    "snr",
    "spectrogram",
    "subtract",
    "suggest_measurements",
    "thd",
    "timed",
    # Digital analysis
    "to_digital",
    "transmission_line_analysis",
    "undershoot",
    "validate_config",
    "velocity_factor",
    "vpp",
]
