"""Digital signal analysis module.

Provides digital signal extraction, edge detection, timing analysis,
signal quality metrics, multi-channel correlation, clock recovery,
bus decoding, edge timing analysis, and signal quality metrics.

- RE-THR-001: Time-Varying Threshold Support
- RE-THR-002: Multi-Level Logic Support
"""

from oscura.analyzers.digital.bus import (
    BusConfig,
    BusDecoder,
    BusTransaction,
    ParallelBusConfig,
    decode_bus,
    sample_at_clock,
)
from oscura.analyzers.digital.clock import (
    BaudRateResult,
    ClockMetrics,
    ClockRecovery,
    detect_baud_rate,
    detect_clock_frequency,
    measure_clock_jitter,
    recover_clock,
)
from oscura.analyzers.digital.correlation import (
    ChannelCorrelator,
    CorrelatedChannels,
    CorrelationResult,
    align_by_trigger,
    correlate_channels,
    resample_to_common_rate,
)
from oscura.analyzers.digital.edges import (
    Edge,
    EdgeDetector,
    EdgeTiming,
    TimingConstraint,
    check_timing_constraints,
    classify_edge_quality,
    interpolate_edge_time,
    measure_edge_timing,
)
from oscura.analyzers.digital.edges import (
    TimingViolation as EdgeTimingViolation,
)
from oscura.analyzers.digital.edges import (
    detect_edges as detect_edges_advanced,
)
from oscura.analyzers.digital.extraction import (
    LOGIC_FAMILIES,
    detect_edges,
    detect_logic_family,
    detect_open_collector,
    get_logic_threshold,
    to_digital,
)
from oscura.analyzers.digital.ic_database import (
    IC_DATABASE,
    ICTiming,
    identify_ic,
    validate_ic_timing,
)
from oscura.analyzers.digital.quality import (
    Glitch,
    NoiseMarginResult,
    Violation,
    detect_glitches,
    detect_violations,
    noise_margin,
    signal_quality_summary,
)
from oscura.analyzers.digital.signal_quality import (
    NoiseMargins,
    SignalIntegrityReport,
    SignalQualityAnalyzer,
    SimpleQualityMetrics,
    TransitionMetrics,
    analyze_signal_integrity,
    measure_noise_margins,
)

# RE-THR-001, RE-THR-002: Adaptive Thresholding and Multi-Level Logic
from oscura.analyzers.digital.thresholds import (
    AdaptiveThresholder,
    AdaptiveThresholdResult,
    MultiLevelDetector,
    MultiLevelResult,
    ThresholdConfig,
    apply_adaptive_threshold,
    calculate_threshold_snr,
    detect_multi_level,
)
from oscura.analyzers.digital.timing import (
    ClockRecoveryResult,
    TimingViolation,
    hold_time,
    phase,
    propagation_delay,
    recover_clock_edge,
    recover_clock_fft,
    setup_time,
    skew,
    slew_rate,
)
from oscura.analyzers.digital.timing_paths import (
    ICStage,
    SetupHoldAnalysis,
    TimingPathResult,
    analyze_setup_hold,
    analyze_timing_path,
    calculate_timing_budget,
    find_critical_paths,
)
from oscura.analyzers.digital.vintage import (
    REPLACEMENT_DATABASE,
    analyze_vintage_logic,
)
from oscura.analyzers.digital.vintage_result import (
    BOMEntry,
    ICIdentificationResult,
    ModernReplacementIC,
    VintageLogicAnalysisResult,
)

__all__ = [
    # IC Database
    "IC_DATABASE",
    # Extraction
    "LOGIC_FAMILIES",
    "REPLACEMENT_DATABASE",
    "AdaptiveThresholdResult",
    # Adaptive Thresholds (RE-THR-001)
    "AdaptiveThresholder",
    # Vintage Logic Analysis
    "BOMEntry",
    # Clock Recovery (DSP-002)
    "BaudRateResult",
    # Bus Decoding (DSP-003)
    "BusConfig",
    "BusDecoder",
    "BusTransaction",
    # Correlation (DSP-001)
    "ChannelCorrelator",
    "ClockMetrics",
    "ClockRecovery",
    # Timing
    "ClockRecoveryResult",
    "CorrelatedChannels",
    "CorrelationResult",
    # Edge Analysis (DSP-004)
    "Edge",
    "EdgeDetector",
    "EdgeTiming",
    "EdgeTimingViolation",
    # Quality
    "Glitch",
    "ICIdentificationResult",
    "ICStage",
    "ICTiming",
    "ModernReplacementIC",
    # Multi-Level Logic (RE-THR-002)
    "MultiLevelDetector",
    "MultiLevelResult",
    "NoiseMarginResult",
    # Signal Quality (DSP-005)
    "NoiseMargins",
    "ParallelBusConfig",
    "SetupHoldAnalysis",
    "SignalIntegrityReport",
    "SignalQualityAnalyzer",
    "SimpleQualityMetrics",
    "ThresholdConfig",
    "TimingConstraint",
    "TimingPathResult",
    "TimingViolation",
    "TransitionMetrics",
    "VintageLogicAnalysisResult",
    "Violation",
    "align_by_trigger",
    "analyze_setup_hold",
    "analyze_signal_integrity",
    "analyze_timing_path",
    "analyze_vintage_logic",
    "apply_adaptive_threshold",
    "calculate_threshold_snr",
    "calculate_timing_budget",
    "check_timing_constraints",
    "classify_edge_quality",
    "correlate_channels",
    "decode_bus",
    "detect_baud_rate",
    "detect_clock_frequency",
    "detect_edges",
    "detect_edges_advanced",
    "detect_glitches",
    "detect_logic_family",
    "detect_multi_level",
    "detect_open_collector",
    "detect_violations",
    "find_critical_paths",
    "get_logic_threshold",
    "hold_time",
    "identify_ic",
    "interpolate_edge_time",
    "measure_clock_jitter",
    "measure_edge_timing",
    "measure_noise_margins",
    "noise_margin",
    "phase",
    "propagation_delay",
    "recover_clock",
    "recover_clock_edge",
    "recover_clock_fft",
    "resample_to_common_rate",
    "sample_at_clock",
    "setup_time",
    "signal_quality_summary",
    "skew",
    "slew_rate",
    "to_digital",
    "validate_ic_timing",
]
