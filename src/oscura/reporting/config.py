"""Configuration for comprehensive analysis report system.

This module defines the simplified configuration schema for the comprehensive
analysis report generator, combining input/output/analysis specs into a single
unified configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class InputType(str, Enum):
    """Supported input data types."""

    WAVEFORM = "waveform"  # Analog waveform (WFM, CSV, NPZ, HDF5)
    DIGITAL = "digital"  # Digital logic signals
    BINARY = "binary"  # Binary packet data
    PCAP = "pcap"  # Network capture
    IQ = "iq"  # I/Q baseband data
    PACKETS = "packets"  # Pre-parsed packets
    SPARAMS = "sparams"  # S-parameter/Touchstone data (.s1p-.s8p)


class AnalysisDomain(str, Enum):
    """Analysis domain categories."""

    WAVEFORM = "waveform"  # Basic waveform measurements
    DIGITAL = "digital"  # Digital signal analysis
    TIMING = "timing"  # Timing measurements
    SPECTRAL = "spectral"  # FFT, PSD, THD, SNR
    STATISTICS = "statistics"  # Statistical analysis
    PATTERNS = "patterns"  # Pattern detection
    JITTER = "jitter"  # Jitter decomposition
    EYE = "eye"  # Eye diagram analysis
    POWER = "power"  # Power analysis
    PROTOCOLS = "protocols"  # Protocol decoding
    SIGNAL_INTEGRITY = "signal_integrity"  # S-params, equalization
    INFERENCE = "inference"  # Auto-inference
    PACKET = "packet"  # Packet metrics
    ENTROPY = "entropy"  # Entropy analysis


@dataclass
class DomainConfig:
    """Configuration for a single analysis domain.

    Attributes:
        enabled: Whether to run this analysis domain.
        parameters: Domain-specific parameters passed to analyzers.
        timeout: Timeout in seconds for this domain (None = use global timeout).
    """

    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout: float | None = None


@dataclass
class DataOutputConfig:
    """Configuration for data output limits and aggregation.

    Controls how data is serialized in reports - whether to truncate,
    aggregate similar values, or output complete data.

    Attributes:
        full_data_mode: If True, output ALL data without truncation.
        max_array_elements: Max array elements (None = unlimited). Ignored if full_data_mode=True.
        max_list_items: Max list items (None = unlimited). Ignored if full_data_mode=True.
        max_bytes_sample: Max bytes to include (None = unlimited). Ignored if full_data_mode=True.
        max_pdf_results_per_domain: Max results per domain in PDF (None = unlimited).
        max_pdf_summary_length: Max summary string length in PDF (None = unlimited).
        smart_aggregation: Enable smart aggregation of repeated/similar values.
        aggregation_threshold: Min identical values to trigger aggregation (default: 5).
    """

    full_data_mode: bool = True  # Default to full data - no truncation
    max_array_elements: int | None = None  # None = unlimited
    max_list_items: int | None = None  # None = unlimited
    max_bytes_sample: int | None = None  # None = unlimited
    max_pdf_results_per_domain: int | None = None  # None = unlimited
    max_pdf_summary_length: int | None = None  # None = unlimited
    smart_aggregation: bool = True  # Enable smart aggregation by default
    aggregation_threshold: int = 5  # Min identical values to trigger aggregation


# Default sample rates for different analysis contexts
# These are used when sample_rate cannot be derived from input data
DEFAULT_SAMPLE_RATE_HZ: float = 1e9  # 1 GHz - high-speed digital/eye diagram
DEFAULT_SAMPLE_RATE_GENERAL_HZ: float = 1e6  # 1 MHz - general waveform analysis
DEFAULT_SAMPLE_RATE_BINARY_HZ: float = 1.0  # 1 Hz (1 sample/s) - binary data


@dataclass
class AnalysisConfig:
    """Unified configuration for comprehensive analysis.

    Combines input specification, analysis selection, and output options
    into a single simplified configuration.

    Attributes:
        domains: List of domains to analyze (None = all applicable domains).
        exclude_domains: Domains to explicitly exclude.
        domain_config: Per-domain configuration overrides.
        output_formats: Output formats to generate (e.g., ["json", "yaml"]).
        index_formats: Index formats to generate (e.g., ["html", "md"]).
        generate_plots: Whether to generate visualization plots.
        plot_format: Plot file format (png, svg, pdf).
        plot_dpi: Plot resolution in DPI.
        copy_input_file: Copy input file to output directory.
        save_intermediate_data: Save intermediate analysis data.
        full_data_mode: Output all data without truncation.
        smart_aggregation: Enable smart aggregation of repeated values.
        data_output: Advanced data output configuration.
        timeout_per_analysis: Timeout per analysis function in seconds (None = no timeout).
        continue_on_error: Continue analysis if individual functions fail.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        parallel_domains: Enable parallel domain execution.
        enable_quality_scoring: Attach quality scores to analysis results.
        max_memory_mb: Maximum memory per analysis in MB (None = auto-detect).
        max_cache_entries: Maximum cached results (prevents cache bloat).
        max_parallel_workers: Maximum parallel threads for analysis.
        chunk_size_mb: Maximum chunk size for large data processing in MB.
        default_sample_rate: Default sample rate (Hz) when not derivable from data.
            Used for analysis functions requiring sample_rate parameter.
            Set to None to require explicit sample_rate in input data.
        title: Report title.
        author: Report author.
        project: Project name.
        notes: Additional notes/description.
        custom_metadata: Custom metadata fields.
    """

    # Analysis selection
    domains: list[AnalysisDomain] | None = None  # None = all applicable
    exclude_domains: list[AnalysisDomain] = field(default_factory=list)
    domain_config: dict[AnalysisDomain, DomainConfig] = field(default_factory=dict)

    # Output formats
    output_formats: list[str] = field(default_factory=lambda: ["json", "yaml"])
    index_formats: list[str] = field(default_factory=lambda: ["html", "md"])

    # Visualization
    generate_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 150

    # Data handling
    copy_input_file: bool = False
    save_intermediate_data: bool = True
    full_data_mode: bool = True
    smart_aggregation: bool = True
    data_output: DataOutputConfig = field(default_factory=DataOutputConfig)

    # Execution control
    timeout_per_analysis: float | None = 30.0
    continue_on_error: bool = True
    log_level: str = "INFO"
    parallel_domains: bool = True  # Enable parallel domain execution
    enable_quality_scoring: bool = True  # Attach quality scores to analysis results

    # Resource limits (MEM-010, MEM-015)
    max_memory_mb: int = 2048  # Max memory per analysis (MB)
    max_cache_entries: int = 100  # Max cached results
    max_parallel_workers: int = 4  # Max parallel threads
    chunk_size_mb: int = 100  # Max chunk size for large data

    # Sample rate configuration
    # Default sample rate used when not available in input data metadata.
    # This is a fallback; sample_rate is preferably derived from:
    # 1. Input data metadata (e.g., WaveformTrace.metadata.sample_rate)
    # 2. Explicit parameter in domain_config
    # 3. This default value
    default_sample_rate: float | None = DEFAULT_SAMPLE_RATE_GENERAL_HZ

    # Metadata
    title: str = ""
    author: str = ""
    project: str = ""
    notes: str = ""
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def get_domain_config(self, domain: AnalysisDomain) -> DomainConfig:
        """Get configuration for a specific domain.

        Args:
            domain: Analysis domain to get configuration for.

        Returns:
            Domain configuration (default if not specified).
        """
        return self.domain_config.get(domain, DomainConfig())

    def is_domain_enabled(self, domain: AnalysisDomain) -> bool:
        """Check if a domain is enabled.

        Args:
            domain: Analysis domain to check.

        Returns:
            True if domain should be analyzed.
        """
        # Check explicit exclusion
        if domain in self.exclude_domains:
            return False

        # Check domain-specific config
        if domain in self.domain_config:
            return self.domain_config[domain].enabled

        # If domains list specified, check membership
        if self.domains is not None:
            return domain in self.domains

        # Otherwise enabled by default
        return True

    def get_effective_sample_rate(
        self, data_sample_rate: float | None = None, context: str = "general"
    ) -> float:
        """Get effective sample rate, preferring data metadata over defaults.

        Priority order:
        1. data_sample_rate (from input data metadata)
        2. self.default_sample_rate (from config)
        3. Context-appropriate default

        Args:
            data_sample_rate: Sample rate from input data metadata (if available).
            context: Analysis context for selecting appropriate default.
                Options: "general" (1 MHz), "highspeed" (1 GHz), "binary" (1 Hz).

        Returns:
            Effective sample rate in Hz.
        """
        if data_sample_rate is not None and data_sample_rate > 0:
            return data_sample_rate

        if self.default_sample_rate is not None and self.default_sample_rate > 0:
            return self.default_sample_rate

        # Context-appropriate defaults
        if context == "highspeed":
            return DEFAULT_SAMPLE_RATE_HZ
        elif context == "binary":
            return DEFAULT_SAMPLE_RATE_BINARY_HZ
        else:
            return DEFAULT_SAMPLE_RATE_GENERAL_HZ


@dataclass
class ProgressInfo:
    """Progress information for callbacks.

    Attributes:
        phase: Current phase (e.g., "loading", "analyzing", "plotting", "saving").
        domain: Current analysis domain (None during non-domain phases).
        function: Current function name (None during non-function phases).
        percent: Progress percentage (0.0 to 100.0).
        message: Human-readable progress message.
        elapsed_seconds: Time elapsed since analysis started.
        estimated_remaining_seconds: Estimated time remaining (None if unknown).
    """

    phase: str
    domain: AnalysisDomain | None
    function: str | None
    percent: float
    message: str
    elapsed_seconds: float
    estimated_remaining_seconds: float | None


@dataclass
class AnalysisError:
    """Record of an analysis error.

    Attributes:
        domain: Analysis domain where error occurred.
        function: Function name that failed.
        error_type: Error type/class name.
        error_message: Error message.
        traceback: Full traceback (None if not captured).
        duration_ms: Time spent before error occurred.
    """

    domain: AnalysisDomain
    function: str
    error_type: str
    error_message: str
    traceback: str | None
    duration_ms: float


@dataclass
class AnalysisResult:
    """Result from comprehensive analysis.

    Contains paths to all generated outputs and summary statistics.

    Attributes:
        output_dir: Root output directory.
        index_html: Path to HTML index (None if not generated).
        index_md: Path to Markdown index (None if not generated).
        index_pdf: Path to PDF index (None if not generated).
        summary_json: Path to JSON summary.
        summary_yaml: Path to YAML summary (None if not generated).
        metadata_json: Path to metadata file.
        config_yaml: Path to saved configuration.
        domain_dirs: Per-domain output directories.
        plot_paths: List of all generated plot files.
        error_log: Path to error log (None if no errors).
        input_file: Input file path (None if from memory).
        input_type: Input data type.
        total_analyses: Total number of analysis functions attempted.
        successful_analyses: Number of successful analyses.
        failed_analyses: Number of failed analyses.
        skipped_analyses: Number of skipped analyses.
        duration_seconds: Total analysis duration.
        domain_summaries: Per-domain summary data.
        errors: List of errors encountered.
    """

    output_dir: Path
    index_html: Path | None
    index_md: Path | None
    index_pdf: Path | None
    summary_json: Path
    summary_yaml: Path | None
    metadata_json: Path
    config_yaml: Path
    domain_dirs: dict[AnalysisDomain, Path]
    plot_paths: list[Path]
    error_log: Path | None
    input_file: str | None
    input_type: InputType
    total_analyses: int
    successful_analyses: int
    failed_analyses: int
    skipped_analyses: int
    duration_seconds: float
    domain_summaries: dict[AnalysisDomain, dict[str, Any]]
    errors: list[AnalysisError]

    def open_index(self) -> None:
        """Open the HTML index in the default web browser.

        Raises:
            FileNotFoundError: If HTML index was not generated.
        """
        if self.index_html is None:
            raise FileNotFoundError("HTML index was not generated")

        import webbrowser

        webbrowser.open(self.index_html.as_uri())

    def get_domain_results(self, domain: AnalysisDomain) -> dict[str, Any]:
        """Get results for a specific domain.

        Args:
            domain: Domain to get results for.

        Returns:
            Domain summary data.

        Raises:
            KeyError: If domain was not analyzed.
        """
        if domain not in self.domain_summaries:
            raise KeyError(f"Domain {domain.value} was not analyzed")

        return self.domain_summaries[domain]

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage.

        Returns:
            Success rate (0.0 to 100.0).
        """
        if self.total_analyses == 0:
            return 0.0
        return (self.successful_analyses / self.total_analyses) * 100.0

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"AnalysisResult("
            f"domains={len(self.domain_summaries)}, "
            f"success={self.successful_analyses}/{self.total_analyses}, "
            f"duration={self.duration_seconds:.1f}s, "
            f"output_dir={self.output_dir})"
        )


# Analysis capability registry - maps domains to available analyzers
# NOTE: Each domain now supports MULTIPLE modules to capture all available functions
# Updated to include all 68 submodules with 318+ total functions across 14 domains
ANALYSIS_CAPABILITIES: dict[AnalysisDomain, dict[str, Any]] = {
    AnalysisDomain.WAVEFORM: {
        "description": "Basic waveform timing and amplitude measurements",
        "modules": ["oscura.analyzers.waveform.measurements"],
        "requires": ["waveform"],
    },
    AnalysisDomain.SPECTRAL: {
        "description": "FFT, PSD, THD, SNR, SFDR, SINAD, ENOB, wavelet analysis",
        "modules": [
            "oscura.analyzers.waveform.spectral",
            "oscura.analyzers.spectral.chunked",
            "oscura.analyzers.spectral.chunked_fft",
            "oscura.analyzers.spectral.chunked_wavelet",
            "oscura.analyzers.waveform.wavelets",
        ],
        "requires": ["waveform"],
    },
    AnalysisDomain.DIGITAL: {
        "description": "Digital signal extraction, edge detection, timing analysis",
        "modules": [
            "oscura.analyzers.digital.extraction",
            "oscura.analyzers.digital.edges",
            "oscura.analyzers.digital.clock",
            "oscura.analyzers.digital.quality",
            "oscura.analyzers.digital.signal_quality",
            "oscura.analyzers.digital.thresholds",
            "oscura.analyzers.digital.bus",
            "oscura.analyzers.digital.correlation",
        ],
        "requires": ["waveform", "digital"],
    },
    AnalysisDomain.TIMING: {
        "description": "Setup/hold time, propagation delay, skew, slew rate",
        "modules": ["oscura.analyzers.digital.timing"],
        "requires": ["waveform", "digital"],
    },
    AnalysisDomain.STATISTICS: {
        "description": "Statistical measures, outlier detection, trend analysis",
        "modules": [
            "oscura.analyzers.statistics.basic",
            "oscura.analyzers.statistics.advanced",
            "oscura.analyzers.statistics.correlation",
            "oscura.analyzers.statistics.distribution",
            "oscura.analyzers.statistics.outliers",
            "oscura.analyzers.statistics.trend",
            "oscura.analyzers.statistical.chunked_corr",
        ],
        "requires": ["waveform", "digital", "binary"],
    },
    AnalysisDomain.ENTROPY: {
        "description": "Entropy analysis, data classification, checksum detection",
        "modules": [
            "oscura.analyzers.statistical.entropy",
            "oscura.analyzers.statistical.classification",
            "oscura.analyzers.statistical.checksum",
            "oscura.analyzers.statistical.ngrams",
        ],
        "requires": ["binary"],
    },
    AnalysisDomain.PATTERNS: {
        "description": "Periodic patterns, motifs, signatures, clustering",
        "modules": [
            "oscura.analyzers.patterns.discovery",
            "oscura.analyzers.patterns.sequences",
            "oscura.analyzers.patterns.periodic",
            "oscura.analyzers.patterns.matching",
            "oscura.analyzers.patterns.clustering",
            "oscura.analyzers.patterns.learning",
        ],
        "requires": ["waveform", "binary", "digital"],
    },
    AnalysisDomain.JITTER: {
        "description": "RJ, DJ, PJ, DDJ, DCD, bathtub curve, TJ at BER",
        "modules": [
            "oscura.analyzers.jitter.measurements",
            "oscura.analyzers.jitter.decomposition",
            "oscura.analyzers.jitter.spectrum",
            "oscura.analyzers.jitter.ber",
        ],
        "requires": ["waveform", "digital"],
    },
    AnalysisDomain.EYE: {
        "description": "Eye diagram generation and metrics",
        "modules": [
            "oscura.analyzers.eye.diagram",
            "oscura.analyzers.eye.metrics",
        ],
        "requires": ["waveform", "digital"],
    },
    AnalysisDomain.POWER: {
        "description": "Power measurements, efficiency, switching loss, ripple",
        "modules": [
            "oscura.analyzers.power.basic",
            "oscura.analyzers.power.ac_power",
            "oscura.analyzers.power.switching",
            "oscura.analyzers.power.conduction",
            "oscura.analyzers.power.efficiency",
            "oscura.analyzers.power.ripple",
            "oscura.analyzers.power.soa",
        ],
        "requires": ["waveform"],
    },
    AnalysisDomain.PROTOCOLS: {
        "description": "Serial protocol decoding (UART, SPI, I2C, CAN, etc.)",
        "modules": [
            "oscura.analyzers.protocols.uart",
            "oscura.analyzers.protocols.spi",
            "oscura.analyzers.protocols.i2c",
            "oscura.analyzers.protocols.can",
            "oscura.analyzers.protocols.can_fd",
            "oscura.analyzers.protocols.lin",
            "oscura.analyzers.protocols.flexray",
            "oscura.analyzers.protocols.manchester",
            "oscura.analyzers.protocols.onewire",
            "oscura.analyzers.protocols.usb",
            "oscura.analyzers.protocols.i2s",
            "oscura.analyzers.protocols.jtag",
            "oscura.analyzers.protocols.swd",
            "oscura.analyzers.protocols.hdlc",
        ],
        "requires": ["digital", "waveform"],
    },
    AnalysisDomain.SIGNAL_INTEGRITY: {
        "description": "S-parameters, de-embedding, equalization",
        "modules": [
            "oscura.analyzers.signal_integrity.sparams",
            "oscura.analyzers.signal_integrity.equalization",
            "oscura.analyzers.signal_integrity.embedding",
        ],
        "requires": ["sparams", "waveform"],
    },
    AnalysisDomain.PACKET: {
        "description": "Packet metrics, throughput, latency, loss, payload analysis",
        "modules": [
            "oscura.analyzers.packet.metrics",
            "oscura.analyzers.packet.parser",
            "oscura.analyzers.packet.payload",
            "oscura.analyzers.packet.stream",
            "oscura.analyzers.packet.daq",
        ],
        "requires": ["packets", "binary"],
    },
    AnalysisDomain.INFERENCE: {
        "description": "Auto-inference, protocol detection, signal classification",
        "modules": [
            "oscura.inference.signal_intelligence",  # classify_signal, assess_signal_quality, suggest_measurements
            "oscura.inference.logic",  # detect_logic_family
            "oscura.inference.protocol",  # detect_protocol
            "oscura.inference.spectral",  # auto_spectral_config
            "oscura.inference.stream",  # reassemble_udp_stream, reassemble_tcp_stream, detect_message_framing
            "oscura.inference.binary",  # detect_magic_bytes, detect_alignment, generate_parser
            "oscura.inference.message_format",  # infer_format, detect_field_types, find_dependencies
            "oscura.inference.sequences",  # detect_sequence_patterns, correlate_requests, find_message_dependencies
            "oscura.inference.alignment",  # align_global, align_local, compute_similarity
            "oscura.inference.state_machine",  # infer_rpni, minimize_dfa
            "oscura.inference.protocol_dsl",  # decode_message, load_protocol
            "oscura.inference.protocol_library",  # get_protocol, list_protocols, get_decoder
        ],
        "requires": ["waveform", "digital", "binary", "packets"],
    },
}


def get_available_analyses(input_type: InputType) -> list[AnalysisDomain]:
    """Get list of analyses applicable to input type.

    Args:
        input_type: Type of input data.

    Returns:
        List of applicable analysis domains.
    """
    type_mapping = {
        InputType.WAVEFORM: "waveform",
        InputType.DIGITAL: "digital",
        InputType.BINARY: "binary",
        InputType.PCAP: "packets",
        InputType.IQ: "waveform",
        InputType.PACKETS: "packets",
        InputType.SPARAMS: "sparams",
    }

    input_category = type_mapping.get(input_type, "waveform")

    applicable = []
    for domain, config in ANALYSIS_CAPABILITIES.items():
        if input_category in config["requires"]:
            applicable.append(domain)

    return applicable


# Type alias for progress callbacks
ProgressCallback = Callable[["ProgressInfo"], None]


__all__ = [
    "ANALYSIS_CAPABILITIES",
    "DEFAULT_SAMPLE_RATE_BINARY_HZ",
    "DEFAULT_SAMPLE_RATE_GENERAL_HZ",
    "DEFAULT_SAMPLE_RATE_HZ",
    "AnalysisConfig",
    "AnalysisDomain",
    "AnalysisError",
    "AnalysisResult",
    "DataOutputConfig",
    "DomainConfig",
    "InputType",
    "ProgressCallback",
    "ProgressInfo",
    "get_available_analyses",
]
