"""Analysis Engine for orchestrating comprehensive analysis execution.

This module provides the AnalysisEngine class that orchestrates running all
applicable analyses on input data, handling progress tracking, timeouts,
and error collection.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import time
import traceback
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from oscura.reporting.argument_preparer import ArgumentPreparer
from oscura.reporting.config import (
    ANALYSIS_CAPABILITIES,
    AnalysisConfig,
    AnalysisDomain,
    AnalysisError,
    InputType,
    ProgressInfo,
    get_available_analyses,
)

logger = logging.getLogger(__name__)


# Functions that require context-specific parameters that cannot be auto-detected
NON_INFERRABLE_FUNCTIONS: set[str] = {
    # INFERENCE domain - require specific data types
    "oscura.inference.protocol_dsl.decode_protocol",
    "oscura.inference.protocol_dsl.match_pattern",
    "oscura.inference.protocol_dsl.validate_message",
    # PACKET domain - require PacketInfo objects
    "oscura.analyzers.packet.timing.analyze_inter_packet_timing",
    "oscura.analyzers.packet.timing.detect_bursts",
    # POWER domain - require voltage+current pairs
    "oscura.analyzers.power.consumption.calculate_power",
    "oscura.analyzers.power.consumption.analyze_power_efficiency",
}


class AnalysisEngine:
    """Engine for orchestrating comprehensive analysis execution.

    The AnalysisEngine accepts input data (from file or in-memory), detects
    the input type, determines applicable analysis domains, and executes
    all relevant analysis functions with progress tracking and error handling.

    Example:
        >>> from oscura.reporting import AnalysisEngine, AnalysisConfig
        >>> config = AnalysisConfig(timeout_per_analysis=30.0)
        >>> engine = AnalysisEngine(config)
        >>> result = engine.run(input_path=Path("data.wfm"))
        >>> print(f"Ran {result['stats']['total_analyses']} analyses")
        >>> print(f"Success rate: {result['stats']['success_rate']:.1f}%")
    """

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        """Initialize the analysis engine.

        Args:
            config: Analysis configuration. If None, uses defaults.
        """
        self.config = config or AnalysisConfig()
        self._start_time = 0.0
        self._input_path: Path | None = None
        self._arg_preparer: ArgumentPreparer | None = None

    def detect_input_type(self, input_path: Path | None, data: Any) -> InputType:
        """Detect input type from file path or data characteristics.

        Args:
            input_path: Path to input file (None if in-memory data).
            data: Input data object.

        Returns:
            Detected input type.

        Raises:
            ValueError: If input type cannot be determined.
        """
        # Try path-based detection first
        if input_path is not None:
            path_type = self._detect_from_extension(input_path)
            if path_type is not None:
                return path_type

        # Fallback to data object characteristics
        data_type = self._detect_from_data_object(data)
        if data_type is not None:
            return data_type

        raise ValueError("Unable to determine input type from path or data characteristics")

    def _detect_from_extension(self, input_path: Path) -> InputType | None:
        """Detect input type from file extension."""
        ext = input_path.suffix.lower()

        # Waveform formats
        if ext in {".wfm", ".csv", ".npz", ".h5", ".hdf5", ".wav", ".tdms"}:
            return InputType.WAVEFORM

        # Digital formats
        if ext in {".vcd", ".sr"}:
            return InputType.DIGITAL

        # Packet formats
        if ext in {".pcap", ".pcapng"}:
            return InputType.PCAP

        # Binary formats
        if ext in {".bin", ".raw"}:
            return InputType.BINARY

        # S-parameter/Touchstone formats
        if ext in {".s1p", ".s2p", ".s3p", ".s4p", ".s5p", ".s6p", ".s7p", ".s8p"}:
            return InputType.SPARAMS

        return None

    def _detect_from_data_object(self, data: Any) -> InputType | None:
        """Detect input type from data object characteristics."""
        # SParameterData
        if hasattr(data, "s_matrix") and hasattr(data, "frequencies"):
            return InputType.SPARAMS

        # WaveformTrace or DigitalTrace
        if hasattr(data, "data") and hasattr(data, "metadata"):
            if hasattr(data.metadata, "is_digital") and data.metadata.is_digital:
                return InputType.DIGITAL
            return InputType.WAVEFORM

        # Raw binary data
        if isinstance(data, bytes | bytearray):
            return InputType.BINARY

        # Packet list
        if isinstance(data, list):
            return InputType.PACKETS

        # NumPy array (assume waveform)
        if isinstance(data, np.ndarray):
            return InputType.WAVEFORM

        return None

    def _initialize_engine(self, input_path: Path | None) -> None:
        """Initialize engine state for analysis run.

        Args:
            input_path: Input file path (or None for in-memory data).
        """
        self._start_time = time.time()
        self._input_path = input_path
        default_sample_rate = self.config.default_sample_rate or 1e6
        self._arg_preparer = ArgumentPreparer(
            input_path=input_path, default_sample_rate=default_sample_rate
        )

    def _check_memory_and_adjust(self) -> None:
        """Check available memory and adjust parallelism if needed."""
        from oscura.core.memory_guard import check_memory_available

        min_required_mb = 500
        if not check_memory_available(min_required_mb):
            logger.warning(
                f"Low memory available (< {min_required_mb} MB). "
                f"Reducing parallel workers to conserve memory."
            )
            self.config.parallel_domains = False

    def _load_input_data(
        self,
        input_path: Path | None,
        data: Any,
        progress_callback: Callable[[ProgressInfo], None] | None,
    ) -> Any:
        """Load input data from file or validate in-memory data.

        Args:
            input_path: Path to input file (or None).
            data: In-memory data (or None).
            progress_callback: Progress callback.

        Returns:
            Loaded or provided data.

        Raises:
            FileNotFoundError: If file not found.
        """
        if progress_callback:
            progress_callback(
                ProgressInfo(
                    phase="loading",
                    domain=None,
                    function=None,
                    percent=0.0,
                    message="Loading input data",
                    elapsed_seconds=0.0,
                    estimated_remaining_seconds=None,
                )
            )

        if data is None:
            if input_path is None or not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            from oscura.loaders import load

            return load(input_path)
        return data

    def _report_detection(
        self,
        input_type: InputType,
        progress_callback: Callable[[ProgressInfo], None] | None,
    ) -> None:
        """Report input type detection progress.

        Args:
            input_type: Detected input type.
            progress_callback: Progress callback.
        """
        if progress_callback:
            progress_callback(
                ProgressInfo(
                    phase="detecting",
                    domain=None,
                    function=None,
                    percent=5.0,
                    message=f"Detected input type: {input_type.value}",
                    elapsed_seconds=time.time() - self._start_time,
                    estimated_remaining_seconds=None,
                )
            )

    def _plan_analysis_domains(
        self,
        input_type: InputType,
        progress_callback: Callable[[ProgressInfo], None] | None,
    ) -> list[AnalysisDomain]:
        """Determine enabled analysis domains.

        Args:
            input_type: Input data type.
            progress_callback: Progress callback.

        Returns:
            List of enabled domains.
        """
        applicable_domains = get_available_analyses(input_type)
        enabled_domains = [d for d in applicable_domains if self.config.is_domain_enabled(d)]

        if progress_callback:
            progress_callback(
                ProgressInfo(
                    phase="planning",
                    domain=None,
                    function=None,
                    percent=10.0,
                    message=f"Planning analysis across {len(enabled_domains)} domains",
                    elapsed_seconds=time.time() - self._start_time,
                    estimated_remaining_seconds=None,
                )
            )
        return enabled_domains

    def _execute_domains_parallel(
        self,
        enabled_domains: list[AnalysisDomain],
        data: Any,
        progress_callback: Callable[[ProgressInfo], None] | None,
    ) -> tuple[dict[AnalysisDomain, dict[str, Any]], list[AnalysisError]]:
        """Execute analysis domains in parallel.

        Args:
            enabled_domains: List of domains to analyze.
            data: Input data.
            progress_callback: Progress callback.

        Returns:
            Tuple of (results dict, errors list).
        """
        import concurrent.futures

        results: dict[AnalysisDomain, dict[str, Any]] = {}
        errors: list[AnalysisError] = []
        total_domains = len(enabled_domains)
        max_workers = min(self.config.max_parallel_workers, len(enabled_domains))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._execute_domain, domain, data): domain
                for domain in enabled_domains
            }

            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                domain = futures[future]
                domain_percent = 10.0 + (completed / total_domains) * 80.0

                if progress_callback:
                    progress_callback(
                        ProgressInfo(
                            phase="analyzing",
                            domain=domain,
                            function=None,
                            percent=domain_percent,
                            message=f"Completed domain: {domain.value}",
                            elapsed_seconds=time.time() - self._start_time,
                            estimated_remaining_seconds=None,
                        )
                    )

                self._handle_domain_future(future, domain, results, errors)

        return results, errors

    def _handle_domain_future(
        self,
        future: Any,
        domain: AnalysisDomain,
        results: dict[AnalysisDomain, dict[str, Any]],
        errors: list[AnalysisError],
    ) -> None:
        """Handle completed domain future.

        Args:
            future: Completed future.
            domain: Domain being analyzed.
            results: Results accumulator.
            errors: Errors accumulator.
        """
        import concurrent.futures

        try:
            timeout_seconds = self.config.timeout_per_analysis or 30.0
            domain_results, domain_errors = future.result(timeout=timeout_seconds * 10)
            if domain_results:
                results[domain] = domain_results
            errors.extend(domain_errors)
        except concurrent.futures.TimeoutError:
            logger.error(f"Domain {domain.value} exceeded timeout")
            errors.append(
                AnalysisError(
                    domain=domain,
                    function=f"{domain.value}.*",
                    error_type="TimeoutError",
                    error_message="Domain execution exceeded timeout",
                    traceback=None,
                    duration_ms=timeout_seconds * 10 * 1000,
                )
            )
        except Exception as e:
            logger.error(f"Domain {domain.value} failed: {e}")
            errors.append(
                AnalysisError(
                    domain=domain,
                    function=f"{domain.value}.*",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc(),
                    duration_ms=0.0,
                )
            )

    def _execute_domains_sequential(
        self,
        enabled_domains: list[AnalysisDomain],
        data: Any,
        progress_callback: Callable[[ProgressInfo], None] | None,
    ) -> tuple[dict[AnalysisDomain, dict[str, Any]], list[AnalysisError]]:
        """Execute analysis domains sequentially.

        Args:
            enabled_domains: List of domains to analyze.
            data: Input data.
            progress_callback: Progress callback.

        Returns:
            Tuple of (results dict, errors list).
        """
        results: dict[AnalysisDomain, dict[str, Any]] = {}
        errors: list[AnalysisError] = []
        total_domains = len(enabled_domains)

        for idx, domain in enumerate(enabled_domains):
            domain_percent = 10.0 + (idx / total_domains) * 80.0

            if progress_callback:
                progress_callback(
                    ProgressInfo(
                        phase="analyzing",
                        domain=domain,
                        function=None,
                        percent=domain_percent,
                        message=f"Analyzing domain: {domain.value}",
                        elapsed_seconds=time.time() - self._start_time,
                        estimated_remaining_seconds=None,
                    )
                )

            domain_results, domain_errors = self._execute_domain(domain, data)
            if domain_results:
                results[domain] = domain_results
            errors.extend(domain_errors)

        return results, errors

    def _calculate_statistics(
        self,
        results: dict[AnalysisDomain, dict[str, Any]],
        errors: list[AnalysisError],
        enabled_domains: list[AnalysisDomain],
        input_type: InputType,
        total_duration: float,
    ) -> dict[str, Any]:
        """Calculate execution statistics.

        Args:
            results: Analysis results.
            errors: Analysis errors.
            enabled_domains: List of enabled domains.
            input_type: Input data type.
            total_duration: Total execution time.

        Returns:
            Statistics dictionary.
        """
        total_analyses = sum(len(dr) for dr in results.values())
        successful_analyses = sum(
            1 for dr in results.values() for v in dr.values() if not isinstance(v, Exception)
        )
        failed_analyses = len(errors)

        return {
            "input_type": input_type.value,
            "total_domains": len(enabled_domains),
            "total_analyses": total_analyses,
            "successful_analyses": successful_analyses,
            "failed_analyses": failed_analyses,
            "success_rate": (successful_analyses / total_analyses * 100.0)
            if total_analyses > 0
            else 0.0,
            "duration_seconds": total_duration,
        }

    def run(
        self,
        input_path: Path | None = None,
        data: Any = None,
        progress_callback: Callable[[ProgressInfo], None] | None = None,
    ) -> dict[str, Any]:
        """Run comprehensive analysis on input data.

        Args:
            input_path: Path to input file (or None for in-memory data).
            data: Input data object (or None to load from input_path).
            progress_callback: Optional callback for progress updates.

        Returns:
            Dictionary with keys:
                - 'results': Dict mapping AnalysisDomain to analysis results
                - 'errors': List of AnalysisError objects
                - 'stats': Execution statistics dict

        Raises:
            ValueError: If neither input_path nor data provided.
            FileNotFoundError: If input_path doesn't exist.

        Example:
            >>> def progress(info: ProgressInfo):
            ...     print(f"{info.phase}: {info.percent:.1f}%")
            >>> result = engine.run(input_path=Path("data.wfm"), progress_callback=progress)
        """
        if input_path is None and data is None:
            raise ValueError("Must provide either input_path or data")

        self._initialize_engine(input_path)
        self._check_memory_and_adjust()
        data = self._load_input_data(input_path, data, progress_callback)
        input_type = self.detect_input_type(input_path, data)
        self._report_detection(input_type, progress_callback)
        enabled_domains = self._plan_analysis_domains(input_type, progress_callback)

        # Execute analyses (parallel or sequential)
        if self.config.parallel_domains and len(enabled_domains) > 1:
            results, errors = self._execute_domains_parallel(
                enabled_domains, data, progress_callback
            )
        else:
            results, errors = self._execute_domains_sequential(
                enabled_domains, data, progress_callback
            )

        # Finalize
        total_duration = time.time() - self._start_time
        if progress_callback:
            progress_callback(
                ProgressInfo(
                    phase="complete",
                    domain=None,
                    function=None,
                    percent=100.0,
                    message="Analysis complete",
                    elapsed_seconds=total_duration,
                    estimated_remaining_seconds=0.0,
                )
            )

        stats = self._calculate_statistics(
            results, errors, enabled_domains, input_type, total_duration
        )

        return {
            "results": results,
            "errors": errors,
            "stats": stats,
        }

    def _execute_domain(
        self, domain: AnalysisDomain, data: Any
    ) -> tuple[dict[str, Any], list[AnalysisError]]:
        """Execute all analyses for a specific domain.

        Args:
            domain: Analysis domain to execute.
            data: Input data object.

        Returns:
            Tuple of (results_dict, errors_list).
        """
        results: dict[str, Any] = {}
        errors: list[AnalysisError] = []

        # Preprocess data for specific domains
        data = self._preprocess_for_domain(domain, data)

        # Get domain capabilities
        cap = ANALYSIS_CAPABILITIES.get(domain, {})
        module_names = cap.get("modules", [])

        # Fallback to old single-module format
        if not module_names:
            single_module = cap.get("module", "")
            if single_module:
                module_names = [single_module]

        if not module_names:
            logger.debug(f"No modules configured for domain {domain.value}")
            return results, errors

        # Get domain-specific config
        domain_config = self.config.get_domain_config(domain)
        timeout = domain_config.timeout or self.config.timeout_per_analysis

        # Track executed functions to prevent duplicates
        executed_functions: set[str] = set()

        # Iterate through all modules for this domain
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                logger.warning(f"Failed to import module {module_name}: {e}")
                if not self.config.continue_on_error:
                    errors.append(
                        AnalysisError(
                            domain=domain,
                            function=module_name,
                            error_type="ImportError",
                            error_message=str(e),
                            traceback=traceback.format_exc(),
                            duration_ms=0.0,
                        )
                    )
                continue

            # Discover public functions in the module
            for func_name, func_obj in inspect.getmembers(module):
                # Skip private functions and non-functions
                if func_name.startswith("_") or not inspect.isfunction(func_obj):
                    continue

                # Skip functions not defined in this module (imported from elsewhere)
                if func_obj.__module__ != module_name:
                    continue

                # Skip if already executed (prevent duplicates)
                func_path = f"{module_name}.{func_name}"
                if func_path in executed_functions:
                    logger.debug(f"Skipping duplicate function: {func_path}")
                    continue
                executed_functions.add(func_path)

                # Execute the function
                try:
                    result = self._execute_function(module_name, func_name, data, timeout)
                    results[f"{module_name}.{func_name}"] = result
                except Exception as e:
                    error = AnalysisError(
                        domain=domain,
                        function=f"{module_name}.{func_name}",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        traceback=traceback.format_exc(),
                        duration_ms=0.0,
                    )
                    errors.append(error)

                    if not self.config.continue_on_error:
                        # Stop execution for this domain
                        return results, errors

        return results, errors

    def _preprocess_for_domain(self, domain: AnalysisDomain, data: Any) -> Any:
        """Preprocess data for domain-specific requirements.

        Some domains require specialized data structures. This method
        converts raw data into the appropriate format.

        Args:
            domain: Target analysis domain.
            data: Input data object.

        Returns:
            Preprocessed data suitable for the domain.
        """
        if domain == AnalysisDomain.EYE:
            # EYE domain requires an EyeDiagram object
            # Try to generate one from waveform data
            return self._preprocess_for_eye_domain(data)

        return data

    def _get_effective_sample_rate(self, data: Any, context: str = "general") -> float:
        """Get effective sample rate from data metadata or config defaults.

        Priority order:
        1. Data metadata (e.g., WaveformTrace.metadata.sample_rate)
        2. AnalysisConfig.default_sample_rate
        3. Context-appropriate default constant

        Args:
            data: Input data object (may have .metadata.sample_rate).
            context: Analysis context for selecting appropriate default.
                Options: "general" (1 MHz), "highspeed" (1 GHz), "binary" (1 Hz).

        Returns:
            Effective sample rate in Hz.

        Note:
            This method logs a debug message when falling back to defaults,
            as sample rate should ideally be provided in the data metadata
            for accurate time-domain analysis.
        """
        # Try to extract from data metadata
        data_sample_rate = None
        if hasattr(data, "metadata") and hasattr(data.metadata, "sample_rate"):
            data_sample_rate = data.metadata.sample_rate
            if data_sample_rate is not None and data_sample_rate > 0:
                return float(data_sample_rate)

        # Use config's get_effective_sample_rate method
        effective_rate = self.config.get_effective_sample_rate(
            data_sample_rate=data_sample_rate,
            context=context,
        )

        # Log when using defaults (indicates missing metadata)
        logger.debug(
            f"Using default sample rate {effective_rate:.2e} Hz (context: {context}). "
            f"For accurate analysis, provide sample_rate in data metadata."
        )

        return effective_rate

    def _preprocess_for_eye_domain(self, data: Any) -> Any:
        """Preprocess data for eye diagram analysis.

        Attempts to generate an EyeDiagram from waveform data using
        automatic unit interval detection via FFT-based period detection
        with fallback to zero-crossing analysis.

        Args:
            data: Input waveform data.

        Returns:
            EyeDiagram object if successful, original data otherwise.
        """
        # Check if already an EyeDiagram
        if hasattr(data, "samples_per_ui") and hasattr(data, "time_axis"):
            return data

        # Try to extract waveform data
        if hasattr(data, "data") and hasattr(data, "metadata"):
            # WaveformTrace
            raw_data = data.data
            sample_rate = getattr(data.metadata, "sample_rate", None)
        elif isinstance(data, np.ndarray):
            raw_data = data
            sample_rate = None
        else:
            # Can't preprocess, return as-is
            return data

        if raw_data is None or len(raw_data) == 0:
            return data

        try:
            from oscura.analyzers.eye.diagram import generate_eye
            from oscura.core.types import TraceMetadata, WaveformTrace

            # Get effective sample rate using config-aware method
            # Use "highspeed" context for eye diagram (typically high-speed serial)
            if sample_rate is None or sample_rate <= 0:
                sample_rate = self._get_effective_sample_rate(data, context="highspeed")

            # Estimate unit interval using FFT-based period detection
            unit_interval = self._detect_unit_interval_fft(raw_data, sample_rate)

            # If FFT detection fails, try zero-crossing analysis
            if unit_interval is None:
                unit_interval = self._detect_unit_interval_zero_crossing(raw_data, sample_rate)

            # If both methods fail, use default fallback
            if unit_interval is None:
                # Fallback: assume 100 UI in the data
                unit_interval = len(raw_data) / sample_rate / 100
                logger.debug("Using default unit interval fallback (100 UI in data)")

            # Ensure unit interval is reasonable
            min_ui = 10 / sample_rate  # At least 10 samples per UI
            max_ui = len(raw_data) / sample_rate / 10  # At least 10 UI in data
            unit_interval = np.clip(unit_interval, min_ui, max_ui)

            # Create a WaveformTrace if we only have raw data
            if not hasattr(data, "data"):
                metadata = TraceMetadata(sample_rate=sample_rate)
                trace = WaveformTrace(data=raw_data.astype(np.float64), metadata=metadata)
            else:
                trace = data

            # Generate eye diagram
            eye_diagram = generate_eye(
                trace=trace,
                unit_interval=unit_interval,
                n_ui=2,
                generate_histogram=True,
            )

            logger.debug(
                f"Generated eye diagram: {eye_diagram.n_traces} traces, "
                f"{eye_diagram.samples_per_ui} samples/UI"
            )
            return eye_diagram

        except Exception as e:
            logger.debug(f"Could not generate eye diagram: {e}")
            # Return original data if preprocessing fails
            return data

    def _detect_unit_interval_fft(
        self, raw_data: np.ndarray[Any, Any], sample_rate: float
    ) -> float | None:
        """Detect unit interval using FFT-based period detection.

        Computes the FFT of the waveform, finds the dominant frequency
        (excluding DC), and calculates the unit interval for NRZ data.

        Args:
            raw_data: Input waveform samples.
            sample_rate: Sample rate in Hz.

        Returns:
            Estimated unit interval in seconds, or None if detection fails.
        """
        try:
            # Remove DC component
            data_ac = raw_data - np.mean(raw_data)

            # Compute FFT
            fft_result = np.fft.rfft(data_ac)
            fft_freqs = np.fft.rfftfreq(len(data_ac), d=1.0 / sample_rate)

            # Get magnitude spectrum (exclude DC bin at index 0)
            magnitude = np.abs(fft_result[1:])
            freqs = fft_freqs[1:]

            if len(magnitude) == 0:
                return None

            # Find dominant frequency (peak in magnitude spectrum)
            peak_idx = np.argmax(magnitude)
            dominant_freq = freqs[peak_idx]

            # For NRZ data, unit interval = 1 / (2 * dominant_freq)
            # For periodic signals like sine waves, unit interval = 1 / dominant_freq
            # We'll use the period as the unit interval for general signals
            if dominant_freq > 0:
                unit_interval = float(1.0 / dominant_freq)

                # Sanity check: dominant frequency should be reasonable
                min_freq = sample_rate / len(raw_data)  # At least one full cycle
                max_freq = sample_rate / 20  # At least 20 samples per cycle

                if min_freq <= dominant_freq <= max_freq:
                    logger.debug(
                        f"FFT detected dominant frequency: {dominant_freq:.2f} Hz, "
                        f"unit interval: {unit_interval * 1e6:.3f} us"
                    )
                    return unit_interval
                else:
                    logger.debug(
                        f"FFT dominant frequency {dominant_freq:.2f} Hz out of range "
                        f"[{min_freq:.2f}, {max_freq:.2f}] Hz"
                    )
                    return None

            return None

        except Exception as e:
            logger.debug(f"FFT-based unit interval detection failed: {e}")
            return None

    def _detect_unit_interval_zero_crossing(
        self, raw_data: np.ndarray[Any, Any], sample_rate: float
    ) -> float | None:
        """Detect unit interval using zero-crossing analysis.

        Estimates the signal period from the average interval between
        zero crossings.

        Args:
            raw_data: Input waveform samples.
            sample_rate: Sample rate in Hz.

        Returns:
            Estimated unit interval in seconds, or None if detection fails.
        """
        try:
            # Find zero crossings
            zero_crossings = np.where(np.diff(np.sign(raw_data - np.mean(raw_data))))[0]

            if len(zero_crossings) > 10:
                # Estimate period from average crossing interval
                avg_half_period = float(np.mean(np.diff(zero_crossings))) / sample_rate
                unit_interval = avg_half_period * 2  # Full period

                logger.debug(
                    f"Zero-crossing detected unit interval: {unit_interval * 1e6:.3f} us "
                    f"({len(zero_crossings)} crossings)"
                )
                return unit_interval
            else:
                logger.debug(f"Insufficient zero crossings ({len(zero_crossings)}) for detection")
                return None

        except Exception as e:
            logger.debug(f"Zero-crossing unit interval detection failed: {e}")
            return None

    def _execute_function(
        self, module_name: str, func_name: str, data: Any, timeout: float | None
    ) -> Any:
        """Execute a single analysis function with quality scoring.

        Args:
            module_name: Name of the module containing the function.
            func_name: Name of the function to execute.
            data: Input data object.
            timeout: Timeout in seconds (None for no timeout).

        Returns:
            Analysis result with optional quality score attached.

        Raises:
            ValueError: If function is non-inferrable or invalid.
        """
        # Check if function is in non-inferrable skip list
        func_path = f"{module_name}.{func_name}"
        if func_path in NON_INFERRABLE_FUNCTIONS:
            logger.debug(f"Skipping non-inferrable function: {func_path}")
            raise ValueError(
                f"Function {func_path} requires context-specific parameters that cannot be auto-detected"
            )

        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        # Prepare function arguments using ArgumentPreparer
        if self._arg_preparer is None:
            raise RuntimeError("ArgumentPreparer not initialized - call run() first")

        args, kwargs = self._arg_preparer.prepare_arguments(func, data)

        if args is None:
            # Function not applicable to this data type
            raise ValueError(f"Function {func_name} not applicable to data type")

        start_time = time.time()

        # Execute with timeout if specified
        if timeout is not None:
            # Note: Python doesn't have built-in function timeout without threads/processes
            # For simplicity, we'll just execute directly and check elapsed time afterward
            # A production implementation would use threading.Timer or signal.alarm
            result = func(*args, **kwargs)

            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(
                    f"Function {module_name}.{func_name} exceeded timeout "
                    f"({elapsed:.2f}s > {timeout:.2f}s)"
                )
        else:
            result = func(*args, **kwargs)

        # Consume generators to avoid serialization issues
        if isinstance(result, types.GeneratorType):
            try:
                result = list(result)
                logger.debug(f"Consumed generator from {module_name}.{func_name}")
            except Exception as e:
                logger.warning(f"Failed to consume generator from {module_name}.{func_name}: {e}")
                result = f"<generator error: {type(e).__name__}>"

        # Add quality scoring if enabled in config
        if self.config.enable_quality_scoring:
            result = self._add_quality_score(result, func_path, data)

        return result

    def _add_quality_score(self, result: Any, method_name: str, data: Any) -> Any:
        """Add quality score to analysis result.

        Args:
            result: Analysis result to score.
            method_name: Name of the analysis method.
            data: Input data object.

        Returns:
            Result with quality score attached (if applicable).
        """
        try:
            from oscura.validation.quality import score_analysis_result

            # Extract raw data array for quality assessment
            if hasattr(data, "data"):
                raw_data = data.data
            elif isinstance(data, np.ndarray):
                raw_data = data
            else:
                # Can't assess quality for non-array data
                return result

            # Score the result
            quality_score = score_analysis_result(
                result=result,
                method_name=method_name,
                data=raw_data,
            )

            # Attach quality score to result if it's a dict
            if isinstance(result, dict):
                result["_quality_score"] = quality_score.to_dict()
            # For other types, wrap in dict
            elif result is not None:
                return {
                    "value": result,
                    "_quality_score": quality_score.to_dict(),
                }

        except Exception as e:
            logger.debug(f"Failed to add quality score: {e}")

        return result


__all__ = [
    "AnalysisEngine",
]
