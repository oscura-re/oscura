"""Argument preparation for analysis functions.

This module handles automatic argument detection and preparation for analysis functions,
including data type detection, parameter inference, and intelligent defaults.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ArgumentPreparer:
    """Prepares arguments for analysis functions automatically.

    This class examines function signatures and prepares appropriate arguments
    from input data, handling type conversions, parameter detection, and
    intelligent defaults.
    """

    def __init__(self, input_path: Path | None = None, default_sample_rate: float = 1e6):
        """Initialize argument preparer.

        Args:
            input_path: Path to input file (used for filename-based detection).
            default_sample_rate: Default sample rate when not available in data.
        """
        self._input_path = input_path
        self._default_sample_rate = default_sample_rate

    def prepare_arguments(
        self, func: Callable[..., Any], data: Any
    ) -> tuple[list[Any] | None, dict[str, Any]]:
        """Prepare arguments for an analysis function.

        Args:
            func: Function to prepare arguments for.
            data: Input data object.

        Returns:
            Tuple of (args_list, kwargs_dict), or (None, {}) if not applicable.
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if not params:
            return [], {}

        first_param = params[0]

        # Handle specialized data types
        if self._is_eye_diagram(data):
            return self._handle_eye_diagram(data, first_param, sig)

        if self._is_sparam_data(data):
            return self._handle_sparam_data(data, first_param, sig)

        # Check type annotation for special handling
        first_param_info = sig.parameters.get(first_param)
        param_annotation = first_param_info.annotation if first_param_info else None
        annotation_str = str(param_annotation) if param_annotation else ""

        # Handle packet data
        if "PacketInfo" in annotation_str or first_param == "packets":
            return self._handle_packet_data(data)

        # Extract raw data and determine if trace
        is_trace, raw_data, sample_rate = self._extract_raw_data(data)

        if raw_data is None or (hasattr(raw_data, "__len__") and len(raw_data) == 0):
            return None, {}

        # Build kwargs with auto-detected parameters
        kwargs = self._build_kwargs(params, sig, raw_data, sample_rate)

        # Handle trace wrapper creation if needed
        if not is_trace and self._needs_trace_wrapper(first_param, annotation_str):
            data, is_trace = self._create_trace_wrapper(raw_data, sample_rate, data)
            if data is None:
                return None, {}

        # Return appropriate arguments based on first parameter
        return self._build_args(first_param, annotation_str, data, is_trace, raw_data, kwargs, sig)

    def _is_eye_diagram(self, data: Any) -> bool:
        """Check if data is an EyeDiagram object."""
        return hasattr(data, "samples_per_ui") and hasattr(data, "time_axis")

    def _is_sparam_data(self, data: Any) -> bool:
        """Check if data is S-parameter data."""
        return hasattr(data, "s_matrix") and hasattr(data, "frequencies")

    def _handle_eye_diagram(
        self, data: Any, first_param: str, sig: inspect.Signature
    ) -> tuple[list[Any] | None, dict[str, Any]]:
        """Handle EyeDiagram data."""
        if first_param == "eye" or "EyeDiagram" in str(sig.parameters.get(first_param, "")):
            return [data], {}
        return None, {}

    def _handle_sparam_data(
        self, data: Any, first_param: str, sig: inspect.Signature
    ) -> tuple[list[Any] | None, dict[str, Any]]:
        """Handle S-parameter data."""
        if first_param in ("s_params", "s_param", "s_data", "sparams"):
            return [data], {}
        if "SParameter" in str(sig.parameters.get(first_param, "")):
            return [data], {}
        return None, {}

    def _handle_packet_data(self, data: Any) -> tuple[list[Any] | None, dict[str, Any]]:
        """Handle packet data - convert to PacketInfo objects if needed."""
        if isinstance(data, list):
            if data and hasattr(data[0], "timestamp"):
                return [data], {}
            elif data and isinstance(data[0], dict):
                try:
                    from oscura.analyzers.packet.metrics import PacketInfo

                    packets = [
                        PacketInfo(
                            timestamp=p.get("timestamp", 0.0),
                            size=p.get("size", 0),
                            sequence=p.get("sequence"),
                        )
                        for p in data
                    ]
                    return [packets], {}
                except Exception as e:
                    logger.debug(f"Failed to convert to PacketInfo: {e}")
                    return None, {}
        return None, {}

    def _extract_raw_data(self, data: Any) -> tuple[bool, Any, float]:
        """Extract raw data array, determine if trace, and get sample rate.

        Returns:
            Tuple of (is_trace, raw_data, sample_rate).
        """
        is_trace = hasattr(data, "data") and hasattr(data, "metadata")

        if is_trace:
            raw_data = data.data
            sample_rate = self._get_sample_rate_from_data(data)
        elif isinstance(data, np.ndarray):
            raw_data = data
            sample_rate = self._default_sample_rate
        elif isinstance(data, bytes | bytearray):
            raw_data = np.frombuffer(data, dtype=np.uint8)
            sample_rate = 1.0  # Binary data context
        else:
            # Try to convert to array
            try:
                raw_data = np.array(data) if hasattr(data, "__iter__") else None
            except (ValueError, TypeError):
                raw_data = None
            sample_rate = self._default_sample_rate

        return is_trace, raw_data, sample_rate

    def _get_sample_rate_from_data(self, data: Any) -> float:
        """Get sample rate from data metadata or use default."""
        if hasattr(data, "metadata") and hasattr(data.metadata, "sample_rate"):
            sample_rate = data.metadata.sample_rate
            if sample_rate is not None and sample_rate > 0:
                return float(sample_rate)
        return self._default_sample_rate

    def _build_kwargs(
        self,
        params: list[str],
        sig: inspect.Signature,
        raw_data: Any,
        sample_rate: float,
    ) -> dict[str, Any]:
        """Build kwargs dictionary with auto-detected parameters."""
        kwargs: dict[str, Any] = {}

        # Add sample rate parameters
        kwargs.update(self._add_sample_rate_params(params, sample_rate))

        # Add digital domain parameters
        kwargs.update(self._add_digital_params(params, sig, raw_data))

        # Add frequency domain parameters
        kwargs.update(self._add_frequency_params(params, sig, raw_data, sample_rate))

        # Add noise/threshold parameters
        kwargs.update(self._add_noise_params(params, sig, raw_data))

        # Add window/width parameters
        kwargs.update(self._add_window_params(params, sig, raw_data, sample_rate))

        return kwargs

    def _add_sample_rate_params(self, params: list[str], sample_rate: float) -> dict[str, Any]:
        """Add sample rate related parameters."""
        kwargs = {}
        if "sample_rate" in params:
            kwargs["sample_rate"] = sample_rate
        if "fs" in params:
            kwargs["fs"] = sample_rate
        if "rate" in params:
            kwargs["rate"] = sample_rate
        return kwargs

    def _add_digital_params(
        self, params: list[str], sig: inspect.Signature, raw_data: Any
    ) -> dict[str, Any]:
        """Add digital domain parameters (baud_rate, logic_family)."""
        kwargs: dict[str, Any] = {}

        # Baud rate detection
        if "baud_rate" in params:
            param_info = sig.parameters.get("baud_rate")
            has_default = (
                param_info is not None and param_info.default is not inspect.Parameter.empty
            )
            if not has_default or (param_info and param_info.default is None):
                detected_baud = self._detect_baud_rate_from_filename()
                if detected_baud is not None:
                    kwargs["baud_rate"] = detected_baud

        # Logic family detection
        if "logic_family" in params:
            param_info = sig.parameters.get("logic_family")
            has_default = (
                param_info is not None and param_info.default is not inspect.Parameter.empty
            )
            if not has_default or (param_info and param_info.default in (None, "auto")):
                try:
                    detected_family: Any = self._detect_logic_family(raw_data)
                    kwargs["logic_family"] = detected_family
                except Exception as e:
                    logger.debug(f"Could not auto-detect logic family: {e}")

        return kwargs

    def _add_frequency_params(
        self, params: list[str], sig: inspect.Signature, raw_data: Any, sample_rate: float
    ) -> dict[str, Any]:
        """Add frequency range parameters."""
        kwargs = {}

        if "freq_min" in params or "freq_max" in params:
            try:
                freq_range = self._detect_frequency_range(raw_data, sample_rate)
                if freq_range is not None:
                    min_freq, max_freq = freq_range

                    if "freq_min" in params:
                        if self._param_needs_value(sig, "freq_min"):
                            kwargs["freq_min"] = min_freq

                    if "freq_max" in params:
                        if self._param_needs_value(sig, "freq_max"):
                            kwargs["freq_max"] = max_freq
            except Exception as e:
                logger.debug(f"Could not auto-detect frequency range: {e}")

        return kwargs

    def _add_noise_params(
        self, params: list[str], sig: inspect.Signature, raw_data: Any
    ) -> dict[str, Any]:
        """Add noise/threshold parameters."""
        kwargs = {}

        if "noise_threshold" in params or "snr_threshold" in params:
            try:
                noise_floor = self._detect_noise_floor(raw_data)
                if noise_floor is not None:
                    if "noise_threshold" in params and self._param_needs_value(
                        sig, "noise_threshold"
                    ):
                        kwargs["noise_threshold"] = noise_floor * 3.0

                    if "snr_threshold" in params and self._param_needs_value(sig, "snr_threshold"):
                        signal_rms = float(np.std(raw_data))
                        if noise_floor > 0:
                            detected_snr = signal_rms / noise_floor
                            kwargs["snr_threshold"] = detected_snr / 2.0
            except Exception as e:
                logger.debug(f"Could not auto-detect noise floor: {e}")

        # Protocol hints for baud rate
        if "baud_rate" in params and "baud_rate" not in kwargs:
            try:
                protocol_hints = self._detect_protocol_hints(raw_data, self._default_sample_rate)
                if "detected_baud" in protocol_hints:
                    if self._param_needs_value(sig, "baud_rate"):
                        kwargs["baud_rate"] = protocol_hints["detected_baud"]
                        logger.debug(
                            f"Using protocol-detected baud rate: {protocol_hints['detected_baud']} bps"
                        )
            except Exception as e:
                logger.debug(f"Could not use protocol hints for baud detection: {e}")

        return kwargs

    def _add_window_params(
        self, params: list[str], sig: inspect.Signature, raw_data: Any, sample_rate: float
    ) -> dict[str, Any]:
        """Add window size and width parameters."""
        kwargs: dict[str, Any] = {}
        data_length = len(raw_data) if hasattr(raw_data, "__len__") else 0

        # Add window_size if needed
        kwargs.update(self._add_window_size_param(params, sig, data_length, kwargs))

        # Add min_width if needed
        kwargs.update(self._add_min_width_param(params, sig, sample_rate, kwargs))

        # Add max_width if needed
        kwargs.update(self._add_max_width_param(params, sig, data_length, sample_rate, kwargs))

        # Add threshold if needed
        kwargs.update(self._add_threshold_param(params, sig, raw_data, kwargs))

        # Add window_duration if needed
        kwargs.update(
            self._add_window_duration_param(params, sig, data_length, sample_rate, kwargs)
        )

        return kwargs

    def _add_window_size_param(
        self, params: list[str], sig: inspect.Signature, data_length: int, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Add window_size parameter."""
        result = {}
        if "window_size" in params:
            if self._param_needs_value(sig, "window_size") and "window_size" not in kwargs:
                window_size: Any = max(10, data_length // 10)
                result["window_size"] = window_size
                logger.debug(f"Using auto-detected window_size: {window_size}")
        return result

    def _add_min_width_param(
        self, params: list[str], sig: inspect.Signature, sample_rate: float, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Add min_width parameter."""
        result = {}
        if "min_width" in params:
            if self._param_needs_value(sig, "min_width") and "min_width" not in kwargs:
                min_width: Any = max(1e-9, 10.0 / sample_rate)
                result["min_width"] = min_width
                logger.debug(f"Using auto-detected min_width: {min_width:.2e}s")
        return result

    def _add_max_width_param(
        self,
        params: list[str],
        sig: inspect.Signature,
        data_length: int,
        sample_rate: float,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Add max_width parameter."""
        result = {}
        if "max_width" in params:
            if self._param_needs_value(sig, "max_width") and "max_width" not in kwargs:
                total_duration = data_length / sample_rate if data_length > 0 else 1e-3
                max_width: Any = min(1e-3, total_duration)
                result["max_width"] = max_width
                logger.debug(f"Using auto-detected max_width: {max_width:.2e}s")
        return result

    def _add_threshold_param(
        self, params: list[str], sig: inspect.Signature, raw_data: Any, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Add threshold parameter."""
        result = {}
        if "threshold" in params and "threshold" not in kwargs:
            param_info = sig.parameters.get("threshold")
            has_default = (
                param_info is not None and param_info.default is not inspect.Parameter.empty
            )
            if not has_default or (param_info and param_info.default in (None, "auto")):
                try:
                    if isinstance(raw_data, np.ndarray) and raw_data.size > 0:
                        threshold: Any = float(np.median(raw_data))
                        result["threshold"] = threshold
                        logger.debug(f"Using auto-detected threshold: {threshold:.3f}")
                except Exception as e:
                    logger.debug(f"Could not auto-detect threshold: {e}")
        return result

    def _add_window_duration_param(
        self,
        params: list[str],
        sig: inspect.Signature,
        data_length: int,
        sample_rate: float,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Add window_duration parameter."""
        result = {}
        if "window_duration" in params:
            if self._param_needs_value(sig, "window_duration") and "window_duration" not in kwargs:
                total_duration = data_length / sample_rate if data_length > 0 else 1.0
                window_duration: Any = min(1.0, total_duration / 10.0)
                result["window_duration"] = window_duration
                logger.debug(f"Using auto-detected window_duration: {window_duration:.3f}s")
        return result

    def _param_needs_value(self, sig: inspect.Signature, param_name: str) -> bool:
        """Check if parameter needs a value (no default or default is None)."""
        param_info = sig.parameters.get(param_name)
        if param_info is None:
            return False
        has_default = param_info.default is not inspect.Parameter.empty
        return not has_default or param_info.default is None

    def _needs_trace_wrapper(self, first_param: str, annotation_str: str) -> bool:
        """Check if function needs trace wrapper."""
        return (
            "Trace" in annotation_str or "WaveformTrace" in annotation_str or first_param == "trace"
        )

    def _create_trace_wrapper(
        self, raw_data: Any, sample_rate: float, original_data: Any
    ) -> tuple[Any, bool]:
        """Create WaveformTrace wrapper for raw array data."""
        try:
            trace_data = np.asarray(raw_data) if isinstance(raw_data, memoryview) else raw_data
            metadata = TraceMetadata(sample_rate=sample_rate)
            data = WaveformTrace(data=trace_data, metadata=metadata)
            logger.debug("Created WaveformTrace wrapper for raw array data")
            return data, True
        except Exception as e:
            logger.debug(f"Could not create trace wrapper: {e}")
            return None, False

    def _build_args(
        self,
        first_param: str,
        annotation_str: str,
        data: Any,
        is_trace: bool,
        raw_data: Any,
        kwargs: dict[str, Any],
        sig: inspect.Signature,
    ) -> tuple[list[Any] | None, dict[str, Any]]:
        """Build the final args list based on first parameter."""
        # If function expects WaveformTrace and we have a trace
        if is_trace and (
            "Trace" in annotation_str or "WaveformTrace" in annotation_str or first_param == "trace"
        ):
            return [data], kwargs

        # Common data parameter names
        if first_param in ("data", "signal", "x", "samples", "waveform"):
            return [raw_data], kwargs

        # Trace expected but not available
        if first_param == "trace" and not is_trace:
            return None, {}

        # Edge timestamps
        if first_param == "edges":
            return self._extract_edges(data, is_trace, kwargs)

        # Period measurements
        if first_param == "periods":
            return self._extract_periods(data, is_trace, kwargs)

        # Bytes data
        if first_param in ("stream", "data") and "bytes" in annotation_str:
            return self._convert_to_bytes(data, raw_data, kwargs)

        if first_param == "bytes" or (first_param == "data" and "bytes" in str(sig)):
            return self._convert_to_bytes(data, raw_data, kwargs)

        # Default: pass raw data
        return [raw_data], kwargs

    def _extract_edges(
        self, data: Any, is_trace: bool, kwargs: dict[str, Any]
    ) -> tuple[list[Any] | None, dict[str, Any]]:
        """Extract edge timestamps from data."""
        try:
            from oscura.analyzers.digital import detect_edges

            if is_trace:
                edges = detect_edges(data)
                edge_times = edges.tolist() if len(edges) > 0 else []
                if len(edge_times) < 3:
                    return None, {}
                return [edge_times], kwargs
        except Exception:
            pass
        return None, {}

    def _extract_periods(
        self, data: Any, is_trace: bool, kwargs: dict[str, Any]
    ) -> tuple[list[Any] | None, dict[str, Any]]:
        """Extract period measurements from data."""
        try:
            if is_trace:
                from oscura.analyzers.waveform.measurements import period

                periods_result = period(data, return_all=True)
                if isinstance(periods_result, np.ndarray) and len(periods_result) >= 3:
                    return [periods_result], kwargs
        except Exception as e:
            logger.debug(f"Could not compute periods: {e}")
        return None, {}

    def _convert_to_bytes(
        self, data: Any, raw_data: Any, kwargs: dict[str, Any]
    ) -> tuple[list[Any] | None, dict[str, Any]]:
        """Convert data to bytes."""
        if isinstance(data, bytes | bytearray):
            return [data], kwargs
        elif isinstance(raw_data, np.ndarray) or hasattr(raw_data, "astype"):
            return [raw_data.astype(np.uint8).tobytes()], kwargs
        return None, {}

    # Detection methods

    def _detect_baud_rate_from_filename(self) -> float | None:
        """Extract baud rate from filename patterns."""
        if self._input_path is None:
            return None

        import re

        patterns = [
            r"(\d+(?:\.\d+)?)[_\s]*[Mm]?baud",
            r"(\d+(?:\.\d+)?)[_\s]*bps",
            r"baud[_-]?(\d+)",
        ]
        filename = self._input_path.stem.lower()

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                matched_text = filename[match.start() : match.end()].lower()
                if "m" in matched_text and "baud" in matched_text:
                    value *= 1_000_000
                logger.debug(
                    f"Detected baud rate from filename '{self._input_path.name}': {value} bps"
                )
                return value

        return None

    def _detect_logic_family(self, data: np.ndarray[Any, Any]) -> str:
        """Detect logic family from voltage levels."""
        vmax = float(np.max(data))
        vmin = float(np.min(data))
        voltage_swing = vmax - vmin

        if voltage_swing < 1.0:
            logic_family = "LVDS"
        elif voltage_swing < 2.0:
            logic_family = "LVCMOS18"
        elif voltage_swing < 3.0:
            logic_family = "LVCMOS25"
        elif voltage_swing < 4.0:
            logic_family = "LVCMOS33"
        else:
            logic_family = "TTL"

        logger.debug(
            f"Detected logic family from voltage swing {voltage_swing:.2f}V: {logic_family}"
        )
        return logic_family

    def _detect_frequency_range(
        self, data: np.ndarray[Any, Any], sample_rate: float
    ) -> tuple[float, float] | None:
        """Detect dominant frequency range from FFT analysis."""
        try:
            fft_result = np.fft.rfft(data - np.mean(data))
            freqs = np.fft.rfftfreq(len(data), d=1.0 / sample_rate)
            magnitude = np.abs(fft_result)

            threshold = 0.1 * np.max(magnitude)
            significant = freqs[magnitude > threshold]

            if len(significant) > 0:
                min_freq = float(np.min(significant))
                max_freq = float(np.max(significant))
                logger.debug(f"Detected frequency range: {min_freq:.2f} Hz - {max_freq:.2f} Hz")
                return (min_freq, max_freq)
            return None
        except Exception as e:
            logger.debug(f"Frequency range detection failed: {e}")
            return None

    def _detect_noise_floor(self, data: np.ndarray[Any, Any]) -> float | None:
        """Estimate noise floor using median absolute deviation."""
        try:
            try:
                from scipy import stats

                mad = stats.median_abs_deviation(data, scale="normal")
                logger.debug(f"Detected noise floor (scipy MAD): {mad:.6f}")
                return float(mad)
            except ImportError:
                median = np.median(data)
                mad = np.median(np.abs(data - median)) * 1.4826
                logger.debug(f"Detected noise floor (numpy MAD): {mad:.6f}")
                return float(mad)
        except Exception as e:
            logger.debug(f"Noise floor detection failed: {e}")
            return None

    def _detect_protocol_hints(
        self, data: np.ndarray[Any, Any], sample_rate: float
    ) -> dict[str, Any]:
        """Detect hints about potential protocols in the signal."""
        hints: dict[str, Any] = {}
        try:
            zero_crossings = np.where(np.diff(np.sign(data - np.mean(data))))[0]
            if len(zero_crossings) > 10:
                intervals = np.diff(zero_crossings) / sample_rate
                avg_interval = float(np.median(intervals))

                common_bauds = [300, 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200]
                for baud in common_bauds:
                    expected_interval = 1.0 / baud
                    if 0.8 < avg_interval / expected_interval < 1.2:
                        hints["detected_baud"] = baud
                        logger.debug(f"Protocol hint: detected baud rate {baud} bps")
                        break

            if len(zero_crossings) > 20:
                interval_std = float(np.std(np.diff(zero_crossings)))
                regularity = "high" if interval_std < 2 else "medium" if interval_std < 5 else "low"
                hints["clock_regularity"] = regularity
                logger.debug(f"Protocol hint: clock regularity {regularity}")

        except Exception as e:
            logger.debug(f"Protocol hints detection failed: {e}")

        return hints
