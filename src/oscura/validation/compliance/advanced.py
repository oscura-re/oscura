"""Advanced EMC compliance features.

This module provides advanced compliance testing capabilities including
limit interpolation, compliance test execution, and quasi-peak detection.


References:
    CISPR 16-1-1: Measuring Apparatus
    FCC Part 15: Unintentional Radiators
    EN 55032: EMC Standard for Multimedia Equipment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.validation.compliance.masks import LimitMask

logger = logging.getLogger(__name__)

__all__ = [
    "ComplianceTestConfig",
    "ComplianceTestRunner",
    "ComplianceTestSuite",
    "InterpolationMethod",
    "LimitInterpolator",
    "QPDetectorBand",
    "QuasiPeakDetector",
    "interpolate_limit",
    "run_compliance_suite",
]


# =============================================================================
# =============================================================================


class InterpolationMethod(Enum):
    """Interpolation methods for limit masks.

    References:
        COMP-005: Limit Interpolation
    """

    LINEAR = "linear"  # Linear interpolation
    LOG_LINEAR = "log-linear"  # Log-linear (dB) interpolation
    CUBIC = "cubic"  # Cubic spline
    STEP = "step"  # Step function (no interpolation)


class LimitInterpolator:
    """Limit mask interpolator.

    Provides accurate interpolation of EMC limits between
    defined frequency points.

    Example:
        >>> from oscura.validation.compliance import load_limit_mask
        >>> mask = load_limit_mask('FCC_Part15_ClassB')
        >>> interp = LimitInterpolator(mask)
        >>> limit_at_100mhz = interp.interpolate(100e6)

    References:
        COMP-005: Limit Interpolation
    """

    def __init__(
        self,
        mask: LimitMask,
        method: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
        extrapolate: bool = False,
    ) -> None:
        """Initialize interpolator.

        Args:
            mask: Limit mask to interpolate
            method: Interpolation method
            extrapolate: Allow extrapolation beyond mask range
        """
        self._mask = mask
        self._method = method
        self._extrapolate = extrapolate

        # Precompute log frequencies for log-linear interpolation
        self._log_freq = np.log10(mask.frequency)
        self._log_limit = mask.limit  # Already in dB

    def interpolate(
        self,
        frequency: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Interpolate limit at given frequency/frequencies.

        Args:
            frequency: Frequency or array of frequencies in Hz

        Returns:
            Interpolated limit value(s)

        Raises:
            ValueError: If frequency outside range and extrapolation disabled
        """
        freq_array = np.atleast_1d(np.asarray(frequency, dtype=np.float64))

        # Validate positive frequencies first
        if np.any(freq_array <= 0):
            raise ValueError("Frequency must be positive")

        # Check range
        f_min, f_max = self._mask.frequency_range
        if not self._extrapolate:
            if np.any(freq_array < f_min) or np.any(freq_array > f_max):
                out_of_range = freq_array[(freq_array < f_min) | (freq_array > f_max)]
                raise ValueError(
                    f"Frequency {out_of_range[0]:.2e} Hz outside mask range "
                    f"[{f_min:.2e}, {f_max:.2e}] Hz. "
                    f"Set extrapolate=True to allow extrapolation."
                )

        if self._method == InterpolationMethod.LINEAR:
            return self._interp_linear(freq_array)
        elif self._method == InterpolationMethod.LOG_LINEAR:
            return self._interp_log_linear(freq_array)
        elif self._method == InterpolationMethod.CUBIC:
            return self._interp_cubic(freq_array)
        else:  # STEP
            return self._interp_step(freq_array)

    def _interp_linear(self, freq: NDArray[np.float64]) -> NDArray[np.float64]:
        """Linear interpolation."""
        return np.interp(freq, self._mask.frequency, self._mask.limit)

    def _interp_log_linear(self, freq: NDArray[np.float64]) -> NDArray[np.float64]:
        """Log-linear interpolation (linear in log-frequency space)."""
        log_freq = np.log10(freq)
        return np.interp(log_freq, self._log_freq, self._log_limit)

    def _interp_cubic(self, freq: NDArray[np.float64]) -> NDArray[np.float64]:
        """Cubic spline interpolation."""
        from scipy.interpolate import CubicSpline

        # Use log-frequency for better behavior
        log_freq = np.log10(freq)
        spline = CubicSpline(self._log_freq, self._log_limit, extrapolate=self._extrapolate)
        result: NDArray[np.float64] = spline(log_freq)
        return result

    def _interp_step(self, freq: NDArray[np.float64]) -> NDArray[np.float64]:
        """Step function (nearest lower point)."""
        result = np.zeros_like(freq)
        for i, f in enumerate(freq):
            idx = np.searchsorted(self._mask.frequency, f, side="right") - 1
            idx = max(0, min(idx, len(self._mask.limit) - 1))
            result[i] = self._mask.limit[idx]
        return result

    def get_limit_at(
        self,
        frequency: float,
        warn_on_extrapolation: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        """Get limit at specific frequency with metadata.

        Args:
            frequency: Frequency in Hz
            warn_on_extrapolation: Emit warning if extrapolating

        Returns:
            (limit_value, metadata) tuple
        """
        f_min, f_max = self._mask.frequency_range
        is_extrapolated = frequency < f_min or frequency > f_max

        if is_extrapolated and warn_on_extrapolation:
            logger.warning(
                f"Extrapolating limit at {frequency:.2e} Hz "
                f"(mask range: {f_min:.2e} to {f_max:.2e} Hz)"
            )

        limit = (
            float(self.interpolate(frequency)[0])
            if not is_extrapolated or self._extrapolate
            else np.nan
        )

        # Find nearest defined points
        idx = np.searchsorted(self._mask.frequency, frequency)
        if idx == 0:
            lower_freq = None
            upper_freq = self._mask.frequency[0]
        elif idx >= len(self._mask.frequency):
            lower_freq = self._mask.frequency[-1]
            upper_freq = None
        else:
            lower_freq = self._mask.frequency[idx - 1]
            upper_freq = self._mask.frequency[idx]

        return limit, {
            "frequency": frequency,
            "method": self._method.value,
            "is_extrapolated": is_extrapolated,
            "is_at_defined_point": frequency in self._mask.frequency,
            "lower_defined_freq": float(lower_freq) if lower_freq is not None else None,
            "upper_defined_freq": float(upper_freq) if upper_freq is not None else None,
        }


def interpolate_limit(
    mask: LimitMask,
    frequency: float | NDArray[np.float64],
    method: str = "log-linear",
) -> NDArray[np.float64]:
    """Convenience function for limit interpolation.

    Args:
        mask: Limit mask
        frequency: Frequency or frequencies in Hz
        method: Interpolation method

    Returns:
        Interpolated limit value(s)

    Example:
        >>> limit = interpolate_limit(mask, 100e6)
    """
    interp = LimitInterpolator(
        mask,
        method=InterpolationMethod(method),
        extrapolate=True,
    )
    return interp.interpolate(frequency)


# =============================================================================
# =============================================================================


@dataclass
class ComplianceTestConfig:
    """Configuration for compliance test.

    Attributes:
        mask_names: List of mask names to test against
        detector_type: Detector type to use
        frequency_range: Frequency range to test
        margin_required_db: Required margin to limit
        include_quasi_peak: Include QP detection
        generate_report: Generate detailed report

    References:
        COMP-006: Compliance Test Execution
    """

    mask_names: list[str] = field(default_factory=lambda: ["FCC_Part15_ClassB"])
    detector_type: str = "peak"
    frequency_range: tuple[float, float] | None = None
    margin_required_db: float = 0.0
    include_quasi_peak: bool = True
    generate_report: bool = True


@dataclass
class ComplianceTestResult:
    """Result of a single compliance test.

    Attributes:
        mask_name: Mask tested against
        passed: Whether test passed
        margin_db: Margin to limit (negative = fail)
        worst_frequency: Worst-case frequency
        violations: List of violations
        detector_used: Detector type used
    """

    mask_name: str
    passed: bool
    margin_db: float
    worst_frequency: float
    violations: list[dict[str, Any]]
    detector_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceTestSuiteResult:
    """Result of compliance test suite.

    Attributes:
        overall_passed: True if all tests passed
        results: Individual test results
        summary: Test summary
    """

    overall_passed: bool
    results: list[ComplianceTestResult]
    summary: dict[str, Any]


class ComplianceTestRunner:
    """Compliance test execution engine.

    Executes compliance tests against multiple masks with
    configurable detection methods.

    Example:
        >>> runner = ComplianceTestRunner()
        >>> runner.add_mask('FCC_Part15_ClassB')
        >>> runner.add_mask('CE_CISPR32_ClassB')
        >>> result = runner.run(spectrum_freq, spectrum_level)

    References:
        COMP-006: Compliance Test Execution
    """

    def __init__(self, config: ComplianceTestConfig | None = None) -> None:
        """Initialize test runner.

        Args:
            config: Test configuration
        """
        self._config = config or ComplianceTestConfig()
        self._masks: list[tuple[str, Any]] = []
        self._qp_detector = QuasiPeakDetector()

    def add_mask(self, mask_name: str) -> ComplianceTestRunner:
        """Add mask to test suite.

        Args:
            mask_name: Mask name to add

        Returns:
            Self for chaining
        """
        from oscura.validation.compliance.masks import load_limit_mask

        mask = load_limit_mask(mask_name)
        self._masks.append((mask_name, mask))
        return self

    def run(
        self,
        frequencies: NDArray[np.float64],
        levels: NDArray[np.float64],
        unit: str = "dBuV",
    ) -> ComplianceTestSuiteResult:
        """Run compliance test suite.

        Args:
            frequencies: Frequency array in Hz
            levels: Level array in specified unit
            unit: Unit of level measurements

        Returns:
            Test suite result
        """
        results: list[ComplianceTestResult] = []

        for _mask_name, mask in self._masks:
            result = self._test_against_mask(frequencies, levels, mask, unit)
            results.append(result)

        overall_passed = all(r.passed for r in results)

        summary = {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "worst_margin_db": min(r.margin_db for r in results) if results else 0,
            "masks_tested": [r.mask_name for r in results],
        }

        return ComplianceTestSuiteResult(
            overall_passed=overall_passed,
            results=results,
            summary=summary,
        )

    def _test_against_mask(
        self,
        frequencies: NDArray[np.float64],
        levels: NDArray[np.float64],
        mask: Any,
        unit: str,
    ) -> ComplianceTestResult:
        """Test against single mask."""
        # Apply frequency range filter
        if self._config.frequency_range:
            f_min, f_max = self._config.frequency_range
            in_range = (frequencies >= f_min) & (frequencies <= f_max)
            frequencies = frequencies[in_range]
            levels = levels[in_range]

        # Limit to mask range
        mask_f_min, mask_f_max = mask.frequency_range
        in_mask = (frequencies >= mask_f_min) & (frequencies <= mask_f_max)
        frequencies = frequencies[in_mask]
        levels = levels[in_mask]

        if len(frequencies) == 0:
            return ComplianceTestResult(
                mask_name=mask.name,
                passed=True,
                margin_db=np.inf,
                worst_frequency=0.0,
                violations=[],
                detector_used=self._config.detector_type,
            )

        # Interpolate limits
        interp = LimitInterpolator(mask)
        limits = interp.interpolate(frequencies)

        # Apply quasi-peak if requested
        if self._config.include_quasi_peak and mask.detector == "quasi-peak":
            levels = self._qp_detector.apply(levels, frequencies)

        # Calculate margin
        margin = limits - levels
        min_margin = float(np.min(margin))
        worst_idx = int(np.argmin(margin))

        # Find violations (considering required margin)
        violations = []
        violation_mask = margin < self._config.margin_required_db
        if np.any(violation_mask):
            for idx in np.where(violation_mask)[0]:
                violations.append(
                    {
                        "frequency": float(frequencies[idx]),
                        "measured": float(levels[idx]),
                        "limit": float(limits[idx]),
                        "excess_db": float(-margin[idx]),
                    }
                )

        passed = len(violations) == 0

        return ComplianceTestResult(
            mask_name=mask.name,
            passed=passed,
            margin_db=min_margin,
            worst_frequency=float(frequencies[worst_idx]),
            violations=violations,
            detector_used=self._config.detector_type,
            metadata={"unit": unit},
        )


class ComplianceTestSuite:
    """Pre-configured compliance test suites.

    Provides standard test configurations for common scenarios.

    References:
        COMP-006: Compliance Test Execution
    """

    @staticmethod
    def residential() -> ComplianceTestRunner:
        """Get residential (Class B) test suite."""
        runner = ComplianceTestRunner(ComplianceTestConfig(include_quasi_peak=True))
        runner.add_mask("FCC_Part15_ClassB")
        runner.add_mask("CE_CISPR32_ClassB")
        return runner

    @staticmethod
    def commercial() -> ComplianceTestRunner:
        """Get commercial (Class A) test suite."""
        runner = ComplianceTestRunner(ComplianceTestConfig(include_quasi_peak=True))
        runner.add_mask("FCC_Part15_ClassA")
        runner.add_mask("CE_CISPR32_ClassA")
        return runner

    @staticmethod
    def military() -> ComplianceTestRunner:
        """Get military (MIL-STD) test suite."""
        runner = ComplianceTestRunner(ComplianceTestConfig(include_quasi_peak=False))
        runner.add_mask("MIL_STD_461G_RE102")
        runner.add_mask("MIL_STD_461G_CE102")
        return runner


def run_compliance_suite(
    frequencies: NDArray[np.float64],
    levels: NDArray[np.float64],
    suite: str = "residential",
) -> ComplianceTestSuiteResult:
    """Run standard compliance test suite.

    Args:
        frequencies: Frequency array in Hz
        levels: Level array in dB
        suite: Suite name ('residential', 'commercial', 'military')

    Returns:
        Test suite result

    Raises:
        ValueError: If suite name is unknown.

    Example:
        >>> result = run_compliance_suite(freq, levels, suite='residential')
        >>> print(f"Passed: {result.overall_passed}")
    """
    if suite == "residential":
        runner = ComplianceTestSuite.residential()
    elif suite == "commercial":
        runner = ComplianceTestSuite.commercial()
    elif suite == "military":
        runner = ComplianceTestSuite.military()
    else:
        raise ValueError(f"Unknown suite: {suite}")

    return runner.run(frequencies, levels)


# =============================================================================
# =============================================================================


class QPDetectorBand(Enum):
    """CISPR 16-1-1 quasi-peak detector bands.

    References:
        CISPR 16-1-1 Table 1
        COMP-007: Quasi-Peak Detection
    """

    BAND_A = "A"  # 9 kHz - 150 kHz
    BAND_B = "B"  # 150 kHz - 30 MHz
    BAND_C = "C"  # 30 MHz - 300 MHz
    BAND_D = "D"  # 300 MHz - 1 GHz


@dataclass
class QPDetectorParams:
    """Quasi-peak detector parameters per CISPR 16-1-1.

    Attributes:
        bandwidth: Measurement bandwidth in Hz
        charge_time: Charge time constant in ms
        discharge_time: Discharge time constant in ms
        mechanical_time: Meter mechanical time constant in ms
    """

    bandwidth: float
    charge_time: float
    discharge_time: float
    mechanical_time: float


class QuasiPeakDetector:
    """CISPR 16-1-1 quasi-peak detector.

    Implements quasi-peak detection per CISPR 16-1-1 standard for
    EMC compliance measurements.

    Example:
        >>> detector = QuasiPeakDetector()
        >>> qp_levels = detector.apply(peak_levels, frequencies)

    References:
        CISPR 16-1-1: Measuring Apparatus
        COMP-007: Quasi-Peak Detection
    """

    # CISPR 16-1-1 detector parameters by band
    BAND_PARAMS = {
        QPDetectorBand.BAND_A: QPDetectorParams(
            bandwidth=200,  # 200 Hz
            charge_time=45,  # ms
            discharge_time=500,  # ms
            mechanical_time=160,  # ms
        ),
        QPDetectorBand.BAND_B: QPDetectorParams(
            bandwidth=9000,  # 9 kHz
            charge_time=1,  # ms
            discharge_time=160,  # ms
            mechanical_time=160,  # ms
        ),
        QPDetectorBand.BAND_C: QPDetectorParams(
            bandwidth=120000,  # 120 kHz
            charge_time=1,  # ms
            discharge_time=550,  # ms
            mechanical_time=100,  # ms
        ),
        QPDetectorBand.BAND_D: QPDetectorParams(
            bandwidth=1000000,  # 1 MHz
            charge_time=1,  # ms
            discharge_time=550,  # ms
            mechanical_time=100,  # ms
        ),
    }

    # Frequency ranges for bands (Hz)
    BAND_RANGES = {
        QPDetectorBand.BAND_A: (9e3, 150e3),
        QPDetectorBand.BAND_B: (150e3, 30e6),
        QPDetectorBand.BAND_C: (30e6, 300e6),
        QPDetectorBand.BAND_D: (300e6, 1e9),
    }

    def __init__(self) -> None:
        """Initialize quasi-peak detector."""
        self._lookup_table: dict[str, NDArray[np.float64]] = {}

    def get_band(self, frequency: float) -> QPDetectorBand | None:
        """Get CISPR band for frequency.

        Args:
            frequency: Frequency in Hz

        Returns:
            Band or None if outside all bands
        """
        for band, (f_min, f_max) in self.BAND_RANGES.items():
            if f_min <= frequency <= f_max:
                return band
        return None

    def get_params(self, frequency: float) -> QPDetectorParams | None:
        """Get detector parameters for frequency.

        Args:
            frequency: Frequency in Hz

        Returns:
            Detector parameters or None
        """
        band = self.get_band(frequency)
        if band is None:
            return None
        return self.BAND_PARAMS[band]

    def apply(
        self,
        peak_levels: NDArray[np.float64],
        frequencies: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply quasi-peak detection to peak levels.

        Args:
            peak_levels: Peak detector levels in dB
            frequencies: Corresponding frequencies in Hz

        Returns:
            Quasi-peak levels in dB

        Note:
            Quasi-peak is always <= peak for repetitive signals.
            The correction factor depends on pulse repetition rate.
        """
        qp_levels = np.copy(peak_levels)

        for i, (level, freq) in enumerate(zip(peak_levels, frequencies, strict=False)):
            band = self.get_band(freq)
            if band is not None:
                # Apply approximate QP correction
                # Real implementation would need actual signal for time-domain processing
                correction = self._get_qp_correction(band)
                qp_levels[i] = level - correction

        return qp_levels

    def _get_qp_correction(self, band: QPDetectorBand) -> float:
        """Get approximate QP correction factor.

        This is a simplified model. Real QP detection requires
        time-domain processing of the actual signal.

        Args:
            band: CISPR band

        Returns:
            Correction factor in dB
        """
        # Approximate corrections for periodic signals
        # Actual correction depends on pulse rate and duty cycle
        corrections = {
            QPDetectorBand.BAND_A: 3.0,
            QPDetectorBand.BAND_B: 6.0,
            QPDetectorBand.BAND_C: 4.0,
            QPDetectorBand.BAND_D: 4.0,
        }
        return corrections.get(band, 0.0)

    def compare_peak_qp(
        self,
        peak_levels: NDArray[np.float64],
        frequencies: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Compare peak and quasi-peak readings.

        Args:
            peak_levels: Peak detector levels
            frequencies: Frequencies

        Returns:
            Comparison results
        """
        qp_levels = self.apply(peak_levels, frequencies)
        difference = peak_levels - qp_levels

        return {
            "peak_levels": peak_levels,
            "qp_levels": qp_levels,
            "difference_db": difference,
            "max_difference_db": float(np.max(difference)),
            "avg_difference_db": float(np.mean(difference)),
            "description": (
                "Quasi-peak is lower than peak for pulsed/repetitive signals. "
                "For CW signals, QP equals peak."
            ),
        }

    def get_bandwidth(self, frequency: float) -> float:
        """Get measurement bandwidth for frequency.

        Args:
            frequency: Frequency in Hz

        Returns:
            Bandwidth in Hz
        """
        params = self.get_params(frequency)
        if params is None:
            # Default to Band B
            return 9000
        return params.bandwidth

    def validate_bandwidth(self, bandwidth: float) -> None:
        """Validate measurement bandwidth.

        Args:
            bandwidth: Bandwidth to validate

        Raises:
            ValueError: If bandwidth is invalid
        """
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive")

        valid_bandwidths = [p.bandwidth for p in self.BAND_PARAMS.values()]
        if bandwidth not in valid_bandwidths:
            logger.warning(
                f"Non-standard bandwidth {bandwidth} Hz. "
                f"Standard CISPR bandwidths: {valid_bandwidths}"
            )
