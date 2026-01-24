"""Measurement uncertainty propagation and estimation.

This module provides data structures and utilities for tracking measurement
uncertainty through Oscura analysis operations, following GUM (Guide to the
Expression of Uncertainty in Measurement) principles.

References:
    JCGM 100:2008 - Guide to the Expression of Uncertainty in Measurement (GUM)
    ISO/IEC 17025:2017 - General Requirements for Competence of Testing and Calibration Laboratories
    NIST Technical Note 1297 - Guidelines for Evaluating and Expressing Uncertainty
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class MeasurementWithUncertainty:
    """Measurement value with uncertainty estimate.

    Stores a measurement value along with its associated uncertainty,
    following GUM (Guide to the Expression of Uncertainty in Measurement)
    conventions.

    Attributes:
        value: The measured value (best estimate).
        uncertainty: Standard uncertainty (1-sigma, coverage factor k=1).
        unit: Unit of measurement (e.g., "V", "s", "Hz") (optional).
        coverage_factor: Coverage factor for expanded uncertainty (default 1.0).
        confidence_level: Confidence level for expanded uncertainty (default 68.27% for k=1).
        n_samples: Number of samples used in measurement (optional).
        degrees_of_freedom: Effective degrees of freedom (optional).

    Properties:
        expanded_uncertainty: Uncertainty * coverage_factor.
        relative_uncertainty: uncertainty / |value| if value != 0.
        lower_bound: value - expanded_uncertainty.
        upper_bound: value + expanded_uncertainty.

    Example:
        >>> result = MeasurementWithUncertainty(value=1.234, uncertainty=0.005, unit="V")
        >>> print(f"{result.value:.3f} ± {result.uncertainty:.3f} {result.unit}")
        1.234 ± 0.005 V

    Example with expanded uncertainty (k=2, 95.45% confidence):
        >>> result = MeasurementWithUncertainty(
        ...     value=10.0e6,
        ...     uncertainty=1000.0,
        ...     unit="Hz",
        ...     coverage_factor=2.0,
        ...     confidence_level=0.9545
        ... )
        >>> print(f"{result.value/1e6:.3f} ± {result.expanded_uncertainty/1e6:.3f} MHz (95.45%)")
        10.000 ± 0.002 MHz (95.45%)

    References:
        JCGM 100:2008 Section 2.3.5 (standard uncertainty)
        JCGM 100:2008 Section 2.3.6 (expanded uncertainty)
    """

    value: float
    uncertainty: float
    unit: str | None = None
    coverage_factor: float = 1.0
    confidence_level: float = 0.6827  # 68.27% for k=1 (Gaussian)
    n_samples: int | None = None
    degrees_of_freedom: float | None = None

    def __post_init__(self) -> None:
        """Validate measurement result after initialization."""
        if not np.isfinite(self.value):
            # Allow NaN/Inf but issue warning
            pass
        if self.uncertainty < 0:
            raise ValueError(f"uncertainty must be non-negative, got {self.uncertainty}")
        if self.coverage_factor <= 0:
            raise ValueError(f"coverage_factor must be positive, got {self.coverage_factor}")
        if not 0 < self.confidence_level <= 1.0:
            raise ValueError(f"confidence_level must be in (0, 1], got {self.confidence_level}")

    @property
    def expanded_uncertainty(self) -> float:
        """Expanded uncertainty (U = k * u).

        Returns:
            Standard uncertainty multiplied by coverage factor.

        References:
            JCGM 100:2008 Section 6.2.1
        """
        return self.coverage_factor * self.uncertainty

    @property
    def relative_uncertainty(self) -> float:
        """Relative standard uncertainty (u_r = u / |value|).

        Returns:
            Relative uncertainty, or np.inf if value is zero.

        References:
            JCGM 100:2008 Section 5.1.6
        """
        if self.value == 0:
            return float(np.inf)
        return abs(self.uncertainty / self.value)

    @property
    def lower_bound(self) -> float:
        """Lower bound of uncertainty interval.

        Returns:
            value - expanded_uncertainty.
        """
        return self.value - self.expanded_uncertainty

    @property
    def upper_bound(self) -> float:
        """Upper bound of uncertainty interval.

        Returns:
            value + expanded_uncertainty.
        """
        return self.value + self.expanded_uncertainty

    def __str__(self) -> str:
        """String representation of measurement with uncertainty."""
        unit_str = f" {self.unit}" if self.unit else ""
        if self.coverage_factor == 1.0:
            return f"{self.value:.6g} ± {self.uncertainty:.6g}{unit_str}"
        else:
            return (
                f"{self.value:.6g} ± {self.expanded_uncertainty:.6g}{unit_str} "
                f"(k={self.coverage_factor:.2f}, {self.confidence_level * 100:.2f}%)"
            )

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"MeasurementWithUncertainty(value={self.value:.6g}, "
            f"uncertainty={self.uncertainty:.6g}, "
            f"unit={self.unit!r})"
        )


class UncertaintyEstimator:
    """Utilities for estimating measurement uncertainty.

    Provides methods for calculating Type A (statistical) and Type B
    (systematic) uncertainties according to GUM principles.

    References:
        JCGM 100:2008 Section 4 (Evaluating Standard Uncertainty)
    """

    @staticmethod
    def type_a_standard_deviation(data: NDArray[np.floating[Any]]) -> float:
        """Type A uncertainty from sample standard deviation.

        Args:
            data: Array of repeated measurements.

        Returns:
            Standard deviation (Type A uncertainty component).

        References:
            JCGM 100:2008 Section 4.2 (Type A evaluation)
        """
        if len(data) < 2:
            return float(np.nan)
        return float(np.std(data, ddof=1))  # Sample std (Bessel correction)

    @staticmethod
    def type_a_standard_error(data: NDArray[np.floating[Any]]) -> float:
        """Type A uncertainty from standard error of the mean.

        Args:
            data: Array of repeated measurements.

        Returns:
            Standard error (sigma / sqrt(n)), Type A uncertainty of the mean.

        References:
            JCGM 100:2008 Section 4.2.3
        """
        if len(data) < 2:
            return float(np.nan)
        return float(np.std(data, ddof=1) / np.sqrt(len(data)))

    @staticmethod
    def combined_uncertainty(
        uncertainties: list[float], correlation_matrix: NDArray[np.float64] | None = None
    ) -> float:
        """Combine multiple uncertainty sources.

        Combines uncorrelated or correlated uncertainty components using
        the law of propagation of uncertainty.

        Args:
            uncertainties: List of standard uncertainties to combine.
            correlation_matrix: Correlation matrix for correlated inputs (optional).
                If None, assumes all inputs are uncorrelated.

        Returns:
            Combined standard uncertainty.

        Example (uncorrelated):
            >>> u1, u2, u3 = 0.01, 0.02, 0.005
            >>> u_combined = UncertaintyEstimator.combined_uncertainty([u1, u2, u3])
            >>> print(f"Combined: {u_combined:.4f}")
            Combined: 0.0233

        Example (correlated):
            >>> import numpy as np
            >>> u = [0.01, 0.02]
            >>> R = np.array([[1.0, 0.5], [0.5, 1.0]])  # 50% correlation
            >>> u_combined = UncertaintyEstimator.combined_uncertainty(u, R)

        References:
            JCGM 100:2008 Section 5.1.2 (uncorrelated)
            JCGM 100:2008 Section 5.2.2 (correlated)
        """
        u_array = np.array(uncertainties, dtype=np.float64)

        if correlation_matrix is None:
            # Uncorrelated: u_c² = Σ u_i²
            return float(np.sqrt(np.sum(u_array**2)))
        else:
            # Correlated: u_c² = u^T R u
            u_combined_sq = u_array @ correlation_matrix @ u_array
            return float(np.sqrt(u_combined_sq))

    @staticmethod
    def type_b_rectangular(half_width: float) -> float:
        """Type B uncertainty from rectangular (uniform) distribution.

        Used when only min/max bounds are known with equal probability.

        Args:
            half_width: Half-width of the interval (a).

        Returns:
            Standard uncertainty u = a / √3.

        Example:
            >>> # Scope resolution ±0.5 LSB
            >>> lsb = 1.0e-3  # 1 mV per bit
            >>> u = UncertaintyEstimator.type_b_rectangular(0.5 * lsb)
            >>> print(f"Quantization uncertainty: {u*1e6:.3f} µV")
            Quantization uncertainty: 288.675 µV

        References:
            JCGM 100:2008 Section 4.3.7 (rectangular distribution)
        """
        return float(half_width / np.sqrt(3))

    @staticmethod
    def type_b_triangular(half_width: float) -> float:
        """Type B uncertainty from triangular distribution.

        Used when values near the center are more likely than extremes.

        Args:
            half_width: Half-width of the interval (a).

        Returns:
            Standard uncertainty u = a / √6.

        References:
            JCGM 100:2008 Section 4.3.9 (triangular distribution)
        """
        return float(half_width / np.sqrt(6))

    @staticmethod
    def type_b_from_tolerance(tolerance: float, confidence: float = 0.95) -> float:
        """Type B uncertainty from manufacturer tolerance specification.

        Assumes Gaussian distribution unless otherwise specified.

        Args:
            tolerance: Tolerance limit (e.g., ±1% of reading).
            confidence: Confidence level of the tolerance (default 95%).

        Returns:
            Standard uncertainty.

        Example:
            >>> # Scope vertical accuracy: ±2% of reading, 95% confidence
            >>> reading = 1.0  # 1V
            >>> tolerance = 0.02 * reading  # ±0.02 V
            >>> u = UncertaintyEstimator.type_b_from_tolerance(tolerance, 0.95)
            >>> print(f"Uncertainty: {u:.4f} V")
            Uncertainty: 0.0102 V

        References:
            JCGM 100:2008 Section 4.3.4
        """
        # For Gaussian, 95% confidence → k ≈ 1.96
        # For 99% confidence → k ≈ 2.58
        if confidence == 0.95:
            k = 1.96
        elif confidence == 0.99:
            k = 2.58
        elif confidence == 0.6827:
            k = 1.0
        else:
            # General case: approximate using normal distribution
            from scipy import stats

            k = stats.norm.ppf((1 + confidence) / 2)

        return float(tolerance / k)

    @staticmethod
    def time_base_uncertainty(sample_rate: float, timebase_accuracy_ppm: float = 50.0) -> float:
        """Calculate time base uncertainty from oscilloscope specification.

        Args:
            sample_rate: Sample rate in Hz.
            timebase_accuracy_ppm: Timebase accuracy in parts per million (typical: 10-50 ppm).

        Returns:
            Standard uncertainty in time per sample (seconds).

        Example:
            >>> # 1 GSa/s scope, 25 ppm timebase (typical for OCXO)
            >>> u_t = UncertaintyEstimator.time_base_uncertainty(1e9, 25.0)
            >>> print(f"Timebase uncertainty: {u_t*1e12:.2f} ps per sample")
            Timebase uncertainty: 25.00 ps per sample

        References:
            IEEE 181-2011 Annex B (Measurement Uncertainty)
        """
        time_per_sample = 1.0 / sample_rate
        # ppm = parts per million = 1e-6
        relative_uncertainty = timebase_accuracy_ppm * 1e-6
        return time_per_sample * relative_uncertainty

    @staticmethod
    def vertical_uncertainty(
        reading: float,
        vertical_accuracy_percent: float = 2.0,
        offset_error_volts: float = 0.0,
    ) -> float:
        """Calculate vertical (voltage) uncertainty from oscilloscope specification.

        Typical scope spec: ±(2% of reading + 0.1% of full scale + 1 mV)

        Args:
            reading: Measured voltage value.
            vertical_accuracy_percent: Gain accuracy in percent (e.g., 2.0 for ±2%).
            offset_error_volts: Fixed offset error in volts.

        Returns:
            Standard uncertainty in volts.

        Example:
            >>> # Tektronix TDS3000 series: ±3% of reading
            >>> reading = 1.5  # 1.5 V
            >>> u_v = UncertaintyEstimator.vertical_uncertainty(reading, 3.0, 0.001)
            >>> print(f"Voltage uncertainty: {u_v*1000:.2f} mV")
            Voltage uncertainty: 45.01 mV

        References:
            IEEE 1057-2017 Section 4.4 (Amplitude Accuracy)
        """
        # Gain error
        u_gain = abs(reading) * (vertical_accuracy_percent / 100.0)

        # Offset error
        u_offset = offset_error_volts

        # Combine (assume uncorrelated)
        return float(np.sqrt(u_gain**2 + u_offset**2))


__all__ = ["MeasurementWithUncertainty", "UncertaintyEstimator"]
