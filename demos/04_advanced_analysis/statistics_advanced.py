"""Advanced Statistical Analysis: Distribution and trend analysis

Demonstrates:
- Distribution fitting (Gaussian, Weibull, etc.)
- Trend detection and forecasting
- Outlier identification
- Statistical process control (SPC)
- Long-term stability analysis

IEEE Standards: IEEE 181-2011
Related Demos:
- 02_basic_analysis/01_waveform_measurements.py
- 04_advanced_analysis/10_correlation.py

Advanced statistical methods for signal quality and process control.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy import stats

from demos.common.base_demo import BaseDemo, run_demo_main
from demos.common.validation import ValidationSuite


class AdvancedStatisticsDemo(BaseDemo):
    """Advanced statistical analysis demonstration."""

    name = "Advanced Statistical Analysis"
    description = "Distribution fitting, trend detection, and SPC"
    category = "advanced_analysis"
    capabilities = [
        "Distribution fitting",
        "Trend detection",
        "Outlier identification",
        "Statistical process control",
        "Stability analysis",
    ]
    ieee_standards = ["IEEE 181-2011"]
    related_demos = ["02_basic_analysis/01_waveform_measurements.py"]

    def generate_data(self) -> None:
        """Generate measurement datasets."""
        # Gaussian distribution
        self.gaussian_data = np.random.normal(100, 5, 1000)

        # Data with trend
        self.trend_data = np.linspace(100, 110, 1000) + np.random.normal(0, 2, 1000)

        # Data with outliers
        self.outlier_data = np.random.normal(100, 5, 1000)
        self.outlier_data[[100, 200, 300, 400]] = [130, 125, 135, 140]

        # Process control data
        self.process_data = np.random.normal(100, 3, 500)

    def run_analysis(self) -> None:
        """Perform statistical analysis."""
        from demos.common.formatting import print_subheader

        print_subheader("Distribution Fitting")
        self.results["distribution"] = self._fit_distribution(self.gaussian_data)

        print_subheader("Trend Detection")
        self.results["trend"] = self._detect_trend(self.trend_data)

        print_subheader("Outlier Detection")
        self.results["outliers"] = self._detect_outliers(self.outlier_data)

        print_subheader("Statistical Process Control")
        self.results["spc"] = self._spc_analysis(self.process_data)

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate statistical analysis."""
        if "distribution" in self.results:
            mean = self.results["distribution"]["mean"]
            suite.check_approximately(
                "gaussian_mean",
                mean,
                100.0,
                2.0,
                "Distribution mean ~100",
            )

        if "outliers" in self.results:
            num_outliers = self.results["outliers"]["num_outliers"]
            suite.check_greater_than(
                "outliers_detected",
                num_outliers,
                0,
                "Outliers detected",
            )

    def _fit_distribution(self, data: np.ndarray) -> dict:
        """Fit distribution to data."""
        from demos.common.formatting import print_info

        # Fit Gaussian
        mu, sigma = stats.norm.fit(data)

        # Goodness of fit (Kolmogorov-Smirnov test)
        _, p_value = stats.kstest(data, "norm", args=(mu, sigma))

        print_info(f"Mean: {mu:.2f}")
        print_info(f"Std dev: {sigma:.2f}")
        print_info(f"KS test p-value: {p_value:.4f}")

        return {"mean": mu, "std": sigma, "p_value": p_value}

    def _detect_trend(self, data: np.ndarray) -> dict:
        """Detect trend in data."""
        from demos.common.formatting import print_info

        x = np.arange(len(data))

        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(x, data)

        print_info(f"Trend slope: {slope:.4f} per sample")
        print_info(f"R²: {r_value**2:.4f}")

        is_trending = abs(r_value) > 0.5

        return {
            "slope": slope,
            "r_squared": r_value**2,
            "is_trending": is_trending,
        }

    def _detect_outliers(self, data: np.ndarray) -> dict:
        """Detect outliers using IQR method."""
        from demos.common.formatting import print_info

        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (data < lower_bound) | (data > upper_bound)
        num_outliers = np.sum(outliers)

        print_info(f"IQR: {iqr:.2f}")
        print_info(f"Lower bound: {lower_bound:.2f}")
        print_info(f"Upper bound: {upper_bound:.2f}")
        print_info(f"Outliers detected: {num_outliers}")

        return {
            "num_outliers": num_outliers,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    def _spc_analysis(self, data: np.ndarray) -> dict:
        """Statistical process control analysis."""
        from demos.common.formatting import print_info

        mean = np.mean(data)
        std = np.std(data)

        # Control limits (3-sigma)
        ucl = mean + 3 * std
        lcl = mean - 3 * std

        # Check for out-of-control points
        out_of_control = (data > ucl) | (data < lcl)
        num_ooc = np.sum(out_of_control)

        # Process capability (Cp)
        usl = mean + 6 * std  # Upper spec limit (example)
        lsl = mean - 6 * std  # Lower spec limit
        cp = (usl - lsl) / (6 * std)

        print_info(f"Mean: {mean:.2f}")
        print_info(f"UCL: {ucl:.2f}")
        print_info(f"LCL: {lcl:.2f}")
        print_info(f"Out-of-control points: {num_ooc}")
        print_info(f"Cp: {cp:.2f}")

        return {
            "mean": mean,
            "ucl": ucl,
            "lcl": lcl,
            "num_ooc": num_ooc,
            "cp": cp,
        }


if __name__ == "__main__":
    sys.exit(run_demo_main(AdvancedStatisticsDemo))
