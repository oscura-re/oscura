"""Result Aggregation: Statistical Analysis Across Batch Results

Demonstrates:
- Statistical aggregation across batch results
- Summary report generation
- Outlier detection in batch results
- Comparative analysis between batches
- Data quality metrics

This demonstration shows how to aggregate and analyze results from
batch processing operations, including statistical summaries, outlier
detection, and comparative analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo


class ResultAggregationDemo(BaseDemo):
    """Demonstrate result aggregation and statistical analysis."""

    def __init__(self) -> None:
        """Initialize result aggregation demonstration."""
        super().__init__(
            name="result_aggregation",
            description="Aggregate and analyze batch results with statistics and outliers",
            capabilities=[
                "batch.result_aggregation",
                "batch.statistical_analysis",
                "batch.outlier_detection",
                "batch.comparative_analysis",
            ],
            related_demos=[
                "09_batch_processing/01_parallel_batch.py",
                "09_batch_processing/03_progress_tracking.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate batch processing results for aggregation.

        Returns:
            Dictionary containing:
            - batch_results: List of processing results
            - batch_metadata: Metadata for each batch
        """
        # Generate two batches of results
        batch1_size = 50
        batch2_size = 50

        # Batch 1: Normal signals
        batch1_results = []
        for i in range(batch1_size):
            frequency = 1e3 + i * 100  # 1 kHz to 6 kHz
            amplitude = 1.0 + np.random.normal(0, 0.05)  # 1V ± 5%
            noise = np.random.normal(0, 0.01)  # 10 mV noise

            batch1_results.append(
                {
                    "file_id": i,
                    "batch": "batch1",
                    "frequency": frequency,
                    "amplitude": amplitude,
                    "rms": amplitude / np.sqrt(2) + noise,
                    "peak": amplitude + abs(noise),
                    "snr_db": 40.0 + np.random.normal(0, 2),
                    "thd_percent": 0.5 + np.random.normal(0, 0.1),
                }
            )

        # Batch 2: Signals with some outliers
        batch2_results = []
        for i in range(batch2_size):
            frequency = 10e3 + i * 200  # 10 kHz to 20 kHz
            amplitude = 2.0 + np.random.normal(0, 0.05)  # 2V ± 5%
            noise = np.random.normal(0, 0.01)

            # Inject outliers (every 10th sample)
            if i % 10 == 0:
                amplitude *= 0.5  # Low amplitude outlier
                noise *= 5  # High noise outlier

            batch2_results.append(
                {
                    "file_id": i + batch1_size,
                    "batch": "batch2",
                    "frequency": frequency,
                    "amplitude": amplitude,
                    "rms": amplitude / np.sqrt(2) + noise,
                    "peak": amplitude + abs(noise),
                    "snr_db": 35.0 + np.random.normal(0, 3),
                    "thd_percent": 1.0 + np.random.normal(0, 0.2),
                }
            )

        return {
            "batch1_results": batch1_results,
            "batch2_results": batch2_results,
            "batch1_size": batch1_size,
            "batch2_size": batch2_size,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the result aggregation demonstration."""
        results: dict[str, Any] = {}

        self.section("Result Aggregation Demonstration")
        self.info("Statistical analysis and aggregation of batch processing results")

        batch1 = data["batch1_results"]
        batch2 = data["batch2_results"]
        all_results = batch1 + batch2

        # Part 1: Basic statistical aggregation
        self.subsection("Part 1: Basic Statistical Aggregation")
        self.info("Compute descriptive statistics across all batch results.")

        stats = self._compute_statistics(all_results)

        self.result("Total results", stats["count"])
        self.result("Mean amplitude", f"{stats['amplitude']['mean']:.4f}", "V")
        self.result("Std amplitude", f"{stats['amplitude']['std']:.4f}", "V")
        self.result("Mean RMS", f"{stats['rms']['mean']:.4f}", "V")
        self.result("Std RMS", f"{stats['rms']['std']:.4f}", "V")
        self.result("Mean SNR", f"{stats['snr_db']['mean']:.2f}", "dB")
        self.result("Mean THD", f"{stats['thd_percent']['mean']:.3f}", "%")
        self.result(
            "Frequency range",
            f"{stats['frequency']['min']:.0f} - {stats['frequency']['max']:.0f}",
            "Hz",
        )

        results["overall_stats"] = stats

        # Part 2: Per-batch aggregation
        self.subsection("Part 2: Per-Batch Aggregation")
        self.info("Compare statistics between different batches.")

        batch1_stats = self._compute_statistics(batch1)
        batch2_stats = self._compute_statistics(batch2)

        self.info("\nBatch 1 Statistics:")
        self.result("  Count", batch1_stats["count"])
        self.result("  Mean amplitude", f"{batch1_stats['amplitude']['mean']:.4f}", "V")
        self.result("  Mean SNR", f"{batch1_stats['snr_db']['mean']:.2f}", "dB")

        self.info("\nBatch 2 Statistics:")
        self.result("  Count", batch2_stats["count"])
        self.result("  Mean amplitude", f"{batch2_stats['amplitude']['mean']:.4f}", "V")
        self.result("  Mean SNR", f"{batch2_stats['snr_db']['mean']:.2f}", "dB")

        results["batch1_stats"] = batch1_stats
        results["batch2_stats"] = batch2_stats

        # Part 3: Outlier detection
        self.subsection("Part 3: Outlier Detection")
        self.info("Detect outliers using statistical methods (Z-score and IQR).")

        # Z-score method (|z| > 3)
        outliers_zscore = self._detect_outliers_zscore(all_results, threshold=3.0)

        # IQR method
        outliers_iqr = self._detect_outliers_iqr(all_results)

        self.result("Total results", len(all_results))
        self.result("Outliers (Z-score)", len(outliers_zscore))
        self.result("Outliers (IQR)", len(outliers_iqr))

        if outliers_zscore:
            self.info("\nSample outliers detected:")
            for outlier in outliers_zscore[:3]:
                self.info(
                    f"  File {outlier['file_id']}: amplitude={outlier['amplitude']:.4f}V, "
                    f"SNR={outlier['snr_db']:.2f}dB"
                )

        results["outliers_zscore"] = outliers_zscore
        results["outliers_iqr"] = outliers_iqr

        # Part 4: Data quality metrics
        self.subsection("Part 4: Data Quality Metrics")
        self.info("Calculate data quality metrics for batch results.")

        quality_metrics = self._calculate_quality_metrics(all_results, outliers_zscore)

        self.result("Total files", quality_metrics["total_files"])
        self.result("Valid files", quality_metrics["valid_files"])
        self.result("Quality rate", f"{quality_metrics['quality_rate']:.1f}", "%")
        self.result("Mean SNR", f"{quality_metrics['mean_snr']:.2f}", "dB")
        self.result("SNR > 30 dB", quality_metrics["high_quality_count"])
        self.result("SNR < 20 dB", quality_metrics["low_quality_count"])

        results["quality_metrics"] = quality_metrics

        # Part 5: Summary report generation
        self.subsection("Part 5: Summary Report Generation")
        self.info("Generate comprehensive summary report.")

        report = self._generate_summary_report(
            all_results, batch1_stats, batch2_stats, outliers_zscore, quality_metrics
        )

        # Save report to output directory
        output_dir = self.get_output_dir()
        report_path = output_dir / "aggregation_report.txt"

        with open(report_path, "w") as f:
            f.write(report)

        self.success(f"Summary report saved to: {report_path}")

        # Display report excerpt
        self.info("\nReport Excerpt:")
        lines = report.split("\n")
        for line in lines[:15]:
            self.info(f"  {line}")
        self.info("  ...")

        results["report"] = report
        results["report_path"] = str(report_path)

        # Part 6: Comparative analysis
        self.subsection("Part 6: Comparative Analysis")
        self.info("Compare key metrics between batches.")

        comparison = self._compare_batches(batch1_stats, batch2_stats)

        self.result("Amplitude ratio (B2/B1)", f"{comparison['amplitude_ratio']:.3f}")
        self.result("SNR difference (B1-B2)", f"{comparison['snr_difference']:.2f}", "dB")
        self.result("THD difference (B2-B1)", f"{comparison['thd_difference']:.3f}", "%")
        self.result("Frequency shift", f"{comparison['frequency_shift']:.0f}", "Hz")

        results["comparison"] = comparison

        self.success("Result aggregation demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate the demonstration results."""
        self.info("Validating result aggregation...")

        # Validate statistics exist
        if not all(key in results for key in ["overall_stats", "batch1_stats", "batch2_stats"]):
            self.error("Missing statistical results")
            return False

        # Validate outlier detection
        if "outliers_zscore" not in results or "outliers_iqr" not in results:
            self.error("Missing outlier detection results")
            return False

        # Validate that some outliers were detected
        if len(results["outliers_zscore"]) == 0:
            self.warning("No outliers detected (expected some in test data)")

        # Validate quality metrics
        if "quality_metrics" not in results:
            self.error("Missing quality metrics")
            return False

        quality = results["quality_metrics"]
        if quality["quality_rate"] < 0 or quality["quality_rate"] > 100:
            self.error(f"Invalid quality rate: {quality['quality_rate']}")
            return False

        # Validate report generation
        if "report" not in results:
            self.error("Missing summary report")
            return False

        if len(results["report"]) < 100:
            self.error("Summary report is too short")
            return False

        # Validate comparison
        if "comparison" not in results:
            self.error("Missing batch comparison")
            return False

        self.success("All result aggregation validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - Use descriptive statistics for overall batch analysis")
        self.info("  - Detect outliers with Z-score (|z| > 3) or IQR methods")
        self.info("  - Calculate data quality metrics for batch validation")
        self.info("  - Generate summary reports for documentation")
        self.info("  - Compare batches to identify systematic differences")

        return True

    def _compute_statistics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute descriptive statistics from results.

        Args:
            results: List of result dictionaries

        Returns:
            Dictionary of statistics
        """
        if not results:
            return {"count": 0}

        # Extract metric arrays
        amplitudes = np.array([r["amplitude"] for r in results])
        rms_values = np.array([r["rms"] for r in results])
        snr_values = np.array([r["snr_db"] for r in results])
        thd_values = np.array([r["thd_percent"] for r in results])
        frequencies = np.array([r["frequency"] for r in results])

        return {
            "count": len(results),
            "amplitude": {
                "mean": float(np.mean(amplitudes)),
                "std": float(np.std(amplitudes)),
                "min": float(np.min(amplitudes)),
                "max": float(np.max(amplitudes)),
                "median": float(np.median(amplitudes)),
            },
            "rms": {
                "mean": float(np.mean(rms_values)),
                "std": float(np.std(rms_values)),
                "min": float(np.min(rms_values)),
                "max": float(np.max(rms_values)),
            },
            "snr_db": {
                "mean": float(np.mean(snr_values)),
                "std": float(np.std(snr_values)),
                "min": float(np.min(snr_values)),
                "max": float(np.max(snr_values)),
            },
            "thd_percent": {
                "mean": float(np.mean(thd_values)),
                "std": float(np.std(thd_values)),
                "min": float(np.min(thd_values)),
                "max": float(np.max(thd_values)),
            },
            "frequency": {
                "mean": float(np.mean(frequencies)),
                "min": float(np.min(frequencies)),
                "max": float(np.max(frequencies)),
            },
        }

    def _detect_outliers_zscore(
        self, results: list[dict[str, Any]], threshold: float = 3.0
    ) -> list[dict[str, Any]]:
        """Detect outliers using Z-score method.

        Args:
            results: List of result dictionaries
            threshold: Z-score threshold for outlier detection

        Returns:
            List of outlier results
        """
        amplitudes = np.array([r["amplitude"] for r in results])
        mean = np.mean(amplitudes)
        std = np.std(amplitudes)

        outliers = []
        for r in results:
            z_score = abs((r["amplitude"] - mean) / std)
            if z_score > threshold:
                outliers.append(r)

        return outliers

    def _detect_outliers_iqr(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Detect outliers using IQR method.

        Args:
            results: List of result dictionaries

        Returns:
            List of outlier results
        """
        amplitudes = np.array([r["amplitude"] for r in results])
        q1 = np.percentile(amplitudes, 25)
        q3 = np.percentile(amplitudes, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = []
        for r in results:
            if r["amplitude"] < lower_bound or r["amplitude"] > upper_bound:
                outliers.append(r)

        return outliers

    def _calculate_quality_metrics(
        self, results: list[dict[str, Any]], outliers: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate data quality metrics.

        Args:
            results: List of all results
            outliers: List of outlier results

        Returns:
            Dictionary of quality metrics
        """
        total = len(results)
        valid = total - len(outliers)

        snr_values = [r["snr_db"] for r in results]
        high_quality = sum(1 for snr in snr_values if snr > 30)
        low_quality = sum(1 for snr in snr_values if snr < 20)

        return {
            "total_files": total,
            "valid_files": valid,
            "quality_rate": 100.0 * valid / total if total > 0 else 0.0,
            "mean_snr": float(np.mean(snr_values)),
            "high_quality_count": high_quality,
            "low_quality_count": low_quality,
        }

    def _generate_summary_report(
        self,
        all_results: list[dict[str, Any]],
        batch1_stats: dict[str, Any],
        batch2_stats: dict[str, Any],
        outliers: list[dict[str, Any]],
        quality: dict[str, Any],
    ) -> str:
        """Generate comprehensive summary report.

        Args:
            all_results: All processing results
            batch1_stats: Batch 1 statistics
            batch2_stats: Batch 2 statistics
            outliers: Detected outliers
            quality: Quality metrics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("BATCH PROCESSING AGGREGATION REPORT")
        report.append("=" * 80)
        report.append("")

        report.append("OVERALL SUMMARY")
        report.append("-" * 80)
        report.append(f"Total files processed: {len(all_results)}")
        report.append(f"Valid files: {quality['valid_files']}")
        report.append(f"Quality rate: {quality['quality_rate']:.1f}%")
        report.append(f"Outliers detected: {len(outliers)}")
        report.append("")

        report.append("BATCH 1 STATISTICS")
        report.append("-" * 80)
        report.append(f"Count: {batch1_stats['count']}")
        report.append(f"Mean amplitude: {batch1_stats['amplitude']['mean']:.4f} V")
        report.append(f"Mean SNR: {batch1_stats['snr_db']['mean']:.2f} dB")
        report.append(f"Mean THD: {batch1_stats['thd_percent']['mean']:.3f} %")
        report.append("")

        report.append("BATCH 2 STATISTICS")
        report.append("-" * 80)
        report.append(f"Count: {batch2_stats['count']}")
        report.append(f"Mean amplitude: {batch2_stats['amplitude']['mean']:.4f} V")
        report.append(f"Mean SNR: {batch2_stats['snr_db']['mean']:.2f} dB")
        report.append(f"Mean THD: {batch2_stats['thd_percent']['mean']:.3f} %")
        report.append("")

        report.append("QUALITY METRICS")
        report.append("-" * 80)
        report.append(f"High quality (SNR > 30 dB): {quality['high_quality_count']}")
        report.append(f"Low quality (SNR < 20 dB): {quality['low_quality_count']}")
        report.append(f"Mean SNR: {quality['mean_snr']:.2f} dB")
        report.append("")

        return "\n".join(report)

    def _compare_batches(
        self, batch1_stats: dict[str, Any], batch2_stats: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare statistics between batches.

        Args:
            batch1_stats: Batch 1 statistics
            batch2_stats: Batch 2 statistics

        Returns:
            Dictionary of comparison metrics
        """
        return {
            "amplitude_ratio": batch2_stats["amplitude"]["mean"]
            / batch1_stats["amplitude"]["mean"],
            "snr_difference": batch1_stats["snr_db"]["mean"] - batch2_stats["snr_db"]["mean"],
            "thd_difference": batch2_stats["thd_percent"]["mean"]
            - batch1_stats["thd_percent"]["mean"],
            "frequency_shift": batch2_stats["frequency"]["mean"]
            - batch1_stats["frequency"]["mean"],
        }


if __name__ == "__main__":
    demo: ResultAggregationDemo = ResultAggregationDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
