"""Automated regression test suite for protocol implementations.

This module provides comprehensive regression testing capabilities for detecting
changes in protocol behavior, tracking metrics over time, and maintaining baseline
test results (golden outputs).

Example:
    >>> from oscura.validation import RegressionTestSuite
    >>> from oscura.analyzers.protocols import UARTDecoder
    >>>
    >>> # Initialize suite and register test
    >>> suite = RegressionTestSuite("my_protocol_tests")
    >>> suite.register_test("uart_decode_hello", UARTDecoder.decode, input_data=b"Hello")
    >>>
    >>> # Capture baseline
    >>> suite.capture_baseline("uart_decode_hello")
    >>>
    >>> # Run regression test
    >>> result = suite.run_test("uart_decode_hello")
    >>> if result.passed:
    ...     print("No regression detected")
    >>> else:
    ...     print(f"Regression found: {result.differences}")
    >>>
    >>> # Generate report
    >>> report = suite.generate_report()
    >>> report.export_html("regression_report.html")

References:
    V0.6.0_COMPLETE_COMPREHENSIVE_PLAN.md: Feature 40 (Regression Testing)
    Software Testing: Regression test automation best practices
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class ComparisonMode(Enum):
    """Comparison mode for different types of protocol outputs.

    Attributes:
        EXACT: Exact byte-for-byte match (deterministic protocols).
        FUZZY: Fuzzy match with tolerance (timing-dependent protocols).
        STATISTICAL: Statistical comparison for noisy measurements.
        FIELD_BY_FIELD: Field-by-field comparison with per-field tolerance.
    """

    EXACT = "exact"
    FUZZY = "fuzzy"
    STATISTICAL = "statistical"
    FIELD_BY_FIELD = "field_by_field"


@dataclass
class RegressionTestResult:
    """Result from a single regression test execution.

    Attributes:
        test_name: Name of the test.
        baseline: Baseline (golden) output.
        current: Current test output.
        differences: List of detected differences.
        passed: True if test passed (no regressions).
        metrics: Performance metrics (execution_time, memory_usage, etc.).
        timestamp: When test was executed.
        comparison_mode: How outputs were compared.
        confidence: Confidence in the result (0.0-1.0).

    Example:
        >>> result = RegressionTestResult(
        ...     test_name="test_decode",
        ...     baseline={"frames": 10},
        ...     current={"frames": 10},
        ...     differences=[],
        ...     passed=True,
        ...     metrics={"execution_time": 0.025}
        ... )
    """

    test_name: str
    baseline: Any
    current: Any
    differences: list[str]
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    comparison_mode: ComparisonMode = ComparisonMode.EXACT
    confidence: float = 1.0


@dataclass
class RegressionReport:
    """Comprehensive regression test report.

    Attributes:
        suite_name: Name of the test suite.
        results: All test results.
        summary: Summary statistics.
        regressions_found: List of test names with regressions.
        timestamp: When report was generated.
        baseline_version: Version of baseline data.
        metadata: Additional report metadata.

    Example:
        >>> report = RegressionReport(
        ...     suite_name="protocol_tests",
        ...     results=[result1, result2],
        ...     summary={"total": 2, "passed": 2, "failed": 0},
        ...     regressions_found=[]
        ... )
    """

    suite_name: str
    results: list[RegressionTestResult]
    summary: dict[str, int | float]
    regressions_found: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    baseline_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def export_json(self, output: Path) -> None:
        """Export report as JSON.

        Args:
            output: Output JSON file path.

        Example:
            >>> report.export_json(Path("report.json"))
        """
        data = {
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "baseline_version": self.baseline_version,
            "summary": self.summary,
            "regressions_found": self.regressions_found,
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "differences": r.differences,
                    "metrics": r.metrics,
                    "timestamp": r.timestamp,
                    "comparison_mode": r.comparison_mode.value,
                    "confidence": r.confidence,
                    "baseline": self._serialize(r.baseline),
                    "current": self._serialize(r.current),
                }
                for r in self.results
            ],
            "metadata": self.metadata,
        }

        output.write_text(json.dumps(data, indent=2))

    def export_html(self, output: Path) -> None:
        """Generate HTML dashboard with visualizations.

        Args:
            output: Output HTML file path.

        Example:
            >>> report.export_html(Path("dashboard.html"))
        """
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>Regression Report: {self.suite_name}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1 { color: #333; }",
            ".summary { background: #f0f0f0; padding: 15px; border-radius: 5px; }",
            ".passed { color: green; font-weight: bold; }",
            ".failed { color: red; font-weight: bold; }",
            ".test { border: 1px solid #ddd; margin: 10px 0; padding: 10px; }",
            ".test-passed { border-left: 4px solid green; }",
            ".test-failed { border-left: 4px solid red; }",
            ".metrics { font-family: monospace; font-size: 0.9em; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Regression Report: {self.suite_name}</h1>",
            f"<p>Generated: {self.timestamp}</p>",
            f"<p>Baseline Version: {self.baseline_version}</p>",
            "<div class='summary'>",
            "<h2>Summary</h2>",
            f"<p>Total Tests: {self.summary['total']}</p>",
            f"<p class='passed'>Passed: {passed}</p>",
            f"<p class='failed'>Failed: {failed}</p>",
            f"<p>Pass Rate: {(passed / max(self.summary['total'], 1)) * 100:.1f}%</p>",
            "</div>",
            "<h2>Test Results</h2>",
        ]

        for r in self.results:
            test_class = "test-passed" if r.passed else "test-failed"
            status = "PASSED" if r.passed else "FAILED"
            html.extend(
                [
                    f"<div class='test {test_class}'>",
                    f"<h3>{r.test_name} - {status}</h3>",
                    f"<p>Comparison Mode: {r.comparison_mode.value}</p>",
                    f"<p>Confidence: {r.confidence:.2f}</p>",
                ]
            )

            if r.differences:
                html.append("<h4>Differences:</h4><ul>")
                for diff in r.differences:
                    html.append(f"<li>{diff}</li>")
                html.append("</ul>")

            if r.metrics:
                html.append("<h4>Metrics:</h4><div class='metrics'>")
                for key, value in r.metrics.items():
                    html.append(f"<p>{key}: {value:.6f}</p>")
                html.append("</div>")

            html.append("</div>")

        html.extend(["</body>", "</html>"])

        output.write_text("\n".join(html))

    def export_csv(self, output: Path) -> None:
        """Export test results as CSV for historical tracking.

        Args:
            output: Output CSV file path.

        Example:
            >>> report.export_csv(Path("history.csv"))
        """
        import csv

        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Test Name",
                    "Passed",
                    "Timestamp",
                    "Execution Time",
                    "Memory Usage",
                    "Differences Count",
                ]
            )
            for r in self.results:
                writer.writerow(
                    [
                        r.test_name,
                        r.passed,
                        r.timestamp,
                        r.metrics.get("execution_time", 0.0),
                        r.metrics.get("memory_usage", 0.0),
                        len(r.differences),
                    ]
                )

    def _serialize(self, obj: Any) -> Any:
        """Serialize object for JSON export.

        Args:
            obj: Object to serialize.

        Returns:
            JSON-serializable representation.
        """
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, bytes):
            return obj.hex()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._serialize(item) for item in obj]
        # Fallback: convert to string
        return str(obj)


class RegressionTestSuite:
    """Automated regression test suite for protocol implementations.

    Manages test baselines, runs regression tests, tracks metrics over time,
    and generates comprehensive reports.

    Example:
        >>> suite = RegressionTestSuite("uart_protocol")
        >>> suite.register_test("decode_basic", decoder.decode, input_data=raw_bytes)
        >>> suite.capture_baseline("decode_basic")
        >>> result = suite.run_test("decode_basic")
        >>> report = suite.generate_report()
    """

    def __init__(
        self,
        suite_name: str,
        baseline_dir: Path | str | None = None,
        auto_update_baselines: bool = False,
    ) -> None:
        """Initialize regression test suite.

        Args:
            suite_name: Name of the test suite.
            baseline_dir: Directory for baseline storage (default: ./baselines/).
            auto_update_baselines: Automatically update baselines on first run.

        Example:
            >>> suite = RegressionTestSuite("my_tests", baseline_dir="test_baselines")
        """
        self.suite_name = suite_name
        self.baseline_dir = Path(baseline_dir) if baseline_dir else Path("baselines")
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.auto_update_baselines = auto_update_baselines

        self.tests: dict[str, dict[str, Any]] = {}
        self.baselines: dict[str, Any] = {}
        self.results: list[RegressionTestResult] = []
        self.metrics_history: dict[str, list[dict[str, float]]] = {}

    def register_test(
        self,
        test_name: str,
        test_function: Callable[..., Any],
        comparison_mode: ComparisonMode = ComparisonMode.EXACT,
        tolerance: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """Register a test for regression tracking.

        Args:
            test_name: Unique test name.
            test_function: Function to test (must be deterministic or use fixed seed).
            comparison_mode: How to compare outputs.
            tolerance: Tolerance for fuzzy/statistical comparisons.
            **kwargs: Arguments to pass to test_function.

        Example:
            >>> suite.register_test(
            ...     "uart_decode",
            ...     decoder.decode,
            ...     comparison_mode=ComparisonMode.EXACT,
            ...     input_data=test_bytes
            ... )
        """
        self.tests[test_name] = {
            "function": test_function,
            "kwargs": kwargs,
            "comparison_mode": comparison_mode,
            "tolerance": tolerance,
        }

    def capture_baseline(self, test_name: str) -> None:
        """Capture baseline (golden output) for a test.

        Args:
            test_name: Name of test to capture baseline for.

        Raises:
            KeyError: If test not registered.

        Example:
            >>> suite.capture_baseline("uart_decode")
        """
        if test_name not in self.tests:
            raise KeyError(f"Test '{test_name}' not registered")

        test = self.tests[test_name]
        output = test["function"](**test["kwargs"])

        self.baselines[test_name] = output
        self._save_baseline(test_name, output)

    def run_test(self, test_name: str) -> RegressionTestResult:
        """Run a single regression test.

        Args:
            test_name: Name of test to run.

        Returns:
            Test result with comparison details.

        Raises:
            KeyError: If test not registered or baseline missing.

        Example:
            >>> result = suite.run_test("uart_decode")
            >>> print(f"Passed: {result.passed}")
        """
        if test_name not in self.tests:
            raise KeyError(f"Test '{test_name}' not registered")

        test = self.tests[test_name]

        # Load baseline
        if test_name not in self.baselines:
            self._load_baseline(test_name)

        if test_name not in self.baselines:
            if self.auto_update_baselines:
                self.capture_baseline(test_name)
            else:
                raise KeyError(f"No baseline for test '{test_name}'")

        baseline = self.baselines[test_name]

        # Run test and measure metrics
        start_time = time.perf_counter()
        current = test["function"](**test["kwargs"])
        execution_time = time.perf_counter() - start_time

        # Compare outputs
        differences, passed, confidence = self._compare_outputs(
            baseline, current, test["comparison_mode"], test["tolerance"]
        )

        # Track metrics
        metrics = {
            "execution_time": execution_time,
        }

        result = RegressionTestResult(
            test_name=test_name,
            baseline=baseline,
            current=current,
            differences=differences,
            passed=passed,
            metrics=metrics,
            comparison_mode=test["comparison_mode"],
            confidence=confidence,
        )

        self.results.append(result)
        self._track_metrics(test_name, metrics)

        return result

    def run_all(self) -> list[RegressionTestResult]:
        """Run all registered tests.

        Returns:
            List of all test results.

        Example:
            >>> results = suite.run_all()
            >>> failed = [r for r in results if not r.passed]
        """
        results = []
        for test_name in self.tests:
            try:
                result = self.run_test(test_name)
                results.append(result)
            except Exception as e:
                # Create failed result for exceptions
                result = RegressionTestResult(
                    test_name=test_name,
                    baseline=None,
                    current=None,
                    differences=[f"Exception: {e}"],
                    passed=False,
                    confidence=0.0,
                )
                results.append(result)

        return results

    def generate_report(self) -> RegressionReport:
        """Generate comprehensive regression report.

        Returns:
            Report with all results and summary statistics.

        Example:
            >>> report = suite.generate_report()
            >>> report.export_html("report.html")
        """
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        regressions = [r.test_name for r in self.results if not r.passed]

        summary = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / max(total, 1)) * 100,
        }

        # Calculate metric trends
        trends = {}
        for test_name, history in self.metrics_history.items():
            if len(history) >= 2:
                recent = history[-1]
                previous = history[-2]
                trends[test_name] = {
                    "execution_time_delta": recent.get("execution_time", 0.0)
                    - previous.get("execution_time", 0.0),
                }

        metadata = {
            "test_count": len(self.tests),
            "baseline_dir": str(self.baseline_dir),
            "trends": trends,
        }

        return RegressionReport(
            suite_name=self.suite_name,
            results=self.results,
            summary=summary,
            regressions_found=regressions,
            metadata=metadata,
        )

    def update_baseline(self, test_name: str) -> None:
        """Update baseline for a test (when behavior change is intentional).

        Args:
            test_name: Name of test to update.

        Example:
            >>> suite.update_baseline("uart_decode")  # Accept new behavior
        """
        if test_name not in self.tests:
            raise KeyError(f"Test '{test_name}' not registered")

        test = self.tests[test_name]
        new_baseline = test["function"](**test["kwargs"])
        self.baselines[test_name] = new_baseline
        self._save_baseline(test_name, new_baseline)

    def _compare_exact(self, baseline: Any, current: Any) -> tuple[list[str], bool, float]:
        """Compare outputs with exact matching.

        Args:
            baseline: Baseline output
            current: Current output

        Returns:
            Tuple of (differences, passed, confidence)
        """
        differences: list[str] = []
        passed = baseline == current
        if not passed:
            differences.append(f"Exact match failed: {baseline} != {current}")
        return differences, passed, 1.0

    def _compare_fuzzy(
        self, baseline: Any, current: Any, tolerance: float
    ) -> tuple[list[str], bool, float]:
        """Compare outputs with fuzzy tolerance.

        Args:
            baseline: Baseline output
            current: Current output
            tolerance: Tolerance for numeric differences

        Returns:
            Tuple of (differences, passed, confidence)
        """
        differences: list[str] = []
        passed = True
        confidence = 1.0

        if isinstance(baseline, (int, float)) and isinstance(current, (int, float)):
            diff = abs(baseline - current)
            if diff > tolerance:
                differences.append(f"Fuzzy match failed: |{baseline} - {current}| > {tolerance}")
                passed = False
                confidence = 1.0 - min(1.0, diff / tolerance)
        elif isinstance(baseline, (list, tuple)) and isinstance(current, (list, tuple)):
            if len(baseline) != len(current):
                differences.append(f"Length mismatch: {len(baseline)} != {len(current)}")
                passed = False
            else:
                for i, (b, c) in enumerate(zip(baseline, current, strict=True)):
                    if isinstance(b, (int, float)) and isinstance(c, (int, float)):
                        if abs(b - c) > tolerance:
                            differences.append(f"Element {i}: |{b} - {c}| > {tolerance}")
                            passed = False
        else:
            # Fallback to exact
            if baseline != current:
                differences.append("Fuzzy comparison not applicable, exact failed")
                passed = False

        return differences, passed, confidence

    def _compare_statistical(
        self, baseline: Any, current: Any, tolerance: float
    ) -> tuple[list[str], bool, float]:
        """Compare outputs using statistical measures.

        Args:
            baseline: Baseline output
            current: Current output
            tolerance: Tolerance for normalized RMSE

        Returns:
            Tuple of (differences, passed, confidence)
        """
        differences: list[str] = []
        passed = True
        confidence = 1.0

        if not (isinstance(baseline, np.ndarray) and isinstance(current, np.ndarray)):
            differences.append("Statistical comparison requires numpy arrays")
            return differences, False, 1.0

        if baseline.shape != current.shape:
            differences.append(f"Shape mismatch: {baseline.shape} != {current.shape}")
            return differences, False, 1.0

        # Use normalized RMSE
        rmse = np.sqrt(np.mean((baseline - current) ** 2))
        baseline_range = np.ptp(baseline) if np.ptp(baseline) > 0 else 1.0
        normalized_rmse = rmse / baseline_range

        if normalized_rmse > tolerance:
            differences.append(
                f"Statistical difference: RMSE={rmse:.6f}, "
                f"normalized={normalized_rmse:.6f} > {tolerance}"
            )
            passed = False
            confidence = 1.0 - min(1.0, normalized_rmse / tolerance)

        return differences, passed, confidence

    def _compare_field_by_field(
        self, baseline: Any, current: Any, tolerance: float
    ) -> tuple[list[str], bool, float]:
        """Compare dictionary outputs field by field.

        Args:
            baseline: Baseline output
            current: Current output
            tolerance: Tolerance for numeric field differences

        Returns:
            Tuple of (differences, passed, confidence)
        """
        differences: list[str] = []
        passed = True

        if not (isinstance(baseline, dict) and isinstance(current, dict)):
            differences.append("Field-by-field comparison requires dictionaries")
            return differences, False, 1.0

        all_keys = set(baseline.keys()) | set(current.keys())
        for key in all_keys:
            if key not in baseline:
                differences.append(f"Field '{key}' missing in baseline")
                passed = False
            elif key not in current:
                differences.append(f"Field '{key}' missing in current")
                passed = False
            else:
                b_val = baseline[key]
                c_val = current[key]
                if isinstance(b_val, (int, float)) and isinstance(c_val, (int, float)):
                    if abs(b_val - c_val) > tolerance:
                        differences.append(f"Field '{key}': |{b_val} - {c_val}| > {tolerance}")
                        passed = False
                elif b_val != c_val:
                    differences.append(f"Field '{key}': {b_val} != {c_val}")
                    passed = False

        return differences, passed, 1.0

    def _compare_outputs(
        self, baseline: Any, current: Any, mode: ComparisonMode, tolerance: float
    ) -> tuple[list[str], bool, float]:
        """Compare baseline and current outputs.

        Args:
            baseline: Baseline output.
            current: Current output.
            mode: Comparison mode.
            tolerance: Tolerance for fuzzy/statistical comparisons.

        Returns:
            Tuple of (differences, passed, confidence).
        """
        if mode == ComparisonMode.EXACT:
            return self._compare_exact(baseline, current)
        elif mode == ComparisonMode.FUZZY:
            return self._compare_fuzzy(baseline, current, tolerance)
        elif mode == ComparisonMode.STATISTICAL:
            return self._compare_statistical(baseline, current, tolerance)
        else:  # FIELD_BY_FIELD - all enum values covered
            return self._compare_field_by_field(baseline, current, tolerance)

    def _save_baseline(self, test_name: str, output: Any) -> None:
        """Save baseline to disk.

        Args:
            test_name: Name of the test.
            output: Baseline output to save.
        """
        baseline_file = self.baseline_dir / f"{test_name}.json"
        data = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "output": self._serialize_for_storage(output),
        }
        baseline_file.write_text(json.dumps(data, indent=2))

    def _load_baseline(self, test_name: str) -> None:
        """Load baseline from disk.

        Args:
            test_name: Name of the test.
        """
        baseline_file = self.baseline_dir / f"{test_name}.json"
        if baseline_file.exists():
            data = json.loads(baseline_file.read_text())
            self.baselines[test_name] = self._deserialize_from_storage(data["output"])

    def _serialize_for_storage(self, obj: Any) -> Any:
        """Serialize object for JSON storage.

        Args:
            obj: Object to serialize.

        Returns:
            JSON-serializable representation.
        """
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, bytes):
            return {"__type__": "bytes", "value": obj.hex()}
        if isinstance(obj, np.ndarray):
            return {"__type__": "ndarray", "value": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, dict):
            return {k: self._serialize_for_storage(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._serialize_for_storage(item) for item in obj]
        # Fallback: convert to string
        return str(obj)

    def _deserialize_from_storage(self, obj: Any) -> Any:
        """Deserialize object from JSON storage.

        Args:
            obj: Serialized object.

        Returns:
            Original object type.
        """
        if isinstance(obj, dict):
            if "__type__" in obj:
                if obj["__type__"] == "bytes":
                    return bytes.fromhex(obj["value"])
                if obj["__type__"] == "ndarray":
                    return np.array(obj["value"], dtype=obj["dtype"])
            return {k: self._deserialize_from_storage(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._deserialize_from_storage(item) for item in obj]
        return obj

    def _track_metrics(self, test_name: str, metrics: dict[str, float]) -> None:
        """Track metrics over time for trend analysis.

        Args:
            test_name: Name of the test.
            metrics: Current metrics.
        """
        if test_name not in self.metrics_history:
            self.metrics_history[test_name] = []
        self.metrics_history[test_name].append(metrics)

    def get_baseline_hash(self, test_name: str) -> str:
        """Get hash of baseline for version tracking.

        Args:
            test_name: Name of the test.

        Returns:
            SHA256 hash of baseline.

        Example:
            >>> baseline_hash = suite.get_baseline_hash("uart_decode")
        """
        if test_name not in self.baselines:
            self._load_baseline(test_name)

        if test_name not in self.baselines:
            return ""

        baseline_json = json.dumps(
            self._serialize_for_storage(self.baselines[test_name]), sort_keys=True
        )
        return hashlib.sha256(baseline_json.encode()).hexdigest()
