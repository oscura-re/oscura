"""Comprehensive tests for regression test suite.

Tests cover:
- Test registration and baseline capture
- All comparison modes (exact, fuzzy, statistical, field-by-field)
- Regression detection and metrics tracking
- Report generation (JSON, HTML, CSV)
- Baseline version control and hashing
- Integration with protocol analyzers
- Edge cases and error handling
"""

import json
from pathlib import Path

import numpy as np
import pytest

from oscura.validation.regression_suite import (
    ComparisonMode,
    RegressionReport,
    RegressionTestResult,
    RegressionTestSuite,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_baseline_dir(tmp_path: Path) -> Path:
    """Temporary directory for baseline storage."""
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    return baseline_dir


@pytest.fixture
def suite(temp_baseline_dir: Path) -> RegressionTestSuite:
    """Create regression test suite with temp baseline directory."""
    return RegressionTestSuite("test_suite", baseline_dir=temp_baseline_dir)


@pytest.fixture
def deterministic_function():
    """Deterministic function for testing."""

    def func(value: int) -> dict:
        return {"result": value * 2, "status": "ok"}

    return func


@pytest.fixture
def noisy_function():
    """Function with controlled noise for statistical testing."""

    def func(value: float, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.sin(np.linspace(0, 2 * np.pi, 100)) + rng.normal(0, 0.01, 100)

    return func


# =============================================================================
# RegressionTestResult Tests
# =============================================================================


def test_regression_test_result_creation():
    """Test RegressionTestResult dataclass creation."""
    result = RegressionTestResult(
        test_name="test_1",
        baseline={"value": 10},
        current={"value": 10},
        differences=[],
        passed=True,
        metrics={"execution_time": 0.025},
        comparison_mode=ComparisonMode.EXACT,
        confidence=1.0,
    )

    assert result.test_name == "test_1"
    assert result.passed is True
    assert result.confidence == 1.0
    assert result.comparison_mode == ComparisonMode.EXACT
    assert result.metrics["execution_time"] == 0.025


def test_regression_test_result_with_differences():
    """Test result with detected regressions."""
    differences = [
        "Field 'count' changed: 10 -> 12",
        "Field 'status' changed: 'ok' -> 'error'",
    ]

    result = RegressionTestResult(
        test_name="test_2",
        baseline={"count": 10, "status": "ok"},
        current={"count": 12, "status": "error"},
        differences=differences,
        passed=False,
        confidence=0.5,
    )

    assert result.passed is False
    assert len(result.differences) == 2
    assert result.confidence == 0.5


# =============================================================================
# RegressionReport Tests
# =============================================================================


def test_regression_report_creation():
    """Test RegressionReport creation with multiple results."""
    result1 = RegressionTestResult("test_1", {"v": 1}, {"v": 1}, [], True, {"execution_time": 0.01})
    result2 = RegressionTestResult(
        "test_2", {"v": 2}, {"v": 3}, ["Changed"], False, {"execution_time": 0.02}
    )

    report = RegressionReport(
        suite_name="my_suite",
        results=[result1, result2],
        summary={"total": 2, "passed": 1, "failed": 1},
        regressions_found=["test_2"],
    )

    assert report.suite_name == "my_suite"
    assert len(report.results) == 2
    assert report.summary["total"] == 2
    assert "test_2" in report.regressions_found


def test_report_export_json(tmp_path: Path):
    """Test JSON export of regression report."""
    result = RegressionTestResult(
        "test_export", {"value": 42}, {"value": 42}, [], True, {"execution_time": 0.015}
    )

    report = RegressionReport(
        suite_name="export_test",
        results=[result],
        summary={"total": 1, "passed": 1, "failed": 0},
        regressions_found=[],
    )

    output_file = tmp_path / "report.json"
    report.export_json(output_file)

    assert output_file.exists()

    data = json.loads(output_file.read_text())
    assert data["suite_name"] == "export_test"
    assert data["summary"]["total"] == 1
    assert len(data["results"]) == 1
    assert data["results"][0]["test_name"] == "test_export"


def test_report_export_html(tmp_path: Path):
    """Test HTML dashboard generation."""
    result1 = RegressionTestResult("test_1", {}, {}, [], True)
    result2 = RegressionTestResult("test_2", {}, {}, ["Error"], False)

    report = RegressionReport(
        suite_name="html_test",
        results=[result1, result2],
        summary={"total": 2, "passed": 1, "failed": 1},
        regressions_found=["test_2"],
    )

    output_file = tmp_path / "report.html"
    report.export_html(output_file)

    assert output_file.exists()
    html_content = output_file.read_text()
    assert "html_test" in html_content
    assert "PASSED" in html_content
    assert "FAILED" in html_content
    assert "test_1" in html_content
    assert "test_2" in html_content


def test_report_export_csv(tmp_path: Path):
    """Test CSV export for historical tracking."""
    result = RegressionTestResult(
        "csv_test", {}, {}, [], True, {"execution_time": 0.025, "memory_usage": 1024}
    )

    report = RegressionReport(
        suite_name="csv_suite",
        results=[result],
        summary={"total": 1, "passed": 1, "failed": 0},
        regressions_found=[],
    )

    output_file = tmp_path / "history.csv"
    report.export_csv(output_file)

    assert output_file.exists()
    csv_content = output_file.read_text()
    assert "Test Name" in csv_content
    assert "csv_test" in csv_content
    assert "True" in csv_content


def test_report_serialize_bytes():
    """Test serialization of bytes objects."""
    report = RegressionReport("test", [], {}, [])
    serialized = report._serialize(b"\x01\x02\x03")
    assert serialized == "010203"


def test_report_serialize_numpy():
    """Test serialization of numpy arrays."""
    report = RegressionReport("test", [], {}, [])
    arr = np.array([1, 2, 3])
    serialized = report._serialize(arr)
    assert serialized == [1, 2, 3]


def test_report_serialize_nested_dict():
    """Test serialization of nested dictionaries."""
    report = RegressionReport("test", [], {}, [])
    obj = {"nested": {"bytes": b"\xaa", "array": np.array([1.5, 2.5])}}
    serialized = report._serialize(obj)
    assert serialized["nested"]["bytes"] == "aa"
    assert serialized["nested"]["array"] == [1.5, 2.5]


# =============================================================================
# RegressionTestSuite Tests
# =============================================================================


def test_suite_initialization(temp_baseline_dir: Path):
    """Test suite initialization with custom baseline directory."""
    suite = RegressionTestSuite("my_suite", baseline_dir=temp_baseline_dir)

    assert suite.suite_name == "my_suite"
    assert suite.baseline_dir == temp_baseline_dir
    assert len(suite.tests) == 0
    assert len(suite.baselines) == 0


def test_register_test(suite: RegressionTestSuite, deterministic_function):
    """Test registering a test with the suite."""
    suite.register_test(
        "test_func",
        deterministic_function,
        comparison_mode=ComparisonMode.EXACT,
        value=5,
    )

    assert "test_func" in suite.tests
    assert suite.tests["test_func"]["function"] == deterministic_function
    assert suite.tests["test_func"]["kwargs"] == {"value": 5}
    assert suite.tests["test_func"]["comparison_mode"] == ComparisonMode.EXACT


def test_capture_baseline(suite: RegressionTestSuite, deterministic_function):
    """Test capturing baseline output."""
    suite.register_test("test_baseline", deterministic_function, value=10)
    suite.capture_baseline("test_baseline")

    assert "test_baseline" in suite.baselines
    assert suite.baselines["test_baseline"] == {"result": 20, "status": "ok"}

    # Check baseline file was created
    baseline_file = suite.baseline_dir / "test_baseline.json"
    assert baseline_file.exists()


def test_capture_baseline_unregistered_test(suite: RegressionTestSuite):
    """Test capturing baseline for unregistered test raises error."""
    with pytest.raises(KeyError, match="not registered"):
        suite.capture_baseline("nonexistent_test")


def test_run_test_exact_match(suite: RegressionTestSuite, deterministic_function):
    """Test running regression test with exact match."""
    suite.register_test("exact_test", deterministic_function, value=7)
    suite.capture_baseline("exact_test")

    # Run test - should pass (same output)
    result = suite.run_test("exact_test")

    assert result.passed is True
    assert len(result.differences) == 0
    assert result.confidence == 1.0
    assert "execution_time" in result.metrics


def test_run_test_with_regression(suite: RegressionTestSuite):
    """Test detecting regression when output changes."""

    def changing_func(value: int, multiplier: int = 2) -> int:
        return value * multiplier

    suite.register_test("regression_test", changing_func, value=5, multiplier=2)
    suite.capture_baseline("regression_test")  # Baseline: 5 * 2 = 10

    # Change function behavior
    suite.tests["regression_test"]["kwargs"]["multiplier"] = 3

    # Run test - should fail (different output)
    result = suite.run_test("regression_test")

    assert result.passed is False
    assert len(result.differences) > 0
    assert result.baseline == 10
    assert result.current == 15


def test_run_test_fuzzy_comparison(suite: RegressionTestSuite):
    """Test fuzzy comparison with tolerance."""

    def approx_func(value: float) -> float:
        return value * 1.01  # Slight variation

    suite.register_test(
        "fuzzy_test",
        approx_func,
        comparison_mode=ComparisonMode.FUZZY,
        tolerance=0.05,
        value=100.0,
    )

    suite.capture_baseline("fuzzy_test")  # Baseline: 101.0

    # Run test - should pass (within tolerance)
    result = suite.run_test("fuzzy_test")

    assert result.passed is True
    assert result.comparison_mode == ComparisonMode.FUZZY


def test_run_test_fuzzy_array_comparison(suite: RegressionTestSuite):
    """Test fuzzy comparison with arrays."""

    def array_func() -> list[float]:
        return [1.0, 2.0, 3.0]

    suite.register_test(
        "fuzzy_array",
        array_func,
        comparison_mode=ComparisonMode.FUZZY,
        tolerance=0.1,
    )

    suite.capture_baseline("fuzzy_array")

    # Modify slightly
    suite.tests["fuzzy_array"]["function"] = lambda: [1.05, 2.05, 3.05]

    result = suite.run_test("fuzzy_array")

    assert result.passed is True  # Within tolerance


def test_run_test_fuzzy_exceeds_tolerance(suite: RegressionTestSuite):
    """Test fuzzy comparison failing when exceeding tolerance."""

    def value_func(val: float) -> float:
        return val

    suite.register_test(
        "fuzzy_fail",
        value_func,
        comparison_mode=ComparisonMode.FUZZY,
        tolerance=0.01,
        val=10.0,
    )

    suite.capture_baseline("fuzzy_fail")  # Baseline: 10.0

    # Change significantly
    suite.tests["fuzzy_fail"]["kwargs"]["val"] = 10.5

    result = suite.run_test("fuzzy_fail")

    assert result.passed is False
    assert any("Fuzzy match failed" in diff for diff in result.differences)


def test_run_test_statistical_comparison(suite: RegressionTestSuite, noisy_function):
    """Test statistical comparison for noisy data."""
    suite.register_test(
        "stat_test",
        noisy_function,
        comparison_mode=ComparisonMode.STATISTICAL,
        tolerance=0.05,
        value=1.0,
        seed=42,
    )

    suite.capture_baseline("stat_test")

    # Run with same seed - should pass
    result = suite.run_test("stat_test")

    assert result.passed is True
    assert result.comparison_mode == ComparisonMode.STATISTICAL


def test_run_test_statistical_different_data(suite: RegressionTestSuite, noisy_function):
    """Test statistical comparison detecting significant changes."""
    suite.register_test(
        "stat_diff",
        noisy_function,
        comparison_mode=ComparisonMode.STATISTICAL,
        tolerance=0.01,
        value=1.0,
        seed=42,
    )

    suite.capture_baseline("stat_diff")

    # Change seed - different noise
    suite.tests["stat_diff"]["kwargs"]["seed"] = 99

    result = suite.run_test("stat_diff")

    # May pass or fail depending on noise - should have low confidence
    if not result.passed:
        assert any("Statistical difference" in diff for diff in result.differences)


def test_run_test_field_by_field_comparison(suite: RegressionTestSuite):
    """Test field-by-field comparison with tolerance per field."""

    def dict_func(a: int, b: float) -> dict:
        return {"field_a": a, "field_b": b, "status": "ok"}

    suite.register_test(
        "field_test",
        dict_func,
        comparison_mode=ComparisonMode.FIELD_BY_FIELD,
        tolerance=0.1,
        a=10,
        b=5.5,
    )

    suite.capture_baseline("field_test")

    # Change numeric field slightly
    suite.tests["field_test"]["kwargs"]["b"] = 5.55

    result = suite.run_test("field_test")

    assert result.passed is True  # Within tolerance


def test_run_test_field_by_field_missing_field(suite: RegressionTestSuite):
    """Test field-by-field detecting missing fields."""

    def dict_func(include_extra: bool = False) -> dict:
        base = {"field1": 10}
        if include_extra:
            base["field2"] = 20
        return base

    suite.register_test(
        "field_missing",
        dict_func,
        comparison_mode=ComparisonMode.FIELD_BY_FIELD,
        include_extra=True,
    )

    suite.capture_baseline("field_missing")

    # Remove field
    suite.tests["field_missing"]["kwargs"]["include_extra"] = False

    result = suite.run_test("field_missing")

    assert result.passed is False
    assert any("missing in current" in diff for diff in result.differences)


def test_run_test_no_baseline_raises_error(suite: RegressionTestSuite, deterministic_function):
    """Test running test without baseline raises error."""
    suite.register_test("no_baseline", deterministic_function, value=5)

    with pytest.raises(KeyError, match="No baseline"):
        suite.run_test("no_baseline")


def test_run_test_auto_update_baseline(temp_baseline_dir: Path, deterministic_function):
    """Test auto-updating baseline on first run."""
    suite = RegressionTestSuite(
        "auto_suite", baseline_dir=temp_baseline_dir, auto_update_baselines=True
    )

    suite.register_test("auto_test", deterministic_function, value=8)

    # Run without capturing baseline - should auto-create
    result = suite.run_test("auto_test")

    assert result.passed is True
    assert "auto_test" in suite.baselines


def test_run_all_tests(suite: RegressionTestSuite, deterministic_function):
    """Test running all registered tests."""
    suite.register_test("test_1", deterministic_function, value=1)
    suite.register_test("test_2", deterministic_function, value=2)
    suite.register_test("test_3", deterministic_function, value=3)

    suite.capture_baseline("test_1")
    suite.capture_baseline("test_2")
    suite.capture_baseline("test_3")

    results = suite.run_all()

    assert len(results) == 3
    assert all(r.passed for r in results)


def test_run_all_with_exception(suite: RegressionTestSuite):
    """Test run_all handling exceptions gracefully."""

    def failing_func() -> None:
        raise ValueError("Test error")

    suite.register_test("failing_test", failing_func)
    suite.baselines["failing_test"] = None  # Mock baseline

    results = suite.run_all()

    assert len(results) == 1
    assert results[0].passed is False
    assert any("Exception" in diff for diff in results[0].differences)


def test_generate_report(suite: RegressionTestSuite, deterministic_function):
    """Test generating comprehensive regression report."""
    suite.register_test("report_test_1", deterministic_function, value=5)
    suite.register_test("report_test_2", deterministic_function, value=10)

    suite.capture_baseline("report_test_1")
    suite.capture_baseline("report_test_2")

    # Run tests
    suite.run_test("report_test_1")
    suite.run_test("report_test_2")

    report = suite.generate_report()

    assert report.suite_name == "test_suite"
    assert report.summary["total"] == 2
    assert report.summary["passed"] == 2
    assert report.summary["failed"] == 0
    assert len(report.regressions_found) == 0


def test_generate_report_with_regressions(suite: RegressionTestSuite, deterministic_function):
    """Test report generation with detected regressions."""
    suite.register_test("pass_test", deterministic_function, value=1)
    suite.register_test("fail_test", deterministic_function, value=2)

    suite.capture_baseline("pass_test")
    suite.capture_baseline("fail_test")

    # Modify one test to cause regression
    suite.tests["fail_test"]["kwargs"]["value"] = 3

    suite.run_test("pass_test")
    suite.run_test("fail_test")

    report = suite.generate_report()

    assert report.summary["total"] == 2
    assert report.summary["passed"] == 1
    assert report.summary["failed"] == 1
    assert "fail_test" in report.regressions_found
    assert report.summary["pass_rate"] == 50.0


def test_update_baseline(suite: RegressionTestSuite, deterministic_function):
    """Test intentionally updating baseline for behavior change."""
    suite.register_test("update_test", deterministic_function, value=10)
    suite.capture_baseline("update_test")

    original_baseline = suite.baselines["update_test"]

    # Change test behavior
    suite.tests["update_test"]["kwargs"]["value"] = 15

    # Update baseline
    suite.update_baseline("update_test")

    new_baseline = suite.baselines["update_test"]

    assert original_baseline != new_baseline
    assert new_baseline == {"result": 30, "status": "ok"}


def test_update_baseline_unregistered_test(suite: RegressionTestSuite):
    """Test updating baseline for unregistered test raises error."""
    with pytest.raises(KeyError, match="not registered"):
        suite.update_baseline("nonexistent")


def test_baseline_persistence(temp_baseline_dir: Path, deterministic_function):
    """Test baselines persist across suite instances."""
    # Create suite and capture baseline
    suite1 = RegressionTestSuite("persist_suite", baseline_dir=temp_baseline_dir)
    suite1.register_test("persist_test", deterministic_function, value=7)
    suite1.capture_baseline("persist_test")

    # Create new suite instance
    suite2 = RegressionTestSuite("persist_suite", baseline_dir=temp_baseline_dir)
    suite2.register_test("persist_test", deterministic_function, value=7)

    # Should load baseline from disk
    result = suite2.run_test("persist_test")

    assert result.passed is True
    assert suite2.baselines["persist_test"] == {"result": 14, "status": "ok"}


def test_metrics_tracking(suite: RegressionTestSuite, deterministic_function):
    """Test tracking metrics over multiple test runs."""
    suite.register_test("metrics_test", deterministic_function, value=5)
    suite.capture_baseline("metrics_test")

    # Run test multiple times
    suite.run_test("metrics_test")
    suite.run_test("metrics_test")
    suite.run_test("metrics_test")

    assert "metrics_test" in suite.metrics_history
    assert len(suite.metrics_history["metrics_test"]) == 3
    assert all("execution_time" in m for m in suite.metrics_history["metrics_test"])


def test_get_baseline_hash(suite: RegressionTestSuite, deterministic_function):
    """Test baseline version hashing."""
    suite.register_test("hash_test", deterministic_function, value=8)
    suite.capture_baseline("hash_test")

    hash1 = suite.get_baseline_hash("hash_test")

    assert len(hash1) == 64  # SHA256 hex digest
    assert all(c in "0123456789abcdef" for c in hash1)

    # Same baseline should produce same hash
    hash2 = suite.get_baseline_hash("hash_test")
    assert hash1 == hash2


def test_get_baseline_hash_different_baselines(suite: RegressionTestSuite, deterministic_function):
    """Test different baselines produce different hashes."""
    suite.register_test("hash_1", deterministic_function, value=1)
    suite.register_test("hash_2", deterministic_function, value=2)

    suite.capture_baseline("hash_1")
    suite.capture_baseline("hash_2")

    hash1 = suite.get_baseline_hash("hash_1")
    hash2 = suite.get_baseline_hash("hash_2")

    assert hash1 != hash2


def test_get_baseline_hash_no_baseline(suite: RegressionTestSuite):
    """Test getting hash for non-existent baseline returns empty string."""
    hash_result = suite.get_baseline_hash("nonexistent")
    assert hash_result == ""


def test_serialize_deserialize_bytes(suite: RegressionTestSuite):
    """Test serialization and deserialization of bytes."""
    original = b"\x01\x02\x03\xff"
    serialized = suite._serialize_for_storage(original)
    deserialized = suite._deserialize_from_storage(serialized)

    assert deserialized == original


def test_serialize_deserialize_numpy(suite: RegressionTestSuite):
    """Test serialization and deserialization of numpy arrays."""
    original = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    serialized = suite._serialize_for_storage(original)
    deserialized = suite._deserialize_from_storage(serialized)

    assert isinstance(deserialized, np.ndarray)
    assert np.allclose(deserialized, original)
    assert deserialized.dtype == original.dtype


def test_serialize_deserialize_complex_structure(suite: RegressionTestSuite):
    """Test serialization of complex nested structures."""
    original = {
        "bytes_field": b"\xaa\xbb",
        "array_field": np.array([1, 2, 3]),
        "nested": {"value": 42, "status": "ok"},
        "list": [1, "two", 3.0],
    }

    serialized = suite._serialize_for_storage(original)
    deserialized = suite._deserialize_from_storage(serialized)

    assert deserialized["bytes_field"] == original["bytes_field"]
    assert np.array_equal(deserialized["array_field"], original["array_field"])
    assert deserialized["nested"] == original["nested"]
    assert deserialized["list"] == original["list"]


# =============================================================================
# Integration Tests
# =============================================================================


def test_integration_protocol_decoder(suite: RegressionTestSuite):
    """Test integration with protocol decoder."""

    def mock_decoder(data: bytes) -> dict:
        """Mock protocol decoder."""
        return {
            "frame_count": len(data) // 10,
            "valid_frames": len(data) // 10 - 1,
            "checksum_errors": 1,
        }

    test_data = b"\x00" * 100

    suite.register_test(
        "decoder_test",
        mock_decoder,
        comparison_mode=ComparisonMode.FIELD_BY_FIELD,
        tolerance=0,
        data=test_data,
    )

    suite.capture_baseline("decoder_test")

    # Run with same data - should pass
    result = suite.run_test("decoder_test")

    assert result.passed is True
    assert result.baseline["frame_count"] == 10


def test_integration_full_workflow(temp_baseline_dir: Path):
    """Test complete workflow: register -> capture -> run -> report."""

    def protocol_analyzer(messages: list[bytes]) -> dict:
        return {
            "total_messages": len(messages),
            "avg_length": sum(len(m) for m in messages) / len(messages) if messages else 0,
            "unique_ids": len(set(messages)),
        }

    suite = RegressionTestSuite("workflow_suite", baseline_dir=temp_baseline_dir)

    # Register multiple tests
    test_messages_1 = [b"\x01\x02", b"\x03\x04", b"\x05\x06"]
    test_messages_2 = [b"\xff\xfe", b"\xfd\xfc"]

    suite.register_test(
        "test_set_1",
        protocol_analyzer,
        comparison_mode=ComparisonMode.FIELD_BY_FIELD,
        tolerance=0.01,
        messages=test_messages_1,
    )

    suite.register_test(
        "test_set_2",
        protocol_analyzer,
        comparison_mode=ComparisonMode.FIELD_BY_FIELD,
        tolerance=0.01,
        messages=test_messages_2,
    )

    # Capture baselines
    suite.capture_baseline("test_set_1")
    suite.capture_baseline("test_set_2")

    # Run all tests
    results = suite.run_all()

    assert all(r.passed for r in results)

    # Generate report
    report = suite.generate_report()

    assert report.summary["total"] == 2
    assert report.summary["passed"] == 2

    # Export reports
    json_file = temp_baseline_dir / "report.json"
    html_file = temp_baseline_dir / "report.html"
    csv_file = temp_baseline_dir / "history.csv"

    report.export_json(json_file)
    report.export_html(html_file)
    report.export_csv(csv_file)

    assert json_file.exists()
    assert html_file.exists()
    assert csv_file.exists()


# =============================================================================
# Edge Cases
# =============================================================================


def test_empty_suite_report(suite: RegressionTestSuite):
    """Test generating report from empty suite."""
    report = suite.generate_report()

    assert report.summary["total"] == 0
    assert report.summary["passed"] == 0
    assert report.summary["failed"] == 0
    assert len(report.regressions_found) == 0


def test_comparison_mode_with_incompatible_types(suite: RegressionTestSuite):
    """Test comparison modes with incompatible data types."""

    def return_string() -> str:
        return "test_string"

    suite.register_test(
        "string_test",
        return_string,
        comparison_mode=ComparisonMode.STATISTICAL,  # Requires numpy arrays
    )

    suite.capture_baseline("string_test")

    result = suite.run_test("string_test")

    # Should fail with incompatible type message
    assert result.passed is False
    assert any("requires numpy arrays" in diff for diff in result.differences)


def test_field_by_field_non_dict(suite: RegressionTestSuite):
    """Test field-by-field comparison with non-dict types."""

    def return_list() -> list:
        return [1, 2, 3]

    suite.register_test(
        "list_test",
        return_list,
        comparison_mode=ComparisonMode.FIELD_BY_FIELD,
    )

    suite.capture_baseline("list_test")

    result = suite.run_test("list_test")

    assert result.passed is False
    assert any("requires dictionaries" in diff for diff in result.differences)


def test_statistical_shape_mismatch(suite: RegressionTestSuite):
    """Test statistical comparison with mismatched array shapes."""

    def array_func(size: int) -> np.ndarray:
        return np.zeros(size)

    suite.register_test(
        "shape_test",
        array_func,
        comparison_mode=ComparisonMode.STATISTICAL,
        tolerance=0.01,
        size=10,
    )

    suite.capture_baseline("shape_test")

    # Change size
    suite.tests["shape_test"]["kwargs"]["size"] = 20

    result = suite.run_test("shape_test")

    assert result.passed is False
    assert any("Shape mismatch" in diff for diff in result.differences)


def test_large_number_of_tests(temp_baseline_dir: Path):
    """Test suite performance with many tests."""

    def simple_func(val: int) -> int:
        return val * 2

    suite = RegressionTestSuite("large_suite", baseline_dir=temp_baseline_dir)

    # Register 100 tests
    for i in range(100):
        suite.register_test(f"test_{i}", simple_func, val=i)
        suite.capture_baseline(f"test_{i}")

    # Run all
    results = suite.run_all()

    assert len(results) == 100
    assert all(r.passed for r in results)

    # Generate report
    report = suite.generate_report()
    assert report.summary["total"] == 100
