"""Comprehensive unit tests for batch analysis workflow.

Requirements tested:
- batch_analyze with parallel and sequential execution
- Error handling and progress callbacks
- DataFrame output formatting
- Thread vs process execution modes

Coverage target: 90%+ of src/oscura/workflows/batch/analyze.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from oscura.workflows.batch.analyze import (
    _build_result_dataframe,
    _create_wrapped_analysis,
    _execute_batch_analysis,
    _execute_parallel,
    _execute_sequential,
    batch_analyze,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestBatchAnalyzeBasic:
    """Test basic batch_analyze functionality."""

    def test_empty_files_list(self):
        """Test batch_analyze with empty files list."""
        result = batch_analyze([], lambda f: {"result": 1})
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_single_file_sequential(self, tmp_path: Path):
        """Test batch_analyze with single file in sequential mode."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"value": 42, "name": "test"}

        result = batch_analyze([test_file], analysis_fn, parallel=False)

        assert len(result) == 1
        assert result.iloc[0]["file"] == str(test_file)
        assert result.iloc[0]["value"] == 42
        assert result.iloc[0]["name"] == "test"
        assert result.iloc[0]["error"] is None

    def test_multiple_files_sequential(self, tmp_path: Path):
        """Test batch_analyze with multiple files sequentially."""
        files = [tmp_path / f"file_{i}.txt" for i in range(5)]
        for f in files:
            f.write_text("data")

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"size": len(Path(filepath).read_text())}

        result = batch_analyze(files, analysis_fn, parallel=False)

        assert len(result) == 5
        assert all(result["size"] == 4)
        assert all(result["error"].isna())

    def test_column_ordering(self, tmp_path: Path):
        """Test that DataFrame has correct column ordering (file first, error last)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"metric_a": 1, "metric_b": 2}

        result = batch_analyze([test_file], analysis_fn, parallel=False)

        columns = result.columns.tolist()
        assert columns[0] == "file"
        assert columns[-1] == "error"
        assert "metric_a" in columns
        assert "metric_b" in columns


@pytest.mark.unit
class TestBatchAnalyzeParallel:
    """Test parallel execution modes."""

    def test_parallel_process_execution(self, tmp_path: Path):
        """Test batch_analyze with parallel execution."""
        files = [tmp_path / f"file_{i}.txt" for i in range(3)]
        for i, f in enumerate(files):
            f.write_text(f"content_{i}")

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"length": len(Path(filepath).read_text())}

        # Use threads to avoid pickling issues with local functions
        result = batch_analyze(files, analysis_fn, parallel=True, workers=2, use_threads=True)

        assert len(result) == 3
        assert all(result["error"].isna())
        assert all(result["length"] == 9)

    def test_parallel_thread_execution(self, tmp_path: Path):
        """Test batch_analyze with parallel thread execution."""
        files = [tmp_path / f"file_{i}.txt" for i in range(3)]
        for i, f in enumerate(files):
            f.write_text(f"data_{i}")

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"chars": len(Path(filepath).read_text())}

        result = batch_analyze(files, analysis_fn, parallel=True, workers=2, use_threads=True)

        assert len(result) == 3
        assert all(result["error"].isna())

    def test_parallel_with_specific_worker_count(self, tmp_path: Path):
        """Test batch_analyze with specified worker count."""
        files = [tmp_path / f"file_{i}.txt" for i in range(4)]
        for f in files:
            f.write_text("test")

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"status": "ok"}

        # Should not raise with workers=1
        result = batch_analyze(files, analysis_fn, parallel=True, workers=1, use_threads=True)
        assert len(result) == 4


@pytest.mark.unit
class TestBatchAnalyzeErrorHandling:
    """Test error handling in batch analysis."""

    def test_analysis_function_exception(self, tmp_path: Path):
        """Test that exceptions in analysis function are captured."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        def failing_analysis(filepath: str | Path) -> dict[str, Any]:
            raise ValueError("Analysis failed")

        result = batch_analyze([test_file], failing_analysis, parallel=False)

        assert len(result) == 1
        assert result.iloc[0]["error"] == "Analysis failed"
        assert result.iloc[0]["file"] == str(test_file)

    def test_mixed_success_and_failure(self, tmp_path: Path):
        """Test batch with mix of successful and failed analyses."""
        files = [tmp_path / f"file_{i}.txt" for i in range(3)]
        for f in files:
            f.write_text("data")

        def conditional_analysis(filepath: str | Path) -> dict[str, Any]:
            if "file_1" in str(filepath):
                raise RuntimeError("File 1 failed")
            return {"result": "success"}

        result = batch_analyze(files, conditional_analysis, parallel=False)

        assert len(result) == 3
        assert result.iloc[0]["error"] is None
        assert "failed" in result.iloc[1]["error"].lower()
        assert result.iloc[2]["error"] is None

    def test_parallel_execution_error_capture(self, tmp_path: Path):
        """Test that parallel execution captures errors correctly."""
        files = [tmp_path / f"file_{i}.txt" for i in range(3)]
        for f in files:
            f.write_text("data")

        def failing_on_second(filepath: str | Path) -> dict[str, Any]:
            if "file_1" in str(filepath):
                raise ValueError("Second file error")
            return {"value": 1}

        result = batch_analyze(files, failing_on_second, parallel=True, workers=2, use_threads=True)

        assert len(result) == 3
        errors = result["error"].dropna()
        assert len(errors) == 1
        assert "error" in errors.iloc[0].lower()

    def test_non_dict_result_handling(self, tmp_path: Path):
        """Test analysis function returning non-dict is wrapped."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        def returns_int(filepath: str | Path) -> int:
            return 42  # type: ignore[return-value]

        result = batch_analyze([test_file], returns_int, parallel=False)  # type: ignore[arg-type]

        assert len(result) == 1
        assert result.iloc[0]["result"] == 42
        assert result.iloc[0]["file"] == str(test_file)


@pytest.mark.unit
class TestBatchAnalyzeProgressCallback:
    """Test progress callback functionality."""

    def test_progress_callback_invoked(self, tmp_path: Path):
        """Test that progress callback is called for each file."""
        files = [tmp_path / f"file_{i}.txt" for i in range(3)]
        for f in files:
            f.write_text("data")

        progress_calls: list[tuple[int, int, str]] = []

        def progress_callback(current: int, total: int, filename: str) -> None:
            progress_calls.append((current, total, filename))

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"status": "ok"}

        batch_analyze(files, analysis_fn, parallel=False, progress_callback=progress_callback)

        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3, str(files[0]))
        assert progress_calls[1] == (2, 3, str(files[1]))
        assert progress_calls[2] == (3, 3, str(files[2]))

    def test_progress_callback_with_parallel(self, tmp_path: Path):
        """Test progress callback works with parallel execution."""
        files = [tmp_path / f"file_{i}.txt" for i in range(3)]
        for f in files:
            f.write_text("data")

        progress_calls: list[tuple[int, int, str]] = []

        def progress_callback(current: int, total: int, filename: str) -> None:
            progress_calls.append((current, total, filename))

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"status": "ok"}

        batch_analyze(
            files,
            analysis_fn,
            parallel=True,
            workers=2,
            use_threads=True,
            progress_callback=progress_callback,
        )

        # Should still get all callbacks (order may vary in parallel)
        assert len(progress_calls) == 3
        assert all(call[1] == 3 for call in progress_calls)


@pytest.mark.unit
class TestBatchAnalyzeConfigPassing:
    """Test passing configuration to analysis function."""

    def test_config_kwargs_passed_to_function(self, tmp_path: Path):
        """Test that additional kwargs are passed to analysis function."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        def analysis_with_config(
            filepath: str | Path, threshold: float = 0.5, mode: str = "fast"
        ) -> dict[str, Any]:
            return {"threshold": threshold, "mode": mode}

        result = batch_analyze(
            [test_file], analysis_with_config, parallel=False, threshold=0.8, mode="accurate"
        )

        assert result.iloc[0]["threshold"] == 0.8
        assert result.iloc[0]["mode"] == "accurate"

    def test_config_with_parallel_execution(self, tmp_path: Path):
        """Test config passing in parallel mode."""
        files = [tmp_path / f"file_{i}.txt" for i in range(2)]
        for f in files:
            f.write_text("data")

        def analysis_with_config(filepath: str | Path, multiplier: int = 1) -> dict[str, Any]:
            return {"result": len(Path(filepath).read_text()) * multiplier}

        result = batch_analyze(
            files, analysis_with_config, parallel=True, use_threads=True, multiplier=3
        )

        assert all(result["result"] == 12)  # 4 chars * 3


@pytest.mark.unit
class TestHelperFunctions:
    """Test internal helper functions."""

    def test_create_wrapped_analysis(self, tmp_path: Path):
        """Test _create_wrapped_analysis creates proper wrapper."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        def analysis_fn(filepath: str | Path, factor: int = 2) -> dict[str, Any]:
            return {"value": factor * 10}

        wrapped = _create_wrapped_analysis(analysis_fn, {"factor": 3})
        result = wrapped(test_file)

        assert result["value"] == 30
        assert result["file"] == str(test_file)
        assert result["error"] is None

    def test_wrapped_analysis_captures_exceptions(self, tmp_path: Path):
        """Test wrapped analysis captures and formats exceptions."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        def failing_fn(filepath: str | Path) -> dict[str, Any]:
            raise RuntimeError("Test error")

        wrapped = _create_wrapped_analysis(failing_fn, {})
        result = wrapped(test_file)

        assert result["file"] == str(test_file)
        assert result["error"] == "Test error"
        assert "result" not in result or result.get("result") is None

    def test_build_result_dataframe(self):
        """Test _build_result_dataframe creates properly ordered DataFrame."""
        results = [
            {"file": "f1.txt", "metric": 1, "error": None},
            {"file": "f2.txt", "metric": 2, "error": "failed"},
        ]

        df = _build_result_dataframe(results)

        assert df.columns[0] == "file"
        assert df.columns[-1] == "error"
        assert len(df) == 2

    def test_build_result_dataframe_empty(self):
        """Test _build_result_dataframe with empty results."""
        df = _build_result_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_execute_batch_analysis_sequential(self, tmp_path: Path):
        """Test _execute_batch_analysis in sequential mode."""
        files = [tmp_path / f"file_{i}.txt" for i in range(2)]
        for f in files:
            f.write_text("data")

        def wrapped_fn(filepath: str | Path) -> dict[str, Any]:
            return {"file": str(filepath), "status": "ok", "error": None}

        results = _execute_batch_analysis(files, wrapped_fn, False, None, False, None)

        assert len(results) == 2
        assert all(r["status"] == "ok" for r in results)

    def test_execute_batch_analysis_parallel_processes(self, tmp_path: Path):
        """Test _execute_batch_analysis with process pool."""
        files = [tmp_path / f"file_{i}.txt" for i in range(2)]
        for f in files:
            f.write_text("data")

        def wrapped_fn(filepath: str | Path) -> dict[str, Any]:
            return {"file": str(filepath), "status": "ok", "error": None}

        results = _execute_batch_analysis(files, wrapped_fn, True, 2, False, None)

        assert len(results) == 2

    def test_execute_sequential(self, tmp_path: Path):
        """Test _execute_sequential helper."""
        files = [tmp_path / f"file_{i}.txt" for i in range(2)]
        for f in files:
            f.write_text("data")

        call_count = [0]

        def wrapped_fn(filepath: str | Path) -> dict[str, Any]:
            call_count[0] += 1
            return {"file": str(filepath), "count": call_count[0], "error": None}

        results = _execute_sequential(files, wrapped_fn, None, 2)

        assert len(results) == 2
        assert results[0]["count"] == 1
        assert results[1]["count"] == 2

    def test_execute_parallel_threads(self, tmp_path: Path):
        """Test _execute_parallel with thread pool."""
        files = [tmp_path / f"file_{i}.txt" for i in range(2)]
        for f in files:
            f.write_text("data")

        def wrapped_fn(filepath: str | Path) -> dict[str, Any]:
            return {"file": str(filepath), "status": "ok", "error": None}

        results = _execute_parallel(files, wrapped_fn, 2, True, None, 2)

        assert len(results) == 2

    def test_execute_parallel_processes(self, tmp_path: Path):
        """Test _execute_parallel with process pool."""
        files = [tmp_path / f"file_{i}.txt" for i in range(2)]
        for f in files:
            f.write_text("data")

        def wrapped_fn(filepath: str | Path) -> dict[str, Any]:
            return {"file": str(filepath), "status": "ok", "error": None}

        results = _execute_parallel(files, wrapped_fn, 2, False, None, 2)

        assert len(results) == 2


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_string_paths_accepted(self, tmp_path: Path):
        """Test that string paths are accepted alongside Path objects."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"path_type": type(filepath).__name__}

        result = batch_analyze([str(test_file)], analysis_fn, parallel=False)

        assert len(result) == 1
        assert result.iloc[0]["file"] == str(test_file)

    def test_mixed_path_types(self, tmp_path: Path):
        """Test mix of string and Path objects in files list."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("data1")
        file2.write_text("data2")

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"length": len(Path(filepath).read_text())}

        result = batch_analyze([str(file1), file2], analysis_fn, parallel=False)

        assert len(result) == 2
        assert result.iloc[0]["length"] == 5
        assert result.iloc[1]["length"] == 5

    def test_large_batch_sequential(self, tmp_path: Path):
        """Test sequential processing of larger batch."""
        files = [tmp_path / f"file_{i}.txt" for i in range(20)]
        for i, f in enumerate(files):
            f.write_text(f"content_{i}")

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"index": int(filepath.stem.split("_")[1])}

        result = batch_analyze(files, analysis_fn, parallel=False)

        assert len(result) == 20
        assert list(result["index"]) == list(range(20))

    def test_analysis_function_returns_empty_dict(self, tmp_path: Path):
        """Test analysis function that returns empty dict."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        def empty_result(filepath: str | Path) -> dict[str, Any]:
            return {}

        result = batch_analyze([test_file], empty_result, parallel=False)

        assert len(result) == 1
        assert result.iloc[0]["file"] == str(test_file)
        assert result.iloc[0]["error"] is None

    def test_analysis_with_none_values(self, tmp_path: Path):
        """Test analysis function returning None values in dict."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        def analysis_with_nones(filepath: str | Path) -> dict[str, Any]:
            return {"value": None, "status": None}

        result = batch_analyze([test_file], analysis_with_nones, parallel=False)

        assert len(result) == 1
        assert pd.isna(result.iloc[0]["value"])
        assert pd.isna(result.iloc[0]["status"])
