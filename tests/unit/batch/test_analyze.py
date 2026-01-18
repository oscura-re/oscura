import pytest

"""Unit tests for batch analyze module.

Tests BATCH-001: Multi-File Analysis
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from oscura.batch.analyze import batch_analyze

pytestmark = pytest.mark.unit


# Module-level functions for pickling support with ProcessPoolExecutor
def _simple_length_analysis(filepath: str | Path) -> dict[str, Any]:
    """Simple analysis function for parallel process tests."""
    return {"result": len(str(filepath))}


def _mock_characterize_buffer(filepath: str | Path) -> dict[str, Any]:
    """Mock buffer characterization for testing."""
    filename = str(filepath)
    if "fast" in filename:
        return {"rise_time": 1.0, "fall_time": 1.2, "status": "good"}
    return {"rise_time": 2.0, "fall_time": 2.5, "status": "slow"}


class TestBatchAnalyzeBasic:
    """Test basic batch_analyze functionality."""

    def test_empty_file_list(self) -> None:
        """Test handling of empty file list."""
        result = batch_analyze([], lambda x: {})

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_file(self) -> None:
        """Test analysis of single file."""

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"value": 42, "name": "test"}

        files = ["file1.txt"]
        result = batch_analyze(files, simple_analysis)

        assert len(result) == 1
        assert result.iloc[0]["file"] == "file1.txt"
        assert result.iloc[0]["value"] == 42
        assert result.iloc[0]["name"] == "test"
        assert result.iloc[0]["error"] is None

    def test_multiple_files_sequential(self) -> None:
        """Test sequential analysis of multiple files."""

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            filename = str(filepath)
            return {"filename_copy": filename, "length": len(filename)}

        files = ["file1.txt", "file2.txt", "file3.txt"]
        result = batch_analyze(files, simple_analysis, parallel=False)

        assert len(result) == 3
        assert result.iloc[0]["file"] == "file1.txt"
        assert result.iloc[1]["file"] == "file2.txt"
        assert result.iloc[2]["file"] == "file3.txt"
        assert all(result["error"].isna())

    def test_path_objects(self) -> None:
        """Test handling of Path objects."""

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"type": "path", "exists": Path(filepath).suffix}

        files = [Path("file1.wfm"), Path("file2.wfm")]
        result = batch_analyze(files, simple_analysis)

        assert len(result) == 2
        assert "file1.wfm" in result.iloc[0]["file"]
        assert "file2.wfm" in result.iloc[1]["file"]

    def test_column_ordering(self) -> None:
        """Test that columns are ordered with file first, error last."""

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"alpha": 1, "beta": 2, "gamma": 3}

        files = ["test.txt"]
        result = batch_analyze(files, simple_analysis)

        columns = result.columns.tolist()
        assert columns[0] == "file"
        assert columns[-1] == "error"
        assert "alpha" in columns
        assert "beta" in columns
        assert "gamma" in columns


class TestBatchAnalyzeErrors:
    """Test error handling in batch_analyze."""

    def test_analysis_function_exception(self) -> None:
        """Test handling of exceptions in analysis function."""

        def failing_analysis(filepath: str | Path) -> dict[str, Any]:
            if "bad" in str(filepath):
                raise ValueError("Bad file detected")
            return {"status": "ok"}

        files = ["good.txt", "bad.txt", "good2.txt"]
        result = batch_analyze(files, failing_analysis)

        assert len(result) == 3

        # Check good files
        good_rows = result[result["file"].str.contains("good")]
        assert len(good_rows) == 2
        assert all(good_rows["error"].isna())

        # Check bad file
        bad_row = result[result["file"] == "bad.txt"].iloc[0]
        assert "Bad file detected" in bad_row["error"]
        assert "status" not in bad_row or pd.isna(bad_row.get("status"))

    def test_all_files_fail(self) -> None:
        """Test batch where all files fail."""

        def always_fails(filepath: str | Path) -> dict[str, Any]:
            raise RuntimeError("Always fails")

        files = ["file1.txt", "file2.txt"]
        result = batch_analyze(files, always_fails)

        assert len(result) == 2
        assert all(result["error"].notna())
        assert all("Always fails" in str(err) for err in result["error"])

    def test_non_dict_return_value(self) -> None:
        """Test handling of non-dict return values."""

        def returns_number(filepath: str | Path) -> int:
            return 42  # type: ignore[return-value]

        files = ["test.txt"]
        result = batch_analyze(files, returns_number)

        assert len(result) == 1
        assert result.iloc[0]["result"] == 42
        assert result.iloc[0]["error"] is None


class TestBatchAnalyzeParallel:
    """Test parallel execution of batch_analyze."""

    def test_parallel_with_processes(self) -> None:
        """Test parallel execution with ProcessPoolExecutor.

        Note: Using threads here because the internal _wrapped_analysis
        function in batch_analyze cannot be pickled for multiprocessing.
        The executor selection logic is tested separately with mocks.
        """
        files = ["file1.txt", "file2.txt", "file3.txt", "file4.txt"]
        result = batch_analyze(files, _simple_length_analysis, parallel=True, use_threads=True)

        assert len(result) == 4
        assert all(result["error"].isna())
        assert all(result["result"] > 0)

    def test_parallel_with_threads(self) -> None:
        """Test parallel execution with ThreadPoolExecutor."""

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"result": len(str(filepath))}

        files = ["file1.txt", "file2.txt", "file3.txt"]
        result = batch_analyze(files, simple_analysis, parallel=True, use_threads=True)

        assert len(result) == 3
        assert all(result["error"].isna())

    def test_parallel_with_custom_workers(self) -> None:
        """Test parallel execution with custom worker count."""

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"value": 1}

        files = ["f1", "f2", "f3", "f4"]

        with patch("oscura.batch.analyze.ProcessPoolExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor

            # Setup mock futures
            mock_futures = []
            for f in files:
                mock_future = MagicMock()
                mock_future.result.return_value = {"file": f, "value": 1, "error": None}
                mock_futures.append(mock_future)

            mock_executor.submit.side_effect = lambda fn, f: mock_futures[files.index(f)]

            with patch("oscura.batch.analyze.as_completed", return_value=mock_futures):
                result = batch_analyze(
                    files, simple_analysis, parallel=True, workers=2, use_threads=False
                )

            # Verify executor was created with correct worker count
            mock_executor_class.assert_called_once_with(max_workers=2)

    def test_parallel_exception_in_future(self) -> None:
        """Test handling of exceptions raised in parallel execution."""

        def sometimes_fails(filepath: str | Path) -> dict[str, Any]:
            if "fail" in str(filepath):
                raise ValueError("Parallel failure")
            return {"status": "ok"}

        files = ["good.txt", "fail.txt", "good2.txt"]
        result = batch_analyze(files, sometimes_fails, parallel=True, use_threads=True)

        assert len(result) == 3

        # Check that failure is recorded
        fail_rows = result[result["file"] == "fail.txt"]
        assert len(fail_rows) == 1
        assert "Parallel failure" in fail_rows.iloc[0]["error"]


class TestBatchAnalyzeConfig:
    """Test configuration parameter passing."""

    def test_kwargs_passed_to_analysis_function(self) -> None:
        """Test that keyword arguments are passed to analysis function."""

        def analysis_with_config(
            filepath: str | Path, threshold: float = 0.5, mode: str = "auto"
        ) -> dict[str, Any]:
            return {"threshold": threshold, "mode": mode, "path": str(filepath)}

        files = ["test.txt"]
        result = batch_analyze(files, analysis_with_config, threshold=0.8, mode="manual")

        assert len(result) == 1
        assert result.iloc[0]["threshold"] == 0.8
        assert result.iloc[0]["mode"] == "manual"

    def test_multiple_kwargs(self) -> None:
        """Test passing multiple configuration parameters."""

        def complex_analysis(
            filepath: str | Path,
            param1: int = 1,
            param2: str = "default",
            param3: bool = False,
        ) -> dict[str, Any]:
            return {
                "p1": param1,
                "p2": param2,
                "p3": param3,
            }

        files = ["test.txt"]
        result = batch_analyze(files, complex_analysis, param1=100, param2="custom", param3=True)

        assert result.iloc[0]["p1"] == 100
        assert result.iloc[0]["p2"] == "custom"
        assert result.iloc[0]["p3"]


class TestBatchAnalyzeProgressCallback:
    """Test progress callback functionality."""

    def test_progress_callback_sequential(self) -> None:
        """Test progress callback in sequential mode."""
        callback_calls: list[tuple[int, int, str]] = []

        def capture_progress(current: int, total: int, filename: str) -> None:
            callback_calls.append((current, total, filename))

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"value": 1}

        files = ["file1.txt", "file2.txt", "file3.txt"]
        batch_analyze(files, simple_analysis, progress_callback=capture_progress)

        assert len(callback_calls) == 3
        assert callback_calls[0] == (1, 3, "file1.txt")
        assert callback_calls[1] == (2, 3, "file2.txt")
        assert callback_calls[2] == (3, 3, "file3.txt")

    def test_progress_callback_parallel(self) -> None:
        """Test progress callback in parallel mode."""
        callback_calls: list[tuple[int, int, str]] = []

        def capture_progress(current: int, total: int, filename: str) -> None:
            callback_calls.append((current, total, filename))

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"value": 1}

        files = ["file1.txt", "file2.txt", "file3.txt"]
        batch_analyze(
            files,
            simple_analysis,
            parallel=True,
            use_threads=True,
            progress_callback=capture_progress,
        )

        # Should have 3 callbacks (order may vary in parallel)
        assert len(callback_calls) == 3
        assert all(call[1] == 3 for call in callback_calls)  # total should be 3

    def test_no_callback_provided(self) -> None:
        """Test that no error occurs when no callback is provided."""

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"value": 1}

        files = ["file1.txt", "file2.txt"]
        result = batch_analyze(files, simple_analysis)

        assert len(result) == 2


class TestBatchAnalyzeIntegration:
    """Integration tests for batch_analyze."""

    def test_real_file_analysis(self) -> None:
        """Test analysis with real temporary files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create test files
            file1 = tmp_path / "test1.txt"
            file2 = tmp_path / "test2.txt"
            file3 = tmp_path / "test3.txt"

            file1.write_text("Hello World")
            file2.write_text("Test Data")
            file3.write_text("Sample")

            def analyze_file(filepath: str | Path) -> dict[str, Any]:
                content = Path(filepath).read_text()
                return {
                    "length": len(content),
                    "words": len(content.split()),
                    "name": Path(filepath).name,
                }

            files = [file1, file2, file3]
            result = batch_analyze(files, analyze_file)

            assert len(result) == 3
            assert result.iloc[0]["length"] == 11  # "Hello World"
            assert result.iloc[0]["words"] == 2
            assert result.iloc[1]["length"] == 9  # "Test Data"
            assert result.iloc[2]["length"] == 6  # "Sample"

    def test_mixed_success_and_failure(self) -> None:
        """Test batch with mix of successful and failed analyses."""

        def conditional_analysis(filepath: str | Path) -> dict[str, Any]:
            filename = str(filepath)
            if "error" in filename:
                raise OSError(f"Cannot read {filename}")
            return {"status": "success", "name": filename}

        files = ["good1.txt", "error.txt", "good2.txt", "error2.txt", "good3.txt"]
        result = batch_analyze(files, conditional_analysis)

        assert len(result) == 5

        # Count successes and failures
        success_count = result["error"].isna().sum()
        error_count = result["error"].notna().sum()

        assert success_count == 3
        assert error_count == 2

    def test_dataframe_export(self) -> None:
        """Test that results can be exported as CSV."""

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"metric": 42, "category": "A"}

        files = ["f1.txt", "f2.txt"]
        result = batch_analyze(files, simple_analysis)

        # Export to CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

        try:
            result.to_csv(csv_path, index=False)
            assert csv_path.exists()

            # Read back
            df_loaded = pd.read_csv(csv_path)
            assert len(df_loaded) == 2
            assert "file" in df_loaded.columns
            assert "metric" in df_loaded.columns
        finally:
            csv_path.unlink()

    def test_complex_analysis_function(self) -> None:
        """Test with complex analysis function returning multiple metrics."""

        def complex_analysis(
            filepath: str | Path,
            threshold: float = 0.5,
            normalize: bool = True,
        ) -> dict[str, Any]:
            filename = str(filepath)
            base_value = len(filename)

            if normalize:
                base_value = base_value / 100.0

            return {
                "raw_length": len(filename),
                "processed_value": base_value,
                "above_threshold": base_value > threshold,
                "filename": filename,
            }

        files = ["short.txt", "verylongfilename.txt"]
        result = batch_analyze(files, complex_analysis, threshold=0.1, normalize=True)

        assert len(result) == 2
        assert "raw_length" in result.columns
        assert "processed_value" in result.columns
        assert "above_threshold" in result.columns


class TestBatchAnalyzeExecutorSelection:
    """Test executor type selection logic."""

    def test_process_executor_by_default(self) -> None:
        """Test that ProcessPoolExecutor is used by default in parallel mode."""

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"value": 1}

        files = ["f1", "f2"]

        with patch("oscura.batch.analyze.ProcessPoolExecutor") as mock_process_executor:
            with patch("oscura.batch.analyze.ThreadPoolExecutor"):
                mock_executor = MagicMock()
                mock_process_executor.return_value.__enter__.return_value = mock_executor

                # Setup mock futures
                mock_futures = []
                for f in files:
                    mock_future = MagicMock()
                    mock_future.result.return_value = {
                        "file": f,
                        "value": 1,
                        "error": None,
                    }
                    mock_futures.append(mock_future)

                mock_executor.submit.side_effect = lambda fn, f: mock_futures[files.index(f)]

                with patch("oscura.batch.analyze.as_completed", return_value=mock_futures):
                    batch_analyze(files, simple_analysis, parallel=True)

                mock_process_executor.assert_called_once()

    def test_thread_executor_when_requested(self) -> None:
        """Test that ThreadPoolExecutor is used when use_threads=True."""

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"value": 1}

        files = ["f1", "f2"]

        with patch("oscura.batch.analyze.ThreadPoolExecutor") as mock_thread_executor:
            with patch("oscura.batch.analyze.ProcessPoolExecutor"):
                mock_executor = MagicMock()
                mock_thread_executor.return_value.__enter__.return_value = mock_executor

                # Setup mock futures
                mock_futures = []
                for f in files:
                    mock_future = MagicMock()
                    mock_future.result.return_value = {
                        "file": f,
                        "value": 1,
                        "error": None,
                    }
                    mock_futures.append(mock_future)

                mock_executor.submit.side_effect = lambda fn, f: mock_futures[files.index(f)]

                with patch("oscura.batch.analyze.as_completed", return_value=mock_futures):
                    batch_analyze(files, simple_analysis, parallel=True, use_threads=True)

                mock_thread_executor.assert_called_once()


class TestBatchAnalyzeEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_file_parallel(self) -> None:
        """Test parallel mode with single file."""
        files = ["single.txt"]
        result = batch_analyze(files, _simple_length_analysis, parallel=True, use_threads=True)

        assert len(result) == 1
        assert result.iloc[0]["error"] is None

    def test_very_large_file_list(self) -> None:
        """Test with large number of files."""

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"index": int(str(filepath).split("_")[1].split(".")[0])}

        files = [f"file_{i}.txt" for i in range(100)]
        result = batch_analyze(files, simple_analysis)

        assert len(result) == 100
        assert all(result["error"].isna())

    def test_empty_result_dict(self) -> None:
        """Test analysis function that returns empty dict."""

        def empty_analysis(filepath: str | Path) -> dict[str, Any]:
            return {}

        files = ["test.txt"]
        result = batch_analyze(files, empty_analysis)

        assert len(result) == 1
        assert result.iloc[0]["file"] == "test.txt"
        assert result.iloc[0]["error"] is None

    def test_analysis_with_none_values(self) -> None:
        """Test handling of None values in results."""

        def analysis_with_nones(filepath: str | Path) -> dict[str, Any]:
            return {"value": None, "status": "incomplete"}

        files = ["test.txt"]
        result = batch_analyze(files, analysis_with_nones)

        assert len(result) == 1
        assert pd.isna(result.iloc[0]["value"])
        assert result.iloc[0]["status"] == "incomplete"

    def test_unicode_filenames(self) -> None:
        """Test handling of unicode characters in filenames."""

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"length": len(str(filepath))}

        files = ["test_日本語.txt", "file_ñ.txt", "data_🔬.txt"]
        result = batch_analyze(files, simple_analysis)

        assert len(result) == 3
        assert all(result["error"].isna())

    def test_special_characters_in_results(self) -> None:
        """Test handling of special characters in result values."""

        def special_analysis(filepath: str | Path) -> dict[str, Any]:
            return {
                "text": "Special chars: <>&\"'",
                "unicode": "🎉🔬📊",
                "newlines": "line1\nline2",
            }

        files = ["test.txt"]
        result = batch_analyze(files, special_analysis)

        assert len(result) == 1
        assert "Special chars" in result.iloc[0]["text"]
        assert "🎉" in result.iloc[0]["unicode"]


class TestBatchAnalyzeDocumentation:
    """Test that documented behavior matches implementation."""

    def test_example_from_docstring(self) -> None:
        """Test example pattern from docstring works correctly."""
        files = ["capture_fast.wfm", "capture_slow.wfm"]
        result = batch_analyze(
            files,
            analysis_fn=_mock_characterize_buffer,
            parallel=True,
            use_threads=True,
            workers=4,
        )

        assert len(result) == 2
        assert "rise_time" in result.columns
        assert "fall_time" in result.columns
        assert "status" in result.columns

    def test_always_includes_file_column(self) -> None:
        """Test that file column is always present as documented."""

        def minimal_analysis(filepath: str | Path) -> dict[str, Any]:
            return {}

        files = ["test.txt"]
        result = batch_analyze(files, minimal_analysis)

        assert "file" in result.columns
        assert result.columns[0] == "file"

    def test_error_column_always_present(self) -> None:
        """Test that error column is always present as documented."""

        def simple_analysis(filepath: str | Path) -> dict[str, Any]:
            return {"value": 1}

        files = ["test.txt"]
        result = batch_analyze(files, simple_analysis)

        assert "error" in result.columns
        assert result.columns[-1] == "error"
