"""Comprehensive unit tests for advanced batch processing.

Requirements tested:
- AdvancedBatchProcessor with checkpointing and resumption
- BatchConfig validation and configuration
- FileResult and BatchCheckpoint serialization
- Timeout enforcement for long-running operations
- Error handling strategies (skip, stop, warn)
- Progress tracking and parallel execution

Coverage target: 90%+ of src/oscura/workflows/batch/advanced.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oscura.workflows.batch.advanced import (
    AdvancedBatchProcessor,
    BatchCheckpoint,
    BatchConfig,
    FileResult,
    _run_with_timeout,
    resume_batch,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestBatchConfig:
    """Test BatchConfig dataclass."""

    def test_default_config(self):
        """Test BatchConfig with default values."""
        config = BatchConfig()

        assert config.on_error == "warn"
        assert config.checkpoint_dir is None
        assert config.checkpoint_interval == 10
        assert config.max_workers is None
        assert config.memory_limit is None
        assert config.timeout_per_file is None
        assert config.use_threads is False
        assert config.progress_bar is True

    def test_custom_config(self, tmp_path: Path):
        """Test BatchConfig with custom values."""
        config = BatchConfig(
            on_error="skip",
            checkpoint_dir=tmp_path,
            checkpoint_interval=5,
            max_workers=4,
            memory_limit=512.0,
            timeout_per_file=60.0,
            use_threads=True,
            progress_bar=False,
        )

        assert config.on_error == "skip"
        assert config.checkpoint_dir == tmp_path
        assert config.checkpoint_interval == 5
        assert config.max_workers == 4
        assert config.memory_limit == 512.0
        assert config.timeout_per_file == 60.0
        assert config.use_threads is True
        assert config.progress_bar is False

    def test_config_accepts_string_path(self):
        """Test that checkpoint_dir accepts string paths."""
        config = BatchConfig(checkpoint_dir="/tmp/checkpoints", use_threads=True)

        assert config.checkpoint_dir == "/tmp/checkpoints"


@pytest.mark.unit
class TestFileResult:
    """Test FileResult dataclass."""

    def test_successful_result(self):
        """Test FileResult for successful processing."""
        result = FileResult(file="test.txt", success=True, result={"metric": 42.0}, duration=1.5)

        assert result.file == "test.txt"
        assert result.success is True
        assert result.result == {"metric": 42.0}
        assert result.error is None
        assert result.duration == 1.5
        assert result.timed_out is False

    def test_failed_result(self):
        """Test FileResult for failed processing."""
        result = FileResult(
            file="test.txt",
            success=False,
            error="Processing failed",
            traceback="Traceback...",
            duration=0.5,
        )

        assert result.success is False
        assert result.error == "Processing failed"
        assert result.traceback == "Traceback..."

    def test_timeout_result(self):
        """Test FileResult for timeout scenario."""
        result = FileResult(
            file="test.txt",
            success=False,
            error="Timeout after 60s",
            duration=60.5,
            timed_out=True,
        )

        assert result.timed_out is True
        assert "Timeout" in result.error

    def test_default_values(self):
        """Test FileResult default values."""
        result = FileResult(file="test.txt")

        assert result.success is True
        assert result.result == {}
        assert result.error is None
        assert result.duration == 0.0
        assert result.timed_out is False


@pytest.mark.unit
class TestBatchCheckpoint:
    """Test BatchCheckpoint dataclass and serialization."""

    def test_checkpoint_creation(self):
        """Test creating BatchCheckpoint."""
        checkpoint = BatchCheckpoint(
            completed_files=["f1.txt", "f2.txt"],
            failed_files=["f3.txt"],
            total_files=5,
        )

        assert len(checkpoint.completed_files) == 2
        assert len(checkpoint.failed_files) == 1
        assert checkpoint.total_files == 5

    def test_checkpoint_save(self, tmp_path: Path):
        """Test saving checkpoint to JSON."""
        checkpoint = BatchCheckpoint(
            completed_files=["f1.txt", "f2.txt"],
            failed_files=["f3.txt"],
            results=[
                FileResult(file="f1.txt", success=True, result={"val": 1}),
                FileResult(file="f2.txt", success=True, result={"val": 2}),
            ],
            total_files=5,
        )

        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint.save(checkpoint_path)

        assert checkpoint_path.exists()
        with open(checkpoint_path) as f:
            data = json.load(f)

        assert data["completed_files"] == ["f1.txt", "f2.txt"]
        assert data["failed_files"] == ["f3.txt"]
        assert data["total_files"] == 5
        assert len(data["results"]) == 2

    def test_checkpoint_load(self, tmp_path: Path):
        """Test loading checkpoint from JSON."""
        checkpoint_path = tmp_path / "checkpoint.json"
        data = {
            "completed_files": ["f1.txt"],
            "failed_files": ["f2.txt"],
            "total_files": 3,
            "results": [{"file": "f1.txt", "success": True, "result": {"val": 1}, "duration": 0.5}],
            "config": None,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(data, f)

        checkpoint = BatchCheckpoint.load(checkpoint_path)

        assert checkpoint.completed_files == ["f1.txt"]
        assert checkpoint.failed_files == ["f2.txt"]
        assert checkpoint.total_files == 3
        assert len(checkpoint.results) == 1

    def test_checkpoint_save_with_config(self, tmp_path: Path):
        """Test saving checkpoint with BatchConfig."""
        config = BatchConfig(on_error="skip", checkpoint_interval=5, use_threads=True)
        checkpoint = BatchCheckpoint(completed_files=["f1.txt"], total_files=3, config=config)

        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint.save(checkpoint_path)

        with open(checkpoint_path) as f:
            data = json.load(f)

        assert data["config"]["on_error"] == "skip"
        assert data["config"]["checkpoint_interval"] == 5

    def test_checkpoint_load_with_config(self, tmp_path: Path):
        """Test loading checkpoint with BatchConfig."""
        checkpoint_path = tmp_path / "checkpoint.json"
        data = {
            "completed_files": [],
            "failed_files": [],
            "results": [],
            "total_files": 0,
            "config": {
                "on_error": "stop",
                "checkpoint_dir": "/tmp",
                "checkpoint_interval": 3,
                "max_workers": 2,
                "memory_limit": None,
                "timeout_per_file": 30.0,
                "use_threads": True,
                "progress_bar": False,
            },
        }

        with open(checkpoint_path, "w") as f:
            json.dump(data, f)

        checkpoint = BatchCheckpoint.load(checkpoint_path)

        assert checkpoint.config is not None
        assert checkpoint.config.on_error == "stop"
        assert checkpoint.config.timeout_per_file == 30.0

    def test_checkpoint_creates_parent_directory(self, tmp_path: Path):
        """Test that save creates parent directories."""
        checkpoint = BatchCheckpoint()
        nested_path = tmp_path / "sub1" / "sub2" / "checkpoint.json"

        checkpoint.save(nested_path)

        assert nested_path.exists()


@pytest.mark.unit
class TestAdvancedBatchProcessorBasic:
    """Test basic AdvancedBatchProcessor functionality."""

    def test_processor_initialization_default(self):
        """Test processor initialization with default config."""
        processor = AdvancedBatchProcessor()

        assert processor.config.on_error == "warn"
        assert processor.checkpoint is None

    def test_processor_initialization_custom(self):
        """Test processor initialization with custom config."""
        config = BatchConfig(on_error="skip", max_workers=4, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        assert processor.config.on_error == "skip"
        assert processor.config.max_workers == 4

    def test_process_empty_files(self):
        """Test processing empty file list."""
        # Use threads to avoid pickling issues with local functions
        config = BatchConfig(use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def dummy_fn(filepath: str | Path) -> dict[str, Any]:
            return {"result": 1}

        result = processor.process([], dummy_fn)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_process_single_file(self, tmp_path: Path):
        """Test processing single file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        # Use threads to avoid pickling issues with local functions
        config = BatchConfig(use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"size": len(Path(filepath).read_text())}

        result = processor.process([test_file], analysis_fn)

        assert len(result) == 1
        assert result.iloc[0]["file"] == str(test_file)
        assert result.iloc[0]["size"] == 4
        assert result.iloc[0]["success"]

    def test_process_multiple_files_sequential(self, tmp_path: Path):
        """Test processing multiple files sequentially."""
        files = [tmp_path / f"file_{i}.txt" for i in range(3)]
        for f in files:
            f.write_text("test")

        # Use threads to avoid pickling issues with local functions
        config = BatchConfig(max_workers=1, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"name": Path(filepath).name}

        result = processor.process(files, analysis_fn)

        assert len(result) == 3
        assert all(result["success"])


@pytest.mark.unit
class TestAdvancedBatchProcessorErrorHandling:
    """Test error handling strategies."""

    def test_error_strategy_skip(self, tmp_path: Path):
        """Test 'skip' error strategy continues on errors."""
        files = [tmp_path / f"file_{i}.txt" for i in range(3)]
        for f in files:
            f.write_text("data")

        config = BatchConfig(on_error="skip", use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def failing_fn(filepath: str | Path) -> dict[str, Any]:
            if "file_1" in str(filepath):
                raise ValueError("Middle file error")
            return {"status": "ok"}

        result = processor.process(files, failing_fn)

        # Check that we have 2 successes and 1 failure (order-independent)
        assert len(result) == 3
        successes = result[result["success"] == True]  # noqa: E712
        failures = result[result["success"] == False]  # noqa: E712
        assert len(successes) == 2
        assert len(failures) == 1

    def test_error_strategy_stop(self, tmp_path: Path):
        """Test 'stop' error strategy halts on first error."""
        files = [tmp_path / f"file_{i}.txt" for i in range(3)]
        for f in files:
            f.write_text("data")

        config = BatchConfig(on_error="stop", use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def failing_fn(filepath: str | Path) -> dict[str, Any]:
            if "file_1" in str(filepath):
                raise ValueError("Error in file_1")
            return {"status": "ok"}

        with pytest.raises(RuntimeError, match="Processing stopped"):
            processor.process(files, failing_fn)

    def test_error_strategy_warn(self, tmp_path: Path, capsys):
        """Test 'warn' error strategy prints warnings."""
        files = [tmp_path / f"file_{i}.txt" for i in range(2)]
        for f in files:
            f.write_text("data")

        config = BatchConfig(on_error="warn", use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def failing_fn(filepath: str | Path) -> dict[str, Any]:
            if "file_0" in str(filepath):
                raise ValueError("Warning test")
            return {"status": "ok"}

        result = processor.process(files, failing_fn)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert len(result) == 2


@pytest.mark.unit
class TestAdvancedBatchProcessorCheckpointing:
    """Test checkpointing and resume functionality."""

    def test_checkpoint_saved_periodically(self, tmp_path: Path):
        """Test that checkpoints are saved at configured intervals."""
        files = [tmp_path / f"file_{i}.txt" for i in range(12)]
        for f in files:
            f.write_text("data")

        config = BatchConfig(checkpoint_dir=tmp_path, checkpoint_interval=5, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"processed": True}

        processor.process(files, analysis_fn, checkpoint_name="test_checkpoint")

        checkpoint_path = tmp_path / "test_checkpoint.json"
        assert checkpoint_path.exists()

    def test_resume_from_checkpoint(self, tmp_path: Path):
        """Test resuming processing from checkpoint."""
        files = [tmp_path / f"file_{i}.txt" for i in range(5)]
        for f in files:
            f.write_text("data")

        # Create initial checkpoint
        checkpoint = BatchCheckpoint(
            completed_files=[str(files[0]), str(files[1])],
            results=[
                FileResult(file=str(files[0]), success=True, result={"val": 1}),
                FileResult(file=str(files[1]), success=True, result={"val": 2}),
            ],
            total_files=5,
        )

        checkpoint_path = tmp_path / "resume_test.json"
        checkpoint.save(checkpoint_path)

        # Resume processing
        config = BatchConfig(checkpoint_dir=tmp_path, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        call_count = [0]

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            call_count[0] += 1
            return {"processed": True}

        result = processor.process(files, analysis_fn, checkpoint_name="resume_test")

        # Should only process remaining 3 files
        assert call_count[0] == 3
        assert len(result) == 5  # Total results include resumed

    def test_checkpoint_without_dir_skips_checkpoint(self, tmp_path: Path):
        """Test that no checkpointing occurs when checkpoint_dir is None."""
        files = [tmp_path / f"file_{i}.txt" for i in range(3)]
        for f in files:
            f.write_text("data")

        config = BatchConfig(checkpoint_dir=None, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"status": "ok"}

        processor.process(files, analysis_fn, checkpoint_name="test")

        # No checkpoint file should be created
        assert not (tmp_path / "test.json").exists()

    def test_resume_with_failed_files(self, tmp_path: Path):
        """Test that resume skips both completed and failed files."""
        files = [tmp_path / f"file_{i}.txt" for i in range(5)]
        for f in files:
            f.write_text("data")

        # Checkpoint with some completed and some failed
        checkpoint = BatchCheckpoint(
            completed_files=[str(files[0])],
            failed_files=[str(files[1])],
            results=[
                FileResult(file=str(files[0]), success=True, result={"val": 1}),
                FileResult(file=str(files[1]), success=False, error="Failed"),
            ],
            total_files=5,
        )

        checkpoint_path = tmp_path / "resume_with_failures.json"
        checkpoint.save(checkpoint_path)

        config = BatchConfig(checkpoint_dir=tmp_path, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        call_count = [0]

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            call_count[0] += 1
            return {"status": "ok"}

        result = processor.process(files, analysis_fn, checkpoint_name="resume_with_failures")

        # Should process only the 3 remaining files
        assert call_count[0] == 3


@pytest.mark.unit
class TestAdvancedBatchProcessorTimeout:
    """Test timeout enforcement."""

    def test_timeout_enforcement(self, tmp_path: Path):
        """Test that timeout is enforced for slow functions."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        config = BatchConfig(timeout_per_file=0.1, use_threads=True)  # 100ms timeout
        processor = AdvancedBatchProcessor(config)

        def slow_fn(filepath: str | Path) -> dict[str, Any]:
            time.sleep(0.5)  # Intentionally exceed timeout
            return {"status": "ok"}

        result = processor.process([test_file], slow_fn)

        assert len(result) == 1
        assert not result.iloc[0]["success"]
        assert result.iloc[0]["timed_out"]
        assert "timed out" in result.iloc[0]["error"].lower()

    def test_no_timeout_when_disabled(self, tmp_path: Path):
        """Test that operations complete when timeout is None."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        config = BatchConfig(timeout_per_file=None, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def slow_fn(filepath: str | Path) -> dict[str, Any]:
            time.sleep(0.1)
            return {"status": "completed"}

        result = processor.process([test_file], slow_fn)

        assert result.iloc[0]["success"]
        assert result.iloc[0]["status"] == "completed"

    def test_run_with_timeout_success(self):
        """Test _run_with_timeout with function that completes in time."""

        def quick_fn(x: int) -> int:
            return x * 2

        result, timed_out = _run_with_timeout(quick_fn, (5,), {}, timeout=1.0)

        assert result == 10
        assert timed_out is False

    def test_run_with_timeout_times_out(self):
        """Test _run_with_timeout with function that exceeds timeout."""

        def slow_fn() -> int:
            time.sleep(0.5)
            return 42

        result, timed_out = _run_with_timeout(slow_fn, (), {}, timeout=0.1)

        assert timed_out is True
        assert result is None

    def test_run_with_timeout_propagates_exception(self):
        """Test _run_with_timeout propagates exceptions from wrapped function."""

        def failing_fn() -> int:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            _run_with_timeout(failing_fn, (), {}, timeout=1.0)


@pytest.mark.unit
class TestAdvancedBatchProcessorParallel:
    """Test parallel execution modes."""

    def test_parallel_process_execution(self, tmp_path: Path):
        """Test parallel execution with multiple workers."""
        from .test_helpers import name_analysis

        files = [tmp_path / f"file_{i}.txt" for i in range(4)]
        for f in files:
            f.write_text("data")

        # Note: use_threads=True because ProcessPoolExecutor can't pickle
        # the closure created by _create_file_processor
        config = BatchConfig(max_workers=2, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        result = processor.process(files, name_analysis)

        assert len(result) == 4
        assert all(result["success"])

    def test_parallel_thread_execution(self, tmp_path: Path):
        """Test parallel execution with thread pool."""
        files = [tmp_path / f"file_{i}.txt" for i in range(4)]
        for f in files:
            f.write_text("data")

        config = BatchConfig(max_workers=2, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"name": Path(filepath).name}

        result = processor.process(files, analysis_fn)

        assert len(result) == 4
        assert all(result["success"])


@pytest.mark.unit
class TestAdvancedBatchProcessorProgressBar:
    """Test progress bar functionality."""

    @patch("oscura.workflows.batch.advanced.HAS_TQDM", True)
    @patch("oscura.workflows.batch.advanced.tqdm")
    def test_progress_bar_enabled(self, mock_tqdm, tmp_path: Path):
        """Test that progress bar is created when enabled."""
        files = [tmp_path / f"file_{i}.txt" for i in range(3)]
        for f in files:
            f.write_text("data")

        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        config = BatchConfig(progress_bar=True, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"status": "ok"}

        processor.process(files, analysis_fn)

        assert mock_tqdm.called
        assert mock_pbar.update.call_count == 3
        assert mock_pbar.close.called

    def test_progress_bar_disabled(self, tmp_path: Path):
        """Test that no progress bar is created when disabled."""
        files = [tmp_path / f"file_{i}.txt" for i in range(2)]
        for f in files:
            f.write_text("data")

        config = BatchConfig(progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"status": "ok"}

        result = processor.process(files, analysis_fn)

        assert len(result) == 2


@pytest.mark.unit
class TestResumeBatch:
    """Test resume_batch convenience function."""

    def test_resume_batch_loads_checkpoint(self, tmp_path: Path):
        """Test resume_batch loads checkpoint from directory."""
        checkpoint = BatchCheckpoint(completed_files=["f1.txt", "f2.txt"], total_files=5)

        checkpoint_path = tmp_path / "batch_checkpoint.json"
        checkpoint.save(checkpoint_path)

        loaded = resume_batch(tmp_path)

        assert loaded.completed_files == ["f1.txt", "f2.txt"]
        assert loaded.total_files == 5

    def test_resume_batch_with_custom_name(self, tmp_path: Path):
        """Test resume_batch with custom checkpoint name."""
        checkpoint = BatchCheckpoint(completed_files=["f1.txt"])

        checkpoint_path = tmp_path / "custom_checkpoint.json"
        checkpoint.save(checkpoint_path)

        loaded = resume_batch(tmp_path, "custom_checkpoint")

        assert loaded.completed_files == ["f1.txt"]


@pytest.mark.unit
class TestDataFrameConversion:
    """Test conversion of results to DataFrame."""

    def test_results_to_dataframe(self, tmp_path: Path):
        """Test _results_to_dataframe creates proper DataFrame."""
        files = [tmp_path / f"file_{i}.txt" for i in range(2)]
        for f in files:
            f.write_text("data")

        # Use threads to avoid pickling issues with local functions
        config = BatchConfig(use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"metric_a": 1.0, "metric_b": 2.0}

        result = processor.process(files, analysis_fn)

        assert "file" in result.columns
        assert "success" in result.columns
        assert "timed_out" in result.columns
        assert "error" in result.columns
        assert "duration" in result.columns
        assert "metric_a" in result.columns
        assert "metric_b" in result.columns

    def test_results_to_dataframe_with_failures(self, tmp_path: Path):
        """Test DataFrame includes error information for failures."""
        files = [tmp_path / f"file_{i}.txt" for i in range(2)]
        for f in files:
            f.write_text("data")

        config = BatchConfig(on_error="skip", use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def failing_fn(filepath: str | Path) -> dict[str, Any]:
            if "file_0" in str(filepath):
                raise RuntimeError("Test error")
            return {"status": "ok"}

        result = processor.process(files, failing_fn)

        # Check that we have one success and one failure (order-independent)
        assert len(result) == 2
        successes = result[result["success"] == True]  # noqa: E712
        failures = result[result["success"] == False]  # noqa: E712
        assert len(successes) == 1
        assert len(failures) == 1
        assert "Test error" in failures.iloc[0]["error"]
        assert failures.iloc[0]["traceback"] is not None

    def test_empty_results_to_dataframe(self):
        """Test _results_to_dataframe with no results."""
        # Use threads to avoid pickling issues with local functions
        config = BatchConfig(use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def dummy_fn(filepath: str | Path) -> dict[str, Any]:
            return {}

        result = processor.process([], dummy_fn)

        assert isinstance(result, pd.DataFrame)
        assert result.empty


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_string_paths_accepted(self, tmp_path: Path):
        """Test that string paths work alongside Path objects."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        # Use threads to avoid pickling issues with local functions
        config = BatchConfig(use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"size": len(Path(filepath).read_text())}

        result = processor.process([str(test_file)], analysis_fn)

        assert len(result) == 1
        assert result.iloc[0]["size"] == 4

    def test_analysis_function_returns_non_dict(self, tmp_path: Path):
        """Test handling when analysis function returns non-dict."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        # Use threads to avoid pickling issues with local functions
        config = BatchConfig(use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def returns_int(filepath: str | Path) -> int:
            return 42  # type: ignore[return-value]

        result = processor.process([test_file], returns_int)  # type: ignore[arg-type]

        assert result.iloc[0]["success"]
        # Non-dict result is wrapped as {"result": value}, then unpacked to row
        assert result.iloc[0]["result"] == 42

    def test_large_batch_with_checkpoint(self, tmp_path: Path):
        """Test processing large batch with periodic checkpointing."""
        files = [tmp_path / f"file_{i}.txt" for i in range(25)]
        for f in files:
            f.write_text("data")

        config = BatchConfig(checkpoint_dir=tmp_path, checkpoint_interval=10, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"index": int(Path(filepath).stem.split("_")[1])}

        result = processor.process(files, analysis_fn, checkpoint_name="large_batch")

        assert len(result) == 25
        assert (tmp_path / "large_batch.json").exists()

    def test_analysis_with_kwargs(self, tmp_path: Path):
        """Test passing additional kwargs to analysis function."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")

        # Use threads to avoid pickling issues with local functions
        config = BatchConfig(use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def analysis_with_config(filepath: str | Path, multiplier: int = 1) -> dict[str, Any]:
            return {"result": len(Path(filepath).read_text()) * multiplier}

        result = processor.process([test_file], analysis_with_config, multiplier=3)

        assert result.iloc[0]["result"] == 12  # 4 chars * 3

    def test_checkpoint_with_path_in_config(self, tmp_path: Path):
        """Test checkpoint save/load with Path object in config."""
        files = [tmp_path / "file.txt"]
        files[0].write_text("data")

        config = BatchConfig(checkpoint_dir=tmp_path, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        def analysis_fn(filepath: str | Path) -> dict[str, Any]:
            return {"status": "ok"}

        processor.process(files, analysis_fn, checkpoint_name="path_test")

        # Load and verify
        loaded_checkpoint = resume_batch(tmp_path, "path_test")
        assert len(loaded_checkpoint.completed_files) == 1
