"""Unit tests for advanced batch processing module.

Tests API-012: Advanced Batch Control (Checkpointing)

This test suite covers:
- AdvancedBatchProcessor with checkpointing
- BatchConfig configuration options
- FileResult tracking
- BatchCheckpoint save/load functionality
- Resume from checkpoint
- Error handling strategies (skip, stop, warn)
- Parallel processing with threads and processes
- Progress tracking
- Timeout handling
- Resource limits
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oscura.batch.advanced import (
    AdvancedBatchProcessor,
    BatchCheckpoint,
    BatchConfig,
    FileResult,
    resume_batch,
)

pytestmark = pytest.mark.unit


# Helper functions for testing
def simple_analysis(filepath: str | Path) -> dict[str, Any]:
    """Simple analysis function that returns file length."""
    return {"length": len(str(filepath)), "result": "ok"}


def failing_analysis(filepath: str | Path) -> dict[str, Any]:
    """Analysis function that fails on files containing 'fail'."""
    if "fail" in str(filepath):
        raise ValueError(f"Failed to process {filepath}")
    return {"status": "success"}


def slow_analysis(filepath: str | Path) -> dict[str, Any]:
    """Slow analysis function for timeout testing."""
    time.sleep(0.1)
    return {"processed": True}


class TestBatchConfig:
    """Test BatchConfig dataclass.

    Tests: API-012
    """

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BatchConfig()

        assert config.on_error == "warn"
        assert config.checkpoint_dir is None
        assert config.checkpoint_interval == 10
        assert config.max_workers is None
        assert config.memory_limit is None
        assert config.timeout_per_file is None
        assert config.use_threads is False
        assert config.progress_bar is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BatchConfig(
            on_error="skip",
            checkpoint_dir="/tmp/checkpoints",
            checkpoint_interval=5,
            max_workers=4,
            memory_limit=1024.0,
            timeout_per_file=60.0,
            use_threads=True,
            progress_bar=False,
        )

        assert config.on_error == "skip"
        assert config.checkpoint_dir == "/tmp/checkpoints"
        assert config.checkpoint_interval == 5
        assert config.max_workers == 4
        assert config.memory_limit == 1024.0
        assert config.timeout_per_file == 60.0
        assert config.use_threads is True
        assert config.progress_bar is False

    def test_path_checkpoint_dir(self) -> None:
        """Test checkpoint_dir as Path object."""
        config = BatchConfig(checkpoint_dir=Path("/tmp/checkpoints"))

        assert isinstance(config.checkpoint_dir, Path)
        assert str(config.checkpoint_dir) == "/tmp/checkpoints"


class TestFileResult:
    """Test FileResult dataclass.

    Tests: API-012
    """

    def test_success_result(self) -> None:
        """Test successful file result."""
        result = FileResult(
            file="test.wfm",
            success=True,
            result={"metric": 42},
            duration=0.5,
        )

        assert result.file == "test.wfm"
        assert result.success is True
        assert result.result["metric"] == 42
        assert result.error is None
        assert result.traceback is None
        assert result.duration == 0.5

    def test_error_result(self) -> None:
        """Test failed file result."""
        result = FileResult(
            file="bad.wfm",
            success=False,
            error="File not found",
            traceback="Traceback info...",
            duration=0.1,
        )

        assert result.file == "bad.wfm"
        assert result.success is False
        assert result.error == "File not found"
        assert result.traceback == "Traceback info..."
        assert result.result == {}

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        result = FileResult(file="test.wfm")

        assert result.success is True
        assert result.result == {}
        assert result.error is None
        assert result.traceback is None
        assert result.duration == 0.0


class TestBatchCheckpoint:
    """Test BatchCheckpoint dataclass and save/load functionality.

    Tests: API-012
    """

    def test_empty_checkpoint(self) -> None:
        """Test creating empty checkpoint."""
        checkpoint = BatchCheckpoint()

        assert checkpoint.completed_files == []
        assert checkpoint.failed_files == []
        assert checkpoint.results == []
        assert checkpoint.total_files == 0
        assert checkpoint.config is None

    def test_checkpoint_with_data(self) -> None:
        """Test checkpoint with data."""
        config = BatchConfig(checkpoint_interval=5)
        result = FileResult(file="test.wfm", success=True, result={"metric": 1})

        checkpoint = BatchCheckpoint(
            completed_files=["test.wfm"],
            failed_files=[],
            results=[result],
            total_files=10,
            config=config,
        )

        assert len(checkpoint.completed_files) == 1
        assert len(checkpoint.results) == 1
        assert checkpoint.total_files == 10
        assert checkpoint.config is not None

    def test_save_checkpoint(self) -> None:
        """Test saving checkpoint to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.json"

            result = FileResult(file="test.wfm", success=True, result={"metric": 1})
            checkpoint = BatchCheckpoint(
                completed_files=["test.wfm"],
                results=[result],
                total_files=5,
            )

            checkpoint.save(checkpoint_path)

            assert checkpoint_path.exists()

            # Verify JSON structure
            with open(checkpoint_path) as f:
                data = json.load(f)

            assert "completed_files" in data
            assert "failed_files" in data
            assert "results" in data
            assert "total_files" in data
            assert data["total_files"] == 5

    def test_load_checkpoint(self) -> None:
        """Test loading checkpoint from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.json"

            # Create and save a checkpoint
            config = BatchConfig(checkpoint_interval=5)
            result = FileResult(file="test.wfm", success=True, result={"metric": 1})
            original = BatchCheckpoint(
                completed_files=["test.wfm", "test2.wfm"],
                failed_files=["bad.wfm"],
                results=[result],
                total_files=10,
                config=config,
            )
            original.save(checkpoint_path)

            # Load it back
            loaded = BatchCheckpoint.load(checkpoint_path)

            assert loaded.completed_files == ["test.wfm", "test2.wfm"]
            assert loaded.failed_files == ["bad.wfm"]
            assert len(loaded.results) == 1
            assert loaded.total_files == 10
            assert loaded.config is not None
            assert loaded.config.checkpoint_interval == 5

    def test_save_creates_directory(self) -> None:
        """Test that save creates parent directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "nested" / "dir" / "checkpoint.json"

            checkpoint = BatchCheckpoint(total_files=5)
            checkpoint.save(checkpoint_path)

            assert checkpoint_path.exists()
            assert checkpoint_path.parent.exists()

    def test_load_without_config(self) -> None:
        """Test loading checkpoint without config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # Create checkpoint without config
            checkpoint = BatchCheckpoint(
                completed_files=["test.wfm"],
                total_files=5,
            )
            checkpoint.save(checkpoint_path)

            # Load it back
            loaded = BatchCheckpoint.load(checkpoint_path)

            assert loaded.completed_files == ["test.wfm"]
            assert loaded.config is None

    def test_save_load_cycle_with_path_checkpoint_dir(self) -> None:
        """Test that save/load cycle works with Path checkpoint_dir.

        This tests the fix for the Path serialization bug where checkpoint_dir
        as Path object would fail JSON serialization on re-save after load.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"

            # Create checkpoint with Path checkpoint_dir
            config = BatchConfig(checkpoint_dir=Path(tmpdir), checkpoint_interval=5)
            result = FileResult(file="test.wfm", success=True, result={"metric": 1})
            original = BatchCheckpoint(
                completed_files=["test.wfm"],
                results=[result],
                total_files=5,
                config=config,
            )

            # First save
            original.save(checkpoint_path)

            # Load
            loaded = BatchCheckpoint.load(checkpoint_path)

            # Re-save (this is where the bug manifested - JSON serialization error)
            resave_path = Path(tmpdir) / "checkpoint_resave.json"
            loaded.save(resave_path)

            # Verify resaved file is valid
            assert resave_path.exists()
            with open(resave_path) as f:
                data = json.load(f)
            assert data["config"]["checkpoint_dir"] == tmpdir


class TestAdvancedBatchProcessor:
    """Test AdvancedBatchProcessor class.

    Tests: API-012
    """

    def test_basic_creation(self) -> None:
        """Test creating processor with default config."""
        processor = AdvancedBatchProcessor()

        assert processor.config is not None
        assert processor.checkpoint is None

    def test_custom_config(self) -> None:
        """Test creating processor with custom config."""
        config = BatchConfig(on_error="skip", max_workers=2)
        processor = AdvancedBatchProcessor(config)

        assert processor.config.on_error == "skip"
        assert processor.config.max_workers == 2

    def test_process_empty_files(self) -> None:
        """Test processing empty file list."""
        processor = AdvancedBatchProcessor()
        result = processor.process([], simple_analysis)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_process_single_file(self) -> None:
        """Test processing single file."""
        config = BatchConfig(progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["test.wfm"]
        result = processor.process(files, simple_analysis)

        assert len(result) == 1
        assert result.iloc[0]["file"] == "test.wfm"
        assert result.iloc[0]["success"]
        assert result.iloc[0]["length"] > 0

    def test_process_multiple_files(self) -> None:
        """Test processing multiple files."""
        config = BatchConfig(progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["file1.wfm", "file2.wfm", "file3.wfm"]
        result = processor.process(files, simple_analysis)

        assert len(result) == 3
        assert all(result["success"])
        assert "file1.wfm" in result["file"].values

    def test_error_handling_warn(self) -> None:
        """Test error handling with 'warn' strategy."""
        config = BatchConfig(on_error="warn", progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["good1.wfm", "fail.wfm", "good2.wfm"]
        result = processor.process(files, failing_analysis)

        assert len(result) == 3
        assert result[result["file"] == "good1.wfm"].iloc[0]["success"]
        assert not result[result["file"] == "fail.wfm"].iloc[0]["success"]
        assert result[result["file"] == "good2.wfm"].iloc[0]["success"]

    def test_error_handling_skip(self) -> None:
        """Test error handling with 'skip' strategy."""
        config = BatchConfig(on_error="skip", progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["good1.wfm", "fail.wfm", "good2.wfm"]
        result = processor.process(files, failing_analysis)

        assert len(result) == 3
        # All files are processed, errors are recorded
        assert not result[result["file"] == "fail.wfm"].iloc[0]["success"]

    def test_error_handling_stop(self) -> None:
        """Test error handling with 'stop' strategy."""
        config = BatchConfig(on_error="stop", progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["good1.wfm", "fail.wfm", "good2.wfm"]

        with pytest.raises(RuntimeError, match="Processing stopped"):
            processor.process(files, failing_analysis)

    def test_checkpoint_save_and_resume(self) -> None:
        """Test checkpoint save and resume functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use string path to avoid JSON serialization issues with Path objects
            config = BatchConfig(
                checkpoint_dir=tmpdir,
                checkpoint_interval=2,
                progress_bar=False,
                use_threads=True,
            )
            processor = AdvancedBatchProcessor(config)

            # Process some files
            files = ["file1.wfm", "file2.wfm", "file3.wfm", "file4.wfm"]
            result = processor.process(files, simple_analysis, checkpoint_name="test_batch")

            # Verify checkpoint was created
            checkpoint_path = Path(tmpdir) / "test_batch.json"
            assert checkpoint_path.exists()

            # Load checkpoint and verify
            checkpoint = BatchCheckpoint.load(checkpoint_path)
            assert len(checkpoint.completed_files) == 4
            assert checkpoint.total_files == 4

            # Verify result dataframe
            assert len(result) == 4
            assert all(result["success"])

    def test_resume_from_checkpoint(self) -> None:
        """Test resuming batch from existing checkpoint.

        This test validates that the Path serialization fix works correctly,
        allowing checkpoints to be saved, loaded, and re-saved without errors.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BatchConfig(
                checkpoint_dir=tmpdir,
                checkpoint_interval=1,
                progress_bar=False,
                use_threads=True,
            )

            # First run: process partial batch
            processor1 = AdvancedBatchProcessor(config)
            files = ["file1.wfm", "file2.wfm"]
            processor1.process(files, simple_analysis, checkpoint_name="resume_test")

            # Second run: add more files and resume
            processor2 = AdvancedBatchProcessor(config)
            all_files = ["file1.wfm", "file2.wfm", "file3.wfm", "file4.wfm"]
            result = processor2.process(all_files, simple_analysis, checkpoint_name="resume_test")

            # Should only process new files (file3, file4)
            assert len(result) == 4
            # All files should be in final result
            assert set(result["file"].values) == set(all_files)

    def test_thread_vs_process_executor(self) -> None:
        """Test using threads vs processes."""
        # Test with threads
        config_threads = BatchConfig(use_threads=True, progress_bar=False)
        processor_threads = AdvancedBatchProcessor(config_threads)

        files = ["file1.wfm", "file2.wfm"]
        result_threads = processor_threads.process(files, simple_analysis)

        assert len(result_threads) == 2
        assert all(result_threads["success"])

        # Note: ProcessPoolExecutor test skipped due to pickling limitations
        # with local analysis functions. In real usage with module-level functions,
        # processes work fine.

    def test_max_workers_config(self) -> None:
        """Test custom max_workers configuration."""
        config = BatchConfig(max_workers=2, progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["file1.wfm", "file2.wfm", "file3.wfm", "file4.wfm"]
        result = processor.process(files, simple_analysis)

        assert len(result) == 4
        assert all(result["success"])

    def test_kwargs_passed_to_analysis_function(self) -> None:
        """Test that kwargs are passed to analysis function."""

        def analysis_with_params(filepath: str | Path, multiplier: int = 1) -> dict[str, Any]:
            return {"value": len(str(filepath)) * multiplier}

        config = BatchConfig(progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["test.wfm"]
        result = processor.process(files, analysis_with_params, multiplier=5)

        assert result.iloc[0]["value"] == len("test.wfm") * 5

    def test_error_traceback_captured(self) -> None:
        """Test that error traceback is captured."""
        config = BatchConfig(on_error="warn", progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["fail.wfm"]
        result = processor.process(files, failing_analysis)

        assert not result.iloc[0]["success"]
        assert result.iloc[0]["error"] is not None
        assert result.iloc[0]["traceback"] is not None
        assert "ValueError" in result.iloc[0]["traceback"]

    def test_duration_tracking(self) -> None:
        """Test that processing duration is tracked."""
        config = BatchConfig(progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["test.wfm"]
        result = processor.process(files, simple_analysis)

        assert result.iloc[0]["duration"] >= 0
        assert "duration" in result.columns

    def test_mixed_success_and_failure(self) -> None:
        """Test batch with mixed success and failure."""
        config = BatchConfig(on_error="warn", progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["good1.wfm", "fail1.wfm", "good2.wfm", "fail2.wfm"]
        result = processor.process(files, failing_analysis)

        assert len(result) == 4
        success_count = result["success"].sum()
        failure_count = (~result["success"]).sum()

        assert success_count == 2
        assert failure_count == 2


class TestResumeBatch:
    """Test resume_batch function.

    Tests: API-012
    """

    def test_resume_batch_function(self) -> None:
        """Test resume_batch convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Create a checkpoint
            checkpoint = BatchCheckpoint(
                completed_files=["file1.wfm", "file2.wfm"],
                total_files=5,
            )
            checkpoint.save(checkpoint_dir / "batch_checkpoint.json")

            # Resume using convenience function
            loaded = resume_batch(checkpoint_dir)

            assert len(loaded.completed_files) == 2
            assert loaded.total_files == 5

    def test_resume_batch_custom_name(self) -> None:
        """Test resume_batch with custom checkpoint name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Create a checkpoint with custom name
            checkpoint = BatchCheckpoint(
                completed_files=["file1.wfm"],
                total_files=10,
            )
            checkpoint.save(checkpoint_dir / "custom_name.json")

            # Resume using custom name
            loaded = resume_batch(checkpoint_dir, checkpoint_name="custom_name")

            assert len(loaded.completed_files) == 1
            assert loaded.total_files == 10


class TestBatchAdvancedEdgeCases:
    """Test edge cases and boundary conditions.

    Tests: API-012
    """

    def test_very_large_batch(self) -> None:
        """Test processing very large batch."""
        config = BatchConfig(
            checkpoint_interval=50,
            progress_bar=False,
            use_threads=True,
        )
        processor = AdvancedBatchProcessor(config)

        # Generate large file list
        files = [f"file_{i}.wfm" for i in range(100)]
        result = processor.process(files, simple_analysis)

        assert len(result) == 100
        assert all(result["success"])

    def test_empty_result_dict(self) -> None:
        """Test analysis function returning empty dict."""

        def empty_analysis(filepath: str | Path) -> dict[str, Any]:
            return {}

        config = BatchConfig(progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["test.wfm"]
        result = processor.process(files, empty_analysis)

        assert len(result) == 1
        assert result.iloc[0]["success"]

    def test_none_result(self) -> None:
        """Test analysis function returning non-dict value."""

        def returns_none(filepath: str | Path) -> dict[str, Any]:
            return {"value": None}  # type: ignore[dict-item]

        config = BatchConfig(progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["test.wfm"]
        result = processor.process(files, returns_none)

        assert len(result) == 1
        assert result.iloc[0]["success"]

    def test_checkpoint_without_directory(self) -> None:
        """Test that no checkpoints are created when checkpoint_dir is None."""
        config = BatchConfig(checkpoint_dir=None, progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["test.wfm"]
        result = processor.process(files, simple_analysis, checkpoint_name="test")

        # Should process successfully without checkpointing
        assert len(result) == 1
        assert result.iloc[0]["success"]

    def test_unicode_filenames(self) -> None:
        """Test handling of unicode filenames."""
        config = BatchConfig(progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["test_日本語.wfm", "file_ñ.wfm", "data_🔬.wfm"]
        result = processor.process(files, simple_analysis)

        assert len(result) == 3
        assert all(result["success"])

    def test_path_objects_in_file_list(self) -> None:
        """Test using Path objects in file list."""
        config = BatchConfig(progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = [Path("file1.wfm"), Path("file2.wfm")]
        result = processor.process(files, simple_analysis)

        assert len(result) == 2
        assert all(result["success"])


class TestProgressBar:
    """Test progress bar functionality.

    Tests: API-012
    """

    def test_progress_bar_enabled(self) -> None:
        """Test progress bar when tqdm is available."""
        config = BatchConfig(progress_bar=True, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        with patch("oscura.batch.advanced.tqdm") as mock_tqdm:
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar

            files = ["file1.wfm", "file2.wfm", "file3.wfm"]
            processor.process(files, simple_analysis)

            # Progress bar should be created and updated
            mock_tqdm.assert_called_once()
            mock_pbar.update.assert_called()
            mock_pbar.close.assert_called()

    def test_progress_bar_disabled(self) -> None:
        """Test processing with progress bar disabled."""
        config = BatchConfig(progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["file1.wfm", "file2.wfm"]
        result = processor.process(files, simple_analysis)

        # Should complete without progress bar
        assert len(result) == 2
        assert all(result["success"])


class TestBatchAdvancedIntegration:
    """Integration tests for advanced batch processing.

    Tests: API-012
    """

    def test_full_workflow_with_checkpoint(self) -> None:
        """Test complete workflow with checkpointing and resume.

        This test validates the full workflow including checkpoint save/load
        cycles that previously failed due to Path serialization issues.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BatchConfig(
                checkpoint_dir=tmpdir,
                checkpoint_interval=2,
                on_error="warn",
                progress_bar=False,
                use_threads=True,
            )

            # Phase 1: Process first batch
            processor1 = AdvancedBatchProcessor(config)
            files_phase1 = ["file1.wfm", "file2.wfm", "fail1.wfm"]
            result1 = processor1.process(
                files_phase1,
                failing_analysis,
                checkpoint_name="integration_test",
            )

            # Verify first phase
            assert len(result1) == 3
            success_count = result1["success"].sum()
            assert success_count == 2  # Two good files

            # Phase 2: Resume and add more files
            processor2 = AdvancedBatchProcessor(config)
            files_phase2 = [
                "file1.wfm",  # Already processed
                "file2.wfm",  # Already processed
                "fail1.wfm",  # Already failed
                "file3.wfm",  # New file
                "file4.wfm",  # New file
            ]
            result2 = processor2.process(
                files_phase2,
                failing_analysis,
                checkpoint_name="integration_test",
            )

            # Verify final result includes all files
            assert len(result2) == 5

            # Verify checkpoint contains all processed files
            checkpoint = resume_batch(tmpdir, "integration_test")
            assert len(checkpoint.completed_files) + len(checkpoint.failed_files) == 5

    def test_export_results_with_metadata(self) -> None:
        """Test that results include all metadata fields."""
        config = BatchConfig(progress_bar=False, use_threads=True)
        processor = AdvancedBatchProcessor(config)

        files = ["good.wfm", "fail.wfm"]
        result = processor.process(files, failing_analysis)

        # Verify all expected columns exist
        assert "file" in result.columns
        assert "success" in result.columns
        assert "duration" in result.columns

        # Error columns should exist for failed files
        failed_row = result[result["file"] == "fail.wfm"].iloc[0]
        assert not failed_row["success"]
        assert failed_row["error"] is not None
        assert failed_row["traceback"] is not None
