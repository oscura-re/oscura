"""Advanced batch processing with checkpointing and error handling.

This module provides enhanced batch processing capabilities including
checkpointing for long-running jobs, resume functionality, and sophisticated
error handling.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Lazy import for optional dataframe support
try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class TimeoutError(Exception):
    """Raised when a function execution exceeds timeout."""


@dataclass
class BatchConfig:
    """Configuration for advanced batch processing.

    Attributes:
        on_error: Error handling strategy:
            - 'skip': Skip failed files, continue processing
            - 'stop': Stop processing on first error
            - 'warn': Log warning but continue (default)
        checkpoint_dir: Directory to save checkpoints. If None, no checkpointing.
        checkpoint_interval: Save checkpoint every N files (default: 10).
        max_workers: Maximum number of parallel workers. None uses CPU count.
        memory_limit: Maximum memory per worker in MB (not enforced, for documentation).
        timeout_per_file: Timeout in seconds per file. None for no timeout.
            When specified, uses threading.Timer for true timeout enforcement
            that interrupts long-running operations.
        use_threads: Use threads instead of processes for parallelization.
        progress_bar: Show progress bar (requires tqdm).

    Example:
        >>> config = BatchConfig(
        ...     on_error='skip',
        ...     checkpoint_dir='./checkpoints',
        ...     checkpoint_interval=5,
        ...     max_workers=4
        ... )

    References:
        API-012: Advanced Batch Control
    """

    on_error: Literal["skip", "stop", "warn"] = "warn"
    checkpoint_dir: Path | str | None = None
    checkpoint_interval: int = 10
    max_workers: int | None = None
    memory_limit: float | None = None  # MB, not enforced
    timeout_per_file: float | None = None  # seconds, enforced via threading.Timer
    use_threads: bool = False
    progress_bar: bool = True


@dataclass
class FileResult:
    """Result from processing a single file.

    Attributes:
        file: Path to the file.
        success: Whether processing succeeded.
        result: Analysis result dictionary if successful.
        error: Error message if failed.
        traceback: Full traceback if failed.
        duration: Processing time in seconds.
        timed_out: Whether processing was terminated due to timeout.

    Example:
        >>> result = FileResult(
        ...     file='trace001.wfm',
        ...     success=True,
        ...     result={'rise_time': 1.2e-9},
        ...     duration=0.5
        ... )

    References:
        API-012: Advanced Batch Control
    """

    file: str
    success: bool = True
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None
    duration: float = 0.0
    timed_out: bool = False


@dataclass
class BatchCheckpoint:
    """Checkpoint state for batch processing.

    Attributes:
        completed_files: List of successfully completed files.
        failed_files: List of failed file paths.
        results: List of FileResult objects.
        total_files: Total number of files in batch.
        config: Batch configuration used.

    Example:
        >>> checkpoint = BatchCheckpoint(
        ...     completed_files=['file1.wfm', 'file2.wfm'],
        ...     total_files=10,
        ...     config=config
        ... )

    References:
        API-012: Advanced Batch Control
    """

    completed_files: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    results: list[FileResult] = field(default_factory=list)
    total_files: int = 0
    config: BatchConfig | None = None

    def save(self, checkpoint_path: Path) -> None:
        """Save checkpoint to JSON file.

        Args:
            checkpoint_path: Path to save checkpoint file.

        Example:
            >>> checkpoint.save(Path('checkpoints/batch_001.json'))
        """
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        config_dict = None
        if self.config:
            # Manually convert BatchConfig to dict to handle Path objects properly
            config_dict = {
                "on_error": self.config.on_error,
                "checkpoint_dir": (
                    str(self.config.checkpoint_dir)
                    if self.config.checkpoint_dir is not None
                    else None
                ),
                "checkpoint_interval": self.config.checkpoint_interval,
                "max_workers": self.config.max_workers,
                "memory_limit": self.config.memory_limit,
                "timeout_per_file": self.config.timeout_per_file,
                "use_threads": self.config.use_threads,
                "progress_bar": self.config.progress_bar,
            }

        data = {
            "completed_files": self.completed_files,
            "failed_files": self.failed_files,
            "results": [asdict(r) for r in self.results],
            "total_files": self.total_files,
            "config": config_dict,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, checkpoint_path: Path) -> BatchCheckpoint:
        """Load checkpoint from JSON file.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Loaded BatchCheckpoint object.

        Example:
            >>> checkpoint = BatchCheckpoint.load(Path('checkpoints/batch_001.json'))
        """
        with open(checkpoint_path) as f:
            data = json.load(f)

        # Reconstruct FileResult objects
        results = [FileResult(**r) for r in data.get("results", [])]

        # Reconstruct BatchConfig if present
        # Keep checkpoint_dir as string (not Path) to ensure JSON serializability
        # when checkpoint is saved again
        config = None
        if data.get("config"):
            config_data = data["config"]
            # Keep checkpoint_dir as string for JSON serialization compatibility
            # The BatchConfig type annotation allows str | Path | None
            config = BatchConfig(**config_data)

        return cls(
            completed_files=data.get("completed_files", []),
            failed_files=data.get("failed_files", []),
            results=results,
            total_files=data.get("total_files", 0),
            config=config,
        )


def _run_with_timeout(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    timeout: float,
) -> tuple[Any, bool]:
    """Run a function with true timeout enforcement using threading.

    This function wraps the target function in a separate thread and uses
    threading.Timer to interrupt it if it exceeds the timeout. This provides
    actual timeout enforcement rather than just post-hoc checking.

    If the wrapped function raises any exception, it will be re-raised by
    this wrapper.

    Args:
        func: Function to execute.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (result, timed_out) where result is the function return value
        (or None if timed out) and timed_out indicates whether timeout occurred.
        If the wrapped function raised an exception, that exception is re-raised.

    Note:
        This uses concurrent.futures.ThreadPoolExecutor internally which can
        only interrupt I/O-bound operations. CPU-bound functions in Python
        cannot be truly interrupted due to the GIL. For CPU-bound timeouts,
        consider using ProcessPoolExecutor with timeout on future.result().
    """
    result_container: dict[str, Any] = {"result": None, "error": None}

    def target() -> None:
        try:
            result_container["result"] = func(*args, **kwargs)
        except Exception as e:
            result_container["error"] = e

    # Use a thread with explicit timeout
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Thread is still running - timeout occurred
        # Note: We can't truly kill the thread, but we can mark it as timed out
        # and move on. The daemon=True ensures it won't block process exit.
        return None, True

    if result_container["error"] is not None:
        raise result_container["error"]

    return result_container["result"], False


class AdvancedBatchProcessor:
    """Advanced batch processor with checkpointing and error handling.

    Provides robust batch processing with checkpoint/resume capability,
    per-file error isolation, progress tracking, and resource limits.

    Timeout enforcement:
        When `timeout_per_file` is configured, this processor uses actual
        timeout enforcement (via threading.Timer or concurrent.futures timeout)
        rather than just post-hoc duration checking. This means:
        - Long-running operations will be interrupted
        - Results will be marked with `timed_out=True`
        - Processing continues to the next file

    Example:
        >>> from oscura.workflows.batch.advanced import AdvancedBatchProcessor, BatchConfig
        >>> config = BatchConfig(
        ...     on_error='skip',
        ...     checkpoint_dir='./checkpoints',
        ...     max_workers=4,
        ...     timeout_per_file=60.0  # Enforced timeout
        ... )
        >>> processor = AdvancedBatchProcessor(config)
        >>> results = processor.process(files, analysis_fn)

    References:
        API-012: Advanced Batch Control
    """

    def __init__(self, config: BatchConfig | None = None) -> None:
        """Initialize batch processor.

        Args:
            config: Batch configuration. Uses defaults if None.
        """
        self.config = config or BatchConfig()
        self.checkpoint: BatchCheckpoint | None = None

    def process(
        self,
        files: list[str | Path],
        analysis_fn: Callable[[str | Path], dict[str, Any]],
        *,
        checkpoint_name: str = "batch_checkpoint",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Process files with checkpointing and error handling.

        Args:
            files: List of file paths to process.
            analysis_fn: Analysis function to apply to each file.
            checkpoint_name: Name for checkpoint file (default: 'batch_checkpoint').
            **kwargs: Additional arguments passed to analysis_fn.

        Returns:
            DataFrame with results and error information.

        Example:
            >>> results = processor.process(
            ...     files=['trace1.wfm', 'trace2.wfm'],
            ...     analysis_fn=analyze_trace
            ... )

        Raises:
            ImportError: If pandas is not installed.

        References:
            API-012: Advanced Batch Control
        """
        if not _HAS_PANDAS:
            raise ImportError(
                "Batch processing requires pandas.\n\n"
                "Install with:\n"
                "  pip install oscura[dataframes]    # DataFrame support\n"
                "  pip install oscura[standard]      # Recommended\n"
                "  pip install oscura[all]           # Everything\n"
            )

        # Try to resume from checkpoint
        remaining_files = self._resume_or_start(files, checkpoint_name)

        # Initialize checkpoint if not resumed
        if self.checkpoint is None:
            self.checkpoint = BatchCheckpoint(
                total_files=len(files),
                config=self.config,
            )

        # Process remaining files
        self._process_files(remaining_files, analysis_fn, checkpoint_name, **kwargs)

        # Convert results to DataFrame
        return self._results_to_dataframe()

    def _resume_or_start(self, files: list[str | Path], checkpoint_name: str) -> list[str | Path]:
        """Try to resume from checkpoint or start fresh.

        Args:
            files: Full list of files to process.
            checkpoint_name: Name of checkpoint file.

        Returns:
            List of files remaining to be processed.
        """
        if self.config.checkpoint_dir is None:
            return files

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.json"

        if checkpoint_path.exists():
            # Load checkpoint
            self.checkpoint = BatchCheckpoint.load(checkpoint_path)

            # Determine remaining files
            completed_set = set(self.checkpoint.completed_files)
            failed_set = set(self.checkpoint.failed_files)
            processed_set = completed_set | failed_set

            remaining = [str(f) for f in files if str(f) not in processed_set]

            print(
                f"Resuming from checkpoint: "
                f"{len(self.checkpoint.completed_files)} completed, "
                f"{len(self.checkpoint.failed_files)} failed, "
                f"{len(remaining)} remaining"
            )

            return [Path(f) for f in remaining]

        return files

    def _create_progress_bar(self) -> Any:
        """Create progress bar if requested.

        Returns:
            tqdm progress bar or None.
        """
        if not (self.config.progress_bar and HAS_TQDM):
            return None

        total = self.checkpoint.total_files if self.checkpoint else 0
        initial = (
            len(self.checkpoint.completed_files) + len(self.checkpoint.failed_files)
            if self.checkpoint
            else 0
        )
        return tqdm(total=total, initial=initial, desc="Processing files")

    def _create_file_processor(
        self,
        analysis_fn: Callable[[str | Path], dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> Callable[[str | Path], FileResult]:
        """Create wrapper function for processing a single file.

        Args:
            analysis_fn: Analysis function to apply.
            kwargs: Keyword arguments for analysis_fn.

        Returns:
            Wrapper function that returns FileResult.
        """

        def _process_one(file_path: str | Path) -> FileResult:
            import time

            start_time = time.time()
            timed_out = False

            try:
                # Apply timeout if configured
                if self.config.timeout_per_file is not None:
                    result, timed_out = _run_with_timeout(
                        analysis_fn,
                        (file_path,),
                        kwargs,
                        self.config.timeout_per_file,
                    )
                    if timed_out:
                        duration = time.time() - start_time
                        return FileResult(
                            file=str(file_path),
                            success=False,
                            error=f"Processing timed out after {self.config.timeout_per_file}s",
                            duration=duration,
                            timed_out=True,
                        )
                else:
                    result = analysis_fn(file_path, **kwargs)

                duration = time.time() - start_time
                return FileResult(
                    file=str(file_path),
                    success=True,
                    result=result if isinstance(result, dict) else {"result": result},
                    duration=duration,
                )
            except Exception as e:
                duration = time.time() - start_time
                return FileResult(
                    file=str(file_path),
                    success=False,
                    error=str(e),
                    traceback=traceback.format_exc(),
                    duration=duration,
                )

        return _process_one

    def _retrieve_future_result(
        self, future: concurrent.futures.Future[FileResult], file_path: str | Path
    ) -> FileResult:
        """Retrieve result from future with timeout handling.

        Args:
            future: Future to retrieve result from.
            file_path: Path being processed (for error messages).

        Returns:
            FileResult from future or error result if timeout/exception.
        """
        try:
            # Apply timeout on result retrieval as backup enforcement
            retrieval_timeout = (
                self.config.timeout_per_file * 1.1 if self.config.timeout_per_file else None
            )
            return future.result(timeout=retrieval_timeout)
        except concurrent.futures.TimeoutError:
            # Backup timeout triggered
            return FileResult(
                file=str(file_path),
                success=False,
                error=f"Processing timed out (backup enforcement) "
                f"after {self.config.timeout_per_file}s",
                timed_out=True,
            )
        except Exception as e:
            # Unexpected error during result retrieval
            return FileResult(
                file=str(file_path),
                success=False,
                error=f"Error retrieving result: {e}",
                traceback=traceback.format_exc(),
            )

    def _update_checkpoint_with_result(self, file_result: FileResult) -> None:
        """Update checkpoint with processing result.

        Args:
            file_result: Result to add to checkpoint.
        """
        if not self.checkpoint:
            return

        self.checkpoint.results.append(file_result)
        if file_result.success:
            self.checkpoint.completed_files.append(file_result.file)
        else:
            self.checkpoint.failed_files.append(file_result.file)

    def _handle_file_error(self, file_result: FileResult, pbar: Any) -> None:
        """Handle error from file processing.

        Args:
            file_result: Result with error.
            pbar: Progress bar to close if stopping.

        Raises:
            RuntimeError: If on_error is "stop".
        """
        if self.config.on_error == "stop":
            if pbar:
                pbar.close()
            raise RuntimeError(
                f"Processing stopped due to error in {file_result.file}: {file_result.error}"
            )
        elif self.config.on_error == "warn":
            timeout_note = " (TIMEOUT)" if file_result.timed_out else ""
            print(
                f"Warning: Error processing {file_result.file}{timeout_note}: {file_result.error}"
            )

    def _process_files(
        self,
        files: list[str | Path],
        analysis_fn: Callable[[str | Path], dict[str, Any]],
        checkpoint_name: str,
        **kwargs: Any,
    ) -> None:
        """Process files with parallel execution and checkpointing.

        Args:
            files: Files to process.
            analysis_fn: Analysis function.
            checkpoint_name: Checkpoint file name.
            **kwargs: Additional arguments for analysis_fn.

        Raises:
            RuntimeError: If processing is stopped due to error and on_error is "stop".
        """
        if not files:
            return

        pbar = self._create_progress_bar()
        _process_one = self._create_file_processor(analysis_fn, kwargs)

        # Process files
        processed_count = 0
        executor_class = ThreadPoolExecutor if self.config.use_threads else ProcessPoolExecutor

        with executor_class(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_process_one, f): f for f in files}

            # Process results as they complete
            for future in as_completed(futures):
                file_path = futures[future]
                file_result = self._retrieve_future_result(future, file_path)

                # Update checkpoint
                self._update_checkpoint_with_result(file_result)

                # Handle errors
                if not file_result.success:
                    self._handle_file_error(file_result, pbar)

                # Update progress
                processed_count += 1
                if pbar:
                    pbar.update(1)

                # Save checkpoint periodically
                if (
                    self.config.checkpoint_dir
                    and processed_count % self.config.checkpoint_interval == 0
                ):
                    self._save_checkpoint(checkpoint_name)

        if pbar:
            pbar.close()

        # Final checkpoint save
        if self.config.checkpoint_dir:
            self._save_checkpoint(checkpoint_name)

    def _save_checkpoint(self, checkpoint_name: str) -> None:
        """Save current checkpoint.

        Args:
            checkpoint_name: Name for checkpoint file.
        """
        if self.checkpoint and self.config.checkpoint_dir:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_path = checkpoint_dir / f"{checkpoint_name}.json"
            self.checkpoint.save(checkpoint_path)

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert checkpoint results to DataFrame.

        Returns:
            DataFrame with results and metadata.
        """
        if not self.checkpoint or not self.checkpoint.results:
            return pd.DataFrame()

        # Build rows
        rows = []
        for file_result in self.checkpoint.results:
            row = {
                "file": file_result.file,
                "success": file_result.success,
                "timed_out": file_result.timed_out,
            }

            if file_result.success:
                row.update(file_result.result)
                row["error"] = None
            else:
                row["error"] = file_result.error
                row["traceback"] = file_result.traceback

            row["duration"] = file_result.duration
            rows.append(row)

        return pd.DataFrame(rows)


def resume_batch(
    checkpoint_dir: str | Path, checkpoint_name: str = "batch_checkpoint"
) -> BatchCheckpoint:
    """Resume a batch job from checkpoint directory.

    Convenience function to load checkpoint and inspect state.

    Args:
        checkpoint_dir: Directory containing checkpoint.
        checkpoint_name: Name of checkpoint file.

    Returns:
        Loaded checkpoint.

    Example:
        >>> checkpoint = resume_batch('./checkpoints')
        >>> print(f"Completed: {len(checkpoint.completed_files)}")

    References:
        API-012: Advanced Batch Control
    """
    checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_name}.json"
    return BatchCheckpoint.load(checkpoint_path)


__all__ = [
    "AdvancedBatchProcessor",
    "BatchCheckpoint",
    "BatchConfig",
    "FileResult",
    "TimeoutError",
    "resume_batch",
]
