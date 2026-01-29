"""Parallel processing for multi-core hardware analysis.

This module provides parallelization strategies for CPU-bound and I/O-bound tasks
in hardware reverse engineering workflows. It enables efficient processing of
multiple signals, protocols, files, and analysis operations across CPU cores.

Key capabilities:
- Process pool for CPU-intensive analysis (FFT, correlation, protocol decoding)
- Thread pool for I/O-bound operations (file loading, network requests)
- Batch processing (multiple files in parallel)
- Pipeline parallelism (different stages running concurrently)
- Data parallelism (split large datasets across workers)
- Automatic worker count based on CPU topology
- Progress tracking with tqdm integration
- Graceful error handling in worker processes

Typical use cases:
- Decode multiple protocol messages simultaneously
- Parallel FFT/spectral analysis on signal chunks
- Load multiple capture files at once
- Generate multiple export formats (Wireshark, Scapy, Kaitai) concurrently
- Batch CRC recovery across message sets
- Parallel side-channel analysis on traces

Performance expectations:
- CPU-bound tasks: ~(N-1)x speedup on N cores
- I/O-bound tasks: ~2-4x speedup with threading
- Mixed workloads: Automatic strategy selection

Example:
    >>> from oscura.utils.performance.parallel import ParallelProcessor, ParallelConfig
    >>> # Configure parallel processor
    >>> config = ParallelConfig(num_workers=4, strategy="process")
    >>> processor = ParallelProcessor(config)
    >>>
    >>> # Parallel protocol decoding
    >>> messages = [...]  # List of message bytes
    >>> def decode_message(msg):
    ...     return protocol_decoder.decode(msg)
    >>> result = processor.map(decode_message, messages)
    >>> print(f"Decoded {len(result.results)} messages in {result.execution_time:.2f}s")
    >>> print(f"Speedup: {result.speedup:.2f}x")
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

# Strategy types for parallelization
StrategyType = Literal["process", "thread", "auto"]


@dataclass
class ParallelConfig:
    """Configuration for parallel processing.

    Attributes:
        num_workers: Number of worker processes/threads. If None, uses CPU count.
            For process strategy: Defaults to cpu_count() - 1 (leave one core free)
            For thread strategy: Defaults to cpu_count() * 2 (I/O-bound tasks)
        strategy: Parallelization strategy.
            "process": Use multiprocessing (CPU-bound tasks like FFT, correlation)
            "thread": Use threading (I/O-bound tasks like file loading)
            "auto": Automatically select based on task characteristics
        batch_size: Number of items per worker batch. If None, auto-calculated.
            Larger batches reduce overhead but may cause load imbalance.
            Smaller batches improve load balancing but increase overhead.
        show_progress: Enable tqdm progress bar for long-running operations.
        timeout: Maximum execution time per task in seconds. None for no timeout.

    Example:
        >>> # CPU-bound tasks (protocol decoding, FFT)
        >>> config = ParallelConfig(num_workers=4, strategy="process")
        >>>
        >>> # I/O-bound tasks (file loading)
        >>> config = ParallelConfig(num_workers=8, strategy="thread")
        >>>
        >>> # Auto-detect strategy
        >>> config = ParallelConfig(strategy="auto")
    """

    num_workers: int | None = None
    strategy: StrategyType = "auto"
    batch_size: int | None = None
    show_progress: bool = False
    timeout: float | None = None

    def __post_init__(self) -> None:
        """Validate configuration and set defaults."""
        if self.num_workers is not None and self.num_workers < 1:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}")

        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")


@dataclass
class WorkerStats:
    """Statistics for individual worker performance.

    Attributes:
        worker_id: Worker identifier (0-indexed).
        tasks_completed: Number of tasks processed by this worker.
        execution_time: Total execution time for this worker in seconds.
        errors: Number of errors encountered by this worker.
    """

    worker_id: int
    tasks_completed: int
    execution_time: float
    errors: int


@dataclass
class ParallelResult:
    """Results from parallel processing operation.

    Attributes:
        results: List of results from each task (same order as input).
        execution_time: Total wall-clock time in seconds.
        speedup: Speedup factor vs sequential execution (estimated or measured).
        worker_stats: Per-worker performance statistics.
        num_workers: Number of workers used.
        strategy: Parallelization strategy used ("process" or "thread").
        errors: List of (index, exception) tuples for failed tasks.

    Example:
        >>> result = processor.map(decode_fn, messages)
        >>> print(f"Processed {len(result.results)} items in {result.execution_time:.2f}s")
        >>> print(f"Speedup: {result.speedup:.2f}x vs sequential")
        >>> print(f"Workers: {result.num_workers}")
        >>> if result.errors:
        ...     print(f"Failed tasks: {len(result.errors)}")
    """

    results: list[Any]
    execution_time: float
    speedup: float
    worker_stats: list[WorkerStats] = field(default_factory=list)
    num_workers: int = 1
    strategy: str = "sequential"
    errors: list[tuple[int, Exception]] = field(default_factory=list)


class ParallelProcessor:
    """Parallel processing manager for hardware analysis tasks.

    This class provides high-level parallelization APIs for common hardware
    reverse engineering workflows. It automatically manages worker pools,
    distributes work efficiently, aggregates results, and tracks progress.

    Supported patterns:
        - map: Apply function to each item independently (embarrassingly parallel)
        - batch_process: Process batches of items with custom aggregation
        - pipeline: Chain multiple processing stages with different strategies

    Worker management:
        - Automatic worker count based on CPU topology
        - Process pools for CPU-bound tasks (bypass GIL)
        - Thread pools for I/O-bound tasks
        - Graceful error handling (tasks fail independently)
        - Timeout support for long-running tasks

    Example:
        >>> from oscura.utils.performance.parallel import ParallelProcessor, ParallelConfig
        >>> # CPU-bound: Parallel FFT analysis
        >>> config = ParallelConfig(num_workers=4, strategy="process")
        >>> processor = ParallelProcessor(config)
        >>> signals = [...]  # List of signal arrays
        >>> result = processor.map(compute_fft, signals)
        >>>
        >>> # I/O-bound: Parallel file loading
        >>> config = ParallelConfig(num_workers=8, strategy="thread")
        >>> processor = ParallelProcessor(config)
        >>> files = [Path("file1.wfm"), Path("file2.wfm")]
        >>> result = processor.map(load_waveform, files)
        >>>
        >>> # Batch processing with custom aggregation
        >>> def process_batch(messages):
        ...     return analyze_protocol_batch(messages)
        >>> result = processor.batch_process(all_messages, batch_fn=process_batch)
    """

    def __init__(self, config: ParallelConfig | None = None) -> None:
        """Initialize parallel processor with configuration.

        Args:
            config: Parallel processing configuration. If None, uses defaults.

        Example:
            >>> # Default configuration (auto-detect strategy)
            >>> processor = ParallelProcessor()
            >>>
            >>> # Custom configuration
            >>> config = ParallelConfig(num_workers=4, strategy="process")
            >>> processor = ParallelProcessor(config)
        """
        self.config = config or ParallelConfig()
        self._cpu_count = mp.cpu_count()

        logger.debug(
            f"ParallelProcessor initialized: strategy={self.config.strategy}, "
            f"workers={self._get_worker_count()}, cpus={self._cpu_count}"
        )

    def map(
        self,
        func: Callable[[Any], Any],
        items: Sequence[Any],
        sequential_time: float | None = None,
    ) -> ParallelResult:
        """Apply function to each item in parallel.

        This is the primary method for embarrassingly parallel tasks where
        each item can be processed independently without shared state.

        Args:
            func: Function to apply to each item. Must be picklable for process strategy.
            items: Sequence of items to process.
            sequential_time: Optional baseline sequential execution time for
                accurate speedup calculation. If None, speedup is estimated.

        Returns:
            ParallelResult with results, timing, and worker statistics.

        Raises:
            ValueError: If items is empty.

        Example:
            >>> # Parallel protocol decoding
            >>> def decode(msg):
            ...     return protocol.decode(msg)
            >>> messages = [b"\\x01\\x02", b"\\x03\\x04", b"\\x05\\x06"]
            >>> result = processor.map(decode, messages)
            >>> decoded = result.results
            >>>
            >>> # With progress tracking
            >>> config = ParallelConfig(show_progress=True)
            >>> processor = ParallelProcessor(config)
            >>> result = processor.map(expensive_function, large_dataset)
        """
        if not items:
            raise ValueError("Cannot process empty item list")

        start_time = time.time()

        # Determine strategy
        strategy = self._resolve_strategy(func, items)
        num_workers = self._get_worker_count(strategy)

        logger.info(
            f"Starting parallel map: {len(items)} items, {num_workers} workers, strategy={strategy}"
        )

        # Execute based on strategy
        if strategy == "process":
            results, errors = self._map_process(func, items, num_workers)
        elif strategy == "thread":
            results, errors = self._map_thread(func, items, num_workers)
        else:
            # Sequential fallback
            results, errors = self._map_sequential(func, items)
            num_workers = 1
            strategy = "sequential"

        execution_time = time.time() - start_time

        # Calculate speedup
        if sequential_time is not None:
            speedup = sequential_time / execution_time
        else:
            # Estimate speedup (assumes linear scaling with slight overhead)
            if strategy == "process":
                speedup = min(num_workers * 0.85, num_workers)  # 15% overhead
            elif strategy == "thread":
                speedup = min(num_workers * 0.6, 4.0)  # I/O-bound limited to ~4x
            else:
                speedup = 1.0

        logger.info(
            f"Parallel map completed: {len(results)} results, "
            f"time={execution_time:.2f}s, speedup={speedup:.2f}x, "
            f"errors={len(errors)}"
        )

        return ParallelResult(
            results=results,
            execution_time=execution_time,
            speedup=speedup,
            num_workers=num_workers,
            strategy=strategy,
            errors=errors,
        )

    def batch_process(
        self,
        items: Sequence[Any],
        batch_fn: Callable[[Sequence[Any]], Any],
        batch_size: int | None = None,
    ) -> ParallelResult:
        """Process items in batches across workers.

        Useful when:
        - Items should be grouped for efficiency (e.g., database bulk inserts)
        - Batch-level aggregation is needed (e.g., statistical analysis)
        - Setup/teardown cost is high (e.g., model loading)

        Args:
            items: Sequence of items to process.
            batch_fn: Function that processes a batch of items and returns result.
            batch_size: Items per batch. If None, auto-calculated based on config.

        Returns:
            ParallelResult with batch results (one per batch).

        Raises:
            ValueError: If items is empty.

        Example:
            >>> # Batch CRC recovery
            >>> def recover_batch_crc(messages):
            ...     return crc_reverser.analyze_batch(messages)
            >>> all_messages = [...]  # 10000 messages
            >>> result = processor.batch_process(
            ...     all_messages,
            ...     batch_fn=recover_batch_crc,
            ...     batch_size=100
            ... )
            >>> # Result contains 100 batch results (100 messages each)
        """
        if not items:
            raise ValueError("Cannot process empty item list")

        # Determine batch size
        if batch_size is None:
            batch_size = self._calculate_batch_size(len(items))

        # Create batches
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        logger.info(
            f"Starting batch processing: {len(items)} items, "
            f"{len(batches)} batches of size ~{batch_size}"
        )

        # Process batches in parallel
        return self.map(batch_fn, batches)

    def pipeline(
        self,
        stages: Sequence[tuple[Callable[[Any], Any], StrategyType]],
        items: Sequence[Any],
    ) -> ParallelResult:
        """Execute multi-stage pipeline with per-stage parallelization.

        Each stage can use a different parallelization strategy based on
        whether it's CPU-bound or I/O-bound. Results flow from one stage
        to the next.

        Args:
            stages: Sequence of (function, strategy) tuples defining pipeline.
                Each function receives output from previous stage.
            items: Initial items to process.

        Returns:
            ParallelResult from final stage.

        Raises:
            ValueError: If stages or items are empty.

        Example:
            >>> # Multi-stage analysis pipeline
            >>> stages = [
            ...     (load_signal, "thread"),      # I/O-bound
            ...     (compute_fft, "process"),     # CPU-bound
            ...     (detect_peaks, "process"),    # CPU-bound
            ...     (export_results, "thread"),   # I/O-bound
            ... ]
            >>> files = [Path("sig1.bin"), Path("sig2.bin")]
            >>> result = processor.pipeline(stages, files)
        """
        if not stages:
            raise ValueError("Pipeline must have at least one stage")
        if not items:
            raise ValueError("Cannot process empty item list")

        current_items = items
        total_time = 0.0

        for i, (func, strategy) in enumerate(stages):
            logger.info(f"Executing pipeline stage {i + 1}/{len(stages)}: strategy={strategy}")

            # Save original strategy and override for this stage
            original_strategy = self.config.strategy
            self.config.strategy = strategy

            result = self.map(func, current_items)

            # Restore original strategy
            self.config.strategy = original_strategy

            current_items = result.results
            total_time += result.execution_time

            if result.errors:
                logger.warning(f"Stage {i + 1} completed with {len(result.errors)} errors")

        logger.info(f"Pipeline completed: {len(stages)} stages, total_time={total_time:.2f}s")

        # Return result from final stage with cumulative timing
        result.execution_time = total_time
        return result

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    def _resolve_strategy(self, func: Callable[[Any], Any], items: Sequence[Any]) -> str:
        """Determine appropriate parallelization strategy.

        Args:
            func: Function to be parallelized.
            items: Items to process.

        Returns:
            Strategy name: "process", "thread", or "sequential".
        """
        if self.config.strategy != "auto":
            return self.config.strategy

        # Auto-detection heuristics
        # 1. Small datasets (<10 items) - sequential is faster
        if len(items) < 10:
            logger.debug("Auto-selecting sequential (small dataset)")
            return "sequential"

        # 2. Function name heuristics
        func_name = func.__name__.lower()
        if any(
            keyword in func_name for keyword in ["fft", "correlate", "decode", "analyze", "compute"]
        ):
            logger.debug(f"Auto-selecting process (CPU-bound function: {func_name})")
            return "process"

        if any(keyword in func_name for keyword in ["load", "read", "fetch", "download"]):
            logger.debug(f"Auto-selecting thread (I/O-bound function: {func_name})")
            return "thread"

        # 3. Default to process for medium+ datasets
        logger.debug("Auto-selecting process (default for parallel workload)")
        return "process"

    def _get_worker_count(self, strategy: str = "process") -> int:
        """Get appropriate worker count for strategy.

        Args:
            strategy: Parallelization strategy.

        Returns:
            Number of workers to use.
        """
        if self.config.num_workers is not None:
            return self.config.num_workers

        # Auto-calculate based on strategy
        if strategy == "process":
            # Leave one core free for OS/orchestration
            return max(1, self._cpu_count - 1)
        elif strategy == "thread":
            # I/O-bound tasks can use more threads than cores
            return self._cpu_count * 2
        else:
            return 1

    def _calculate_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size for dataset.

        Args:
            total_items: Total number of items to process.

        Returns:
            Optimal batch size.
        """
        if self.config.batch_size is not None:
            return self.config.batch_size

        num_workers = self._get_worker_count()

        # Target: 4 batches per worker (allows good load balancing)
        optimal_batches = num_workers * 4
        batch_size = max(1, total_items // optimal_batches)

        logger.debug(
            f"Auto-calculated batch_size={batch_size} "
            f"({total_items} items / {optimal_batches} batches)"
        )

        return batch_size

    def _map_process(
        self,
        func: Callable[[Any], Any],
        items: Sequence[Any],
        num_workers: int,
    ) -> tuple[list[Any], list[tuple[int, Exception]]]:
        """Execute map using process pool.

        Args:
            func: Function to apply.
            items: Items to process.
            num_workers: Number of worker processes.

        Returns:
            Tuple of (results list, errors list).
        """
        results: list[Any] = [None] * len(items)
        errors: list[tuple[int, Exception]] = []

        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                futures = {executor.submit(func, item): idx for idx, item in enumerate(items)}

                # Collect results with optional progress bar
                iterator: Iterable[Any]
                if self.config.show_progress:
                    try:
                        from tqdm import tqdm

                        iterator = tqdm(futures, total=len(items), desc="Processing")
                    except ImportError:
                        logger.warning("tqdm not available, progress bar disabled")
                        iterator = futures
                else:
                    iterator = futures

                for future in iterator:
                    idx = futures[future]
                    try:
                        result = future.result(timeout=self.config.timeout)
                        results[idx] = result
                    except Exception as e:
                        logger.warning(f"Task {idx} failed: {e}")
                        errors.append((idx, e))
                        results[idx] = None

        except Exception as e:
            logger.error(f"Process pool execution failed: {e}")
            raise

        return results, errors

    def _map_thread(
        self,
        func: Callable[[Any], Any],
        items: Sequence[Any],
        num_workers: int,
    ) -> tuple[list[Any], list[tuple[int, Exception]]]:
        """Execute map using thread pool.

        Args:
            func: Function to apply.
            items: Items to process.
            num_workers: Number of worker threads.

        Returns:
            Tuple of (results list, errors list).
        """
        results: list[Any] = [None] * len(items)
        errors: list[tuple[int, Exception]] = []

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                futures = {executor.submit(func, item): idx for idx, item in enumerate(items)}

                # Collect results with optional progress bar
                iterator: Iterable[Any]
                if self.config.show_progress:
                    try:
                        from tqdm import tqdm

                        iterator = tqdm(futures, total=len(items), desc="Processing")
                    except ImportError:
                        logger.warning("tqdm not available, progress bar disabled")
                        iterator = futures
                else:
                    iterator = futures

                for future in iterator:
                    idx = futures[future]
                    try:
                        result = future.result(timeout=self.config.timeout)
                        results[idx] = result
                    except Exception as e:
                        logger.warning(f"Task {idx} failed: {e}")
                        errors.append((idx, e))
                        results[idx] = None

        except Exception as e:
            logger.error(f"Thread pool execution failed: {e}")
            raise

        return results, errors

    def _map_sequential(
        self,
        func: Callable[[Any], Any],
        items: Sequence[Any],
    ) -> tuple[list[Any], list[tuple[int, Exception]]]:
        """Execute map sequentially (fallback/baseline).

        Args:
            func: Function to apply.
            items: Items to process.

        Returns:
            Tuple of (results list, errors list).
        """
        results: list[Any] = []
        errors: list[tuple[int, Exception]] = []

        for idx, item in enumerate(items):
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                logger.warning(f"Task {idx} failed: {e}")
                errors.append((idx, e))
                results.append(None)

        return results, errors


__all__ = [
    "ParallelConfig",
    "ParallelProcessor",
    "ParallelResult",
    "WorkerStats",
]
