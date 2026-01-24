"""Parallel processing utilities for optimization and analysis.

This module provides utilities for efficient parallel execution of analysis tasks
using both thread and process-based parallelism.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from oscura.core.exceptions import AnalysisError

T = TypeVar("T")
R = TypeVar("R")

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ParallelResult(Generic[R]):
    """Result from parallel execution.

    Attributes:
        results: List of results from all tasks.
        execution_time: Total execution time in seconds.
        success_count: Number of successfully completed tasks.
        error_count: Number of failed tasks.
        errors: List of exceptions encountered.

    Example:
        >>> result = parallel_map(fn, items)
        >>> print(f"Completed {result.success_count}/{len(items)}")
    """

    results: list[R]
    execution_time: float
    success_count: int
    error_count: int
    errors: list[Exception] | None = None


@dataclass
class WorkerPool:
    """Configuration for worker pool management.

    Attributes:
        max_workers: Maximum number of workers.
        use_threads: Use threads (True) or processes (False).
        timeout: Timeout per task in seconds.
        chunk_size: Number of items per worker chunk.

    Example:
        >>> pool = WorkerPool(max_workers=4, use_threads=True, timeout=30)
    """

    max_workers: int = 4
    use_threads: bool = True
    timeout: float | None = None
    chunk_size: int = 1


def get_optimal_workers(max_workers: int | None = None) -> int:
    """Get optimal number of workers for current system.

    Uses CPU count by default, respecting max_workers limit.

    Args:
        max_workers: Maximum workers to use. None for all CPUs.

    Returns:
        Optimal number of workers.

    Example:
        >>> workers = get_optimal_workers(max_workers=8)
    """
    import os

    cpu_count = os.cpu_count() or 1
    if max_workers is None:
        return cpu_count
    return min(max_workers, cpu_count)


def parallel_map(
    func: Callable[[T], R],
    iterable: Iterable[T],
    *,
    max_workers: int | None = None,
    use_threads: bool = True,
    timeout: float | None = None,
    collect_errors: bool = True,
) -> ParallelResult[R]:
    """Apply function to items in parallel.

    Maps a function over an iterable using either threads or processes.

    Args:
        func: Function to apply to each item.
        iterable: Items to process.
        max_workers: Maximum concurrent workers.
        use_threads: Use threads (True) or processes (False).
        timeout: Timeout per task in seconds.
        collect_errors: Collect errors instead of raising.

    Returns:
        ParallelResult with results and execution stats.

    Raises:
        AnalysisError: If collect_errors=False and a task fails.

    Example:
        >>> def process_item(x):
        ...     return x * 2
        >>> result = parallel_map(process_item, range(100))
        >>> print(f"Completed: {result.success_count}")

    References:
        OPT-001: Parallel Execution Framework
    """
    import time

    items = list(iterable)
    if not items:
        return ParallelResult(results=[], execution_time=0.0, success_count=0, error_count=0)

    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    max_workers = get_optimal_workers(max_workers)

    start_time = time.time()
    results: list[R] = [None] * len(items)  # type: ignore[list-item]
    errors: list[Exception] = []
    success_count = 0
    error_count = 0

    with executor_class(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): i for i, item in enumerate(items)}

        for future in as_completed(futures, timeout=timeout):
            idx = futures[future]
            try:
                results[idx] = future.result()
                success_count += 1
            except Exception as e:
                error_count += 1
                errors.append(e)

                if not collect_errors:
                    execution_time = time.time() - start_time
                    raise AnalysisError(f"Task {idx} failed: {e!s}") from e

    execution_time = time.time() - start_time

    return ParallelResult(
        results=results,
        execution_time=execution_time,
        success_count=success_count,
        error_count=error_count,
        errors=errors if errors else None,
    )


def parallel_reduce(
    func: Callable[[T], R],
    iterable: Iterable[T],
    reducer: Callable[[list[R]], Any],
    *,
    max_workers: int | None = None,
    use_threads: bool = True,
    timeout: float | None = None,
) -> Any:
    """Map and reduce results in parallel.

    Applies function to items in parallel, then reduces results.

    Args:
        func: Function to apply to each item.
        iterable: Items to process.
        reducer: Function to reduce list of results.
        max_workers: Maximum concurrent workers.
        use_threads: Use threads (True) or processes (False).
        timeout: Timeout per task in seconds.

    Returns:
        Reduced result.

    Example:
        >>> def compute(x):
        ...     return x * 2
        >>> result = parallel_reduce(
        ...     compute,
        ...     range(100),
        ...     reducer=lambda x: sum(x)
        ... )

    References:
        OPT-001: Parallel Execution Framework
    """
    result = parallel_map(
        func,
        iterable,
        max_workers=max_workers,
        use_threads=use_threads,
        timeout=timeout,
        collect_errors=False,
    )

    return reducer(result.results)


def batch_parallel_map(
    func: Callable[[list[T]], list[R]],
    iterable: Iterable[T],
    *,
    batch_size: int = 100,
    max_workers: int | None = None,
    use_threads: bool = True,
    timeout: float | None = None,
) -> ParallelResult[R]:
    """Apply function to batches of items in parallel.

    Processes items in batches, useful when function benefits from
    batch processing.

    Args:
        func: Function accepting list of items.
        iterable: Items to process.
        batch_size: Number of items per batch.
        max_workers: Maximum concurrent workers.
        use_threads: Use threads (True) or processes (False).
        timeout: Timeout per batch in seconds.

    Returns:
        ParallelResult with flattened results.

    Example:
        >>> def process_batch(items):
        ...     return [x * 2 for x in items]
        >>> result = batch_parallel_map(
        ...     process_batch,
        ...     range(1000),
        ...     batch_size=100
        ... )

    References:
        OPT-001: Parallel Execution Framework
    """
    import time

    items = list(iterable)
    if not items:
        return ParallelResult(results=[], execution_time=0.0, success_count=0, error_count=0)

    # Create batches
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    start_time = time.time()
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    max_workers = get_optimal_workers(max_workers)

    all_results: list[R] = []
    errors: list[Exception] = []
    success_count = 0
    error_count = 0

    with executor_class(max_workers=max_workers) as executor:
        futures = {executor.submit(func, batch): i for i, batch in enumerate(batches)}

        for future in as_completed(futures, timeout=timeout):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                success_count += 1
            except Exception as e:
                error_count += 1
                errors.append(e)

    execution_time = time.time() - start_time

    return ParallelResult(
        results=all_results,
        execution_time=execution_time,
        success_count=success_count,
        error_count=error_count,
        errors=errors if errors else None,
    )


def parallel_filter(
    func: Callable[[T], bool],
    iterable: Iterable[T],
    *,
    max_workers: int | None = None,
    use_threads: bool = True,
    timeout: float | None = None,
) -> ParallelResult[T]:
    """Filter items in parallel.

    Applies predicate to items in parallel, filtering results.

    Args:
        func: Predicate function returning True to keep item.
        iterable: Items to filter.
        max_workers: Maximum concurrent workers.
        use_threads: Use threads (True) or processes (False).
        timeout: Timeout per task in seconds.

    Returns:
        ParallelResult with filtered items.

    Example:
        >>> def is_even(x):
        ...     return x % 2 == 0
        >>> result = parallel_filter(is_even, range(100))

    References:
        OPT-001: Parallel Execution Framework
    """
    import time

    items = list(iterable)
    if not items:
        return ParallelResult(results=[], execution_time=0.0, success_count=0, error_count=0)

    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    max_workers = get_optimal_workers(max_workers)

    start_time = time.time()
    results: list[T] = []
    errors: list[Exception] = []
    success_count = 0
    error_count = 0

    with executor_class(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): item for item in items}

        for future in as_completed(futures, timeout=timeout):
            item = futures[future]
            try:
                if future.result():
                    results.append(item)
                success_count += 1
            except Exception as e:
                error_count += 1
                errors.append(e)

    execution_time = time.time() - start_time

    return ParallelResult(
        results=results,
        execution_time=execution_time,
        success_count=success_count,
        error_count=error_count,
        errors=errors if errors else None,
    )


def chunked_parallel_map(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    data: NDArray[np.float64],
    *,
    chunk_size: int = 10000,
    max_workers: int | None = None,
    use_threads: bool = True,
    timeout: float | None = None,
) -> NDArray[np.float64]:
    """Apply function to chunks of array data in parallel.

    Useful for processing large arrays where parallelization overhead
    is justified.

    Args:
        func: Function accepting 1D array chunk.
        data: Array to process.
        chunk_size: Number of samples per chunk.
        max_workers: Maximum concurrent workers.
        use_threads: Use threads (True) or processes (False).
        timeout: Timeout per chunk in seconds.

    Returns:
        Processed array (concatenated chunks).

    Raises:
        AnalysisError: If processing fails.

    Example:
        >>> def process_chunk(chunk):
        ...     return np.fft.fft(chunk)
        >>> result = chunked_parallel_map(process_chunk, data, chunk_size=1000)

    References:
        OPT-001: Parallel Execution Framework
    """
    if len(data) == 0:
        return np.array([])

    if len(data) <= chunk_size:
        return func(data)

    # Create chunks
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    max_workers = get_optimal_workers(max_workers)

    results: list[NDArray[np.float64]] = []

    with executor_class(max_workers=max_workers) as executor:
        futures = {executor.submit(func, chunk): i for i, chunk in enumerate(chunks)}

        for future in as_completed(futures, timeout=timeout):
            try:
                results.append(future.result())
            except Exception as e:
                raise AnalysisError(f"Chunk processing failed: {e!s}") from e

    return np.concatenate(results)


__all__ = [
    "ParallelResult",
    "WorkerPool",
    "batch_parallel_map",
    "chunked_parallel_map",
    "get_optimal_workers",
    "parallel_filter",
    "parallel_map",
    "parallel_reduce",
]
