"""Oscura Benchmark Command - Performance Benchmarking.

Provides CLI for performance benchmarking of analysis operations.


Example:
    $ oscura benchmark --operations all
    $ oscura benchmark --operations decode --protocol uart
"""

from __future__ import annotations

import logging
import time
from typing import Any

import click

logger = logging.getLogger("oscura.cli.benchmark")


@click.command()
@click.option(
    "--operations",
    type=click.Choice(["all", "load", "decode", "fft", "measurements"], case_sensitive=False),
    default="all",
    help="Operations to benchmark.",
)
@click.option(
    "--protocol",
    type=str,
    default="uart",
    help="Protocol for decode benchmark.",
)
@click.option(
    "--iterations",
    type=int,
    default=10,
    help="Number of iterations per benchmark.",
)
@click.option(
    "--output",
    type=click.Choice(["json", "table"], case_sensitive=False),
    default="table",
    help="Output format.",
)
@click.pass_context
def benchmark(
    ctx: click.Context,
    operations: str,
    protocol: str,
    iterations: int,
    output: str,
) -> None:
    """Run performance benchmarks.

    Measures performance of core operations like loading, decoding,
    FFT, and measurements.

    Args:
        ctx: Click context object.
        operations: Operations to benchmark.
        protocol: Protocol for decode benchmark.
        iterations: Number of iterations.
        output: Output format.

    Examples:

        \b
        # Benchmark all operations
        $ oscura benchmark --operations all

        \b
        # Benchmark specific operation
        $ oscura benchmark --operations decode --protocol uart --iterations 100
    """
    verbose = ctx.obj.get("verbose", 0)

    if verbose:
        logger.info(f"Running benchmark: {operations}")

    try:
        results: dict[str, Any] = {"iterations": iterations, "benchmarks": {}}

        # Generate test data
        test_data = _generate_test_data()

        # Run benchmarks
        if operations in ["all", "load"]:
            results["benchmarks"]["load"] = _benchmark_load(test_data, iterations)

        if operations in ["all", "decode"]:
            results["benchmarks"]["decode"] = _benchmark_decode(test_data, protocol, iterations)

        if operations in ["all", "fft"]:
            results["benchmarks"]["fft"] = _benchmark_fft(test_data, iterations)

        if operations in ["all", "measurements"]:
            results["benchmarks"]["measurements"] = _benchmark_measurements(test_data, iterations)

        # Output results
        if output == "json":
            import json

            click.echo(json.dumps(results, indent=2))
        else:
            _print_table(results)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if verbose > 1:
            raise
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _generate_test_data() -> Any:
    """Generate test data for benchmarking.

    Returns:
        Test waveform trace.
    """
    import numpy as np

    from oscura.core.types import TraceMetadata, WaveformTrace

    # Generate 100k sample waveform
    samples = 100000
    sample_rate = 1e6
    t = np.arange(samples) / sample_rate

    # Mix of sine waves
    data = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 5000 * t)

    return WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))


def _benchmark_load(test_data: Any, iterations: int) -> dict[str, Any]:
    """Benchmark data loading.

    Args:
        test_data: Test data.
        iterations: Number of iterations.

    Returns:
        Benchmark results.
    """
    import tempfile
    from pathlib import Path

    import numpy as np

    from oscura.loaders import load

    # Save test data to temp file (use .npz which is supported)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        temp_path = Path(f.name)
        np.savez(temp_path, data=test_data.data)

    try:
        start = time.time()
        for _ in range(iterations):
            _ = load(str(temp_path))
        elapsed = time.time() - start

        return {
            "total_time": f"{elapsed:.3f}s",
            "avg_time": f"{elapsed / iterations * 1000:.2f}ms",
            "throughput": f"{iterations / elapsed:.1f} ops/sec",
        }
    finally:
        temp_path.unlink()


def _benchmark_decode(test_data: Any, protocol: str, iterations: int) -> dict[str, Any]:
    """Benchmark protocol decoding.

    Args:
        test_data: Test data.
        protocol: Protocol name.
        iterations: Number of iterations.

    Returns:
        Benchmark results.
    """
    import numpy as np

    from oscura.core.types import DigitalTrace

    # Convert to digital
    threshold = np.mean(test_data.data)
    digital = test_data.data > threshold
    digital_trace = DigitalTrace(data=digital, metadata=test_data.metadata)

    start = time.time()
    for _ in range(iterations):
        if protocol.lower() == "uart":
            from oscura.analyzers.protocols.uart import UARTDecoder

            decoder = UARTDecoder(baudrate=9600)
            _ = list(decoder.decode(digital_trace))

    elapsed = time.time() - start

    return {
        "protocol": protocol,
        "total_time": f"{elapsed:.3f}s",
        "avg_time": f"{elapsed / iterations * 1000:.2f}ms",
        "throughput": f"{iterations / elapsed:.1f} ops/sec",
    }


def _benchmark_fft(test_data: Any, iterations: int) -> dict[str, Any]:
    """Benchmark FFT computation.

    Args:
        test_data: Test data.
        iterations: Number of iterations.

    Returns:
        Benchmark results.
    """
    from oscura.analyzers.waveform.spectral import fft

    start = time.time()
    for _ in range(iterations):
        _ = fft(test_data)
    elapsed = time.time() - start

    return {
        "total_time": f"{elapsed:.3f}s",
        "avg_time": f"{elapsed / iterations * 1000:.2f}ms",
        "throughput": f"{iterations / elapsed:.1f} ops/sec",
    }


def _benchmark_measurements(test_data: Any, iterations: int) -> dict[str, Any]:
    """Benchmark waveform measurements.

    Args:
        test_data: Test data.
        iterations: Number of iterations.

    Returns:
        Benchmark results.
    """
    from oscura.analyzers.waveform.measurements import fall_time, rise_time

    start = time.time()
    for _ in range(iterations):
        _ = rise_time(test_data)
        _ = fall_time(test_data)
    elapsed = time.time() - start

    return {
        "total_time": f"{elapsed:.3f}s",
        "avg_time": f"{elapsed / iterations * 1000:.2f}ms",
        "throughput": f"{iterations / elapsed:.1f} ops/sec",
    }


def _print_table(results: dict[str, Any]) -> None:
    """Print results as table.

    Args:
        results: Benchmark results.
    """
    click.echo("\n=== Benchmark Results ===\n")
    click.echo(f"Iterations: {results['iterations']}\n")

    for name, bench in results["benchmarks"].items():
        click.echo(f"{name.upper()}:")
        for key, value in bench.items():
            click.echo(f"  {key}: {value}")
        click.echo()
