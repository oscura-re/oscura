"""Generate performance visualization charts from profiling results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_results(results_dir: Path) -> dict:
    """Load all JSON result files."""
    results = {}
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            results[json_file.stem] = json.load(f)
    return results


def plot_loading_performance(metrics: list[dict], output_dir: Path) -> None:
    """Create loading performance curves."""
    # Extract data
    file_sizes_mb = [m["file_size_mb"] for m in metrics]
    throughput_msa = [m["throughput_msa_per_sec"] for m in metrics]
    duration_ms = [m["duration_seconds"] * 1000 for m in metrics]

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Throughput vs File Size
    ax1.scatter(file_sizes_mb, throughput_msa, alpha=0.6, s=50)
    ax1.set_xlabel("File Size (MB)")
    ax1.set_ylabel("Throughput (MSa/s)")
    ax1.set_title("Loading Throughput vs File Size")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")

    # Add trend line
    log_sizes = np.log10(file_sizes_mb)
    coeffs = np.polyfit(log_sizes, throughput_msa, 2)
    x_trend = np.logspace(np.log10(min(file_sizes_mb)), np.log10(max(file_sizes_mb)), 100)
    y_trend = np.polyval(coeffs, np.log10(x_trend))
    ax1.plot(x_trend, y_trend, "r--", alpha=0.5, label="Trend")
    ax1.legend()

    # Plot 2: Load Time vs File Size
    ax2.scatter(file_sizes_mb, duration_ms, alpha=0.6, s=50, color="green")
    ax2.set_xlabel("File Size (MB)")
    ax2.set_ylabel("Load Time (ms)")
    ax2.set_title("Load Time vs File Size")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    # Add linear trend line in log-log space
    log_durations = np.log10(duration_ms)
    coeffs2 = np.polyfit(log_sizes, log_durations, 1)
    y_trend2 = np.polyval(coeffs2, np.log10(x_trend))
    ax2.plot(x_trend, 10**y_trend2, "r--", alpha=0.5, label=f"Slope: {coeffs2[0]:.2f}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "loading_performance.png", dpi=150)
    print(f"Saved: {output_dir / 'loading_performance.png'}")
    plt.close()


def plot_memory_efficiency(metrics: list[dict], output_dir: Path) -> None:
    """Create memory efficiency visualization."""
    # Extract data
    samples = [m["samples_processed"] for m in metrics]
    memory_mb = [m["peak_memory_mb"] for m in metrics]
    file_sizes_mb = [m["file_size_mb"] for m in metrics]

    # Calculate theoretical memory (float64 = 8 bytes/sample)
    theoretical_mb = [s * 8 / 1024 / 1024 for s in samples]

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Actual vs Theoretical Memory
    ax1.scatter(samples, memory_mb, alpha=0.6, s=50, label="Actual")
    ax1.plot(samples, theoretical_mb, "r--", alpha=0.5, label="Theoretical (8 bytes/sample)")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Memory (MB)")
    ax1.set_title("Memory Usage vs Data Size")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()

    # Plot 2: Memory Overhead Percentage
    overhead_pct = [
        (a / t - 1) * 100 if t > 0 else 0 for a, t in zip(memory_mb, theoretical_mb, strict=False)
    ]
    ax2.scatter(samples, overhead_pct, alpha=0.6, s=50, color="orange")
    ax2.axhline(y=100, color="r", linestyle="--", alpha=0.5, label="100% overhead")
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Memory Overhead (%)")
    ax2.set_title("Memory Overhead vs Data Size")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "memory_efficiency.png", dpi=150)
    print(f"Saved: {output_dir / 'memory_efficiency.png'}")
    plt.close()


def plot_complexity_curves(complexity: dict, output_dir: Path) -> None:
    """Create algorithm complexity visualization."""
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot statistics complexity
    stats_data = complexity["statistics"]["timings"]
    sizes = [d["size"] for d in stats_data]
    times_ms = [d["time"] * 1000 for d in stats_data]
    exponent = complexity["statistics"]["complexity_exponent"]

    ax1.scatter(sizes, times_ms, alpha=0.6, s=50)
    ax1.set_xlabel("Samples (n)")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title(f"Statistics Computation: O(n^{exponent:.2f})")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Add fitted curve
    log_sizes = np.log10(sizes)
    log_times = np.log10(times_ms)
    coeffs = np.polyfit(log_sizes, log_times, 1)
    x_fit = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
    y_fit = 10 ** np.polyval(coeffs, np.log10(x_fit))
    ax1.plot(x_fit, y_fit, "r--", alpha=0.5, label=f"Fitted: O(n^{coeffs[0]:.2f})")
    ax1.legend()

    # Plot time_vector complexity
    tv_data = complexity["time_vector"]["timings"]
    sizes2 = [d["size"] for d in tv_data]
    times_us = [d["time"] * 1e6 for d in tv_data]
    exponent2 = complexity["time_vector"]["complexity_exponent"]

    ax2.scatter(sizes2, times_us, alpha=0.6, s=50, color="green")
    ax2.set_xlabel("Samples (n)")
    ax2.set_ylabel("Time (μs)")
    ax2.set_title(f"Time Vector Generation: O(n^{exponent2:.2f})")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    # Add fitted curve
    log_times2 = np.log10(times_us)
    coeffs2 = np.polyfit(log_sizes, log_times2, 1)
    y_fit2 = 10 ** np.polyval(coeffs2, np.log10(x_fit))
    ax2.plot(x_fit, y_fit2, "r--", alpha=0.5, label=f"Fitted: O(n^{coeffs2[0]:.2f})")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "complexity_curves.png", dpi=150)
    print(f"Saved: {output_dir / 'complexity_curves.png'}")
    plt.close()


def plot_scalability(scalability: dict, output_dir: Path) -> None:
    """Create scalability comparison chart."""
    methods = ["Sequential", "Thread Pool", "Process Pool"]
    times = [
        scalability["sequential_total"],
        scalability["thread_total"],
        scalability["process_total"],
    ]
    speedups = [
        1.0,
        scalability["speedup_thread"],
        scalability["speedup_process"],
    ]

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Execution Time
    colors = ["green", "orange", "red"]
    bars1 = ax1.bar(methods, times, color=colors, alpha=0.7)
    ax1.set_ylabel("Total Time (s)")
    ax1.set_title("Execution Time by Method")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}s",
            ha="center",
            va="bottom",
        )

    # Plot 2: Speedup
    colors2 = ["green", "orange", "red"]
    bars2 = ax2.bar(methods, speedups, color=colors2, alpha=0.7)
    ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Baseline")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("Speedup vs Sequential")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend()

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}x",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "scalability.png", dpi=150)
    print(f"Saved: {output_dir / 'scalability.png'}")
    plt.close()


def main():
    """Generate all visualization charts."""
    results_dir = Path(__file__).parent / "results"
    output_dir = results_dir

    print("Loading results...")
    results = load_results(results_dir)

    print("\nGenerating visualizations...")

    # 1. Loading performance
    if "performance_metrics" in results:
        plot_loading_performance(results["performance_metrics"], output_dir)

    # 2. Memory efficiency
    if "performance_metrics" in results:
        plot_memory_efficiency(results["performance_metrics"], output_dir)

    # 3. Complexity curves
    if "complexity_analysis" in results:
        plot_complexity_curves(results["complexity_analysis"], output_dir)

    # 4. Scalability
    if "scalability" in results:
        plot_scalability(results["scalability"], output_dir)

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
