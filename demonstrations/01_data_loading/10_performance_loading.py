"""Performance-Optimized Loading Techniques

Demonstrates memory-efficient loading strategies for huge waveform files:
- Standard loading (eager, all data in RAM)
- Memory-mapped loading (zero-copy via OS page cache)
- Lazy loading (deferred data access)

IEEE Standards: IEEE 181-2011 (Waveform and Vector Measurements)
Related Demos:
- 01_data_loading/01_oscilloscopes.py
- 01_data_loading/06_streaming_large_files.py
- 02_basic_analysis/01_waveform_measurements.py

This demonstration shows:
1. How to generate large synthetic test files
2. Performance benchmarking of loading strategies
3. Memory usage comparison
4. When to use each method (decision tree)
5. Chunked processing for huge files
"""

from __future__ import annotations

import gc
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    format_table,
)
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.loaders import load_trace_lazy
from oscura.loaders.mmap_loader import load_mmap, should_use_mmap


class PerformanceLoadingDemo(BaseDemo):
    """Demonstrate performance-optimized loading strategies."""

    def __init__(self) -> None:
        """Initialize performance loading demonstration."""
        super().__init__(
            name="performance_loading",
            description="Benchmark and compare memory-efficient loading strategies",
            capabilities=[
                "oscura.loaders.load_mmap",
                "oscura.loaders.load_trace_lazy",
                "oscura.loaders.should_use_mmap",
                "MmapWaveformTrace.iter_chunks",
                "LazyWaveformTrace slicing",
                "Performance benchmarking",
            ],
            ieee_standards=["IEEE 181-2011"],
            related_demos=[
                "01_data_loading/01_oscilloscopes.py",
                "01_data_loading/06_streaming_large_files.py",
            ],
        )

        # Test file sizes (in millions of samples)
        self.file_sizes = {
            "small": 1,  # 1M samples = 8MB
            "medium": 10,  # 10M samples = 80MB
            "large": 100,  # 100M samples = 800MB
        }

        self.temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.test_files: dict[str, Path] = {}

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic test files of various sizes."""
        self.info("Generating large synthetic test files...")

        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)

        sample_rate = 1e9  # 1 GSa/s

        for size_name, num_samples_millions in self.file_sizes.items():
            file_path = temp_path / f"test_{size_name}.npy"

            # Generate synthetic waveform: 1kHz sine + noise
            self.info(f"  Generating {size_name}: {num_samples_millions}M samples...")

            # Generate in chunks to avoid memory issues during generation
            chunk_size = 1_000_000  # 1M samples per chunk
            chunks = []

            t_offset = 0.0
            for _ in range(num_samples_millions):
                t = np.linspace(
                    t_offset,
                    t_offset + chunk_size / sample_rate,
                    chunk_size,
                    dtype=np.float32,
                )
                # 1kHz sine wave
                chunk = np.sin(2 * np.pi * 1e3 * t, dtype=np.float32)
                # Add noise
                chunk += 0.01 * np.random.randn(chunk_size).astype(np.float32)
                chunks.append(chunk)
                t_offset += chunk_size / sample_rate

            # Concatenate and save
            data = np.concatenate(chunks)
            np.save(file_path, data)

            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            self.info(
                f"  ✓ Created {size_name}: {file_size_mb:.1f} MB ({num_samples_millions}M samples)"
            )

            self.test_files[size_name] = file_path

            # Clean up memory
            del data, chunks
            gc.collect()

        return {
            "sample_rate": sample_rate,
            "temp_dir": temp_path,
            "file_paths": self.test_files,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the performance loading demonstration."""
        sample_rate = data["sample_rate"]
        file_paths = data["file_paths"]

        # Overview
        self.section("Performance Loading Strategies Overview")
        self.info("""
Three primary strategies for loading waveform data:

1. STANDARD LOADING (Eager)
   - Loads entire file into RAM immediately
   - Fast access after loading
   - Best for: Files < 100 MB, random access patterns
   - Memory: Full file size in RAM

2. MEMORY-MAPPED LOADING (Mmap)
   - Uses OS page cache for zero-copy access
   - Data loaded on-demand in pages
   - Best for: Files > 1 GB, sequential or chunked access
   - Memory: Only accessed pages in RAM

3. LAZY LOADING
   - Metadata loaded immediately, data deferred
   - Supports efficient slicing
   - Best for: Exploring metadata, selective data access
   - Memory: Only loaded slices in RAM
        """)

        # Benchmark each file size
        results = {}

        for size_name in ["small", "medium", "large"]:
            file_path = file_paths[size_name]
            self.section(f"Benchmarking {size_name.upper()} File")
            results[size_name] = self._benchmark_file(file_path, sample_rate)

        # Performance comparison
        self.section("Performance Comparison")
        self._display_performance_comparison(results)

        # Memory usage analysis
        self.section("Memory Usage Analysis")
        self._analyze_memory_usage(results)

        # Chunked processing demonstration
        self.section("Chunked Processing for Huge Files")
        large_file = file_paths["large"]
        self._demonstrate_chunked_processing(large_file, sample_rate)

        # Decision tree
        self.section("When to Use Each Method")
        self._show_decision_tree()

        return results

    def _benchmark_file(
        self,
        file_path: Path,
        sample_rate: float,
    ) -> dict[str, Any]:
        """Benchmark all loading strategies for a file."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        self.subsection(f"File: {file_path.name} ({file_size_mb:.1f} MB)")

        # Check recommendation
        should_mmap = should_use_mmap(file_path)
        self.info(f"Recommended mmap: {should_mmap}")
        self.info("")

        results: dict[str, Any] = {
            "file_size_mb": file_size_mb,
            "timings": {},
        }

        # Benchmark 1: Standard loading
        self.info("1. Standard Loading (Eager)...")
        gc.collect()

        start = time.perf_counter()
        data = np.load(file_path)
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace_standard = WaveformTrace(data=data, metadata=metadata)
        load_time_standard = time.perf_counter() - start

        # Access benchmark
        start = time.perf_counter()
        _ = trace_standard.data[1000:2000]  # Access 1000 samples
        access_time_standard = time.perf_counter() - start

        self.result("  Load time", f"{load_time_standard:.4f}", "seconds")
        self.result("  Access time (1K samples)", f"{access_time_standard * 1000:.4f}", "ms")
        self.result("  Memory (approx)", f"{file_size_mb:.1f}", "MB")
        self.info("")

        results["timings"]["standard"] = {
            "load_time": load_time_standard,
            "access_time": access_time_standard,
            "memory_mb": file_size_mb,
        }

        # Clean up
        del trace_standard, data
        gc.collect()

        # Benchmark 2: Memory-mapped loading
        self.info("2. Memory-Mapped Loading (Mmap)...")
        gc.collect()

        start = time.perf_counter()
        trace_mmap = load_mmap(file_path, sample_rate=sample_rate)
        load_time_mmap = time.perf_counter() - start

        # Access benchmark
        start = time.perf_counter()
        _ = trace_mmap[1000:2000]  # Access 1000 samples
        access_time_mmap = time.perf_counter() - start

        self.result("  Load time", f"{load_time_mmap:.4f}", "seconds")
        self.result("  Access time (1K samples)", f"{access_time_mmap * 1000:.4f}", "ms")
        self.result("  Memory (approx)", "<1", "MB (on-demand)")
        self.info("")

        results["timings"]["mmap"] = {
            "load_time": load_time_mmap,
            "access_time": access_time_mmap,
            "memory_mb": 0.1,  # Minimal overhead
        }

        # Clean up
        trace_mmap.close()
        del trace_mmap
        gc.collect()

        # Benchmark 3: Lazy loading
        self.info("3. Lazy Loading...")
        gc.collect()

        start = time.perf_counter()
        trace_lazy = load_trace_lazy(file_path, sample_rate=sample_rate, lazy=True)
        load_time_lazy = time.perf_counter() - start

        # Access benchmark (triggers data load)
        start = time.perf_counter()
        lazy_slice = trace_lazy[1000:2000]
        # Access the data to actually load it
        if hasattr(lazy_slice, "data"):
            _ = lazy_slice.data
        access_time_lazy = time.perf_counter() - start

        self.result("  Load time (metadata only)", f"{load_time_lazy:.4f}", "seconds")
        self.result("  Access time (1K samples)", f"{access_time_lazy * 1000:.4f}", "ms")
        self.result("  Memory (approx)", "0.01", "MB (metadata only)")
        self.info("")

        results["timings"]["lazy"] = {
            "load_time": load_time_lazy,
            "access_time": access_time_lazy,
            "memory_mb": 0.01,  # Metadata only
        }

        # Clean up
        trace_lazy.close()
        del trace_lazy, lazy_slice
        gc.collect()

        return results

    def _display_performance_comparison(self, results: dict[str, Any]) -> None:
        """Display performance comparison table."""
        self.subsection("Load Time Comparison")

        table_data = []
        for size_name in ["small", "medium", "large"]:
            if size_name in results:
                timings = results[size_name]["timings"]
                table_data.append(
                    [
                        size_name.capitalize(),
                        f"{results[size_name]['file_size_mb']:.1f}",
                        f"{timings['standard']['load_time']:.4f}",
                        f"{timings['mmap']['load_time']:.4f}",
                        f"{timings['lazy']['load_time']:.4f}",
                    ]
                )

        headers = ["File Size", "Size (MB)", "Standard (s)", "Mmap (s)", "Lazy (s)"]
        self.info(format_table(table_data, headers))
        self.info("")

        # Access time comparison
        self.subsection("Access Time Comparison (1K samples)")

        table_data = []
        for size_name in ["small", "medium", "large"]:
            if size_name in results:
                timings = results[size_name]["timings"]
                table_data.append(
                    [
                        size_name.capitalize(),
                        f"{timings['standard']['access_time'] * 1000:.4f}",
                        f"{timings['mmap']['access_time'] * 1000:.4f}",
                        f"{timings['lazy']['access_time'] * 1000:.4f}",
                    ]
                )

        headers = ["File Size", "Standard (ms)", "Mmap (ms)", "Lazy (ms)"]
        self.info(format_table(table_data, headers))
        self.info("")

        # Key insights
        self.info("Key Insights:")
        self.info("  • Standard loading time increases linearly with file size")
        self.info("  • Mmap loading is nearly instant (metadata only)")
        self.info("  • Lazy loading is fastest for metadata inspection")
        self.info("  • Access time is similar for all methods (OS caching)")
        self.info("  • Memory usage is critical factor for huge files")

    def _analyze_memory_usage(self, results: dict[str, Any]) -> None:
        """Analyze memory usage for each strategy."""
        self.subsection("Memory Footprint")

        table_data = []
        for size_name in ["small", "medium", "large"]:
            if size_name in results:
                timings = results[size_name]["timings"]
                file_size_mb = results[size_name]["file_size_mb"]
                table_data.append(
                    [
                        size_name.capitalize(),
                        f"{file_size_mb:.1f}",
                        f"{timings['standard']['memory_mb']:.1f}",
                        f"{timings['mmap']['memory_mb']:.2f}",
                        f"{timings['lazy']['memory_mb']:.2f}",
                    ]
                )

        headers = ["File Size", "File (MB)", "Standard (MB)", "Mmap (MB)", "Lazy (MB)"]
        self.info(format_table(table_data, headers))
        self.info("")

        self.info("Memory Analysis:")
        self.info("  • Standard: Full file loaded into RAM")
        self.info("  • Mmap: Only accessed pages in RAM (OS managed)")
        self.info("  • Lazy: Only loaded slices in RAM")
        self.info("")
        self.info("For 10GB file:")
        self.info("  • Standard: 10GB RAM required (may fail with OOM)")
        self.info("  • Mmap: <100MB typical usage (depends on access pattern)")
        self.info("  • Lazy: <10MB for metadata + accessed slices")

    def _demonstrate_chunked_processing(
        self,
        file_path: Path,
        sample_rate: float,
    ) -> None:
        """Demonstrate chunked processing for huge files."""
        self.subsection("Chunked Processing with Mmap")

        trace = load_mmap(file_path, sample_rate=sample_rate)

        self.info(f"File: {file_path.name}")
        self.info(f"Total samples: {trace.length:,}")
        self.info(f"Duration: {trace.duration:.4f} seconds")
        self.info("")

        # Process in chunks
        chunk_size = 1_000_000  # 1M samples per chunk
        self.info(f"Processing in {chunk_size:,} sample chunks...")

        chunk_count = 0
        max_values = []

        start = time.perf_counter()

        for chunk in trace.iter_chunks(chunk_size=chunk_size):
            chunk_count += 1
            # Compute max value in chunk
            max_val = float(np.max(chunk))
            max_values.append(max_val)

            if chunk_count % 10 == 0:
                self.info(f"  Processed {chunk_count} chunks...")

        processing_time = time.perf_counter() - start

        self.info("")
        self.result("Total chunks processed", chunk_count)
        self.result("Processing time", f"{processing_time:.4f}", "seconds")
        self.result(
            "Throughput",
            f"{trace.length / processing_time / 1e6:.1f}",
            "Msamples/s",
        )
        self.result("Global max value", f"{max(max_values):.4f}")
        self.info("")

        self.info("Chunked Processing Benefits:")
        self.info("  • Process files larger than available RAM")
        self.info("  • Constant memory usage regardless of file size")
        self.info("  • Supports overlapping chunks for windowed analysis")
        self.info("  • Compatible with parallel processing")

        trace.close()

    def _show_decision_tree(self) -> None:
        """Show decision tree for choosing loading strategy."""
        self.subsection("Loading Strategy Decision Tree")

        self.info("""
DECISION TREE: Which loading strategy should I use?

┌─────────────────────────────────────────────────────────────────┐
│ START: How large is your file?                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         < 100 MB        100MB - 1GB       > 1 GB
              │               │               │
              ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ STANDARD │    │  CHECK   │    │   MMAP   │
        │ LOADING  │    │ USE CASE │    │ REQUIRED │
        └──────────┘    └──────────┘    └──────────┘
              │               │               │
              │        ┌──────┴──────┐        │
              │        │             │        │
              │    Random      Sequential     │
              │    Access      or Chunked     │
              │        │             │        │
              │        ▼             ▼        │
              │   ┌─────────┐  ┌─────────┐   │
              │   │ STANDARD│  │  MMAP   │   │
              │   │ or LAZY │  │         │   │
              │   └─────────┘  └─────────┘   │
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │ Additional Considerations:                  │
        │                                             │
        │ Use LAZY when:                              │
        │  • Only need metadata initially             │
        │  • Selective data access (slicing)          │
        │  • Exploring multiple files                 │
        │                                             │
        │ Use MMAP when:                              │
        │  • File > 1 GB                              │
        │  • Sequential or chunked processing         │
        │  • Memory constrained environment           │
        │  • Processing on server with other apps     │
        │                                             │
        │ Use STANDARD when:                          │
        │  • File < 100 MB                            │
        │  • Random access to full dataset            │
        │  • Plenty of RAM available                  │
        │  • Maximum performance needed               │
        └─────────────────────────────────────────────┘

EXAMPLES:

1. Exploring 50GB oscilloscope capture:
   trace = load_mmap("huge_capture.npy", sample_rate=1e9)
   for chunk in trace.iter_chunks(chunk_size=1_000_000):
       analyze(chunk)

2. Checking metadata of multiple large files:
   for file in files:
       trace = load_trace_lazy(file, sample_rate=1e9, lazy=True)
       print(f"{file}: {trace.duration}s, {trace.length} samples")
       trace.close()

3. Small file with full analysis:
   trace = load("small_capture.wfm")  # Standard loading
   spectrum = compute_fft(trace.data)
   measurements = analyze_waveform(trace.data)

4. Selective access to large file:
   trace = load_trace_lazy("large.npy", sample_rate=1e9, lazy=True)
   subset = trace[1000000:2000000]  # Only loads this slice
   result = analyze(subset.data)
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate performance loading results."""
        self.info("Validating performance benchmarks...")

        all_valid = True

        # Validate all file sizes were benchmarked
        for size_name in ["small", "medium", "large"]:
            if size_name not in results:
                self.error(f"Missing benchmark for {size_name} file")
                all_valid = False
                continue

            timings = results[size_name]["timings"]

            # Validate all strategies were tested
            for strategy in ["standard", "mmap", "lazy"]:
                if strategy not in timings:
                    self.error(f"Missing {strategy} timing for {size_name}")
                    all_valid = False
                    continue

                # Check timing data exists
                if "load_time" not in timings[strategy]:
                    self.error(f"Missing load_time for {strategy}/{size_name}")
                    all_valid = False

                if "access_time" not in timings[strategy]:
                    self.error(f"Missing access_time for {strategy}/{size_name}")
                    all_valid = False

            # Validate expected performance characteristics
            if all(s in timings for s in ["standard", "mmap", "lazy"]):
                # Mmap load should be faster than standard for large files
                if size_name == "large":
                    if timings["mmap"]["load_time"] >= timings["standard"]["load_time"]:
                        self.warning(
                            f"Mmap load not faster than standard for {size_name} "
                            f"({timings['mmap']['load_time']:.4f}s vs "
                            f"{timings['standard']['load_time']:.4f}s)"
                        )

                # Lazy load should be fastest (metadata only)
                if timings["lazy"]["load_time"] > 0.1:
                    self.warning(
                        f"Lazy load unexpectedly slow for {size_name}: "
                        f"{timings['lazy']['load_time']:.4f}s"
                    )

        if all_valid:
            self.success("All performance benchmarks completed successfully!")
            self.info("""
Next steps for working with huge files:

1. CHOOSE APPROPRIATE STRATEGY
   from oscura.loaders.mmap_loader import should_use_mmap
   if should_use_mmap("huge_file.npy"):
       trace = load_mmap("huge_file.npy", sample_rate=1e9)
   else:
       trace = load("huge_file.npy")

2. CHUNKED PROCESSING
   trace = load_mmap("huge_file.npy", sample_rate=1e9)
   for chunk in trace.iter_chunks(chunk_size=1_000_000):
       result = process_chunk(chunk)

3. LAZY EXPLORATION
   trace = load_trace_lazy("file.npy", sample_rate=1e9, lazy=True)
   print(f"Duration: {trace.duration}s")
   subset = trace[1000000:2000000]  # Only load what you need

4. MEMORY MONITORING
   import psutil
   process = psutil.Process()
   print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
            """)
        else:
            self.error("Some performance validations failed!")

        # Cleanup temporary files
        if self.temp_dir:
            self.temp_dir.cleanup()
            self.info("Cleaned up temporary test files")

        return all_valid


if __name__ == "__main__":
    demo = PerformanceLoadingDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
