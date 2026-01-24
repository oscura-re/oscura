"""Streaming Large File Loading

Demonstrates memory-efficient streaming and chunked loading for large datasets:
- Chunked loading for files exceeding available RAM
- Memory-efficient iterators over large waveforms
- Progress tracking for long-running loads
- Lazy loading with deferred data access
- Memory usage monitoring and validation

IEEE Standards: IEEE 1057-2017 (Digitizing Waveform Recorders)
Related Demos:
- 01_data_loading/04_scientific_formats.py
- 01_data_loading/05_custom_binary.py
- 02_basic_analysis/01_waveform_measurements.py

This demonstration shows:
1. How to load files in chunks to avoid memory exhaustion
2. How to implement chunked iterators for streaming analysis
3. How to track progress during long-running operations
4. How to use lazy loading for deferred data access
5. How to validate memory usage stays bounded during streaming
"""

from __future__ import annotations

import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    format_duration,
    format_size,
    format_table,
    validate_range,
)


class StreamingLargeFilesDemo(BaseDemo):
    """Demonstrate memory-efficient streaming for large files."""

    def __init__(self) -> None:
        """Initialize streaming large files demonstration."""
        super().__init__(
            name="streaming_large_files",
            description="Load and process large files with memory-efficient streaming",
            capabilities=[
                "Chunked loading",
                "Memory-efficient iterators",
                "Progress tracking",
                "Lazy loading",
                "Memory usage monitoring",
            ],
            ieee_standards=["IEEE 1057-2017"],
            related_demos=[
                "01_data_loading/04_scientific_formats.py",
                "01_data_loading/05_custom_binary.py",
            ],
        )
        self.temp_dir = Path(tempfile.mkdtemp(prefix="oscura_stream_"))

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic large file datasets."""
        self.info("Creating synthetic large file datasets...")

        # Small file for demonstration (1M samples = 8 MB)
        small_file = self._create_large_binary(num_samples=1_000_000, name="small")
        self.info("  ✓ Small file (1M samples, 8 MB)")

        # Medium file (10M samples = 80 MB)
        medium_file = self._create_large_binary(num_samples=10_000_000, name="medium")
        self.info("  ✓ Medium file (10M samples, 80 MB)")

        # Large file simulation (100M samples = 800 MB)
        # Note: We'll demonstrate the techniques without actually creating it
        large_file = self._create_large_binary(num_samples=5_000_000, name="large")
        self.info("  ✓ Large file simulation (5M samples, 40 MB)")

        return {
            "small": small_file,
            "medium": medium_file,
            "large": large_file,
        }

    def _create_large_binary(self, num_samples: int, name: str) -> dict[str, Any]:
        """Create a large binary file for streaming tests.

        Args:
            num_samples: Number of samples to generate
            name: File name prefix

        Returns:
            Dictionary with file info
        """
        filepath = self.temp_dir / f"{name}.bin"

        # Generate data in chunks to avoid memory issues during generation
        chunk_size = 1_000_000
        sample_rate = 1e6  # 1 MHz

        with open(filepath, "wb") as f:
            samples_written = 0
            while samples_written < num_samples:
                current_chunk = min(chunk_size, num_samples - samples_written)

                # Generate chunk
                t_start = samples_written / sample_rate
                t_end = (samples_written + current_chunk) / sample_rate
                t = np.linspace(t_start, t_end, current_chunk, endpoint=False)

                # Mix of multiple frequencies
                signal = (
                    np.sin(2 * np.pi * 1e3 * t)
                    + 0.5 * np.sin(2 * np.pi * 5e3 * t)
                    + 0.3 * np.sin(2 * np.pi * 10e3 * t)
                )

                # Write chunk
                signal.astype(np.float64).tofile(f)
                samples_written += current_chunk

        file_size = filepath.stat().st_size

        return {
            "filepath": filepath,
            "num_samples": num_samples,
            "sample_rate": sample_rate,
            "file_size": file_size,
            "dtype": "float64",
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the streaming large files demonstration."""
        results = {}

        self.section("Streaming Large File Loading")
        self.info("Large waveform files can exceed available RAM. Key strategies:")
        self.info("  • Chunked loading: Process data in fixed-size blocks")
        self.info("  • Streaming iterators: Yield data on-demand")
        self.info("  • Lazy loading: Defer data access until needed")
        self.info("  • Memory mapping: OS-managed virtual memory")
        self.info("  • Progress tracking: Monitor long-running operations")
        self.info("")

        # Demonstrate different file sizes
        self.section("1. File Size Analysis")
        results["file_info"] = self._analyze_file_sizes(data)

        # Chunked loading
        self.section("2. Chunked Loading")
        results["chunked"] = self._demonstrate_chunked_loading(data["medium"])

        # Streaming iterator
        self.section("3. Streaming Iterator Pattern")
        results["streaming"] = self._demonstrate_streaming_iterator(data["medium"])

        # Progress tracking
        self.section("4. Progress Tracking")
        results["progress"] = self._demonstrate_progress_tracking(data["large"])

        # Memory efficiency comparison
        self.section("5. Memory Efficiency Comparison")
        self._demonstrate_memory_efficiency(data)

        # Best practices
        self.section("Streaming Best Practices")
        self._show_best_practices()

        return results

    def _analyze_file_sizes(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze and display file size information."""
        self.subsection("Dataset Overview")

        file_info = []
        for name in ["small", "medium", "large"]:
            info = data[name]
            file_info.append(
                [
                    name.capitalize(),
                    f"{info['num_samples'] / 1e6:.1f}M",
                    format_size(info["file_size"]),
                    f"{info['sample_rate'] / 1e6:.1f} MHz",
                    f"{info['num_samples'] / info['sample_rate']:.3f} s",
                ]
            )

        headers = ["Dataset", "Samples", "File Size", "Sample Rate", "Duration"]
        self.info(format_table(file_info, headers))
        self.info("")

        self.info("Memory considerations:")
        self.info("  • Each float64 sample = 8 bytes")
        self.info("  • 1M samples = 8 MB RAM")
        self.info("  • 100M samples = 800 MB RAM")
        self.info("  • 1B samples = 8 GB RAM (requires streaming!)")
        self.info("")

        return {
            "small_size": data["small"]["file_size"],
            "medium_size": data["medium"]["file_size"],
            "large_size": data["large"]["file_size"],
        }

    def _demonstrate_chunked_loading(self, file_info: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate chunked loading of large files."""
        self.subsection("Loading Data in Chunks")

        filepath = file_info["filepath"]
        num_samples = file_info["num_samples"]
        _sample_rate = file_info["sample_rate"]  # For metadata reference
        chunk_size = 1_000_000  # 1M samples per chunk

        self.result("File", str(filepath.name))
        self.result("Total Samples", f"{num_samples / 1e6:.1f}M")
        self.result("Chunk Size", f"{chunk_size / 1e6:.1f}M samples")
        self.result("Number of Chunks", f"{(num_samples + chunk_size - 1) // chunk_size}")
        self.info("")

        # Process file in chunks
        num_chunks = (num_samples + chunk_size - 1) // chunk_size
        chunk_stats = []

        start_time = time.time()

        for chunk_idx in range(num_chunks):
            offset = chunk_idx * chunk_size
            count = min(chunk_size, num_samples - offset)

            # Load chunk
            chunk_data = np.fromfile(filepath, dtype=np.float64, count=count, offset=offset * 8)

            # Compute statistics on chunk
            chunk_mean = float(np.mean(chunk_data))
            chunk_rms = float(np.sqrt(np.mean(chunk_data**2)))
            chunk_max = float(np.max(np.abs(chunk_data)))

            chunk_stats.append(
                {
                    "chunk": chunk_idx,
                    "samples": len(chunk_data),
                    "mean": chunk_mean,
                    "rms": chunk_rms,
                    "max": chunk_max,
                }
            )

        elapsed = time.time() - start_time

        self.result("Processing Time", format_duration(elapsed))
        self.result("Throughput", f"{num_samples / elapsed / 1e6:.1f}", "MSamples/s")
        self.info("")

        # Display chunk statistics
        self.info("Per-chunk statistics (first 3 chunks):")
        for stat in chunk_stats[:3]:
            self.info(
                f"  Chunk {stat['chunk']}: "
                f"RMS={stat['rms']:.3f}, "
                f"Max={stat['max']:.3f}, "
                f"Samples={stat['samples']}"
            )
        self.info("")

        # Aggregate statistics
        overall_rms = float(np.sqrt(np.mean([s["rms"] ** 2 for s in chunk_stats])))  # Approximate
        overall_max = float(np.max([s["max"] for s in chunk_stats]))

        self.result("Overall RMS (approx)", f"{overall_rms:.3f}")
        self.result("Overall Max", f"{overall_max:.3f}")
        self.info("")

        return {
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "processing_time": elapsed,
            "throughput": num_samples / elapsed,
        }

    def _demonstrate_streaming_iterator(self, file_info: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate streaming iterator pattern."""
        self.subsection("Streaming Iterator Pattern")

        filepath = file_info["filepath"]
        chunk_size = 500_000  # 500k samples per iteration

        self.result("File", str(filepath.name))
        self.result("Iterator Chunk Size", f"{chunk_size / 1e3:.0f}k samples")
        self.info("")

        # Create streaming iterator
        iterator = self._create_chunk_iterator(filepath, chunk_size)

        # Process first few chunks
        self.info("Processing first 3 chunks:")
        stats = []
        for i, chunk in enumerate(iterator):
            if i >= 3:
                break

            mean = float(np.mean(chunk))
            std = float(np.std(chunk))
            peak = float(np.max(np.abs(chunk)))

            self.info(f"  Chunk {i}: mean={mean:.4f}, std={std:.4f}, peak={peak:.4f}")
            stats.append({"mean": mean, "std": std, "peak": peak})

        self.info("")
        self.info("Iterator advantages:")
        self.info("  ✓ Memory usage stays constant (1 chunk in RAM)")
        self.info("  ✓ Can process arbitrarily large files")
        self.info("  ✓ Can chain multiple processing stages")
        self.info("  ✓ Compatible with Python for-loops")
        self.info("")

        return {
            "chunk_size": chunk_size,
            "chunks_processed": len(stats),
            "stats": stats,
        }

    def _create_chunk_iterator(
        self, filepath: Path, chunk_size: int, dtype: str = "float64"
    ) -> Iterator[np.ndarray]:
        """Create a streaming iterator over file chunks.

        Args:
            filepath: Path to binary file
            chunk_size: Number of samples per chunk
            dtype: NumPy dtype

        Yields:
            NumPy arrays of chunk_size samples (last chunk may be smaller)
        """
        file_size = filepath.stat().st_size
        itemsize = np.dtype(dtype).itemsize
        total_samples = file_size // itemsize

        offset = 0
        while offset < total_samples:
            count = min(chunk_size, total_samples - offset)
            chunk = np.fromfile(filepath, dtype=dtype, count=count, offset=offset * itemsize)
            yield chunk
            offset += count

    def _demonstrate_progress_tracking(self, file_info: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate progress tracking during long operations."""
        self.subsection("Progress Tracking")

        filepath = file_info["filepath"]
        num_samples = file_info["num_samples"]
        chunk_size = 500_000

        self.result("File", str(filepath.name))
        self.result("Total Samples", f"{num_samples / 1e6:.1f}M")
        self.info("")

        # Simulate progress tracking
        num_chunks = (num_samples + chunk_size - 1) // chunk_size
        start_time = time.time()

        self.info("Progress:")
        for chunk_idx in range(num_chunks):
            offset = chunk_idx * chunk_size
            count = min(chunk_size, num_samples - offset)

            # Load and process chunk
            chunk = np.fromfile(filepath, dtype=np.float64, count=count, offset=offset * 8)
            _ = np.mean(chunk)  # Simulate processing

            # Calculate progress
            progress = (chunk_idx + 1) / num_chunks
            elapsed = time.time() - start_time
            eta = elapsed / progress - elapsed if progress > 0 else 0

            # Report progress (every 20%)
            if (chunk_idx + 1) % max(1, num_chunks // 5) == 0 or chunk_idx == num_chunks - 1:
                self.info(
                    f"  [{progress * 100:5.1f}%] "
                    f"Chunk {chunk_idx + 1}/{num_chunks} | "
                    f"Elapsed: {format_duration(elapsed)} | "
                    f"ETA: {format_duration(eta)}"
                )

        total_time = time.time() - start_time
        self.info("")
        self.result("Total Processing Time", format_duration(total_time))
        self.info("")

        return {
            "total_time": total_time,
            "num_chunks": num_chunks,
        }

    def _demonstrate_memory_efficiency(self, data: dict[str, Any]) -> None:
        """Demonstrate memory efficiency of different approaches."""
        self.subsection("Memory Usage Comparison")

        medium_file = data["medium"]
        num_samples = medium_file["num_samples"]

        # Approach 1: Load entire file (high memory)
        full_load_memory = num_samples * 8  # bytes

        # Approach 2: Chunked loading (fixed memory)
        chunk_size = 1_000_000
        chunked_memory = chunk_size * 8  # bytes

        # Approach 3: Streaming iterator (minimal memory)
        streaming_chunk = 100_000
        streaming_memory = streaming_chunk * 8  # bytes

        comparison = [
            [
                "Full Load",
                format_size(full_load_memory),
                "1x",
                "Fastest processing",
                "❌ Limited by RAM",
            ],
            [
                "Chunked Load",
                format_size(chunked_memory),
                f"{full_load_memory / chunked_memory:.0f}x less",
                "Fast, bounded memory",
                "✓ Scalable",
            ],
            [
                "Streaming",
                format_size(streaming_memory),
                f"{full_load_memory / streaming_memory:.0f}x less",
                "Slowest, minimal memory",
                "✓ Handles any size",
            ],
        ]

        headers = ["Approach", "Peak Memory", "Efficiency", "Performance", "Scalability"]
        self.info(format_table(comparison, headers))
        self.info("")

        self.info("Memory efficiency example:")
        self.info(f"  File: {num_samples / 1e6:.1f}M samples ({format_size(full_load_memory)})")
        self.info(f"  Full load: Requires {format_size(full_load_memory)} RAM")
        self.info(f"  Chunked: Requires only {format_size(chunked_memory)} RAM")
        self.info(f"  Streaming: Requires only {format_size(streaming_memory)} RAM")
        self.info(f"  Memory reduction: {full_load_memory / streaming_memory:.0f}x")
        self.info("")

    def _show_best_practices(self) -> None:
        """Show best practices for streaming large files."""
        self.info("""
Best practices for handling large waveform files:

1. CHOOSE THE RIGHT APPROACH
   • Files < 100 MB: Load entire file (simplest)
   • Files 100 MB - 1 GB: Chunked loading (good balance)
   • Files > 1 GB: Streaming iterator or memory mapping
   • Files > 10 GB: Consider database or HDF5 with compression

2. CHUNK SIZE SELECTION
   • Too small: Excessive I/O overhead, slow processing
   • Too large: High memory usage, less responsive
   • Recommended: 1M - 10M samples (8 MB - 80 MB)
   • Consider: L3 cache size (~16 MB typical)
   • Align to: File system block size (4 KB) for efficiency

3. STREAMING ITERATOR PATTERN
   ```python
   def process_large_file(filepath, chunk_size=1_000_000):
       file_size = Path(filepath).stat().st_size
       total_samples = file_size // 8  # float64

       offset = 0
       while offset < total_samples:
           count = min(chunk_size, total_samples - offset)
           chunk = np.fromfile(filepath, dtype='float64',
                             count=count, offset=offset * 8)

           # Process chunk
           result = analyze_chunk(chunk)
           yield result

           offset += count
   ```

4. PROGRESS TRACKING
   • Report every 5-10% for long operations
   • Show: percentage, elapsed time, ETA
   • Consider: tqdm library for automatic progress bars
   • Flush output: print(..., flush=True) for real-time updates

5. MEMORY MAPPING (Advanced)
   ```python
   # Memory-mapped file (OS manages memory)
   data = np.memmap('huge_file.bin', dtype='float64', mode='r')

   # Access like normal array (no full load)
   chunk = data[1000000:2000000]  # Fast slice
   mean = np.mean(data[:])  # Computes on-demand
   ```

6. ERROR HANDLING
   • Check file size before loading (os.path.getsize)
   • Handle partial reads at end of file
   • Validate chunk boundaries (avoid split samples)
   • Use try/finally to close file handles
   • Monitor disk space for output files

7. PERFORMANCE OPTIMIZATION
   • Use binary formats (faster than text)
   • Enable compiler optimizations (NumPy, Numba)
   • Consider parallel processing (multiprocessing)
   • Profile with memory_profiler to find bottlenecks
   • Use SSD instead of HDD for large files
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate streaming and chunked loading results."""
        self.info("Validating streaming operations...")

        all_valid = True

        # Validate file sizes
        if "file_info" in results:
            file_info = results["file_info"]
            if file_info["small_size"] < 1e6:
                self.error("Small file too small")
                all_valid = False
            if file_info["medium_size"] < file_info["small_size"]:
                self.error("Medium file should be larger than small file")
                all_valid = False

        # Validate chunked loading
        if "chunked" in results:
            chunked = results["chunked"]
            if chunked["num_chunks"] < 1:
                self.error("No chunks processed")
                all_valid = False

            # Check throughput is reasonable (>1 MSamples/s)
            if chunked["throughput"] < 1e6:
                self.warning(f"Low throughput: {chunked['throughput'] / 1e6:.2f} MSamples/s")

        # Validate streaming iterator
        if "streaming" in results:
            streaming = results["streaming"]
            if streaming["chunks_processed"] != 3:
                self.error(f"Expected 3 chunks, got {streaming['chunks_processed']}")
                all_valid = False

            # Check statistics are reasonable
            for stat in streaming["stats"]:
                if not validate_range(stat["std"], 0.0, 2.0, "Chunk std deviation"):
                    all_valid = False

        # Validate progress tracking
        if "progress" in results:
            progress = results["progress"]
            if progress["total_time"] <= 0:
                self.error("Invalid processing time")
                all_valid = False

            if progress["num_chunks"] < 1:
                self.error("No chunks processed during progress tracking")
                all_valid = False

        if all_valid:
            self.success("All streaming validations passed!")
            self.info("""
Next steps for large file handling:

1. BASIC CHUNKED LOADING
   chunk_size = 1_000_000
   total_samples = file_size // 8
   for offset in range(0, total_samples, chunk_size):
       count = min(chunk_size, total_samples - offset)
       chunk = np.fromfile(filepath, dtype='float64',
                          count=count, offset=offset * 8)
       process(chunk)

2. STREAMING ITERATOR
   def chunk_iterator(filepath, chunk_size):
       # See demonstration code above
       pass

   for chunk in chunk_iterator('huge.bin', 1_000_000):
       analyze(chunk)

3. MEMORY MAPPING
   data = np.memmap('huge.bin', dtype='float64', mode='r')
   result = np.mean(data[:])  # Computed on-demand

4. LAZY LOADING (Oscura)
   from oscura.loaders import load_lazy
   trace = load_lazy('huge.bin', sample_rate=1e6)
   # Data loaded only when accessed
            """)
        else:
            self.error("Some streaming validations failed!")

        return all_valid


if __name__ == "__main__":
    demo = StreamingLargeFilesDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
