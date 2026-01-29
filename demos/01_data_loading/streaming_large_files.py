"""Streaming and Large File Loading

Demonstrates memory-efficient loading of large files:
- Chunk-based processing for multi-GB files
- Lazy loading patterns
- Memory-mapped file access
- Streaming data analysis without full load

Essential for processing large captures without running out of memory.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    ValidationSuite,
    format_size,
    generate_sine_wave,
)


class StreamingLargeFilesDemo(BaseDemo):
    """Demonstrate streaming and large file loading techniques."""

    def __init__(self) -> None:
        """Initialize streaming large files demonstration."""
        super().__init__(
            name="streaming_large_files",
            description="Memory-efficient loading of large data files",
            capabilities=[
                "Chunk-based processing",
                "Memory-mapped file access",
                "Lazy loading patterns",
                "Streaming analysis",
            ],
            ieee_standards=[],
            related_demos=[
                "05_custom_binary.py",
                "09_lazy_loading.py",
            ],
        )
        self.temp_dir = Path(tempfile.mkdtemp(prefix="oscura_streaming_"))

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic large file data."""
        self.info("Creating synthetic large file (simulated)...")

        # Create moderately-sized file for demonstration
        large_file = self._create_large_binary_file()
        self.info(f"  ✓ Large binary file created: {format_size(large_file['size'])}")

        return {
            "large_file": large_file,
        }

    def _create_large_binary_file(self) -> dict[str, Any]:
        """Create a large binary file for streaming demonstration."""
        # Create 10 MB file (simulates multi-GB in practice)
        sample_rate = 1e6  # 1 MHz
        duration = 1.25  # 1.25 seconds = 1.25M samples * 8 bytes = 10 MB

        filepath = self.temp_dir / "large_capture.bin"

        # Generate in chunks to simulate large file creation
        chunk_size = 125000  # 125k samples per chunk
        num_chunks = 10

        with open(filepath, "wb") as f:
            for i in range(num_chunks):
                # Generate chunk with varying frequency
                freq = 1e3 * (i + 1)  # 1-10 kHz
                chunk_duration = chunk_size / sample_rate
                chunk = generate_sine_wave(freq, 1.0, chunk_duration, sample_rate)
                chunk.data.astype(np.float64).tofile(f)

        file_size = filepath.stat().st_size

        return {
            "filepath": filepath,
            "size": file_size,
            "sample_rate": sample_rate,
            "total_samples": chunk_size * num_chunks,
            "chunk_size": chunk_size,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the streaming large files demonstration."""
        results = {}

        self.subsection("Streaming Large File Techniques")
        self.info("For multi-GB files, loading all data at once can exhaust memory.")
        self.info("Techniques for memory-efficient processing:")
        self.info("  • Chunk-based processing: Process file in fixed-size blocks")
        self.info("  • Memory-mapped files: OS-managed paging")
        self.info("  • Lazy loading: Load only when accessed")
        self.info("  • Streaming analysis: Single-pass algorithms")
        self.info("")

        # Chunk-based processing
        self.subsection("1. Chunk-Based Processing")
        results["chunked"] = self._demonstrate_chunked_processing(data["large_file"])

        # Memory-mapped access
        self.subsection("2. Memory-Mapped File Access")
        results["mmap"] = self._demonstrate_mmap_access(data["large_file"])

        # Streaming statistics
        self.subsection("3. Streaming Statistics (Single Pass)")
        results["streaming"] = self._demonstrate_streaming_stats(data["large_file"])

        # Best practices
        self.subsection("Large File Best Practices")
        self._show_best_practices()

        return results

    def _demonstrate_chunked_processing(self, file_info: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate chunk-based file processing."""
        filepath = file_info["filepath"]
        chunk_size = file_info["chunk_size"]

        self.result("File Size", format_size(file_info["size"]))
        self.result("Chunk Size", f"{chunk_size} samples")

        # Process file in chunks
        max_values = []
        chunk_count = 0

        with open(filepath, "rb") as f:
            while True:
                # Read chunk
                chunk_bytes = f.read(chunk_size * 8)  # 8 bytes per float64
                if not chunk_bytes:
                    break

                # Parse chunk
                chunk_data = np.frombuffer(chunk_bytes, dtype=np.float64)
                max_values.append(float(np.max(np.abs(chunk_data))))
                chunk_count += 1

        self.result("Chunks Processed", chunk_count)
        self.result("Max Amplitude", f"{max(max_values):.4f}", "V")

        return {
            "num_chunks": chunk_count,
            "max_amplitude": max(max_values),
        }

    def _demonstrate_mmap_access(self, file_info: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate memory-mapped file access."""
        filepath = file_info["filepath"]
        total_samples = file_info["total_samples"]

        self.info("Memory-mapped files allow random access without loading entire file")

        # Use numpy's memmap for demonstration
        mmap_data = np.memmap(filepath, dtype=np.float64, mode="r", shape=(total_samples,))

        # Access random samples (OS loads pages on-demand)
        sample_indices = [0, total_samples // 4, total_samples // 2, total_samples - 1]
        samples = [float(mmap_data[i]) for i in sample_indices]

        self.result("File Samples", total_samples)
        self.result("Memory Mapped", "Yes (lazy pages)")
        self.result("Random Access", f"{len(sample_indices)} samples read")

        # Clean up memmap
        del mmap_data

        return {
            "total_samples": total_samples,
            "samples_accessed": len(sample_indices),
        }

    def _demonstrate_streaming_stats(self, file_info: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate streaming statistics calculation."""
        filepath = file_info["filepath"]
        chunk_size = 10000  # Smaller chunks for streaming

        self.info("Calculate statistics in single pass without storing all data")

        # Streaming mean and RMS calculation
        total_samples = 0
        running_sum = 0.0
        running_sum_squares = 0.0

        with open(filepath, "rb") as f:
            while True:
                chunk_bytes = f.read(chunk_size * 8)
                if not chunk_bytes:
                    break

                chunk_data = np.frombuffer(chunk_bytes, dtype=np.float64)
                total_samples += len(chunk_data)
                running_sum += float(np.sum(chunk_data))
                running_sum_squares += float(np.sum(chunk_data**2))

        # Calculate final statistics
        mean = running_sum / total_samples
        rms = np.sqrt(running_sum_squares / total_samples)

        self.result("Total Samples Processed", total_samples)
        self.result("Mean (streaming)", f"{mean:.6f}", "V")
        self.result("RMS (streaming)", f"{rms:.6f}", "V")

        return {
            "total_samples": total_samples,
            "mean": mean,
            "rms": float(rms),
        }

    def _show_best_practices(self) -> None:
        """Show best practices for large file handling."""
        self.info("""
Large File Processing Best Practices:

1. CHOOSE RIGHT TECHNIQUE
   - <100 MB: Load entirely into memory (simplest)
   - 100 MB - 1 GB: Chunk-based processing
   - 1 GB - 10 GB: Memory-mapped files
   - >10 GB: Streaming algorithms + database

2. CHUNK SIZE SELECTION
   - Too small: Overhead dominates (I/O per chunk)
   - Too large: Memory pressure, long pauses
   - Sweet spot: 1-10 MB chunks (OS page multiple)
   - Match chunk to cache line size for best performance

3. ALGORITHM DESIGN
   - Prefer single-pass streaming algorithms
   - Use running statistics (mean, variance)
   - Avoid algorithms requiring random access
   - Consider approximate algorithms (sketches)

4. MEMORY MANAGEMENT
   - Explicitly delete large arrays when done
   - Use context managers for file handles
   - Monitor memory usage during development
   - Set memory limits for production code
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate streaming large files results."""
        suite = ValidationSuite()

        # Validate chunked processing
        if "chunked" in results:
            suite.check_true(results["chunked"]["num_chunks"] > 5, "Chunks processed")
            suite.check_true(
                results["chunked"]["max_amplitude"] > 0.5, "Max amplitude in reasonable range"
            )

        # Validate mmap access
        if "mmap" in results:
            suite.check_true(results["mmap"]["total_samples"] > 100000, "Sufficient samples")

        # Validate streaming stats
        if "streaming" in results:
            suite.check_true(abs(results["streaming"]["mean"]) < 0.1, "Mean near zero")
            suite.check_true(
                0.5 < results["streaming"]["rms"] < 1.0, "RMS in expected range"
            )

        if suite.all_passed():
            self.success("All streaming validations passed!")
            self.info("\nKey Takeaways:")
            self.info("  - Use chunking for files > 100 MB")
            self.info("  - Memory-mapped files for random access")
            self.info("  - Streaming algorithms avoid memory limits")
        else:
            self.error("Some validations failed")

        return suite.all_passed()


if __name__ == "__main__":
    demo = StreamingLargeFilesDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
