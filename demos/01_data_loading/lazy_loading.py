"""Lazy Loading and Evaluation

Demonstrates lazy loading patterns for deferred data access:
- Lazy evaluation of expensive operations
- Deferred file loading until access
- On-demand computation strategies
- Memory optimization techniques

Essential for interactive analysis and large dataset workflows.
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


class LazyLoadingDemo(BaseDemo):
    """Demonstrate lazy loading and evaluation patterns."""

    def __init__(self) -> None:
        """Initialize lazy loading demonstration."""
        super().__init__(
            name="lazy_loading",
            description="Lazy loading and deferred evaluation patterns",
            capabilities=[
                "Lazy file loading",
                "Deferred computation",
                "On-demand data access",
                "Memory optimization",
            ],
            ieee_standards=[],
            related_demos=[
                "06_streaming_large_files.py",
                "07_multi_channel.py",
            ],
        )
        self.temp_dir = Path(tempfile.mkdtemp(prefix="oscura_lazy_"))

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic data files for lazy loading."""
        self.info("Creating synthetic data files...")

        # Multiple data files that won't all be loaded
        files = []
        for i in range(5):
            filepath = self.temp_dir / f"channel_{i}.bin"
            freq = 1e3 * (i + 1)  # 1-5 kHz
            signal = generate_sine_wave(freq, 1.0, 0.01, 100e3)
            signal.data.astype(np.float64).tofile(filepath)
            files.append(
                {
                    "filepath": filepath,
                    "channel": i,
                    "frequency": freq,
                    "size": filepath.stat().st_size,
                }
            )
            self.info(f"  ✓ Channel {i} file: {format_size(filepath.stat().st_size)}")

        return {
            "files": files,
            "sample_rate": 100e3,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the lazy loading demonstration."""
        results = {}

        self.subsection("Lazy Loading Patterns")
        self.info("Lazy loading defers expensive operations until needed:")
        self.info("  • File I/O: Open files only when accessed")
        self.info("  • Computation: Calculate results on-demand")
        self.info("  • Memory: Load only required data")
        self.info("  • Interactive: Fast startup, gradual loading")
        self.info("")

        # Demonstrate lazy file proxy
        self.subsection("1. Lazy File Proxy")
        results["lazy_proxy"] = self._demonstrate_lazy_proxy(data)

        # Demonstrate deferred computation
        self.subsection("2. Deferred Computation")
        results["deferred"] = self._demonstrate_deferred_computation(data)

        # Demonstrate selective loading
        self.subsection("3. Selective Loading")
        results["selective"] = self._demonstrate_selective_loading(data)

        # Best practices
        self.subsection("Lazy Loading Best Practices")
        self._show_best_practices()

        return results

    def _demonstrate_lazy_proxy(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate lazy file proxy pattern."""
        files = data["files"]

        self.info("Lazy proxy pattern: File not loaded until accessed")

        class LazyFileProxy:
            """Proxy that loads file only when data is accessed."""

            def __init__(self, filepath: Path, sample_rate: float):
                self.filepath = filepath
                self.sample_rate = sample_rate
                self._data = None
                self._loaded = False

            @property
            def data(self) -> np.ndarray:
                """Load data on first access."""
                if not self._loaded:
                    self._data = np.fromfile(self.filepath, dtype=np.float64)
                    self._loaded = True
                return self._data

            @property
            def is_loaded(self) -> bool:
                """Check if data has been loaded."""
                return self._loaded

        # Create lazy proxies for all files
        proxies = [LazyFileProxy(f["filepath"], data["sample_rate"]) for f in files]

        self.result("Files Created", len(proxies))
        self.result("Files Loaded", sum(p.is_loaded for p in proxies))

        # Access only one file
        _ = proxies[2].data  # Load channel 2

        self.result("After Accessing Channel 2", sum(p.is_loaded for p in proxies))

        return {
            "total_files": len(proxies),
            "loaded_files": sum(p.is_loaded for p in proxies),
        }

    def _demonstrate_deferred_computation(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate deferred computation pattern."""
        self.info("Deferred computation: Expensive calculations done on-demand")

        class DeferredAnalysis:
            """Deferred FFT computation."""

            def __init__(self, filepath: Path):
                self.filepath = filepath
                self._fft_result = None
                self._computed = False

            def get_fft(self) -> np.ndarray:
                """Compute FFT only when requested."""
                if not self._computed:
                    data = np.fromfile(self.filepath, dtype=np.float64)
                    self._fft_result = np.fft.rfft(data)
                    self._computed = True
                return self._fft_result

            @property
            def is_computed(self) -> bool:
                """Check if FFT has been computed."""
                return self._computed

        # Create deferred analysis objects
        analyses = [DeferredAnalysis(f["filepath"]) for f in data["files"]]

        self.result("Analyses Created", len(analyses))
        self.result("FFTs Computed", sum(a.is_computed for a in analyses))

        # Compute FFT for one channel
        _ = analyses[0].get_fft()

        self.result("After Computing Channel 0 FFT", sum(a.is_computed for a in analyses))

        return {
            "total_analyses": len(analyses),
            "computed_ffts": sum(a.is_computed for a in analyses),
        }

    def _demonstrate_selective_loading(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate selective loading based on criteria."""
        files = data["files"]

        self.info("Selective loading: Load only files matching criteria")

        # Only load files with frequency > 3 kHz
        freq_threshold = 3e3
        selected_files = [f for f in files if f["frequency"] > freq_threshold]

        self.result("Total Files", len(files))
        self.result("Frequency Threshold", f"{freq_threshold / 1e3:.0f}", "kHz")
        self.result("Files Selected", len(selected_files))

        # Load only selected files
        loaded_data = []
        total_bytes = 0
        for f in selected_files:
            data_array = np.fromfile(f["filepath"], dtype=np.float64)
            loaded_data.append(data_array)
            total_bytes += f["size"]

        self.result("Data Loaded", format_size(total_bytes))
        self.result("Memory Saved", format_size(sum(f["size"] for f in files) - total_bytes))

        return {
            "total_files": len(files),
            "loaded_files": len(selected_files),
            "bytes_loaded": total_bytes,
        }

    def _show_best_practices(self) -> None:
        """Show best practices for lazy loading."""
        self.info("""
Lazy Loading Best Practices:

1. WHEN TO USE LAZY LOADING
   - Interactive applications (fast startup)
   - Large datasets (avoid memory exhaustion)
   - Optional computations (may not be needed)
   - Exploratory analysis (unknown access patterns)

2. IMPLEMENTATION PATTERNS
   - Property decorators: Transparent lazy access
   - Proxy classes: Separate loading logic
   - Context managers: Explicit lifecycle
   - Generators: Streaming data access

3. CACHING STRATEGIES
   - Cache computed results (avoid recomputation)
   - LRU eviction: Keep recent data in memory
   - Size limits: Cap maximum memory usage
   - Invalidation: Clear cache when source changes

4. TRADEOFFS
   - Pros: Lower memory, faster startup
   - Cons: Unpredictable latency, complex code
   - Use profiling to identify bottlenecks
   - Balance between eager and lazy loading
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate lazy loading results."""
        suite = ValidationSuite()

        # Validate lazy proxy
        if "lazy_proxy" in results:
            suite.check_equal(results["lazy_proxy"]["total_files"], 5, "Total files")
            suite.check_equal(results["lazy_proxy"]["loaded_files"], 1, "Only accessed file loaded")

        # Validate deferred computation
        if "deferred" in results:
            suite.check_equal(results["deferred"]["total_analyses"], 5, "Total analyses")
            suite.check_equal(
                results["deferred"]["computed_ffts"], 1, "Only requested FFT computed"
            )

        # Validate selective loading
        if "selective" in results:
            suite.check_true(
                results["selective"]["loaded_files"] < results["selective"]["total_files"],
                "Selective loading reduced file count",
            )

        if suite.all_passed():
            self.success("All lazy loading validations passed!")
            self.info("\nKey Benefits:")
            self.info("  - Reduced memory usage")
            self.info("  - Faster application startup")
            self.info("  - On-demand resource allocation")
        else:
            self.error("Some validations failed")

        return suite.all_passed()


if __name__ == "__main__":
    demo = LazyLoadingDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
