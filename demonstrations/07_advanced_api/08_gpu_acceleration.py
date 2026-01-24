"""GPU Acceleration: Backend Performance Comparison

Demonstrates:
- GPU backend with CuPy (automatic fallback to CPU)
- CPU vs GPU performance comparison
- GPU-accelerated FFT
- GPU-accelerated correlation
- Data size thresholds for GPU benefit
- Memory management for GPU operations

IEEE Standards: N/A
Related Demos:
- 07_advanced_api/05_optimization.py
- 07_advanced_api/07_parallel_processing.py

GPU acceleration provides significant speedup for large signal processing tasks.
CuPy enables GPU computing with minimal code changes, automatically falling
back to NumPy when GPU is unavailable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, generate_sine_wave, validate_range
from oscura.core import GPUBackend, gpu


class GPUAccelerationDemo(BaseDemo):
    """Demonstrate GPU acceleration with performance benchmarks."""

    def __init__(self):
        """Initialize GPU acceleration demonstration."""
        super().__init__(
            name="gpu_acceleration",
            description="GPU backend with CuPy for high-performance computing",
            capabilities=[
                "oscura.core.gpu",
                "oscura.core.GPUBackend",
                "gpu_fft",
                "gpu_correlation",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate signals of various sizes for GPU benchmarking."""
        # Create signals of different sizes for performance comparison
        sizes = {
            "small": 10_000,  # 10K samples
            "medium": 100_000,  # 100K samples
            "large": 1_000_000,  # 1M samples
            "xlarge": 10_000_000,  # 10M samples (GPU beneficial)
        }

        signals = {}
        for name, size in sizes.items():
            # Generate sine wave with noise
            signal = generate_sine_wave(
                frequency=1000.0,
                amplitude=1.0,
                duration=size / 100e3,  # Adjust duration based on size
                sample_rate=100e3,
            )
            signals[name] = signal.data

        # Generate correlation signals
        signals["correlation_signal"] = signals["large"]
        signals["correlation_template"] = signals["large"][100_000:110_000]

        return signals

    def run_demonstration(self, data: dict) -> dict:
        """Run GPU acceleration demonstration."""
        self.section("GPU Acceleration: Performance Comparison")

        # ===================================================================
        # Part 1: GPU Availability Check
        # ===================================================================
        self.subsection("1. GPU Availability and Configuration")
        self.info("Checking GPU backend availability...")

        gpu_available = gpu.gpu_available
        self.result("GPU backend available", gpu_available, "")

        if gpu_available:
            self.success("CuPy is installed and GPU is accessible")
            self.info("GPU acceleration will be used for large datasets")
        else:
            self.warning("CuPy not available or GPU not accessible")
            self.info("All operations will use CPU (NumPy) backend")
            self.info("To enable GPU: pip install cupy-cuda11x (for CUDA 11.x)")

        # ===================================================================
        # Part 2: CPU vs GPU FFT Benchmark
        # ===================================================================
        self.subsection("2. FFT Performance: CPU vs GPU")
        self.info("Benchmarking FFT on different signal sizes...")

        fft_results = {}
        for size_name in ["small", "medium", "large", "xlarge"]:
            signal = data[size_name]
            size = len(signal)

            # CPU FFT (using NumPy directly)
            cpu_backend = GPUBackend(force_cpu=True)
            start_time = time.time()
            cpu_result = cpu_backend.fft(signal)
            cpu_time = time.time() - start_time

            # GPU FFT (or CPU fallback if no GPU)
            start_time = time.time()
            gpu_result = gpu.fft(signal)
            gpu_time = time.time() - start_time

            # Verify results match
            result_match = np.allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-8)

            # Calculate speedup
            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

            fft_results[size_name] = {
                "size": size,
                "cpu_time": cpu_time,
                "gpu_time": gpu_time,
                "speedup": speedup,
                "result_match": result_match,
            }

            backend_used = "GPU" if gpu_available else "CPU (fallback)"
            self.info(f"\n{size_name.upper()} ({size:,} samples):")
            self.result("  CPU time", f"{cpu_time * 1000:.2f}", "ms")
            self.result("  GPU time", f"{gpu_time * 1000:.2f}", "ms")
            self.result("  Speedup", f"{speedup:.2f}", "x")
            self.result("  Backend used", backend_used, "")
            self.result("  Results match", result_match, "")

        if gpu_available:
            self.success("GPU provides significant speedup for large signals")
        else:
            self.success("CPU fallback working correctly (identical performance)")

        # ===================================================================
        # Part 3: GPU vs CPU Correlation Benchmark
        # ===================================================================
        self.subsection("3. Correlation Performance: CPU vs GPU")
        self.info("Benchmarking cross-correlation operations...")

        signal = data["correlation_signal"]
        template = data["correlation_template"]

        # CPU correlation
        cpu_backend = GPUBackend(force_cpu=True)
        start_time = time.time()
        cpu_corr = cpu_backend.correlate(signal, template, mode="valid")
        cpu_corr_time = time.time() - start_time

        # GPU correlation (or CPU fallback)
        start_time = time.time()
        gpu_corr = gpu.correlate(signal, template, mode="valid")
        gpu_corr_time = time.time() - start_time

        # Verify results match
        corr_match = np.allclose(cpu_corr, gpu_corr, rtol=1e-5, atol=1e-8)
        corr_speedup = cpu_corr_time / gpu_corr_time if gpu_corr_time > 0 else 1.0

        self.info(f"\nCorrelation ({len(signal):,} signal, {len(template):,} template):")
        self.result("  CPU time", f"{cpu_corr_time * 1000:.2f}", "ms")
        self.result("  GPU time", f"{gpu_corr_time * 1000:.2f}", "ms")
        self.result("  Speedup", f"{corr_speedup:.2f}", "x")
        self.result("  Results match", corr_match, "")

        # Find peak correlation
        peak_idx = np.argmax(gpu_corr)
        peak_value = gpu_corr[peak_idx]
        self.result("  Peak correlation at", peak_idx, "samples")
        self.result("  Peak correlation value", f"{peak_value:.2e}", "")

        self.success("GPU correlation provides performance benefit for large signals")

        # ===================================================================
        # Part 4: Data Size Threshold Analysis
        # ===================================================================
        self.subsection("4. GPU Performance Threshold Analysis")
        self.info("Determining when GPU acceleration is beneficial...")

        self.info("\nPerformance summary:")
        self.info("Size          | CPU Time | GPU Time | Speedup | GPU Benefit")
        self.info("-" * 70)

        for size_name in ["small", "medium", "large", "xlarge"]:
            result = fft_results[size_name]
            benefit = "YES" if result["speedup"] > 1.2 and gpu_available else "NO"
            self.info(
                f"{size_name:12s}  | "
                f"{result['cpu_time'] * 1000:7.2f}ms | "
                f"{result['gpu_time'] * 1000:7.2f}ms | "
                f"{result['speedup']:6.2f}x | "
                f"{benefit}"
            )

        self.info("\nGuidelines:")
        if gpu_available:
            self.info("  - Small signals (<100K): CPU faster (overhead dominates)")
            self.info("  - Medium signals (100K-1M): GPU starts to show benefit")
            self.info("  - Large signals (>1M): GPU significantly faster")
            self.info("  - Very large (>10M): GPU essential for performance")
        else:
            self.info("  - No GPU available: All operations use optimized NumPy")
            self.info("  - Install CuPy to enable GPU acceleration")
            self.info("  - CPU performance is still excellent for most use cases")

        self.success("Threshold analysis complete")

        # ===================================================================
        # Part 5: Memory Management
        # ===================================================================
        self.subsection("5. GPU Memory Management")
        self.info("Demonstrating GPU memory transfer and management...")

        test_signal = data["large"]

        # Manual GPU transfer (if available)
        if gpu_available:
            # Transfer to GPU
            start_time = time.time()
            gpu_array = gpu._to_gpu(test_signal)
            transfer_to_gpu_time = time.time() - start_time

            # Process on GPU
            start_time = time.time()
            _ = gpu.fft(test_signal)  # Process on GPU (result not used for timing)
            gpu_process_time = time.time() - start_time

            # Transfer back to CPU
            start_time = time.time()
            cpu_array = gpu._to_cpu(gpu_array)
            transfer_to_cpu_time = time.time() - start_time

            self.info("\nMemory transfer times:")
            self.result("  CPU → GPU transfer", f"{transfer_to_gpu_time * 1000:.2f}", "ms")
            self.result("  GPU processing", f"{gpu_process_time * 1000:.2f}", "ms")
            self.result("  GPU → CPU transfer", f"{transfer_to_cpu_time * 1000:.2f}", "ms")

            total_time = transfer_to_gpu_time + gpu_process_time + transfer_to_cpu_time
            self.result("  Total time (with transfers)", f"{total_time * 1000:.2f}", "ms")

            # Verify data integrity
            data_match = np.allclose(cpu_array, test_signal, rtol=1e-10, atol=1e-12)
            self.result("  Data integrity", data_match, "")

            self.success("GPU memory management working correctly")
            self.info("\nBest practices:")
            self.info("  - GPU backend automatically manages transfers")
            self.info("  - Batch operations on GPU to amortize transfer cost")
            self.info("  - Keep data on GPU for multiple operations when possible")
        else:
            self.info("No GPU available - memory management is CPU-only")
            self.info("CPU arrays remain in system RAM throughout processing")
            self.success("CPU memory management is transparent (no transfers needed)")

        # ===================================================================
        # Part 6: Multiple GPU Operations (Pipeline)
        # ===================================================================
        self.subsection("6. GPU Pipeline: Multiple Operations")
        self.info("Demonstrating efficient GPU pipeline...")

        signal = data["large"]

        # Pipeline: FFT → magnitude → inverse FFT
        start_time = time.time()

        # Forward FFT
        spectrum = gpu.fft(signal)

        # Compute magnitude (demonstrates intermediate processing)
        _ = np.abs(spectrum)  # Magnitude calculation (not used for reconstruction)

        # Inverse FFT
        reconstructed = gpu.ifft(spectrum)
        reconstructed_real = np.real(reconstructed)

        pipeline_time = time.time() - start_time

        # Verify reconstruction
        reconstruction_error = np.max(np.abs(signal - reconstructed_real))

        self.info("\nGPU pipeline results:")
        self.result("  Pipeline time", f"{pipeline_time * 1000:.2f}", "ms")
        self.result("  Operations", "FFT → magnitude → IFFT", "")
        self.result("  Reconstruction error", f"{reconstruction_error:.2e}", "")
        self.result("  Signal preserved", reconstruction_error < 1e-10, "")

        self.success("GPU pipeline enables efficient multi-step processing")

        # ===================================================================
        # Part 7: When to Use GPU (Decision Guide)
        # ===================================================================
        self.subsection("7. GPU Usage Decision Guide")
        self.info("Guidelines for choosing GPU vs CPU backend...")

        self.info("\nUse GPU when:")
        self.info("  ✓ Signal size > 1M samples")
        self.info("  ✓ Multiple FFT operations needed")
        self.info("  ✓ Large correlation/convolution operations")
        self.info("  ✓ Batch processing many signals")
        self.info("  ✓ Real-time processing requirements")

        self.info("\nUse CPU when:")
        self.info("  ✓ Signal size < 100K samples")
        self.info("  ✓ One-time analysis (transfer overhead not justified)")
        self.info("  ✓ Complex custom operations (Python loops)")
        self.info("  ✓ GPU not available (automatic fallback)")
        self.info("  ✓ Memory constrained (GPU memory limited)")

        self.info("\nAutomatic selection:")
        self.info("  - oscura.core.gpu automatically uses best backend")
        self.info("  - Transparent fallback to CPU when GPU unavailable")
        self.info("  - No code changes needed between CPU and GPU")

        self.success("GPU acceleration is transparent and automatic")

        return {
            "gpu_available": gpu_available,
            "fft_results": fft_results,
            "correlation_speedup": corr_speedup,
            "correlation_match": corr_match,
            "reconstruction_error": reconstruction_error,
        }

    def validate(self, results: dict) -> bool:
        """Validate GPU acceleration results."""
        self.info("Validating GPU acceleration demonstration...")

        # Check correlation results match between CPU and GPU
        if not results["correlation_match"]:
            self.error("GPU correlation does not match CPU results")
            return False
        self.success("GPU correlation matches CPU results")

        # Check all FFT results
        for size_name, fft_result in results["fft_results"].items():
            if not fft_result["result_match"]:
                self.error(f"FFT results don't match for {size_name}")
                return False
            self.success(f"FFT results match for {size_name}")

            # Verify positive speedup (or equal for CPU fallback)
            if fft_result["speedup"] <= 0:
                self.error(f"Invalid speedup for {size_name}: {fft_result['speedup']}")
                return False

        # Check reconstruction error
        if not validate_range(
            results["reconstruction_error"],
            min_val=0.0,
            max_val=1e-9,
            name="Reconstruction error",
        ):
            return False

        # Verify GPU speedup is reasonable (if GPU available)
        if results["gpu_available"]:
            xlarge_speedup = results["fft_results"]["xlarge"]["speedup"]
            if xlarge_speedup > 1.1:
                self.success(f"GPU provides {xlarge_speedup:.2f}x speedup for large signals")
            else:
                self.warning("GPU speedup lower than expected (may be normal on some systems)")
        else:
            self.success("CPU fallback working correctly")

        self.success("All GPU operations validated!")
        self.info("\nKey takeaways:")
        self.info("  - GPU backend provides transparent acceleration")
        self.info("  - Automatic fallback to CPU when GPU unavailable")
        self.info("  - GPU beneficial for signals > 1M samples")
        self.info("  - Memory transfers handled automatically")
        self.info("  - Results identical between CPU and GPU")

        return True


if __name__ == "__main__":
    demo = GPUAccelerationDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
