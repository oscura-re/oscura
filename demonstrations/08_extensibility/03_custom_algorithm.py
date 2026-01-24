"""Custom Algorithms: Register and use custom FFT, filter, and analysis algorithms

Demonstrates:
- oscura.register_algorithm() - Register a custom algorithm
- oscura.get_algorithm() - Retrieve a registered algorithm
- oscura.get_algorithms() - List algorithms in a category
- Creating custom FFT algorithms
- Creating custom filter algorithms
- Creating custom analysis algorithms
- Practical integration examples

IEEE Standards: N/A
Related Demos:
- 08_extensibility/02_custom_measurement.py
- 02_basic_analysis/01_waveform_measurements.py
- 02_basic_analysis/03_spectral_analysis.py

Algorithm registration allows you to create custom signal processing algorithms
and integrate them seamlessly with Oscura's analysis pipeline. This is a P0
CRITICAL feature for extensibility.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

import oscura as osc
from demonstrations.common import BaseDemo, generate_sine_wave, validate_approximately

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


class CustomAlgorithmDemo(BaseDemo):
    """Demonstrates custom algorithm registration and usage."""

    def __init__(self) -> None:
        """Initialize custom algorithm demonstration."""
        super().__init__(
            name="custom_algorithm",
            description="Create, register, and use custom algorithms",
            capabilities=[
                "oscura.register_algorithm",
                "oscura.get_algorithm",
                "oscura.get_algorithms",
                "Custom FFT algorithms",
                "Custom filter algorithms",
                "Custom analysis algorithms",
            ],
            related_demos=[
                "08_extensibility/02_custom_measurement.py",
                "02_basic_analysis/03_spectral_analysis.py",
            ],
        )

    def generate_test_data(self) -> dict[str, WaveformTrace]:
        """Generate test signals for algorithm demonstrations.

        Returns:
            Dictionary with test traces:
            - 'sine_1khz': 1 kHz sine wave at 1V amplitude
            - 'sine_10khz': 10 kHz sine wave at 0.5V amplitude
            - 'noisy_sine': 1 kHz sine with 0.1V noise
        """
        # 1 kHz sine wave - fundamental signal
        sine_1khz = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=1.0,  # 1V peak
            duration=0.05,  # 50ms
            sample_rate=100e3,  # 100 kHz sampling
        )

        # 10 kHz sine wave - high frequency
        sine_10khz = generate_sine_wave(
            frequency=10000.0,  # 10 kHz
            amplitude=0.5,  # 0.5V peak
            duration=0.05,  # 50ms
            sample_rate=100e3,  # 100 kHz sampling
        )

        # Noisy 1 kHz sine
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, len(sine_1khz.data))
        noisy_data = sine_1khz.data + noise
        noisy_sine = osc.WaveformTrace(
            data=noisy_data,
            metadata=sine_1khz.metadata,
        )

        return {
            "sine_1khz": sine_1khz,
            "sine_10khz": sine_10khz,
            "noisy_sine": noisy_sine,
        }

    def run_demonstration(self, data: dict[str, WaveformTrace]) -> dict[str, Any]:
        """Run custom algorithm demonstration."""
        sine_1khz = data["sine_1khz"]
        sine_10khz = data["sine_10khz"]
        noisy_sine = data["noisy_sine"]
        results: dict[str, Any] = {}

        # ===== Section 1: Understanding Algorithm Categories =====
        self.section("Part 1: Algorithm Registry Overview")
        self.subsection("Available Algorithm Categories")
        self.info("Oscura supports custom algorithms in multiple categories:")
        self.info("  - fft: Fast Fourier Transform implementations")
        self.info("  - filter: Signal filtering algorithms")
        self.info("  - peak_finder: Peak detection algorithms")
        self.info("  - edge_detector: Edge detection algorithms")
        self.info("  - window_func: Window functions for spectral analysis")
        self.info("  - And more custom categories as needed")

        # ===== Section 2: Creating Custom FFT Algorithm =====
        self.section("Part 2: Custom FFT Algorithm")
        self.subsection("Example 1: Simple FFT with zero-padding")
        self.info("Custom FFT algorithm with configurable zero-padding")

        def simple_fft_zeropad(
            data: np.ndarray[Any, Any],
            pad_factor: float = 4.0,
            **kwargs: Any,
        ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
            """Simple FFT with zero-padding for improved frequency resolution.

            Args:
                data: Input signal samples
                pad_factor: Padding factor (default 4x)

            Returns:
                Tuple of (frequencies, magnitudes)
            """
            sample_rate = kwargs.get("sample_rate", 1.0)
            nfft = int(len(data) * pad_factor)

            # Perform zero-padded FFT
            fft_result = np.fft.fft(data, n=nfft)
            magnitudes = np.abs(fft_result) / len(data)
            frequencies = np.fft.fftfreq(nfft, 1 / sample_rate)

            # Return only positive frequencies
            positive_idx = frequencies >= 0
            return frequencies[positive_idx], magnitudes[positive_idx]

        # Register the custom FFT algorithm
        self.info("Registering: osc.register_algorithm('simple_zeropad', ...)")
        osc.register_algorithm(
            name="simple_zeropad",
            func=simple_fft_zeropad,
            category="fft",
        )
        self.success("✓ Registered custom FFT algorithm: simple_zeropad")
        results["fft_registered"] = True

        # ===== Section 3: Creating Custom Filter Algorithm =====
        self.section("Part 3: Custom Filter Algorithm")
        self.subsection("Example 2: Moving Average Filter")
        self.info("Simple moving average filter - smooths signal by averaging neighbors")

        def moving_average_filter(
            data: np.ndarray[Any, Any],
            window_size: int = 5,
            **kwargs: Any,
        ) -> np.ndarray[Any, Any]:
            """Moving average filter for signal smoothing.

            Args:
                data: Input signal samples
                window_size: Number of samples to average (default 5)

            Returns:
                Filtered signal
            """
            if window_size < 1:
                raise ValueError("window_size must be >= 1")

            # Create kernel for moving average
            kernel = np.ones(window_size) / window_size

            # Apply convolution (using 'same' to maintain size)
            filtered = np.convolve(data, kernel, mode="same")
            return filtered

        # Register the custom filter
        self.info("Registering: osc.register_algorithm('moving_avg', ...)")
        osc.register_algorithm(
            name="moving_avg",
            func=moving_average_filter,
            category="filter",
        )
        self.success("✓ Registered custom filter algorithm: moving_avg")
        results["filter_registered"] = True

        # ===== Section 4: Creating Custom Analysis Algorithm =====
        self.section("Part 4: Custom Analysis Algorithm")
        self.subsection("Example 3: Peak-to-Average Power Ratio (PAPR)")
        self.info("PAPR (Crest Factor) for signal quality analysis")

        def calculate_papr(
            data: np.ndarray[Any, Any],
            **kwargs: Any,
        ) -> float:
            """Calculate Peak-to-Average Power Ratio.

            PAPR is the ratio of peak power to average power.
            Used for signal quality and clipping analysis.

            Args:
                data: Input signal samples

            Returns:
                PAPR in linear (not dB)
            """
            peak_power = float(np.max(np.abs(data) ** 2))
            avg_power = float(np.mean(data**2))

            if avg_power == 0:
                return 0.0

            return peak_power / avg_power

        # Register the custom analysis algorithm
        self.info("Registering: osc.register_algorithm('papr', ...)")
        osc.register_algorithm(
            name="papr",
            func=calculate_papr,
            category="analysis",
        )
        self.success("✓ Registered custom analysis algorithm: papr")
        results["analysis_registered"] = True

        # ===== Section 5: Listing Registered Algorithms =====
        self.section("Part 5: Discovering Registered Algorithms")
        self.subsection("Listing Algorithms by Category")

        # List FFT algorithms
        fft_algos = osc.get_algorithms("fft")
        self.result("FFT algorithms registered", len(fft_algos))
        self.info(f"  Available: {', '.join(fft_algos)}")

        # List filter algorithms
        filter_algos = osc.get_algorithms("filter")
        self.result("Filter algorithms registered", len(filter_algos))
        self.info(f"  Available: {', '.join(filter_algos)}")

        # List analysis algorithms
        analysis_algos = osc.get_algorithms("analysis")
        self.result("Analysis algorithms registered", len(analysis_algos))
        self.info(f"  Available: {', '.join(analysis_algos)}")

        results["fft_algos"] = len(fft_algos)
        results["filter_algos"] = len(filter_algos)
        results["analysis_algos"] = len(analysis_algos)

        # ===== Section 6: Using Custom Algorithms =====
        self.section("Part 6: Using Custom Algorithms")
        self.subsection("Custom FFT with Zero-Padding")
        self.info("Comparing standard FFT with custom zero-padded FFT")

        # Get the registered algorithm
        custom_fft = osc.get_algorithm("fft", "simple_zeropad")
        self.info("Retrieved algorithm: osc.get_algorithm('fft', 'simple_zeropad')")

        # Apply custom FFT
        freq_custom, mag_custom = custom_fft(
            sine_1khz.data,
            sample_rate=sine_1khz.metadata.sample_rate,
        )

        # Show results
        peak_freq_idx = np.argmax(mag_custom)
        peak_freq = freq_custom[peak_freq_idx]
        self.result("Peak frequency detected", f"{peak_freq:.2f}", "Hz")
        self.result("Resolution improvement", f"{4.0:.1f}x", "zero-padding")

        results["custom_fft_peak"] = float(peak_freq)

        # ===== Section 7: Custom Filter Demonstration =====
        self.section("Part 7: Custom Filter Application")
        self.subsection("Smoothing Noisy Signal")
        self.info("Applying moving average filter to noisy signal")

        # Get the filter algorithm
        ma_filter = osc.get_algorithm("filter", "moving_avg")
        self.info("Retrieved algorithm: osc.get_algorithm('filter', 'moving_avg')")

        # Apply custom filter with different window sizes
        filtered_small = ma_filter(noisy_sine.data, window_size=3)
        filtered_medium = ma_filter(noisy_sine.data, window_size=7)
        filtered_large = ma_filter(noisy_sine.data, window_size=15)

        # Calculate improvement metrics
        original_noise = np.std(noisy_sine.data)
        noise_small = np.std(noisy_sine.data - filtered_small)
        noise_medium = np.std(noisy_sine.data - filtered_medium)
        noise_large = np.std(noisy_sine.data - filtered_large)

        self.result("Original noise std dev", f"{original_noise:.4f}")
        self.result("Filtered (window=3) std dev", f"{noise_small:.4f}")
        self.result("Filtered (window=7) std dev", f"{noise_medium:.4f}")
        self.result("Filtered (window=15) std dev", f"{noise_large:.4f}")

        results["original_noise"] = float(original_noise)
        results["filtered_noise"] = float(noise_large)

        # ===== Section 8: Custom Analysis Demonstration =====
        self.section("Part 8: Custom Analysis Algorithm")
        self.subsection("Peak-to-Average Power Ratio (PAPR)")
        self.info("Measuring signal quality using custom PAPR algorithm")

        # Get the analysis algorithm
        papr_func = osc.get_algorithm("analysis", "papr")
        self.info("Retrieved algorithm: osc.get_algorithm('analysis', 'papr')")

        # Calculate PAPR for different signals
        papr_sine = papr_func(sine_1khz.data)
        papr_noisy = papr_func(noisy_sine.data)
        papr_10khz = papr_func(sine_10khz.data)

        # Convert to dB for display
        papr_sine_db = 10 * np.log10(papr_sine)
        papr_noisy_db = 10 * np.log10(papr_noisy)
        papr_10khz_db = 10 * np.log10(papr_10khz)

        self.result("PAPR (1 kHz sine)", f"{papr_sine:.3f}", f"({papr_sine_db:.1f} dB)")
        self.result("PAPR (1 kHz noisy)", f"{papr_noisy:.3f}", f"({papr_noisy_db:.1f} dB)")
        self.result("PAPR (10 kHz sine)", f"{papr_10khz:.3f}", f"({papr_10khz_db:.1f} dB)")

        self.info("\nNote: Pure sine wave PAPR ≈ 2.0 (3.0 dB)")
        self.info("Noisy signals have higher PAPR due to noise peaks")

        results["papr_sine"] = float(papr_sine)
        results["papr_noisy"] = float(papr_noisy)

        # ===== Section 9: Integration Example =====
        self.section("Part 9: Integration with Oscura Pipeline")
        self.subsection("Complete Analysis Using Custom Algorithms")
        self.info("Demonstrating custom algorithms in a complete workflow")

        # Use custom FFT for spectral analysis
        freq, mag = custom_fft(
            sine_10khz.data,
            sample_rate=sine_10khz.metadata.sample_rate,
        )

        # Use custom filter to smooth the spectrum
        mag_filtered = ma_filter(mag, window_size=5)

        # Use custom analysis on filtered spectrum
        spectrum_papr = papr_func(mag_filtered)

        self.result("10 kHz signal analysis complete", "Success")
        self.result("Peak frequency", f"{freq[np.argmax(mag)]:.1f}", "Hz")
        self.result("Spectrum PAPR", f"{spectrum_papr:.3f}")

        # ===== Section 10: Error Handling and Best Practices =====
        self.section("Part 10: Best Practices and Error Handling")
        self.subsection("Duplicate Registration Prevention")
        self.info("Attempting to register algorithm with existing name...")

        try:
            osc.register_algorithm(
                name="simple_zeropad",  # Already registered
                func=simple_fft_zeropad,
                category="fft",
            )
            self.warning("ERROR: Should have raised ValueError")
            results["duplicate_error"] = False
        except ValueError as e:
            self.success(f"✓ Correctly caught duplicate: {e!s}")
            results["duplicate_error"] = True

        self.subsection("Type Validation")
        self.info("Custom algorithms are validated on registration")
        self.info("  - Must be callable")
        self.info("  - Should accept **kwargs for extensibility")
        self.info("  - Should have clear docstrings")

        # Final summary
        self.section("Summary")
        self.success("All custom algorithms registered and tested!")
        self.info(f"\nRegistered {results['fft_algos']} FFT algorithms")
        self.info(f"Registered {results['filter_algos']} filter algorithms")
        self.info(f"Registered {results['analysis_algos']} analysis algorithms")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate the results."""
        self.info("Validating custom algorithm demonstrations...")

        # Check registrations
        if not results.get("fft_registered"):
            self.error("FFT algorithm registration failed")
            return False

        if not results.get("filter_registered"):
            self.error("Filter algorithm registration failed")
            return False

        if not results.get("analysis_registered"):
            self.error("Analysis algorithm registration failed")
            return False

        # Check algorithm counts
        if results.get("fft_algos", 0) < 1:
            self.error("No FFT algorithms registered")
            return False

        if results.get("filter_algos", 0) < 1:
            self.error("No filter algorithms registered")
            return False

        if results.get("analysis_algos", 0) < 1:
            self.error("No analysis algorithms registered")
            return False

        # Validate FFT peak detection (~1000 Hz)
        if not validate_approximately(
            results.get("custom_fft_peak", 0),
            1000.0,
            tolerance=0.05,
            name="FFT peak frequency",
        ):
            return False

        # Validate noise reduction
        if results.get("original_noise", 1.0) < results.get("filtered_noise", 1.0):
            self.error("Filter did not reduce noise")
            return False

        # Validate PAPR values (should be ~2.0 for sine)
        papr_sine = results.get("papr_sine", 0)
        if not validate_approximately(papr_sine, 2.0, tolerance=0.1, name="PAPR (sine)"):
            return False

        # Check error handling
        if not results.get("duplicate_error"):
            self.error("Duplicate registration error handling failed")
            return False

        self.success("All validations passed!")
        self.info("\nNext steps:")
        self.info("  - Explore algorithm registration for your domain")
        self.info("  - Create custom algorithms for specialized analysis")
        self.info("  - Combine custom algorithms with built-in functions")
        self.info("  - See 08_extensibility/02_custom_measurement.py for measurements")

        return True


if __name__ == "__main__":
    demo = CustomAlgorithmDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
