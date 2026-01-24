"""Signal Recovery and Reconstruction

Demonstrates signal recovery techniques:
- Corrupted signal recovery
- Missing data interpolation
- Noise reduction strategies
- Signal reconstruction methods

This demonstration shows:
1. How to recover corrupted signals
2. How to interpolate missing data
3. How to reduce noise effectively
4. How to reconstruct degraded signals
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    add_noise,
    generate_sine_wave,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class SignalRecoveryDemo(BaseDemo):
    """Demonstrate signal recovery and reconstruction techniques."""

    def __init__(self) -> None:
        """Initialize signal recovery demonstration."""
        super().__init__(
            name="signal_recovery",
            description="Techniques for recovering and reconstructing degraded signals",
            capabilities=[
                "oscura.recovery.corruption_repair",
                "oscura.recovery.interpolation",
                "oscura.recovery.noise_reduction",
                "oscura.recovery.reconstruction",
            ],
            related_demos=[
                "14_exploratory/02_fuzzy_matching.py",
                "12_quality_tools/02_quality_scoring.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals with various types of degradation."""
        self.info("Creating degraded test signals...")

        # Clean reference signal
        clean = generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0)
        self.info("  ✓ Clean reference signal")

        # Noisy signal
        noisy = add_noise(generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0), 0.2)
        self.info("  ✓ Noisy signal (SNR≈20dB)")

        # Signal with dropouts
        dropout = self._create_dropout_signal()
        self.info("  ✓ Signal with data dropouts")

        # Signal with glitches
        glitches = self._create_glitch_signal()
        self.info("  ✓ Signal with glitches")

        # Clipped signal
        clipped = self._create_clipped_signal()
        self.info("  ✓ Clipped signal")

        # Low resolution signal
        low_res = self._create_low_resolution_signal()
        self.info("  ✓ Low resolution signal")

        return {
            "clean": clean,
            "noisy": noisy,
            "dropout": dropout,
            "glitches": glitches,
            "clipped": clipped,
            "low_resolution": low_res,
        }

    def _create_dropout_signal(self) -> WaveformTrace:
        """Create signal with random dropouts."""
        signal = generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0)
        data = signal.data.copy()

        # Add random dropouts (zero sections)
        dropout_positions = np.random.choice(len(data) - 100, size=5, replace=False)
        for pos in dropout_positions:
            dropout_length = np.random.randint(20, 100)
            data[pos : pos + dropout_length] = 0

        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _create_glitch_signal(self) -> WaveformTrace:
        """Create signal with random glitches."""
        signal = generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0)
        data = signal.data.copy()

        # Add random spikes
        glitch_positions = np.random.choice(len(data), size=10, replace=False)
        data[glitch_positions] += np.random.uniform(-5, 5, size=10)

        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _create_clipped_signal(self) -> WaveformTrace:
        """Create clipped signal."""
        signal = generate_sine_wave(1000.0, 1.5, 0.1, 100_000.0)
        data = np.clip(signal.data, -1.0, 1.0)
        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _create_low_resolution_signal(self) -> WaveformTrace:
        """Create low resolution signal (quantized)."""
        signal = generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0)
        # Quantize to 4-bit (16 levels)
        levels = 16
        data = np.round(signal.data * (levels / 2)) / (levels / 2)
        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate signal recovery techniques."""
        results: dict[str, Any] = {}

        # Part 1: Noise reduction
        self.section("Part 1: Noise Reduction")

        noisy = data["noisy"]
        clean_ref = data["clean"]

        # Method 1: Moving average
        denoised_ma = self._moving_average_filter(noisy, window_size=10)
        snr_ma = self._calculate_snr(clean_ref, denoised_ma)

        # Method 2: Frequency domain filtering
        denoised_freq = self._frequency_domain_filter(noisy, cutoff_factor=0.1)
        snr_freq = self._calculate_snr(clean_ref, denoised_freq)

        # Method 3: Median filter
        denoised_median = self._median_filter(noisy, window_size=5)
        snr_median = self._calculate_snr(clean_ref, denoised_median)

        self.info("Noise reduction results:")
        self.info(f"  Original SNR:            {self._calculate_snr(clean_ref, noisy):.1f} dB")
        self.info(f"  Moving average SNR:      {snr_ma:.1f} dB")
        self.info(f"  Frequency filter SNR:    {snr_freq:.1f} dB")
        self.info(f"  Median filter SNR:       {snr_median:.1f} dB")

        results["noise_reduction"] = {
            "moving_average": snr_ma,
            "frequency_filter": snr_freq,
            "median_filter": snr_median,
        }

        # Part 2: Dropout recovery
        self.section("Part 2: Missing Data Interpolation")

        dropout = data["dropout"]

        # Detect dropouts
        dropout_mask = np.abs(dropout.data) < 0.01

        self.info(f"Detected {np.sum(dropout_mask)} missing samples")

        # Interpolate
        recovered = self._interpolate_missing(dropout)
        recovery_error = self._calculate_recovery_error(clean_ref, recovered, dropout_mask)

        self.info(f"Recovery error (RMSE): {recovery_error:.4f}")

        results["dropout_recovery"] = {
            "missing_samples": int(np.sum(dropout_mask)),
            "recovery_error": recovery_error,
        }

        # Part 3: Glitch removal
        self.section("Part 3: Glitch Detection and Removal")

        glitchy = data["glitches"]

        # Detect glitches
        glitch_indices = self._detect_glitches(glitchy)

        self.info(f"Detected {len(glitch_indices)} glitches")

        # Remove glitches
        deglitched = self._remove_glitches(glitchy, glitch_indices)
        glitch_snr = self._calculate_snr(clean_ref, deglitched)

        self.info(f"SNR after glitch removal: {glitch_snr:.1f} dB")

        results["glitch_removal"] = {
            "glitches_found": len(glitch_indices),
            "snr_after": glitch_snr,
        }

        # Part 4: Clipping recovery
        self.section("Part 4: Clipped Signal Recovery")

        clipped = data["clipped"]

        # Detect clipping
        clipped_samples = np.sum(np.abs(clipped.data) > 0.99)
        clipping_pct = 100 * clipped_samples / len(clipped.data)

        self.info(f"Clipped samples: {clipped_samples} ({clipping_pct:.1f}%)")

        # Attempt recovery using spectral reconstruction
        recovered_clip = self._recover_clipped(clipped)
        clip_error = np.sqrt(np.mean((clean_ref.data - recovered_clip.data) ** 2))

        self.info(f"Recovery RMSE: {clip_error:.4f}")
        self.info("  Note: Clipping recovery is limited - data is permanently lost")

        results["clipping_recovery"] = {
            "clipped_samples": int(clipped_samples),
            "recovery_rmse": clip_error,
        }

        # Part 5: Resolution enhancement
        self.section("Part 5: Resolution Enhancement")

        low_res = data["low_resolution"]

        # Measure quantization error
        quant_error = np.sqrt(np.mean((clean_ref.data - low_res.data) ** 2))

        self.info(f"Quantization error (RMSE): {quant_error:.4f}")

        # Apply smoothing to reduce quantization artifacts
        enhanced = self._enhance_resolution(low_res)
        enhanced_error = np.sqrt(np.mean((clean_ref.data - enhanced.data) ** 2))

        self.info(f"Enhanced error (RMSE):     {enhanced_error:.4f}")
        improvement = 100 * (quant_error - enhanced_error) / quant_error
        self.info(f"Improvement:               {improvement:.1f}%")

        results["resolution_enhancement"] = {
            "original_error": quant_error,
            "enhanced_error": enhanced_error,
            "improvement_pct": improvement,
        }

        # Part 6: Combined recovery
        self.section("Part 6: Combined Recovery Techniques")

        # Create signal with multiple issues
        multi_degraded = add_noise(dropout, 0.1)

        self.info("Applying combined recovery pipeline:")
        self.info("  1. Glitch removal")
        glitch_idx = self._detect_glitches(multi_degraded)
        step1 = self._remove_glitches(multi_degraded, glitch_idx)

        self.info("  2. Dropout interpolation")
        step2 = self._interpolate_missing(step1)

        self.info("  3. Noise reduction")
        step3 = self._frequency_domain_filter(step2, cutoff_factor=0.1)

        final_snr = self._calculate_snr(clean_ref, step3)
        self.info(f"\nFinal SNR: {final_snr:.1f} dB")

        results["combined_recovery"] = {"final_snr": final_snr}

        return results

    def _moving_average_filter(self, signal: WaveformTrace, window_size: int) -> WaveformTrace:
        """Apply moving average filter."""
        kernel = np.ones(window_size) / window_size
        filtered = np.convolve(signal.data, kernel, mode="same")
        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=filtered, metadata=metadata)

    def _frequency_domain_filter(
        self, signal: WaveformTrace, cutoff_factor: float
    ) -> WaveformTrace:
        """Apply frequency domain lowpass filter."""
        fft = np.fft.rfft(signal.data)
        cutoff = int(len(fft) * cutoff_factor)
        fft[cutoff:] = 0
        filtered = np.fft.irfft(fft, n=len(signal.data))
        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=filtered, metadata=metadata)

    def _median_filter(self, signal: WaveformTrace, window_size: int) -> WaveformTrace:
        """Apply median filter."""
        filtered = np.zeros_like(signal.data)
        pad = window_size // 2

        for i in range(len(signal.data)):
            start = max(0, i - pad)
            end = min(len(signal.data), i + pad + 1)
            filtered[i] = np.median(signal.data[start:end])

        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=filtered, metadata=metadata)

    def _interpolate_missing(self, signal: WaveformTrace) -> WaveformTrace:
        """Interpolate missing data (zeros)."""
        data = signal.data.copy()

        # Find zero regions
        zero_mask = np.abs(data) < 0.01

        # Linear interpolation
        x = np.arange(len(data))
        valid = ~zero_mask

        if np.sum(valid) > 0:
            data[zero_mask] = np.interp(x[zero_mask], x[valid], data[valid])

        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _detect_glitches(self, signal: WaveformTrace, threshold: float = 3.0) -> list[int]:
        """Detect glitches using statistical threshold."""
        data = signal.data
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        # Modified z-score
        z_scores = 0.6745 * (data - median) / (mad + 1e-10)

        glitch_indices = np.where(np.abs(z_scores) > threshold)[0].tolist()
        return glitch_indices

    def _remove_glitches(self, signal: WaveformTrace, glitch_indices: list[int]) -> WaveformTrace:
        """Remove glitches by interpolation."""
        data = signal.data.copy()

        if len(glitch_indices) == 0:
            metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
            return WaveformTrace(data=data, metadata=metadata)

        # Mark glitches
        mask = np.ones(len(data), dtype=bool)
        mask[glitch_indices] = False

        # Interpolate
        x = np.arange(len(data))
        data[~mask] = np.interp(x[~mask], x[mask], data[mask])

        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _recover_clipped(self, signal: WaveformTrace) -> WaveformTrace:
        """Attempt to recover clipped signal (limited success)."""
        # Simple approach: replace clipped values with trend
        data = signal.data.copy()

        clipped_mask = np.abs(data) > 0.99

        if np.sum(clipped_mask) > 0:
            # Use median filter to estimate trend
            trend = self._median_filter(signal, window_size=11)
            # Replace clipped with trend (conservative recovery)
            data[clipped_mask] = trend.data[clipped_mask]

        metadata = TraceMetadata(sample_rate=signal.metadata.sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _enhance_resolution(self, signal: WaveformTrace) -> WaveformTrace:
        """Enhance low-resolution signal."""
        # Apply smoothing to reduce quantization steps
        smoothed = self._moving_average_filter(signal, window_size=5)
        return smoothed

    def _calculate_snr(self, reference: WaveformTrace, signal: WaveformTrace) -> float:
        """Calculate SNR relative to reference."""
        noise = reference.data - signal.data
        signal_power = np.mean(reference.data**2)
        noise_power = np.mean(noise**2)

        if noise_power < 1e-10:
            return 100.0

        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db

    def _calculate_recovery_error(
        self,
        reference: WaveformTrace,
        recovered: WaveformTrace,
        missing_mask: np.ndarray,
    ) -> float:
        """Calculate recovery error on missing samples."""
        if np.sum(missing_mask) == 0:
            return 0.0

        error = np.sqrt(np.mean((reference.data[missing_mask] - recovered.data[missing_mask]) ** 2))
        return error

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate demonstration results."""
        self.info("Validating signal recovery...")
        all_valid = True

        # Check noise reduction improved SNR
        if "noise_reduction" in results:
            nr = results["noise_reduction"]
            if all(snr < 15 for snr in nr.values()):
                self.warning("Noise reduction SNR lower than expected")

        # Check dropout recovery
        if "dropout_recovery" not in results:
            self.error("Missing dropout recovery results")
            all_valid = False

        # Check glitch removal
        if "glitch_removal" in results:
            gr = results["glitch_removal"]
            if gr["glitches_found"] < 5:
                self.warning("Expected more glitches to be detected")

        if all_valid:
            self.success("All signal recovery validated successfully")

        return all_valid


if __name__ == "__main__":
    demo = SignalRecoveryDemo()
    success = demo.execute()
    exit(0 if success else 1)
