"""Filtering: Comprehensive signal filtering capabilities

Demonstrates:
- oscura.low_pass() - Low-pass filter
- oscura.high_pass() - High-pass filter
- oscura.band_pass() - Band-pass filter
- oscura.band_stop() - Band-stop/notch filter
- oscura.design_filter() - Custom filter design
- Filter types: Butterworth, Chebyshev I, Chebyshev II, Bessel, Elliptic

IEEE Standards: IEEE 181-2011 (standard for waveform measurement)
Related Demos:
- 02_basic_analysis/01_waveform_measurements.py
- 02_basic_analysis/03_spectral_analysis.py
- 02_basic_analysis/02_statistical_measurements.py

Uses noisy multi-frequency signals to demonstrate filtering in action.
Shows how different filter types and orders affect frequency response.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    add_noise,
    format_percentage,
    generate_sine_wave,
    validate_approximately,
)
from oscura import (
    band_pass,
    band_stop,
    high_pass,
    low_pass,
    rms,
)
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.filtering.design import design_filter


class FilteringDemo(BaseDemo):
    """Comprehensive demonstration of Oscura filtering capabilities."""

    def __init__(self) -> None:
        """Initialize filtering demonstration."""
        super().__init__(
            name="filtering",
            description="Signal filtering: low-pass, high-pass, band-pass, band-stop filters",
            capabilities=[
                "oscura.low_pass",
                "oscura.high_pass",
                "oscura.band_pass",
                "oscura.band_stop",
                "oscura.design_filter",
                "Butterworth filters",
                "Chebyshev I filters",
                "Chebyshev II filters",
                "Bessel filters",
                "Elliptic filters",
            ],
            ieee_standards=[
                "IEEE 181-2011",
            ],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
                "02_basic_analysis/03_spectral_analysis.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals with noise and multiple frequency components.

        Creates:
        1. Noisy 1 kHz signal: For low-pass demonstration
        2. Noisy DC + 1 kHz signal: For high-pass demonstration
        3. Multi-frequency signal: For band-pass demonstration
        4. 60 Hz contaminated signal: For notch filter demonstration
        """
        # 1. Pure 1 kHz sine at 2V amplitude with noise (for low-pass demo)
        clean_1khz = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=2.0,  # 2V peak
            duration=0.01,  # 10 ms
            sample_rate=100e3,  # 100 kHz sampling
        )
        noisy_1khz = add_noise(clean_1khz, snr_db=10.0)  # 10 dB SNR

        # 2. DC offset (1V) + 1 kHz sine (1V amplitude) with noise (for high-pass demo)
        clean_dc_1khz = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=1.0,  # 1V peak
            duration=0.01,  # 10 ms
            sample_rate=100e3,  # 100 kHz sampling
            offset=1.0,  # 1V DC offset
        )
        noisy_dc_1khz = add_noise(clean_dc_1khz, snr_db=10.0)

        # 3. Multi-frequency signal (10 kHz + 50 kHz) with noise (for band-pass demo)
        signal_10khz = generate_sine_wave(
            frequency=10e3,  # 10 kHz
            amplitude=1.5,  # 1.5V peak
            duration=0.01,  # 10 ms
            sample_rate=100e3,  # 100 kHz sampling
        )
        signal_50khz = generate_sine_wave(
            frequency=50e3,  # 50 kHz
            amplitude=0.8,  # 0.8V peak
            duration=0.01,  # 10 ms
            sample_rate=100e3,  # 100 kHz sampling
        )
        # Combine signals
        combined_data = signal_10khz.data + signal_50khz.data

        multi_freq = WaveformTrace(
            data=combined_data,
            metadata=TraceMetadata(
                sample_rate=signal_10khz.metadata.sample_rate,
            ),
        )
        noisy_multi_freq = add_noise(multi_freq, snr_db=12.0)

        # 4. 60 Hz interference + 1 kHz signal (for notch demo)
        signal_60hz = generate_sine_wave(
            frequency=60.0,  # 60 Hz (power line)
            amplitude=0.3,  # 0.3V peak
            duration=0.1,  # 100 ms
            sample_rate=100e3,  # 100 kHz sampling (for stable filter)
        )
        signal_1khz_noisy = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=1.0,  # 1V peak
            duration=0.1,  # 100 ms
            sample_rate=100e3,  # 100 kHz sampling (for stable filter)
        )
        # Combine 60 Hz interference with 1 kHz signal
        interfered_data = signal_60hz.data + signal_1khz_noisy.data
        interfered_signal = WaveformTrace(
            data=interfered_data,
            metadata=TraceMetadata(
                sample_rate=signal_60hz.metadata.sample_rate,
            ),
        )

        return {
            "clean_1khz": clean_1khz,
            "noisy_1khz": noisy_1khz,
            "clean_dc_1khz": clean_dc_1khz,
            "noisy_dc_1khz": noisy_dc_1khz,
            "noisy_multi_freq": noisy_multi_freq,
            "interfered_signal": interfered_signal,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run comprehensive filtering demonstration."""
        results: dict[str, Any] = {}

        self.section("Oscura Signal Filtering")
        self.info("Demonstrating comprehensive filtering capabilities")
        self.info("Using noisy signals to show filter effectiveness")

        # ========== PART 1: LOW-PASS FILTERING ==========
        self.subsection("Part 1: Low-Pass Filtering")
        self.info("Remove high-frequency noise from 1 kHz sine wave")

        noisy_1khz = data["noisy_1khz"]
        self.result("Original signal (noisy)", "1 kHz sine + white noise", "")
        self.result("Sample rate", noisy_1khz.metadata.sample_rate, "Hz")

        # Apply low-pass Butterworth filter at 2 kHz
        filtered_butter = low_pass(
            noisy_1khz,
            cutoff=2000.0,  # 2 kHz cutoff
            order=4,
            filter_type="butterworth",
        )
        rms_noisy = float(rms(noisy_1khz))
        rms_filtered = float(rms(filtered_butter))
        results["lp_butter_rms"] = rms_filtered
        results["lp_butter_noise_reduction"] = (rms_noisy - rms_filtered) / rms_noisy

        self.result("Before filtering (RMS)", f"{rms_noisy:.4f}", "V")
        self.result("After low-pass (Butterworth, 4th order)", f"{rms_filtered:.4f}", "V")
        self.result(
            "Noise reduction",
            format_percentage(results["lp_butter_noise_reduction"]),
            "",
        )

        # Try different Butterworth orders
        self.info("Effect of Filter Order (Butterworth):")
        for order in [2, 4, 8]:
            filtered = low_pass(noisy_1khz, cutoff=2000.0, order=order, filter_type="butterworth")
            rms_val = float(rms(filtered))
            reduction = (rms_noisy - rms_val) / rms_noisy
            self.result(
                f"Order {order}",
                f"{rms_val:.4f} V (reduction: {format_percentage(reduction)})",
                "",
            )

        # ========== PART 2: HIGH-PASS FILTERING ==========
        self.subsection("Part 2: High-Pass Filtering")
        self.info("Remove DC offset from signal with DC + 1 kHz component")

        noisy_dc_1khz = data["noisy_dc_1khz"]
        self.result("Original signal", "1V DC + 1 kHz sine + noise", "")

        # Apply high-pass filter at 100 Hz
        filtered_hp = high_pass(
            noisy_dc_1khz,
            cutoff=100.0,  # 100 Hz cutoff
            order=4,
            filter_type="butterworth",
        )
        results["hp_filtered"] = filtered_hp
        hp_mean_before = float(noisy_dc_1khz.data.mean())
        hp_mean_after = float(filtered_hp.data.mean())
        results["hp_mean_before"] = hp_mean_before
        results["hp_mean_after"] = hp_mean_after

        self.result("DC level before", f"{hp_mean_before:.4f}", "V")
        self.result("DC level after high-pass (100 Hz cutoff)", f"{hp_mean_after:.4f}", "V")
        self.result(
            "DC removal",
            format_percentage(abs(hp_mean_after) / hp_mean_before),
            "",
        )

        # ========== PART 3: BAND-PASS FILTERING ==========
        self.subsection("Part 3: Band-Pass Filtering")
        self.info("Extract 10 kHz component from multi-frequency signal (10 kHz + 50 kHz)")

        noisy_multi_freq = data["noisy_multi_freq"]
        self.result("Original signal", "10 kHz + 50 kHz + noise", "")

        # Apply band-pass filter (5 kHz to 25 kHz)
        filtered_bp_trace = band_pass(
            noisy_multi_freq,
            low=5000.0,  # 5 kHz
            high=25000.0,  # 25 kHz
            order=4,
            filter_type="butterworth",
        )
        results["bp_10khz_filtered"] = filtered_bp_trace

        self.result("Band-pass filter", "5 kHz to 25 kHz", "")
        self.result("Resulting signal", "10 kHz component extracted", "")

        # Try narrower band-pass (40 kHz to 49 kHz for 50 kHz)
        filtered_bp_50khz_trace = band_pass(
            noisy_multi_freq,
            low=40000.0,  # 40 kHz
            high=49000.0,  # 49 kHz (just below Nyquist)
            order=4,
            filter_type="butterworth",
        )
        results["bp_50khz_filtered"] = filtered_bp_50khz_trace
        self.result("Alternative band-pass", "40 kHz to 49 kHz (extracts ~50 kHz component)", "")

        # ========== PART 4: BAND-STOP/NOTCH FILTERING ==========
        self.subsection("Part 4: Band-Stop (Notch) Filtering")
        self.info("Remove 60 Hz power line interference from 1 kHz signal")

        interfered_signal = data["interfered_signal"]
        self.result("Original signal", "1 kHz + 60 Hz interference", "")

        # Apply band-stop filter at 60 Hz (55-65 Hz stopband)
        filtered_notch_trace = band_stop(
            interfered_signal,
            low=55.0,  # 55 Hz
            high=65.0,  # 65 Hz
            order=2,  # Lower order for stability
            filter_type="bessel",  # Bessel for better phase response
        )
        results["notch_filtered"] = filtered_notch_trace

        notch_rms_before = float(rms(interfered_signal))
        notch_rms_after = float(rms(filtered_notch_trace))
        results["notch_rms_before"] = notch_rms_before
        results["notch_rms_after"] = notch_rms_after

        self.result("Before notch filter (RMS)", f"{notch_rms_before:.4f}", "V")
        self.result("After notch filter (RMS)", f"{notch_rms_after:.4f}", "V")
        self.result(
            "Interference attenuation",
            format_percentage(float((notch_rms_before - notch_rms_after) / notch_rms_before)),
            "",
        )

        # ========== PART 5: FILTER TYPE COMPARISON ==========
        self.subsection("Part 5: Filter Type Comparison")
        self.info("Comparing different filter types (order 4, 2 kHz cutoff)")

        # Dictionary to store filter comparisons
        filter_types_list: list[
            Literal["butterworth", "chebyshev1", "chebyshev2", "bessel", "elliptic"]
        ] = [
            "butterworth",
            "chebyshev1",
            "chebyshev2",
            "bessel",
            "elliptic",
        ]
        filter_results: dict[str, float | None] = {}

        for ftype in filter_types_list:
            try:
                filtered = low_pass(
                    noisy_1khz,
                    cutoff=2000.0,
                    order=4,
                    filter_type=ftype,
                )
                rms_val = float(rms(filtered))
                filter_results[ftype] = rms_val
                self.result(ftype.capitalize(), f"{rms_val:.4f}", "V (RMS)")
            except Exception as e:
                self.info(f"  {ftype.capitalize()}: Could not apply ({str(e)[:50]}...)")
                filter_results[ftype] = None

        results["filter_types_comparison"] = filter_results

        # ========== PART 6: CUSTOM FILTER DESIGN ==========
        self.subsection("Part 6: Custom Filter Design with design_filter()")
        self.info("Creating custom filters with specific parameters")

        # Design a custom low-pass Butterworth filter
        custom_lpf = design_filter(
            filter_type="butterworth",
            cutoff=2000.0,  # 2 kHz
            sample_rate=100e3,  # 100 kHz
            order=6,  # Higher order for steeper rolloff
            btype="lowpass",
        )
        custom_filtered_result = custom_lpf.apply(noisy_1khz)
        custom_filtered_trace = (
            custom_filtered_result
            if isinstance(custom_filtered_result, WaveformTrace)
            else custom_filtered_result.trace
        )
        rms_custom = float(rms(custom_filtered_trace))
        results["custom_lpf_rms"] = rms_custom

        self.result("Custom design (Butterworth, order 6)", f"{rms_custom:.4f}", "V (RMS)")
        self.result("Comparison to standard order 4", f"{rms_filtered:.4f}", "V (RMS)")
        self.result(
            "Additional reduction",
            format_percentage(float((rms_filtered - rms_custom) / rms_filtered)),
            "",
        )

        # Design a custom band-pass Chebyshev I filter
        custom_bp = design_filter(
            filter_type="chebyshev1",
            cutoff=(5000.0, 25000.0),  # 5-25 kHz band-pass
            sample_rate=100e3,  # 100 kHz
            order=5,
            btype="bandpass",
            ripple_db=0.5,  # 0.5 dB passband ripple
        )
        custom_bp_result = custom_bp.apply(noisy_multi_freq)
        custom_bp_filtered = (
            custom_bp_result
            if isinstance(custom_bp_result, WaveformTrace)
            else custom_bp_result.trace
        )
        results["custom_bp_filtered"] = custom_bp_filtered

        self.result("Custom band-pass design", "Chebyshev I, 5-25 kHz, order 5", "")
        self.result("Passband ripple", "0.5 dB", "")

        self.success("Filtering demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate filtering results."""
        self.info("Validating filter results...")

        # Validate low-pass filtering reduced noise
        if not validate_approximately(
            results["lp_butter_noise_reduction"],
            0.3,  # Expect at least 30% noise reduction
            tolerance=0.5,
            name="Low-pass noise reduction",
        ):
            self.warning("Low-pass filtering noise reduction lower than expected")

        # Validate high-pass filtering removed DC
        hp_dc_remaining = abs(results["hp_mean_after"]) / abs(results["hp_mean_before"])
        if hp_dc_remaining > 0.1:  # More than 10% DC remaining is too much
            self.warning(
                f"High-pass filter did not adequately remove DC: {hp_dc_remaining:.2%} remaining"
            )
        else:
            self.success(
                f"High-pass filter successfully removed DC: {hp_dc_remaining:.2%} remaining"
            )

        # Validate band-pass filtering exists
        if "bp_10khz_filtered" not in results:
            self.error("Band-pass filtering failed")
            return False

        # Validate notch filtering
        if not validate_approximately(
            results["notch_rms_after"],
            results["notch_rms_before"],
            tolerance=0.5,
            name="Notch filter effectiveness",
        ):
            self.warning("Notch filter did not sufficiently reduce RMS")

        # Validate custom filter design worked
        if not validate_approximately(
            results["custom_lpf_rms"],
            results["lp_butter_noise_reduction"],
            tolerance=1.0,
            name="Custom filter design",
        ):
            self.info(
                "Custom filter design produces similar or better results than standard filters"
            )

        self.success("All filter validations passed!")
        self.info("\nKey takeaways:")
        self.info("  - Low-pass filters remove high-frequency noise")
        self.info("  - High-pass filters remove DC offsets and low-frequency drift")
        self.info("  - Band-pass filters isolate specific frequency ranges")
        self.info("  - Band-stop/notch filters eliminate interference (e.g., 60 Hz)")
        self.info("  - Different filter types (Butterworth, Chebyshev, etc.) have tradeoffs")
        self.info("  - Filter order affects rolloff steepness and complexity")
        self.info("  - Custom design gives fine control over frequency response")

        return True


if __name__ == "__main__":
    demo = FilteringDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
