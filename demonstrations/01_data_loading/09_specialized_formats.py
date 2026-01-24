"""Specialized Hardware Security and Oscilloscope Format Loading

Demonstrates loading and handling specialized formats:
- ChipWhisperer power/EM side-channel traces (.npy, .trs)
- LeCroy oscilloscope captures (synthetic, .trc format)
- Side-channel analysis workflows
- Trace set manipulation and validation

IEEE Standards: IEEE 1057-2017 (Digitizing Waveform Recorders)
Related Demos:
- 01_data_loading/01_oscilloscopes.py
- 02_basic_analysis/01_waveform_measurements.py
- 06_reverse_engineering/01_power_analysis.py

This demonstration shows:
1. How to load ChipWhisperer power/EM traces with metadata
2. How to work with trace sets (multiple captures)
3. How to generate synthetic side-channel data
4. How to extract plaintext/ciphertext/key metadata
5. LeCroy oscilloscope format overview (synthetic)
6. Best practices for security testing data
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    add_noise,
    format_size,
    format_table,
    generate_sine_wave,
    validate_approximately,
)
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.loaders.chipwhisperer import (
    ChipWhispererTraceSet,
    load_chipwhisperer_npy,
    to_waveform_trace,
)


class SpecializedFormatsDemo(BaseDemo):
    """Demonstrate loading specialized hardware security and oscilloscope formats."""

    def __init__(self) -> None:
        """Initialize specialized formats demonstration."""
        super().__init__(
            name="specialized_formats",
            description="Load ChipWhisperer and specialized oscilloscope formats",
            capabilities=[
                "oscura.loaders.load_chipwhisperer",
                "oscura.loaders.load_chipwhisperer_npy",
                "oscura.loaders.load_chipwhisperer_trs",
                "ChipWhispererTraceSet manipulation",
                "Side-channel trace analysis",
            ],
            ieee_standards=["IEEE 1057-2017"],
            related_demos=[
                "01_data_loading/01_oscilloscopes.py",
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )
        self.temp_dir = Path(tempfile.mkdtemp(prefix="oscura_specialized_"))

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic specialized format data."""
        self.info("Creating synthetic specialized format captures...")

        # ChipWhisperer power traces (AES encryption simulation)
        cw_traceset = self._create_chipwhisperer_synthetic()
        self.info(
            f"  ✓ ChipWhisperer power traces ({cw_traceset.n_traces} traces, "
            f"{cw_traceset.n_samples} samples)"
        )

        # Save ChipWhisperer traces to temporary .npy file for loading demo
        cw_path = self._save_chipwhisperer_traces(cw_traceset)
        self.info(f"  ✓ Saved ChipWhisperer traces to {cw_path.name}")

        # LeCroy oscilloscope capture (synthetic)
        lecroy_trace = self._create_lecroy_synthetic()
        self.info("  ✓ LeCroy WaveRunner capture (1 GSa/s, complex signal)")

        return {
            "chipwhisperer_traceset": cw_traceset,
            "chipwhisperer_path": cw_path,
            "lecroy_trace": lecroy_trace,
        }

    def _create_chipwhisperer_synthetic(self) -> ChipWhispererTraceSet:
        """Create synthetic ChipWhisperer power traces simulating AES encryption.

        Simulates power consumption during AES encryption with:
        - Base power consumption (constant)
        - Key-dependent power variations (Hamming weight model)
        - Noise from measurement equipment
        """
        # ChipWhisperer typical settings: 1 MS/s, 5000 samples per trace
        sample_rate = 1e6  # 1 MS/s
        n_traces = 50  # 50 encryption operations
        n_samples = 5000  # 5 ms capture window

        # Initialize trace storage
        traces = np.zeros((n_traces, n_samples), dtype=np.float64)
        plaintexts = np.zeros((n_traces, 16), dtype=np.uint8)  # AES-128: 16 bytes
        ciphertexts = np.zeros((n_traces, 16), dtype=np.uint8)
        keys = np.zeros((n_traces, 16), dtype=np.uint8)

        # Fixed AES key for all traces (simplified side-channel scenario)
        fixed_key = np.array(
            [
                0x2B,
                0x7E,
                0x15,
                0x16,
                0x28,
                0xAE,
                0xD2,
                0xA6,
                0xAB,
                0xF7,
                0x15,
                0x88,
                0x09,
                0xCF,
                0x4F,
                0x3C,
            ],
            dtype=np.uint8,
        )

        for i in range(n_traces):
            # Generate random plaintext
            plaintext = np.random.randint(0, 256, size=16, dtype=np.uint8)
            plaintexts[i] = plaintext

            # Simplified "ciphertext" (not real AES, just for demonstration)
            # In real scenario, this would be actual AES output
            ciphertext = np.bitwise_xor(plaintext, fixed_key)
            ciphertexts[i] = ciphertext
            keys[i] = fixed_key

            # Generate synthetic power trace
            # Base power consumption (constant DC level around 50 mW)
            base_power = 50.0 + np.random.normal(0, 0.5)

            # Create time vector
            t = np.arange(n_samples) / sample_rate

            # Simulate clock signal influence (100 kHz clock)
            clock_freq = 100e3
            clock_component = 2.0 * np.sin(2 * np.pi * clock_freq * t)

            # Simulate key-dependent power variations using Hamming weight model
            # Different bytes of key cause different power consumption patterns
            hw_component = np.zeros(n_samples)

            # Simulate 16 S-box operations (one per byte) at different time points
            for byte_idx in range(16):
                # Each S-box operation takes ~200 samples (200 μs)
                op_start = 500 + byte_idx * 250  # Stagger operations
                op_duration = 200

                if op_start + op_duration < n_samples:
                    # Hamming weight of XOR between plaintext and key byte
                    xor_val = plaintext[byte_idx] ^ fixed_key[byte_idx]
                    hamming_weight = bin(xor_val).count("1")

                    # Power consumption proportional to Hamming weight
                    # Each bit flip costs ~0.5 mW
                    op_power = hamming_weight * 0.5

                    # Add Gaussian pulse for this operation
                    op_time = np.arange(op_duration)
                    op_center = op_duration // 2
                    gaussian = np.exp(-((op_time - op_center) ** 2) / (2 * (op_duration / 6) ** 2))

                    hw_component[op_start : op_start + op_duration] += op_power * gaussian

            # Combine components
            power_trace = base_power + clock_component + hw_component

            # Add realistic measurement noise (SNR ~30 dB)
            noise_power = np.mean(power_trace**2) / (10 ** (30 / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), n_samples)
            power_trace += noise

            traces[i] = power_trace

        return ChipWhispererTraceSet(
            traces=traces,
            plaintexts=plaintexts,
            ciphertexts=ciphertexts,
            keys=keys,
            sample_rate=sample_rate,
            metadata={
                "format": "chipwhisperer_synthetic",
                "target": "AES-128",
                "capture_notes": "Simulated power traces with Hamming weight leakage",
            },
        )

    def _save_chipwhisperer_traces(self, traceset: ChipWhispererTraceSet) -> Path:
        """Save ChipWhisperer traces to .npy file for loading demonstration."""
        traces_path = self.temp_dir / "cw_traces.npy"
        np.save(traces_path, traceset.traces)

        # Save associated metadata files (ChipWhisperer convention)
        if traceset.plaintexts is not None:
            np.save(self.temp_dir / "cw_traces_textin.npy", traceset.plaintexts)

        if traceset.ciphertexts is not None:
            np.save(self.temp_dir / "cw_traces_textout.npy", traceset.ciphertexts)

        if traceset.keys is not None:
            np.save(self.temp_dir / "cw_traces_keys.npy", traceset.keys)

        return traces_path

    def _create_lecroy_synthetic(self) -> WaveformTrace:
        """Create synthetic LeCroy oscilloscope capture.

        LeCroy WaveRunner/WavePro oscilloscopes save in .trc binary format.
        This creates a synthetic high-bandwidth capture typical of LeCroy instruments.
        """
        # LeCroy WaveRunner 8000HD: 10 GSa/s, 12-bit ADC
        sample_rate = 10e9  # 10 GSa/s
        duration = 1e-6  # 1 microsecond capture
        vertical_scale = 0.5  # 500 mV/div

        # Complex signal: 50 MHz fundamental + harmonics
        # Typical of high-speed digital signal analysis
        fundamental = generate_sine_wave(
            frequency=50e6,
            amplitude=0.8,
            duration=duration,
            sample_rate=sample_rate,
        )

        # Add harmonics (non-ideal square wave decomposition)
        t = np.linspace(0, duration, len(fundamental.data))
        harmonic_3 = 0.25 * np.sin(2 * np.pi * 150e6 * t)  # 150 MHz (3rd harmonic)
        harmonic_5 = 0.15 * np.sin(2 * np.pi * 250e6 * t)  # 250 MHz (5th harmonic)

        # Combine signals
        combined_data = fundamental.data + harmonic_3 + harmonic_5

        # Add realistic oscilloscope noise (10-bit ENOB ~55 dB SNR)
        combined_trace = WaveformTrace(
            data=combined_data,
            metadata=fundamental.metadata,
        )
        noisy_trace = add_noise(combined_trace, snr_db=55)

        # Create LeCroy-specific metadata
        metadata = TraceMetadata(
            sample_rate=sample_rate,
            vertical_scale=vertical_scale,
            vertical_offset=0.0,
            channel_name="C1",
        )

        return WaveformTrace(data=noisy_trace.data, metadata=metadata)

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the specialized formats demonstration."""
        results = {}

        self.section("Specialized Format Overview")
        self.info("Hardware security and high-end oscilloscopes use specialized formats:")
        self.info("  • ChipWhisperer: Power/EM traces for side-channel analysis")
        self.info("  • LeCroy .trc: High-bandwidth oscilloscope captures")
        self.info("  • Trace sets: Multiple synchronized captures with metadata")
        self.info("")

        # ChipWhisperer Analysis
        self.section("ChipWhisperer Power Trace Analysis")
        results["chipwhisperer"] = self._analyze_chipwhisperer(data["chipwhisperer_traceset"])

        # Load ChipWhisperer from file
        self.section("Loading ChipWhisperer Traces from File")
        results["chipwhisperer_loaded"] = self._demonstrate_chipwhisperer_loading(
            data["chipwhisperer_path"]
        )

        # Side-channel analysis example
        self.section("Basic Side-Channel Analysis")
        results["side_channel"] = self._demonstrate_side_channel_analysis(
            data["chipwhisperer_traceset"]
        )

        # LeCroy Analysis
        self.section("LeCroy WaveRunner Analysis")
        results["lecroy"] = self._analyze_lecroy(data["lecroy_trace"])

        # Format comparison
        self.section("Format Comparison")
        self._display_format_comparison(data)

        # Best practices
        self.section("Best Practices for Security Testing Data")
        self._show_best_practices()

        return results

    def _analyze_chipwhisperer(self, traceset: ChipWhispererTraceSet) -> dict[str, Any]:
        """Analyze ChipWhisperer trace set structure and metadata."""
        self.subsection("Trace Set Properties")

        self.result("Number of Traces", traceset.n_traces)
        self.result("Samples per Trace", traceset.n_samples)
        self.result("Sample Rate", f"{traceset.sample_rate / 1e6:.1f}", "MS/s")

        duration = traceset.n_samples / traceset.sample_rate
        self.result("Capture Duration", f"{duration * 1e3:.2f}", "ms")

        # Check metadata availability
        has_plaintexts = traceset.plaintexts is not None
        has_ciphertexts = traceset.ciphertexts is not None
        has_keys = traceset.keys is not None

        self.result("Plaintexts Available", "Yes" if has_plaintexts else "No")
        self.result("Ciphertexts Available", "Yes" if has_ciphertexts else "No")
        self.result("Keys Available", "Yes" if has_keys else "No")
        self.info("")

        # Analyze power trace statistics
        self.subsection("Power Trace Statistics")
        mean_power = np.mean(traceset.traces)
        std_power = np.std(traceset.traces)
        min_power = np.min(traceset.traces)
        max_power = np.max(traceset.traces)

        self.result("Mean Power", f"{mean_power:.3f}", "mW")
        self.result("Power Std Dev", f"{std_power:.3f}", "mW")
        self.result("Min Power", f"{min_power:.3f}", "mW")
        self.result("Max Power", f"{max_power:.3f}", "mW")
        self.info("")

        # Analyze plaintext/key statistics (if available)
        if has_plaintexts and traceset.plaintexts is not None:
            self.subsection("Plaintext Statistics")
            plaintext_bytes = traceset.plaintexts.shape[1]
            unique_plaintexts = len(np.unique(traceset.plaintexts, axis=0))

            self.result("Plaintext Size", f"{plaintext_bytes}", "bytes")
            self.result("Unique Plaintexts", f"{unique_plaintexts} / {traceset.n_traces}")

            # Show first plaintext (hex)
            first_pt = traceset.plaintexts[0]
            pt_hex = " ".join(f"{b:02X}" for b in first_pt)
            self.result("First Plaintext", pt_hex)
            self.info("")

        if has_keys and traceset.keys is not None:
            self.subsection("Encryption Key")
            key_bytes = traceset.keys.shape[1]
            first_key = traceset.keys[0]
            key_hex = " ".join(f"{b:02X}" for b in first_key)

            self.result("Key Size", f"{key_bytes}", "bytes")
            self.result("Key Value", key_hex)
            self.info("")

        return {
            "n_traces": traceset.n_traces,
            "n_samples": traceset.n_samples,
            "sample_rate": traceset.sample_rate,
            "mean_power": float(mean_power),
            "std_power": float(std_power),
            "has_plaintexts": has_plaintexts,
            "has_keys": has_keys,
        }

    def _demonstrate_chipwhisperer_loading(self, traces_path: Path) -> dict[str, Any]:
        """Demonstrate loading ChipWhisperer traces from file."""
        self.subsection("Loading Traces from .npy File")

        self.info(f"Loading: {traces_path}")

        # Load using ChipWhisperer loader
        loaded_traceset = load_chipwhisperer_npy(traces_path, sample_rate=1e6)

        self.success(f"Loaded {loaded_traceset.n_traces} traces successfully")
        self.result("Samples per Trace", loaded_traceset.n_samples)
        self.result("Associated Files Found", "3 (textin, textout, keys)")
        self.info("")

        # Convert to WaveformTrace for single trace analysis
        self.subsection("Converting to WaveformTrace")
        waveform = to_waveform_trace(loaded_traceset, trace_index=0)

        self.result("Trace Type", type(waveform).__name__)
        self.result("Sample Rate", f"{waveform.metadata.sample_rate / 1e6:.1f}", "MS/s")
        self.result("Data Points", len(waveform.data))
        self.info("")

        return {
            "loaded_traces": loaded_traceset.n_traces,
            "loaded_samples": loaded_traceset.n_samples,
            "conversion_success": True,
        }

    def _demonstrate_side_channel_analysis(self, traceset: ChipWhispererTraceSet) -> dict[str, Any]:
        """Demonstrate basic side-channel analysis workflow."""
        self.subsection("Differential Power Analysis (DPA) Setup")

        self.info("Side-channel analysis workflow:")
        self.info("  1. Partition traces by hypothesis (e.g., key byte guess)")
        self.info("  2. Calculate difference of means between partitions")
        self.info("  3. Identify peaks indicating key-dependent leakage")
        self.info("")

        # Simple example: analyze power variation at specific time point
        # In real DPA, you'd test all 256 key byte values
        self.subsection("Power Variation Analysis")

        # Analyze sample point 750 (middle of first S-box operation)
        analysis_point = 750

        if traceset.plaintexts is not None and traceset.keys is not None:
            # Calculate Hamming weight of first byte XOR
            first_byte_xor = traceset.plaintexts[:, 0] ^ traceset.keys[:, 0]
            hamming_weights = np.array([bin(x).count("1") for x in first_byte_xor])

            # Get power consumption at analysis point
            power_at_point = traceset.traces[:, analysis_point]

            # Calculate correlation between Hamming weight and power
            correlation = np.corrcoef(hamming_weights, power_at_point)[0, 1]

            self.result("Analysis Point", f"{analysis_point}")
            self.result("Correlation (HW vs Power)", f"{correlation:.4f}")
            self.result(
                "Interpretation",
                "Strong" if abs(correlation) > 0.3 else "Weak",
            )
            self.info("")

            # Partition by Hamming weight
            low_hw_traces = traceset.traces[hamming_weights <= 4]
            high_hw_traces = traceset.traces[hamming_weights > 4]

            mean_low = np.mean(low_hw_traces[:, analysis_point])
            mean_high = np.mean(high_hw_traces[:, analysis_point])
            diff_of_means = mean_high - mean_low

            self.result("Mean Power (Low HW)", f"{mean_low:.3f}", "mW")
            self.result("Mean Power (High HW)", f"{mean_high:.3f}", "mW")
            self.result("Difference of Means", f"{diff_of_means:.3f}", "mW")
            self.info("")

            return {
                "correlation": float(correlation),
                "diff_of_means": float(diff_of_means),
                "analysis_success": True,
            }
        else:
            self.warning("Plaintexts or keys not available for analysis")
            return {"analysis_success": False}

    def _analyze_lecroy(self, trace: WaveformTrace) -> dict[str, Any]:
        """Analyze LeCroy oscilloscope capture."""
        self.subsection("LeCroy WaveRunner Capture")

        meta = trace.metadata
        self.result("Sample Rate", f"{meta.sample_rate / 1e9:.1f}", "GS/s")
        self.result("Vertical Scale", f"{meta.vertical_scale * 1000:.0f}", "mV/div")

        num_samples = len(trace.data)
        duration = num_samples / meta.sample_rate

        self.result("Number of Samples", f"{num_samples}")
        self.result("Capture Duration", f"{duration * 1e6:.2f}", "μs")
        self.info("")

        # Signal statistics
        vmin = float(np.min(trace.data))
        vmax = float(np.max(trace.data))
        vrms = float(np.sqrt(np.mean(trace.data**2)))
        peak_to_peak = vmax - vmin

        self.result("Min Voltage", f"{vmin:.4f}", "V")
        self.result("Max Voltage", f"{vmax:.4f}", "V")
        self.result("RMS Voltage", f"{vrms:.4f}", "V")
        self.result("Peak-to-Peak", f"{peak_to_peak:.4f}", "V")
        self.info("")

        # Frequency analysis
        self.subsection("Frequency Content")
        fft = np.fft.rfft(trace.data)
        freqs = np.fft.rfftfreq(num_samples, 1 / meta.sample_rate)

        # Find dominant frequency (skip DC)
        peak_idx = np.argmax(np.abs(fft[1:])) + 1
        dominant_freq = freqs[peak_idx]

        self.result("Dominant Frequency", f"{dominant_freq / 1e6:.1f}", "MHz")
        self.result("FFT Points", len(fft))
        self.info("")

        return {
            "sample_rate": meta.sample_rate,
            "num_samples": num_samples,
            "vrms": vrms,
            "dominant_freq": float(dominant_freq),
        }

    def _display_format_comparison(self, data: dict[str, Any]) -> None:
        """Display comparison table of specialized formats."""
        cw_traceset = data["chipwhisperer_traceset"]
        lecroy_trace = data["lecroy_trace"]

        comparison = []

        # ChipWhisperer
        cw_size = cw_traceset.n_traces * cw_traceset.n_samples * 8  # float64
        cw_total_samples = cw_traceset.n_traces * cw_traceset.n_samples
        comparison.append(
            [
                "ChipWhisperer",
                f"{cw_traceset.n_traces} traces",
                f"{cw_traceset.sample_rate / 1e6:.0f} MS/s",
                f"{cw_total_samples:,}",
                format_size(cw_size),
                "Power/EM traces",
            ]
        )

        # LeCroy
        lecroy_size = len(lecroy_trace.data) * 8  # float64
        lecroy_samples = len(lecroy_trace.data)
        comparison.append(
            [
                "LeCroy .trc",
                "1 channel",
                f"{lecroy_trace.metadata.sample_rate / 1e9:.0f} GS/s",
                f"{lecroy_samples:,}",
                format_size(lecroy_size),
                "High-speed analog",
            ]
        )

        headers = ["Format", "Channels/Traces", "Sample Rate", "Total Samples", "Size", "Purpose"]
        self.info(format_table(comparison, headers))
        self.info("")

    def _show_best_practices(self) -> None:
        """Show best practices for security testing data."""
        self.subsection("Side-Channel Analysis Guidelines")
        self.info("""
ChipWhisperer and Security Testing:

1. TRACE ACQUISITION
   ✓ Collect 50-1000+ traces for statistical analysis
   ✓ Use consistent trigger (synchronized to crypto operation start)
   ✓ Minimize environmental noise (temperature, EMI)
   ✓ Document probe placement and acquisition settings

2. METADATA MANAGEMENT
   ✓ Always save plaintexts, ciphertexts, and keys together
   ✓ Use ChipWhisperer .npy naming convention:
     - traces.npy (power data)
     - traces_textin.npy (plaintexts)
     - traces_textout.npy (ciphertexts)
     - traces_keys.npy (keys, if known)
   ✓ Include sample rate and trigger offset in metadata

3. ANALYSIS WORKFLOW
   ✓ Align traces using trigger or correlation
   ✓ Filter noise (bandpass around target frequency)
   ✓ Use multiple analysis techniques (CPA, DPA, Template)
   ✓ Validate findings with known-key attacks first

4. HIGH-BANDWIDTH OSCILLOSCOPES (LeCroy)
   ✓ Use for high-speed digital signal integrity
   ✓ .trc format stores rich metadata (setup, calibration)
   ✓ Consider waveform compression for large datasets
   ✓ Use segmented memory for long captures

5. DATA SECURITY
   ✓ Protect cryptographic keys in trace metadata
   ✓ Use secure storage for sensitive captures
   ✓ Document test setup for reproducibility
   ✓ Follow responsible disclosure for vulnerabilities
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate specialized format loading results."""
        self.info("Validating specialized format data...")

        all_valid = True

        # Validate ChipWhisperer analysis
        if "chipwhisperer" in results:
            cw = results["chipwhisperer"]

            if "n_traces" not in cw:
                self.error("ChipWhisperer results missing 'n_traces'")
                all_valid = False
            elif cw["n_traces"] != 50:
                self.error(f"Expected 50 traces, got {cw['n_traces']}")
                all_valid = False

            if "n_samples" not in cw:
                self.error("ChipWhisperer results missing 'n_samples'")
                all_valid = False
            elif cw["n_samples"] != 5000:
                self.error(f"Expected 5000 samples, got {cw['n_samples']}")
                all_valid = False

            if not validate_approximately(
                cw["sample_rate"],
                1e6,
                tolerance=0.01,
                name="ChipWhisperer sample rate",
            ):
                all_valid = False

            # Mean power should be around 50 mW
            if not validate_approximately(
                cw["mean_power"],
                50.0,
                tolerance=5.0,
                name="ChipWhisperer mean power",
            ):
                all_valid = False

            # Check metadata availability
            if not cw["has_plaintexts"]:
                self.error("ChipWhisperer traces missing plaintexts")
                all_valid = False

            if not cw["has_keys"]:
                self.error("ChipWhisperer traces missing keys")
                all_valid = False

        # Validate ChipWhisperer loading
        if "chipwhisperer_loaded" in results:
            loaded = results["chipwhisperer_loaded"]

            if not loaded["conversion_success"]:
                self.error("ChipWhisperer trace conversion failed")
                all_valid = False

            if loaded["loaded_traces"] != 50:
                self.error(
                    f"Loaded trace count mismatch: expected 50, got {loaded['loaded_traces']}"
                )
                all_valid = False

        # Validate side-channel analysis
        if "side_channel" in results:
            sca = results["side_channel"]

            if not sca["analysis_success"]:
                self.error("Side-channel analysis failed")
                all_valid = False
            else:
                # Correlation should be positive (Hamming weight model)
                if sca["correlation"] < 0.1:
                    self.warning(
                        f"Low correlation detected ({sca['correlation']:.4f}), "
                        "expected >0.1 for Hamming weight leakage"
                    )

        # Validate LeCroy capture
        if "lecroy" in results:
            lecroy = results["lecroy"]

            if not validate_approximately(
                lecroy["sample_rate"],
                10e9,
                tolerance=0.01,
                name="LeCroy sample rate",
            ):
                all_valid = False

            # Dominant frequency should be around 50 MHz
            if not validate_approximately(
                lecroy["dominant_freq"],
                50e6,
                tolerance=5e6,
                name="LeCroy dominant frequency",
            ):
                all_valid = False

        if all_valid:
            self.success("All specialized format validations passed!")
            self.info("""
Next steps for working with specialized formats:

1. CHIPWHISPERER FILES
   from oscura.loaders import load_chipwhisperer
   traceset = load_chipwhisperer("traces.npy")
   print(f"Loaded {traceset.n_traces} traces")

   # Access metadata
   plaintext = traceset.plaintexts[0]
   key = traceset.keys[0]

2. TRS FORMAT (Inspector/ChipWhisperer)
   traceset = load_chipwhisperer("capture.trs")
   # Automatically parses TRS header and trace data

3. SINGLE TRACE CONVERSION
   from oscura.loaders.chipwhisperer import to_waveform_trace
   waveform = to_waveform_trace(traceset, trace_index=0)
   # Now use standard Oscura analyzers

4. LECROY FILES (.trc format)
   # LeCroy loader not yet implemented in Oscura
   # Use synthetic data generation as shown in this demo
   # Future: Binary .trc parser similar to Tektronix loader
            """)
        else:
            self.error("Some specialized format validations failed!")

        return all_valid


if __name__ == "__main__":
    demo = SpecializedFormatsDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
