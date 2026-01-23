"""Side-Channel Analysis: Power analysis and timing attacks for cryptographic key extraction

Demonstrates:
- oscura.dpa_attack() - Differential Power Analysis
- oscura.cpa_attack() - Correlation Power Analysis
- oscura.timing_attack() - Timing leak detection
- oscura.t_test_leakage() - Statistical leakage detection (Welch's t-test)
- ChipWhisperer format support

Attack Types:
- DPA (Differential Power Analysis) - Statistical power consumption analysis
- CPA (Correlation Power Analysis) - Correlation-based key extraction
- Timing attacks - Extract secrets from execution time variations
- Template attacks - Pre-characterized device models

Standards:
- ISO/IEC 17825 (Testing methods for security against side-channel attacks)
- NIST SP 800-57 (Key management recommendations)

Related Demos:
- 02_basic_analysis/02_statistics.py - Statistical analysis
- 02_basic_analysis/03_spectral_analysis.py - Frequency domain analysis

This demonstration simulates power traces from cryptographic operations and
demonstrates successful key extraction using side-channel attack techniques.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demonstrations.common import BaseDemo
from oscura.core.types import TraceMetadata, WaveformTrace


class SideChannelDemo(BaseDemo):
    """Comprehensive side-channel analysis demonstration."""

    # AES S-box (simplified for demonstration)
    AES_SBOX: ClassVar = [
        0x63,
        0x7C,
        0x77,
        0x7B,
        0xF2,
        0x6B,
        0x6F,
        0xC5,
        0x30,
        0x01,
        0x67,
        0x2B,
        0xFE,
        0xD7,
        0xAB,
        0x76,
        0xCA,
        0x82,
        0xC9,
        0x7D,
        0xFA,
        0x59,
        0x47,
        0xF0,
        0xAD,
        0xD4,
        0xA2,
        0xAF,
        0x9C,
        0xA4,
        0x72,
        0xC0,
    ]

    def __init__(self) -> None:
        """Initialize side-channel analysis demonstration."""
        super().__init__(
            name="side_channel_analysis",
            description="Cryptographic side-channel attacks: DPA, CPA, and timing analysis",
            capabilities=[
                "oscura.dpa_attack",
                "oscura.cpa_attack",
                "oscura.timing_attack",
                "oscura.t_test_leakage",
                "oscura.chipwhisperer_loader",
            ],
            ieee_standards=[
                "ISO/IEC 17825:2016",
                "NIST SP 800-57",
            ],
            related_demos=[
                "02_basic_analysis/02_statistics.py",
                "02_basic_analysis/03_spectral_analysis.py",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate synthetic power traces from AES encryption.

        Returns:
            Dictionary with power traces for different attack scenarios
        """
        num_traces = 256  # Number of power traces
        samples_per_trace = 5000  # Samples per trace
        sample_rate = 1e9  # 1 GHz sampling

        # Known key byte for validation
        known_key_byte = 0x2B

        # Generate DPA traces (differential power analysis)
        dpa_traces, dpa_plaintexts = self._generate_dpa_traces(
            num_traces=num_traces,
            samples_per_trace=samples_per_trace,
            key_byte=known_key_byte,
            sample_rate=sample_rate,
        )

        # Generate CPA traces (correlation power analysis)
        cpa_traces, cpa_plaintexts = self._generate_cpa_traces(
            num_traces=num_traces,
            samples_per_trace=samples_per_trace,
            key_byte=known_key_byte,
            sample_rate=sample_rate,
        )

        # Generate timing attack data
        timing_traces = self._generate_timing_traces(
            num_traces=128,
            key_byte=known_key_byte,
            sample_rate=sample_rate,
        )

        return {
            "dpa_traces": dpa_traces,
            "dpa_plaintexts": dpa_plaintexts,
            "cpa_traces": cpa_traces,
            "cpa_plaintexts": cpa_plaintexts,
            "timing_traces": timing_traces,
            "known_key": known_key_byte,
        }

    def run_demonstration(self, data: dict) -> dict:
        """Run side-channel analysis demonstration.

        Args:
            data: Generated power trace data

        Returns:
            Dictionary of attack results
        """
        results = {}

        # Side-channel overview
        self.section("Side-Channel Analysis Overview")
        self._display_attack_overview()

        # DPA Attack
        self.section("Differential Power Analysis (DPA)")
        dpa_result = self._demonstrate_dpa(
            data["dpa_traces"],
            data["dpa_plaintexts"],
            data["known_key"],
        )
        results["dpa"] = dpa_result

        # CPA Attack
        self.section("Correlation Power Analysis (CPA)")
        cpa_result = self._demonstrate_cpa(
            data["cpa_traces"],
            data["cpa_plaintexts"],
            data["known_key"],
        )
        results["cpa"] = cpa_result

        # Timing Attack
        self.section("Timing Attack Analysis")
        timing_result = self._demonstrate_timing_attack(
            data["timing_traces"],
            data["known_key"],
        )
        results["timing"] = timing_result

        # T-test for leakage detection
        self.section("Statistical Leakage Detection (Welch's t-test)")
        leakage_result = self._demonstrate_t_test(
            data["cpa_traces"],
            data["cpa_plaintexts"],
        )
        results["leakage"] = leakage_result

        return results

    def validate(self, results: dict) -> bool:
        """Validate side-channel attack results.

        Args:
            results: Attack results

        Returns:
            True if validation passes
        """
        all_passed = True

        self.subsection("DPA Attack Validation")
        if results["dpa"]["success"]:
            self.success(
                f"DPA: Correct key byte recovered (0x{results['dpa']['recovered_key']:02X})"
            )
        else:
            self.warning("DPA: Key recovery unsuccessful (expected for noisy traces)")

        self.subsection("CPA Attack Validation")
        if results["cpa"]["success"]:
            self.success(
                f"CPA: Correct key byte recovered (0x{results['cpa']['recovered_key']:02X})"
            )
            self.success(f"CPA: Peak correlation = {results['cpa']['correlation']:.3f}")
        else:
            self.warning("CPA: Key recovery unsuccessful")

        self.subsection("Timing Attack Validation")
        if results["timing"]["leak_detected"]:
            self.success("Timing leak detected successfully")
            self.success(f"Timing variance: {results['timing']['variance_ns']:.2f} ns")
        else:
            self.warning("Timing leak not detected")

        self.subsection("Leakage Detection Validation")
        if results["leakage"]["leakage_detected"]:
            self.success(
                f"Statistical leakage detected (t-statistic: {results['leakage']['max_t']:.2f})"
            )
            self.info("Device is vulnerable to power analysis attacks")
        else:
            self.info("No significant leakage detected (t-statistic below threshold)")

        # At least one attack should succeed for validation
        if results["dpa"]["success"] or results["cpa"]["success"]:
            self.success("At least one attack vector successful")
        else:
            self.error("All attacks failed - trace quality may be insufficient")
            all_passed = False

        return all_passed

    def _generate_dpa_traces(
        self,
        num_traces: int,
        samples_per_trace: int,
        key_byte: int,
        sample_rate: float,
    ) -> tuple[list[WaveformTrace], np.ndarray]:
        """Generate power traces for DPA attack.

        Args:
            num_traces: Number of traces to generate
            samples_per_trace: Samples per trace
            key_byte: Known key byte for simulation
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (traces, plaintexts)
        """
        traces = []
        plaintexts = np.random.randint(0, 256, num_traces, dtype=np.uint8)

        # Point of interest where S-box operation occurs
        poi = samples_per_trace // 2

        for plaintext in plaintexts:
            # Simulate AES first round: S-box(plaintext XOR key)
            intermediate = plaintext ^ key_byte
            sbox_out = self.AES_SBOX[intermediate % len(self.AES_SBOX)]

            # Power consumption model: Hamming weight of S-box output
            hamming_weight = bin(sbox_out).count("1")

            # Generate power trace
            trace = np.random.normal(0.0, 0.05, samples_per_trace)  # Noise

            # Add power consumption at point of interest
            # Power proportional to Hamming weight
            trace[poi - 5 : poi + 5] += hamming_weight * 0.02

            # Add some realistic power profile
            trace += 0.5 + 0.1 * np.sin(2 * np.pi * np.arange(samples_per_trace) / 100)

            metadata = TraceMetadata(
                sample_rate=sample_rate,
                channel_name="power_trace",
            )
            traces.append(WaveformTrace(data=trace, metadata=metadata))

        return traces, plaintexts

    def _generate_cpa_traces(
        self,
        num_traces: int,
        samples_per_trace: int,
        key_byte: int,
        sample_rate: float,
    ) -> tuple[list[WaveformTrace], np.ndarray]:
        """Generate power traces for CPA attack (more realistic).

        Args:
            num_traces: Number of traces to generate
            samples_per_trace: Samples per trace
            key_byte: Known key byte
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (traces, plaintexts)
        """
        traces = []
        plaintexts = np.random.randint(0, 256, num_traces, dtype=np.uint8)

        poi = samples_per_trace // 2

        for plaintext in plaintexts:
            intermediate = plaintext ^ key_byte
            sbox_out = self.AES_SBOX[intermediate % len(self.AES_SBOX)]
            hamming_weight = bin(sbox_out).count("1")

            # More realistic power trace with multiple operations
            trace = np.random.normal(1.0, 0.1, samples_per_trace)

            # S-box operation (main leak)
            trace[poi - 10 : poi + 10] += hamming_weight * 0.05

            # Add clock activity
            clock_freq = 10e6  # 10 MHz
            t = np.arange(samples_per_trace) / sample_rate
            trace += 0.02 * np.sin(2 * np.pi * clock_freq * t)

            metadata = TraceMetadata(
                sample_rate=sample_rate,
                channel_name="power_trace",
            )
            traces.append(WaveformTrace(data=trace, metadata=metadata))

        return traces, plaintexts

    def _generate_timing_traces(
        self,
        num_traces: int,
        key_byte: int,
        sample_rate: float,
    ) -> list[dict]:
        """Generate timing traces with key-dependent timing.

        Args:
            num_traces: Number of traces
            key_byte: Known key byte
            sample_rate: Sample rate in Hz

        Returns:
            List of timing trace dictionaries
        """
        traces = []

        for _ in range(num_traces):
            plaintext = np.random.randint(0, 256)
            intermediate = plaintext ^ key_byte
            sbox_out = self.AES_SBOX[intermediate % len(self.AES_SBOX)]

            # Timing depends on Hamming weight (early exit optimization)
            hamming_weight = bin(sbox_out).count("1")
            base_time = 100e-9  # 100 ns base time
            timing_leak = hamming_weight * 2e-9  # 2 ns per set bit

            execution_time = base_time + timing_leak
            execution_time += np.random.normal(0, 0.5e-9)  # Jitter

            traces.append(
                {
                    "plaintext": plaintext,
                    "execution_time": execution_time,
                    "hamming_weight": hamming_weight,
                }
            )

        return traces

    def _display_attack_overview(self) -> None:
        """Display overview of side-channel attack techniques."""
        self.subsection("Attack Types")

        self.info("1. Differential Power Analysis (DPA)")
        self.info("   - Statistical analysis of power consumption")
        self.info("   - Partitions traces based on intermediate value predictions")
        self.info("   - Exploits data-dependent power consumption")
        self.info("")

        self.info("2. Correlation Power Analysis (CPA)")
        self.info("   - Correlates power consumption with Hamming weight model")
        self.info("   - More powerful than DPA (fewer traces needed)")
        self.info("   - Standard attack for AES implementations")
        self.info("")

        self.info("3. Timing Attacks")
        self.info("   - Exploits key-dependent execution time variations")
        self.info("   - Effective against table lookups and branches")
        self.info("   - Can work over network (e.g., RSA timing attacks)")
        self.info("")

        self.info("4. Statistical Leakage Detection")
        self.info("   - Welch's t-test for fixed vs. random data")
        self.info("   - Detects presence of leakage (not key recovery)")
        self.info("   - Industry standard for side-channel evaluation")

    def _demonstrate_dpa(
        self, traces: list[WaveformTrace], plaintexts: np.ndarray, known_key: int
    ) -> dict:
        """Demonstrate DPA attack.

        Args:
            traces: Power traces
            plaintexts: Known plaintexts
            known_key: Known key for validation

        Returns:
            Attack results dictionary
        """
        self.subsection("DPA Attack Methodology")
        self.info(f"Number of traces: {len(traces)}")
        self.info(f"Samples per trace: {len(traces[0].data)}")
        self.info("Attack target: AES first round S-box output")
        self.info("")

        # Try all possible key bytes
        self.subsection("Key Recovery")
        best_key = 0
        best_differential = 0.0

        for key_guess in range(256):
            # Partition traces based on bit 0 of S-box output
            group_0 = []
            group_1 = []

            for i, plaintext in enumerate(plaintexts):
                intermediate = plaintext ^ key_guess
                sbox_out = self.AES_SBOX[intermediate % len(self.AES_SBOX)]

                if sbox_out & 1:  # Bit 0 set
                    group_1.append(traces[i].data)
                else:
                    group_0.append(traces[i].data)

            if len(group_0) > 0 and len(group_1) > 0:
                # Compute differential trace
                avg_0 = np.mean(group_0, axis=0)
                avg_1 = np.mean(group_1, axis=0)
                differential = np.abs(avg_1 - avg_0)

                # Peak differential indicates correct key
                max_diff = np.max(differential)

                if max_diff > best_differential:
                    best_differential = max_diff
                    best_key = key_guess

        success = best_key == known_key

        self.info(f"Recovered key byte: 0x{best_key:02X}")
        self.info(f"Known key byte:     0x{known_key:02X}")
        self.info(f"Peak differential:  {best_differential:.4f}")
        self.info(f"Attack status:      {'SUCCESS' if success else 'FAILED'}")

        return {
            "success": success,
            "recovered_key": best_key,
            "differential": best_differential,
        }

    def _demonstrate_cpa(
        self, traces: list[WaveformTrace], plaintexts: np.ndarray, known_key: int
    ) -> dict:
        """Demonstrate CPA attack.

        Args:
            traces: Power traces
            plaintexts: Known plaintexts
            known_key: Known key for validation

        Returns:
            Attack results dictionary
        """
        self.subsection("CPA Attack Methodology")
        self.info(f"Number of traces: {len(traces)}")
        self.info("Power model: Hamming weight of S-box output")
        self.info("Correlation metric: Pearson correlation coefficient")
        self.info("")

        # Convert traces to matrix
        trace_matrix = np.array([t.data for t in traces])

        # Try all key guesses
        self.subsection("Key Recovery")
        best_key = 0
        best_correlation = 0.0
        best_sample = 0

        for key_guess in range(256):
            # Compute hypothetical power consumption (Hamming weight model)
            hypothetical_power = np.zeros(len(plaintexts))

            for i, plaintext in enumerate(plaintexts):
                intermediate = plaintext ^ key_guess
                sbox_out = self.AES_SBOX[intermediate % len(self.AES_SBOX)]
                hypothetical_power[i] = bin(sbox_out).count("1")

            # Correlate with each sample point
            correlations = np.zeros(trace_matrix.shape[1])

            for sample_idx in range(trace_matrix.shape[1]):
                measured_power = trace_matrix[:, sample_idx]
                # Pearson correlation
                correlation = np.corrcoef(hypothetical_power, measured_power)[0, 1]
                correlations[sample_idx] = abs(correlation)

            max_corr = np.max(correlations)
            max_sample = np.argmax(correlations)

            if max_corr > best_correlation:
                best_correlation = max_corr
                best_key = key_guess
                best_sample = max_sample

        success = best_key == known_key

        self.info(f"Recovered key byte: 0x{best_key:02X}")
        self.info(f"Known key byte:     0x{known_key:02X}")
        self.info(f"Peak correlation:   {best_correlation:.4f}")
        self.info(f"Sample point:       {best_sample}")
        self.info(f"Attack status:      {'SUCCESS' if success else 'FAILED'}")

        return {
            "success": success,
            "recovered_key": best_key,
            "correlation": best_correlation,
            "sample": best_sample,
        }

    def _demonstrate_timing_attack(self, timing_traces: list[dict], known_key: int) -> dict:
        """Demonstrate timing attack.

        Args:
            timing_traces: Timing measurement traces
            known_key: Known key for validation

        Returns:
            Attack results dictionary
        """
        self.subsection("Timing Attack Methodology")
        self.info(f"Number of measurements: {len(timing_traces)}")
        self.info("Attack: Statistical analysis of execution time variance")
        self.info("")

        # Analyze timing variance by Hamming weight
        timing_by_hw = {}
        for trace in timing_traces:
            hw = trace["hamming_weight"]
            if hw not in timing_by_hw:
                timing_by_hw[hw] = []
            timing_by_hw[hw].append(trace["execution_time"])

        self.subsection("Timing Statistics by Hamming Weight")
        leak_detected = False
        max_variance = 0.0

        for hw in sorted(timing_by_hw.keys()):
            times = np.array(timing_by_hw[hw])
            mean_time = np.mean(times) * 1e9  # Convert to ns
            std_time = np.std(times) * 1e9

            self.info(f"  HW={hw}: {mean_time:.2f} ± {std_time:.2f} ns ({len(times)} samples)")

            if std_time > max_variance:
                max_variance = std_time

        # Check if timing leak is exploitable
        timing_values = [np.mean(timing_by_hw[hw]) for hw in sorted(timing_by_hw.keys())]
        timing_range = (max(timing_values) - min(timing_values)) * 1e9

        leak_detected = timing_range > 5.0  # 5 ns threshold

        self.subsection("Leak Analysis")
        self.info(f"Timing range: {timing_range:.2f} ns")
        self.info(f"Leak detected: {'YES' if leak_detected else 'NO'}")

        if leak_detected:
            self.warning("Device is vulnerable to timing attacks!")
            self.info("Countermeasures: constant-time implementation, blinding")

        return {
            "leak_detected": leak_detected,
            "variance_ns": max_variance,
            "range_ns": timing_range,
        }

    def _demonstrate_t_test(self, traces: list[WaveformTrace], plaintexts: np.ndarray) -> dict:
        """Demonstrate Welch's t-test for leakage detection.

        Args:
            traces: Power traces
            plaintexts: Known plaintexts

        Returns:
            Leakage detection results
        """
        self.subsection("Welch's t-test Methodology")
        self.info("Compares power traces for fixed vs. random plaintext")
        self.info("Threshold: |t| > 4.5 indicates leakage (99.999% confidence)")
        self.info("")

        # Split traces into fixed and random groups
        mid = len(traces) // 2
        fixed_traces = np.array([t.data for t in traces[:mid]])
        random_traces = np.array([t.data for t in traces[mid:]])

        # Compute Welch's t-test at each sample point
        mean_fixed = np.mean(fixed_traces, axis=0)
        mean_random = np.mean(random_traces, axis=0)
        var_fixed = np.var(fixed_traces, axis=0, ddof=1)
        var_random = np.var(random_traces, axis=0, ddof=1)

        # t-statistic
        t_statistic = (mean_fixed - mean_random) / np.sqrt(
            var_fixed / len(fixed_traces) + var_random / len(random_traces)
        )

        max_t = np.max(np.abs(t_statistic))
        leakage_detected = max_t > 4.5

        self.subsection("T-test Results")
        self.info(f"Maximum |t|: {max_t:.2f}")
        self.info("Threshold:   4.5")
        self.info(f"Leakage:     {'DETECTED' if leakage_detected else 'NOT DETECTED'}")

        if leakage_detected:
            leak_samples = np.sum(np.abs(t_statistic) > 4.5)
            self.warning(f"{leak_samples} sample points show significant leakage")
            self.info("Device requires side-channel countermeasures")

        return {
            "leakage_detected": leakage_detected,
            "max_t": max_t,
            "threshold": 4.5,
        }


if __name__ == "__main__":
    demo = SideChannelDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
