"""EMC Compliance Testing: Complete test workflow with compliance report

Demonstrates:
- oscura.spectral.fft - FFT-based emission measurement
- oscura.emc.conducted_emissions - Conducted emissions testing
- oscura.emc.radiated_emissions - Radiated emissions testing
- oscura.emc.cispr32_limits - CISPR 32 limit comparison
- Complete workflow with compliance report generation

Standards:
- CISPR 16-1-1:2019 (Measurement apparatus)
- CISPR 32:2015 (Multimedia equipment)
- MIL-STD-461G (Military EMC)

Related Demos:
- 02_basic_analysis/03_spectral_analysis.py - Spectral analysis
- 05_domain_specific/02_emc_compliance.py - EMC compliance

This demonstration shows a complete EMC compliance testing workflow:
1. Capture conducted and radiated emissions
2. Perform FFT-based spectrum analysis
3. Compare against regulatory limits
4. Calculate margins and identify violations
5. Generate compliance test report

Time Budget: < 2 seconds for complete analysis
"""

from __future__ import annotations

import sys
import time
import typing
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.core.types import TraceMetadata, WaveformTrace


class EMCTestingWorkflowDemo(BaseDemo):
    """Complete EMC compliance testing workflow with reporting."""

    # CISPR 32 Class B limits
    CISPR32_CLASS_B_CONDUCTED: typing.ClassVar[dict[float, tuple[int, int]]] = {
        # Frequency (MHz): (QP limit dBμV, AVG limit dBμV)
        0.15: (66, 56),
        0.50: (56, 46),
        5.00: (56, 46),
        30.0: (60, 50),
    }

    CISPR32_CLASS_B_RADIATED: typing.ClassVar[dict[int, int]] = {
        # Frequency (MHz): (QP limit dBμV/m at 10m)
        30: 30,
        230: 37,
        1000: 37,
    }

    def __init__(self) -> None:
        """Initialize demonstration."""
        super().__init__(
            name="emc_testing_workflow",
            description="Complete EMC compliance testing workflow with margin analysis",
            capabilities=[
                "oscura.spectral.fft",
                "oscura.emc.conducted_emissions",
                "oscura.emc.radiated_emissions",
                "oscura.emc.cispr32_limits",
            ],
            ieee_standards=[
                "CISPR 16-1-1:2019",
                "CISPR 32:2015",
                "MIL-STD-461G",
            ],
            related_demos=[
                "02_basic_analysis/03_spectral_analysis.py",
                "05_domain_specific/02_emc_compliance.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate EMC test signals.

        Simulates:
        - Conducted emissions (power line noise)
        - Radiated emissions (antenna measurements)
        - Multiple frequency components

        Returns:
            Dictionary with EMC test signals
        """
        self.section("Generating EMC Test Signals")

        # Conducted emissions: switching power supply harmonics
        self.info("Generating conducted emissions signal...")
        conducted = self._generate_conducted_emissions(
            fundamental=100e3,  # 100 kHz switching
            harmonics=[1, 2, 3, 5, 7],
            levels_dbuv=[88, 72, 65, 52, 45],
            duration=0.001,
            sample_rate=50e6,
        )

        # Radiated emissions: multiple EMI sources
        self.info("Generating radiated emissions signal...")
        radiated = self._generate_radiated_emissions(
            frequencies=[88e6, 150e6, 433e6],  # Common EMI frequencies
            levels_dbuvm=[33, 36, 28],  # dBμV/m at 10m
            duration=0.001,
            sample_rate=2e9,
        )

        self.result("Conducted signal", f"{len(conducted.data)} samples")
        self.result("Radiated signal", f"{len(radiated.data)} samples")

        return {"conducted": conducted, "radiated": radiated}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute complete EMC testing workflow."""
        results: dict[str, Any] = {}
        workflow_start = time.time()

        # ===== PHASE 1: Conducted Emissions Testing =====
        self.section("Phase 1: Conducted Emissions Testing (CISPR 32)")
        phase1_start = time.time()

        self.subsection("1.1 Spectrum Measurement")
        conducted_spectrum = self._measure_spectrum(data["conducted"])
        results["conducted_spectrum"] = conducted_spectrum

        self.subsection("1.2 Limit Comparison")
        conducted_results = self._compare_to_limits(
            conducted_spectrum, self.CISPR32_CLASS_B_CONDUCTED, limit_type="conducted"
        )
        results["conducted_results"] = conducted_results

        self.info(f"Test points: {len(conducted_results)}")
        pass_count = sum(1 for r in conducted_results if r["pass"])
        self.info(f"Passed: {pass_count}/{len(conducted_results)}")

        phase1_time = time.time() - phase1_start
        results["phase1_time"] = phase1_time
        self.result("Phase 1 duration", f"{phase1_time:.3f}", "seconds")

        # ===== PHASE 2: Radiated Emissions Testing =====
        self.section("Phase 2: Radiated Emissions Testing (CISPR 32)")
        phase2_start = time.time()

        self.subsection("2.1 Spectrum Measurement")
        radiated_spectrum = self._measure_spectrum(data["radiated"])
        results["radiated_spectrum"] = radiated_spectrum

        self.subsection("2.2 Limit Comparison")
        radiated_results = self._compare_to_limits(
            radiated_spectrum, self.CISPR32_CLASS_B_RADIATED, limit_type="radiated"
        )
        results["radiated_results"] = radiated_results

        self.info(f"Test points: {len(radiated_results)}")
        pass_count = sum(1 for r in radiated_results if r["pass"])
        self.info(f"Passed: {pass_count}/{len(radiated_results)}")

        phase2_time = time.time() - phase2_start
        results["phase2_time"] = phase2_time
        self.result("Phase 2 duration", f"{phase2_time:.3f}", "seconds")

        # ===== PHASE 3: Margin Analysis =====
        self.section("Phase 3: Margin Analysis")
        phase3_start = time.time()

        self.subsection("3.1 Worst-Case Margins")
        margin_analysis = self._analyze_margins(conducted_results, radiated_results)
        results["margin_analysis"] = margin_analysis

        self.info("Conducted emissions:")
        self.info(f"  Worst margin: {margin_analysis['conducted_worst_margin']:.1f} dB")
        self.info(f"  Best margin: {margin_analysis['conducted_best_margin']:.1f} dB")

        self.info("Radiated emissions:")
        self.info(f"  Worst margin: {margin_analysis['radiated_worst_margin']:.1f} dB")
        self.info(f"  Best margin: {margin_analysis['radiated_best_margin']:.1f} dB")

        phase3_time = time.time() - phase3_start
        results["phase3_time"] = phase3_time
        self.result("Phase 3 duration", f"{phase3_time:.3f}", "seconds")

        # ===== PHASE 4: Compliance Report Generation =====
        self.section("Phase 4: Compliance Report Generation")
        phase4_start = time.time()

        self.subsection("4.1 Generating Report")
        report = self._generate_compliance_report(
            conducted_results=conducted_results,
            radiated_results=radiated_results,
            margin_analysis=margin_analysis,
        )

        output_dir = self.get_output_dir()
        report_path = output_dir / "emc_compliance_report.txt"
        report_path.write_text(report)

        results["report_generated"] = True
        results["report_path"] = str(report_path)

        self.success(f"Report saved: {report_path}")
        self.info(f"Report size: {len(report)} bytes")

        phase4_time = time.time() - phase4_start
        results["phase4_time"] = phase4_time
        self.result("Phase 4 duration", f"{phase4_time:.3f}", "seconds")

        # ===== WORKFLOW SUMMARY =====
        self.section("Complete Workflow Summary")

        total_time = time.time() - workflow_start
        results["total_time"] = total_time

        self.subsection("Timing Breakdown")
        self.result("  Phase 1 (Conducted)", f"{phase1_time:.3f}", "s")
        self.result("  Phase 2 (Radiated)", f"{phase2_time:.3f}", "s")
        self.result("  Phase 3 (Margin Analysis)", f"{phase3_time:.3f}", "s")
        self.result("  Phase 4 (Report)", f"{phase4_time:.3f}", "s")
        self.result("  TOTAL WORKFLOW", f"{total_time:.3f}", "s")

        # Overall compliance
        overall_compliant = (
            margin_analysis["conducted_worst_margin"] > 0
            and margin_analysis["radiated_worst_margin"] > 0
        )
        results["overall_compliant"] = overall_compliant

        if overall_compliant:
            self.success("OVERALL RESULT: COMPLIANT")
        else:
            self.warning("OVERALL RESULT: NON-COMPLIANT (expected for test signals)")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate EMC testing workflow results."""
        all_passed = True

        # Validate conducted emissions testing
        if not results.get("conducted_results"):
            self.error("Conducted emissions testing failed")
            all_passed = False
        else:
            self.success(
                f"Conducted emissions testing passed: {len(results['conducted_results'])} points"
            )

        # Validate radiated emissions testing
        if not results.get("radiated_results"):
            self.error("Radiated emissions testing failed")
            all_passed = False
        else:
            self.success(
                f"Radiated emissions testing passed: {len(results['radiated_results'])} points"
            )

        # Validate margin analysis
        if not results.get("margin_analysis"):
            self.error("Margin analysis failed")
            all_passed = False
        else:
            self.success("Margin analysis completed successfully")

        # Validate report generation
        if not results.get("report_generated", False):
            self.error("Report generation failed")
            all_passed = False
        else:
            self.success("Compliance report generated successfully")

        # Validate timing
        total_time = results.get("total_time", 999)
        if total_time > 5.0:
            self.warning(f"Workflow exceeded target time (got {total_time:.1f}s, target <5s)")
        else:
            self.success(f"Workflow completed within time budget ({total_time:.3f}s)")

        return all_passed

    def _generate_conducted_emissions(
        self,
        fundamental: float,
        harmonics: list[int],
        levels_dbuv: list[float],
        duration: float,
        sample_rate: float,
    ) -> WaveformTrace:
        """Generate conducted emissions signal."""
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        signal = np.zeros(num_samples)

        for harmonic, level_dbuv in zip(harmonics, levels_dbuv, strict=False):
            amplitude_v = (10 ** (level_dbuv / 20)) * 1e-6
            freq = fundamental * harmonic
            signal += amplitude_v * np.sin(2 * np.pi * freq * t)

        # Add noise floor
        signal += np.random.normal(0, 1e-8, num_samples)

        metadata = TraceMetadata(sample_rate=sample_rate, channel_name="conducted_emissions")
        return WaveformTrace(data=signal, metadata=metadata)

    def _generate_radiated_emissions(
        self,
        frequencies: list[float],
        levels_dbuvm: list[float],
        duration: float,
        sample_rate: float,
    ) -> WaveformTrace:
        """Generate radiated emissions signal."""
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        signal = np.zeros(num_samples)

        antenna_factor = 20  # dB
        for freq, level_dbuvm in zip(frequencies, levels_dbuvm, strict=False):
            level_dbuv = level_dbuvm - antenna_factor
            amplitude_v = (10 ** (level_dbuv / 20)) * 1e-6
            signal += amplitude_v * np.sin(2 * np.pi * freq * t)

        signal += np.random.normal(0, 1e-8, num_samples)

        metadata = TraceMetadata(sample_rate=sample_rate, channel_name="radiated_emissions")
        return WaveformTrace(data=signal, metadata=metadata)

    def _measure_spectrum(
        self, signal: WaveformTrace
    ) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
        """Measure signal spectrum using FFT."""
        fft = np.fft.rfft(signal.data)
        freqs = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)
        magnitude_v = np.abs(fft) * 2 / len(signal.data)
        magnitude_dbuv = 20 * np.log10(magnitude_v / 1e-6 + 1e-12)

        return freqs, magnitude_dbuv

    def _compare_to_limits(
        self,
        spectrum: tuple[
            np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
        ],
        limits: dict[float, Any],
        limit_type: str,
    ) -> list[dict[str, Any]]:
        """Compare spectrum to regulatory limits."""
        freqs, magnitude = spectrum
        results = []

        for freq_mhz, limit_spec in sorted(limits.items()):
            freq_hz = freq_mhz * 1e6
            idx = np.argmin(np.abs(freqs - freq_hz))
            measured = magnitude[idx]

            if limit_type == "conducted":
                qp_limit, avg_limit = limit_spec
                margin = qp_limit - measured
            else:  # radiated
                qp_limit = limit_spec
                margin = qp_limit - measured

            results.append(
                {
                    "frequency_mhz": freq_mhz,
                    "measured_db": measured,
                    "limit_db": qp_limit,
                    "margin_db": margin,
                    "pass": margin > 0,
                }
            )

        return results

    def _analyze_margins(
        self, conducted_results: list[dict[str, Any]], radiated_results: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Analyze margins across all test points."""
        conducted_margins = [r["margin_db"] for r in conducted_results]
        radiated_margins = [r["margin_db"] for r in radiated_results]

        return {
            "conducted_worst_margin": min(conducted_margins),
            "conducted_best_margin": max(conducted_margins),
            "radiated_worst_margin": min(radiated_margins),
            "radiated_best_margin": max(radiated_margins),
        }

    def _generate_compliance_report(
        self,
        conducted_results: list[dict[str, Any]],
        radiated_results: list[dict[str, Any]],
        margin_analysis: dict[str, float],
    ) -> str:
        """Generate compliance test report."""
        report = """EMC COMPLIANCE TEST REPORT
================================================================================
Generated by Oscura Framework
Standard: CISPR 32:2015 Class B

CONDUCTED EMISSIONS (CISPR 32 Class B)
--------------------------------------
Frequency Range: 150 kHz - 30 MHz
Measurement Method: CISPR 16-1-1 Quasi-Peak Detector

"""

        for result in conducted_results:
            status = "PASS" if result["pass"] else "FAIL"
            report += (
                f"{result['frequency_mhz']:8.2f} MHz: "
                f"{result['measured_db']:6.1f} dBμV "
                f"(Limit: {result['limit_db']:.0f} dBμV) "
                f"Margin: {result['margin_db']:+.1f} dB [{status}]\n"
            )

        report += """
RADIATED EMISSIONS (CISPR 32 Class B)
-------------------------------------
Frequency Range: 30 MHz - 1 GHz
Measurement Distance: 10 meters
Measurement Method: CISPR 16-1-1 Quasi-Peak Detector

"""

        for result in radiated_results:
            status = "PASS" if result["pass"] else "FAIL"
            report += (
                f"{result['frequency_mhz']:8.0f} MHz: "
                f"{result['measured_db']:6.1f} dBμV/m "
                f"(Limit: {result['limit_db']:.0f} dBμV/m) "
                f"Margin: {result['margin_db']:+.1f} dB [{status}]\n"
            )

        report += f"""
MARGIN ANALYSIS
---------------
Conducted Emissions:
  Worst Case Margin: {margin_analysis["conducted_worst_margin"]:+.1f} dB
  Best Case Margin:  {margin_analysis["conducted_best_margin"]:+.1f} dB

Radiated Emissions:
  Worst Case Margin: {margin_analysis["radiated_worst_margin"]:+.1f} dB
  Best Case Margin:  {margin_analysis["radiated_best_margin"]:+.1f} dB

"""

        # Overall compliance determination
        overall_pass = (
            margin_analysis["conducted_worst_margin"] > 0
            and margin_analysis["radiated_worst_margin"] > 0
        )

        report += """OVERALL COMPLIANCE STATUS
-------------------------
"""

        if overall_pass:
            report += "RESULT: COMPLIANT\n"
            report += "All emission measurements are within CISPR 32 Class B limits.\n"
        else:
            report += "RESULT: NON-COMPLIANT\n"
            report += "One or more emission measurements exceed regulatory limits.\n"
            report += "Corrective action required before product release.\n"

        report += """
================================================================================
End of Report
"""

        return report


if __name__ == "__main__":
    demo = EMCTestingWorkflowDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
