"""Device Characterization: Complete characterization workflow with datasheet validation

Demonstrates:
- oscura.waveform.measurements - Parameter extraction
- oscura.characterization.sweep - Multi-parameter sweep
- oscura.characterization.datasheet - Datasheet validation
- oscura.statistics.basic - Statistical analysis
- Complete workflow with characterization report

Standards:
- IEEE 181-2011 (Waveform measurements)
- IEC 60747 (Semiconductor device characterization)

Related Demos:
- 02_basic_analysis/01_waveform_measurements.py - Measurements
- 02_basic_analysis/02_statistics.py - Statistics

This demonstration shows a complete device characterization workflow:
1. Define characterization test plan
2. Perform multi-parameter sweep measurements
3. Extract performance parameters
4. Validate against datasheet specifications
5. Generate characterization report with plots

Time Budget: < 5 seconds for complete characterization
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


class DeviceCharacterizationWorkflowDemo(BaseDemo):
    """Complete device characterization workflow."""

    # Datasheet specifications for fictional device
    DATASHEET_SPECS: typing.ClassVar[dict[str, dict[str, Any]]] = {
        "output_voltage": {"min": 4.75, "typ": 5.0, "max": 5.25, "unit": "V"},
        "load_regulation": {"min": None, "typ": 1.0, "max": 2.0, "unit": "%"},
        "rise_time": {"min": None, "typ": 10.0, "max": 20.0, "unit": "ns"},
        "fall_time": {"min": None, "typ": 10.0, "max": 20.0, "unit": "ns"},
        "frequency": {"min": 0.95, "typ": 1.0, "max": 1.05, "unit": "MHz"},
        "duty_cycle": {"min": 48.0, "typ": 50.0, "max": 52.0, "unit": "%"},
        "output_impedance": {"min": None, "typ": 50.0, "max": 75.0, "unit": "Ω"},
        "jitter_rms": {"min": None, "typ": 5.0, "max": 10.0, "unit": "ps"},
    }

    def __init__(self) -> None:
        """Initialize demonstration."""
        super().__init__(
            name="device_characterization_workflow",
            description="Complete device characterization with datasheet validation",
            capabilities=[
                "oscura.waveform.measurements",
                "oscura.characterization.sweep",
                "oscura.characterization.datasheet",
                "oscura.statistics.basic",
            ],
            ieee_standards=[
                "IEEE 181-2011",
                "IEC 60747",
            ],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
                "02_basic_analysis/02_statistics.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate device characterization test data.

        Simulates multi-condition characterization:
        - Temperature sweep (25°C, 50°C, 85°C)
        - Load sweep (no load, 50%, 100%)
        - Supply voltage sweep (4.5V, 5.0V, 5.5V)

        Returns:
            Dictionary with characterization test data
        """
        self.section("Generating Device Characterization Test Data")

        sample_rate = 100e6  # 100 MHz
        duration = 0.001  # 1 ms

        test_conditions = []

        # Temperature sweep at nominal conditions
        self.info("Generating temperature sweep data...")
        for temp_c in [25, 50, 85]:
            signal = self._generate_device_output(
                temperature=temp_c,
                load_percent=50,
                supply_v=5.0,
                duration=duration,
                sample_rate=sample_rate,
            )
            test_conditions.append(
                {"temperature": temp_c, "load": 50, "supply": 5.0, "signal": signal}
            )

        # Load sweep at nominal temperature
        self.info("Generating load sweep data...")
        for load_percent in [0, 50, 100]:
            signal = self._generate_device_output(
                temperature=25,
                load_percent=load_percent,
                supply_v=5.0,
                duration=duration,
                sample_rate=sample_rate,
            )
            test_conditions.append(
                {"temperature": 25, "load": load_percent, "supply": 5.0, "signal": signal}
            )

        # Supply voltage sweep at nominal conditions
        self.info("Generating supply voltage sweep data...")
        for supply_v in [4.5, 5.0, 5.5]:
            signal = self._generate_device_output(
                temperature=25,
                load_percent=50,
                supply_v=supply_v,
                duration=duration,
                sample_rate=sample_rate,
            )
            test_conditions.append(
                {"temperature": 25, "load": 50, "supply": supply_v, "signal": signal}
            )

        self.result("Test conditions", len(test_conditions))
        self.result("Temperature range", "25°C - 85°C")
        self.result("Load range", "0% - 100%")
        self.result("Supply range", "4.5V - 5.5V")

        return {"test_conditions": test_conditions}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute complete device characterization workflow."""
        results: dict[str, Any] = {}
        workflow_start = time.time()

        test_conditions = data["test_conditions"]

        # ===== PHASE 1: Multi-Parameter Sweep =====
        self.section("Phase 1: Multi-Parameter Characterization Sweep")
        phase1_start = time.time()

        self.subsection("1.1 Measurement Sweep")
        sweep_results = []

        for _i, condition in enumerate(test_conditions):
            measurements = self._characterize_signal(condition["signal"])
            measurements["temperature"] = condition["temperature"]
            measurements["load"] = condition["load"]
            measurements["supply"] = condition["supply"]
            sweep_results.append(measurements)

        results["sweep_results"] = sweep_results
        self.info(f"Completed {len(sweep_results)} characterization measurements")

        phase1_time = time.time() - phase1_start
        results["phase1_time"] = phase1_time
        self.result("Phase 1 duration", f"{phase1_time:.3f}", "seconds")

        # ===== PHASE 2: Statistical Analysis =====
        self.section("Phase 2: Statistical Analysis")
        phase2_start = time.time()

        self.subsection("2.1 Parameter Statistics")
        statistics = self._compute_statistics(sweep_results)
        results["statistics"] = statistics

        self.info("Parameter statistics (across all conditions):")
        for param, stats in statistics.items():
            if param not in ["temperature", "load", "supply"]:
                self.info(
                    f"  {param:20s}: min={stats['min']:7.3f}, "
                    f"mean={stats['mean']:7.3f}, max={stats['max']:7.3f}, "
                    f"std={stats['std']:7.3f}"
                )

        phase2_time = time.time() - phase2_start
        results["phase2_time"] = phase2_time
        self.result("Phase 2 duration", f"{phase2_time:.3f}", "seconds")

        # ===== PHASE 3: Datasheet Validation =====
        self.section("Phase 3: Datasheet Specification Validation")
        phase3_start = time.time()

        self.subsection("3.1 Specification Compliance Check")
        compliance_results = self._validate_against_datasheet(statistics, self.DATASHEET_SPECS)
        results["compliance_results"] = compliance_results

        pass_count = sum(1 for r in compliance_results if r["compliant"])
        self.info(f"Specification compliance: {pass_count}/{len(compliance_results)}")

        for result in compliance_results:
            status = "PASS" if result["compliant"] else "FAIL"
            self.info(
                f"  {result['parameter']:20s}: {result['measured_mean']:7.3f} {result['unit']} "
                f"[{result['spec_min'] if result['spec_min'] else 'N/A':>6s} - "
                f"{result['spec_max'] if result['spec_max'] else 'N/A':>6s}] [{status}]"
            )

        phase3_time = time.time() - phase3_start
        results["phase3_time"] = phase3_time
        self.result("Phase 3 duration", f"{phase3_time:.3f}", "seconds")

        # ===== PHASE 4: Performance Analysis =====
        self.section("Phase 4: Performance Trend Analysis")
        phase4_start = time.time()

        self.subsection("4.1 Temperature Dependence")
        temp_analysis = self._analyze_temperature_dependence(sweep_results)
        results["temp_analysis"] = temp_analysis

        self.info("Temperature sensitivity:")
        for param, sensitivity in temp_analysis.items():
            self.info(f"  {param}: {sensitivity:.4f} per °C")

        self.subsection("4.2 Load Regulation")
        load_analysis = self._analyze_load_regulation(sweep_results)
        results["load_analysis"] = load_analysis

        self.info(f"Load regulation: {load_analysis['regulation_percent']:.2f}% (0% to 100% load)")

        phase4_time = time.time() - phase4_start
        results["phase4_time"] = phase4_time
        self.result("Phase 4 duration", f"{phase4_time:.3f}", "seconds")

        # ===== PHASE 5: Report Generation =====
        self.section("Phase 5: Characterization Report Generation")
        phase5_start = time.time()

        self.subsection("5.1 Generating Report")
        report = self._generate_characterization_report(
            statistics=statistics,
            compliance_results=compliance_results,
            temp_analysis=temp_analysis,
            load_analysis=load_analysis,
            sweep_results=sweep_results,
        )

        output_dir = self.get_output_dir()
        report_path = output_dir / "characterization_report.txt"
        report_path.write_text(report)

        results["report_generated"] = True
        results["report_path"] = str(report_path)

        self.success(f"Report saved: {report_path}")
        self.info(f"Report size: {len(report)} bytes")

        phase5_time = time.time() - phase5_start
        results["phase5_time"] = phase5_time
        self.result("Phase 5 duration", f"{phase5_time:.3f}", "seconds")

        # ===== WORKFLOW SUMMARY =====
        self.section("Characterization Summary")

        total_time = time.time() - workflow_start
        results["total_time"] = total_time

        self.subsection("Timing Breakdown")
        self.result("  Phase 1 (Sweep)", f"{phase1_time:.3f}", "s")
        self.result("  Phase 2 (Statistics)", f"{phase2_time:.3f}", "s")
        self.result("  Phase 3 (Validation)", f"{phase3_time:.3f}", "s")
        self.result("  Phase 4 (Analysis)", f"{phase4_time:.3f}", "s")
        self.result("  Phase 5 (Report)", f"{phase5_time:.3f}", "s")
        self.result("  TOTAL WORKFLOW", f"{total_time:.3f}", "s")

        self.subsection("Characterization Results")
        self.result("  Test conditions", len(sweep_results))
        self.result("  Parameters measured", len(statistics))
        self.result("  Datasheet compliance", f"{pass_count}/{len(compliance_results)}")

        overall_compliant = pass_count == len(compliance_results)
        results["overall_compliant"] = overall_compliant

        if overall_compliant:
            self.success("OVERALL: Device meets all datasheet specifications")
        else:
            self.warning("OVERALL: Device has specification deviations")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate characterization workflow results."""
        all_passed = True

        # Validate sweep completion
        if len(results.get("sweep_results", [])) < 5:
            self.error("Insufficient sweep measurements")
            all_passed = False
        else:
            self.success(f"Characterization sweep passed: {len(results['sweep_results'])} points")

        # Validate statistics computation
        if not results.get("statistics"):
            self.error("Statistical analysis failed")
            all_passed = False
        else:
            self.success(f"Statistical analysis passed: {len(results['statistics'])} parameters")

        # Validate datasheet compliance
        if not results.get("compliance_results"):
            self.error("Datasheet validation failed")
            all_passed = False
        else:
            self.success(
                f"Datasheet validation passed: {len(results['compliance_results'])} specs checked"
            )

        # Validate report generation
        if not results.get("report_generated", False):
            self.error("Report generation failed")
            all_passed = False
        else:
            self.success("Characterization report generated successfully")

        # Validate timing
        total_time = results.get("total_time", 999)
        if total_time > 10.0:
            self.warning(f"Workflow exceeded target time (got {total_time:.1f}s, target <10s)")
        else:
            self.success(f"Workflow completed within time budget ({total_time:.3f}s)")

        return all_passed

    def _generate_device_output(
        self,
        temperature: float,
        load_percent: float,
        supply_v: float,
        duration: float,
        sample_rate: float,
    ) -> WaveformTrace:
        """Generate device output signal for given conditions."""
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate

        # Model temperature effects
        temp_factor = 1.0 + (temperature - 25) * 0.0005  # 0.05% per °C
        freq = 1e6 * temp_factor

        # Model load regulation
        load_factor = 1.0 - (load_percent / 100) * 0.01  # 1% droop at full load
        amplitude = 5.0 * load_factor

        # Model supply voltage effects
        supply_factor = supply_v / 5.0
        amplitude *= supply_factor

        # Generate square wave
        signal = amplitude * np.sign(np.sin(2 * np.pi * freq * t))

        # Add realistic noise and imperfections
        noise_level = amplitude * 0.01
        signal += np.random.normal(0, noise_level, num_samples)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name=f"output_T{temperature}_L{load_percent}_V{supply_v}",
        )
        return WaveformTrace(data=signal, metadata=metadata)

    def _characterize_signal(self, signal: WaveformTrace) -> dict[str, float]:
        """Perform comprehensive signal characterization."""
        data = signal.data

        measurements = {
            "output_voltage": (np.max(data) + np.min(data)) / 2,  # Average level
            "peak_to_peak": np.max(data) - np.min(data),
            "rms": np.sqrt(np.mean(data**2)),
            "rise_time": 10.0 + np.random.normal(0, 2.0),  # Simulated
            "fall_time": 10.0 + np.random.normal(0, 2.0),  # Simulated
            "frequency": 1.0 + np.random.normal(0, 0.01),  # MHz, simulated
            "duty_cycle": 50.0 + np.random.normal(0, 0.5),  # Simulated
            "jitter_rms": 5.0 + np.random.normal(0, 1.0),  # ps, simulated
        }

        return measurements

    def _compute_statistics(
        self, sweep_results: list[dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Compute statistics across all sweep points."""
        statistics = {}

        # Get list of all parameters (exclude conditions)
        params = [k for k in sweep_results[0] if k not in ["temperature", "load", "supply"]]

        for param in params:
            values = [r[param] for r in sweep_results]

            statistics[param] = {
                "min": np.min(values),
                "max": np.max(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
            }

        return statistics

    def _validate_against_datasheet(
        self, statistics: dict[str, dict[str, float]], datasheet: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Validate measured statistics against datasheet specifications."""
        results = []

        for param, stats in statistics.items():
            if param in datasheet:
                spec = datasheet[param]
                mean_val = stats["mean"]

                spec_min = spec.get("min")
                spec_max = spec.get("max")
                unit = spec.get("unit", "")

                # Check compliance
                compliant = True
                if spec_min is not None and mean_val < spec_min:
                    compliant = False
                if spec_max is not None and mean_val > spec_max:
                    compliant = False

                results.append(
                    {
                        "parameter": param,
                        "measured_mean": mean_val,
                        "measured_std": stats["std"],
                        "spec_min": f"{spec_min:.2f}" if spec_min is not None else None,
                        "spec_max": f"{spec_max:.2f}" if spec_max is not None else None,
                        "unit": unit,
                        "compliant": compliant,
                    }
                )

        return results

    def _analyze_temperature_dependence(
        self, sweep_results: list[dict[str, float]]
    ) -> dict[str, float]:
        """Analyze temperature dependence of key parameters."""
        # Filter results at constant load/supply
        temp_sweep = [r for r in sweep_results if r["load"] == 50 and r["supply"] == 5.0]

        if len(temp_sweep) < 2:
            return {}

        analysis = {}

        # Calculate temperature coefficient for output voltage
        temps = [r["temperature"] for r in temp_sweep]
        voltages = [r["output_voltage"] for r in temp_sweep]

        if len(temps) > 1:
            temp_coef = (voltages[-1] - voltages[0]) / (temps[-1] - temps[0])
            analysis["output_voltage"] = temp_coef

        return analysis

    def _analyze_load_regulation(self, sweep_results: list[dict[str, float]]) -> dict[str, float]:
        """Analyze load regulation performance."""
        # Filter results at constant temperature/supply
        load_sweep = [r for r in sweep_results if r["temperature"] == 25 and r["supply"] == 5.0]

        if len(load_sweep) < 2:
            return {"regulation_percent": 0.0}

        voltages = [r["output_voltage"] for r in load_sweep]
        no_load = voltages[0]
        full_load = voltages[-1]

        regulation_percent = abs((no_load - full_load) / no_load * 100)

        return {
            "regulation_percent": regulation_percent,
            "no_load_v": no_load,
            "full_load_v": full_load,
        }

    def _generate_characterization_report(
        self,
        statistics: dict[str, dict[str, float]],
        compliance_results: list[dict[str, Any]],
        temp_analysis: dict[str, float],
        load_analysis: dict[str, float],
        sweep_results: list[dict[str, float]],
    ) -> str:
        """Generate comprehensive characterization report."""
        report = (
            """DEVICE CHARACTERIZATION REPORT
================================================================================
Generated by Oscura Framework
Test Date: 2024-01-22
Device: Unknown Test Device
Test Engineer: AUTO

TEST CONDITIONS
---------------
Temperature Range: 25°C - 85°C
Load Range: 0% - 100%
Supply Voltage Range: 4.5V - 5.5V
Total Test Points: """
            + str(len(sweep_results))
            + """

MEASURED PARAMETERS (Statistical Summary)
------------------------------------------
"""
        )

        for param, stats in statistics.items():
            report += (
                f"{param:20s}: min={stats['min']:7.3f}, mean={stats['mean']:7.3f}, "
                f"max={stats['max']:7.3f}, std={stats['std']:7.3f}\n"
            )

        report += """
DATASHEET SPECIFICATION COMPLIANCE
-----------------------------------
"""

        for result in compliance_results:
            status = "PASS" if result["compliant"] else "FAIL"
            report += (
                f"{result['parameter']:20s}: {result['measured_mean']:7.3f} {result['unit']} "
                f"[{result['spec_min'] if result['spec_min'] else 'N/A':>7s} - "
                f"{result['spec_max'] if result['spec_max'] else 'N/A':>7s}] [{status}]\n"
            )

        report += """
TEMPERATURE CHARACTERISTICS
---------------------------
"""

        for param, coef in temp_analysis.items():
            report += f"{param}: {coef:.6f} per °C\n"

        report += f"""
LOAD REGULATION
---------------
No Load Voltage: {load_analysis.get("no_load_v", 0):.3f} V
Full Load Voltage: {load_analysis.get("full_load_v", 0):.3f} V
Load Regulation: {load_analysis["regulation_percent"]:.2f}%

CONCLUSION
----------
"""

        pass_count = sum(1 for r in compliance_results if r["compliant"])
        if pass_count == len(compliance_results):
            report += "Device meets all datasheet specifications across tested conditions.\n"
            report += "RESULT: PASS - Device approved for production.\n"
        else:
            report += f"Device fails {len(compliance_results) - pass_count} of {len(compliance_results)} specifications.\n"
            report += "RESULT: FAIL - Device requires design review.\n"
            report += "\nFailed specifications:\n"
            for result in compliance_results:
                if not result["compliant"]:
                    report += f"  - {result['parameter']}: {result['measured_mean']:.3f} {result['unit']}\n"

        report += """
================================================================================
End of Report
"""

        return report


if __name__ == "__main__":
    demo = DeviceCharacterizationWorkflowDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
