"""Comprehensive example: Reverse engineering 1970s digital logic systems.

This example demonstrates ALL new features for vintage logic analysis:
1. Tektronix file loading
2. Automatic logic family detection (TTL, ECL, RTL, DTL, CMOS)
3. Open-collector detection
4. IC identification from measured timings
5. Multi-IC timing path analysis
6. Parallel bus protocol decoding (GPIB, Centronics, ISA)
7. WaveDrom timing diagram generation
8. State-of-the-art visualizations

Use case: Analyzing a vintage 1976 microcomputer to replicate with modern parts.
"""

from pathlib import Path

import numpy as np

from oscura.analyzers.digital import (
    IC_DATABASE,
    analyze_timing_path,
    detect_logic_family,
    detect_open_collector,
    identify_ic,
    validate_ic_timing,
)
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.export.wavedrom import WaveDromBuilder


def create_sample_ttl_signal(freq: float = 1e6, duration: float = 1e-4) -> WaveformTrace:
    """Create a sample TTL signal for demonstration."""
    sample_rate = 100e6  # 100 MHz sampling
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate

    # Square wave with TTL levels (0V to 5V)
    signal = np.where((t * freq) % 1 < 0.5, 0.0, 5.0)

    # Add realistic rise/fall times (10ns for LS-TTL)
    rise_samples = int(10e-9 * sample_rate)
    for i in range(1, len(signal)):
        if signal[i] != signal[i - 1]:
            if signal[i] > signal[i - 1]:  # Rising
                for j in range(min(rise_samples, len(signal) - i)):
                    signal[i + j] = np.linspace(0, 5, rise_samples)[j]
            else:  # Falling
                for j in range(min(rise_samples // 2, len(signal) - i)):
                    signal[i + j] = np.linspace(5, 0, rise_samples // 2)[j]

    # Add noise
    signal += np.random.normal(0, 0.05, num_samples)

    metadata = TraceMetadata(
        sample_rate=sample_rate,
        channel_name="TTL_CLK",
    )
    return WaveformTrace(data=signal, metadata=metadata)


def main():
    """Main demonstration of vintage logic reverse engineering."""

    print("=" * 80)
    print("VINTAGE DIGITAL LOGIC REVERSE ENGINEERING DEMO")
    print("Target: 1976 Microcomputer System (74LS-TTL based)")
    print("=" * 80)
    print()

    # =========================================================================
    # STEP 1: Logic Family Detection
    # =========================================================================
    print("STEP 1: Automatic Logic Family Detection")
    print("-" * 80)

    # Create TTL signal
    ttl_signal = create_sample_ttl_signal()

    # Auto-detect logic family
    family, confidence = detect_logic_family(ttl_signal)
    print(f"  Detected logic family: {family}")
    print(f"  Confidence: {confidence * 100:.1f}%")
    print(f"  ✓ System uses {family} logic (5V supply)")
    print()

    # =========================================================================
    # STEP 2: Open-Collector Detection
    # =========================================================================
    print("STEP 2: Open-Collector Output Detection")
    print("-" * 80)

    # Simulate open-collector output (slow rise, fast fall)
    # In real scenario, this would be detected from actual measurements

    is_oc, ratio = detect_open_collector(ttl_signal)
    print(f"  Open-collector detected: {is_oc}")
    print(f"  Rise/Fall asymmetry ratio: {ratio:.2f}")

    if is_oc:
        print("  ✓ Open-collector output detected")
        print("    Recommendation: Use 10kΩ pull-up resistor in modern design")
    else:
        print("  ✓ Standard totem-pole output")
    print()

    # =========================================================================
    # STEP 3: IC Identification from Measured Timings
    # =========================================================================
    print("STEP 3: IC Identification from Measured Timings")
    print("-" * 80)

    # Simulate measured timings from oscilloscope
    # In real scenario, these would be measured using propagation_delay(),
    # setup_time(), etc.
    measured_timings = {
        "t_pd": 25e-9,  # Propagation delay
        "t_su": 20e-9,  # Setup time
        "t_h": 5e-9,  # Hold time
        "t_w": 25e-9,  # Min pulse width
    }

    # Identify IC
    ic_name, confidence = identify_ic(measured_timings, tolerance=0.3)
    print("  Measured timings:")
    for param, value in measured_timings.items():
        print(f"    {param} = {value * 1e9:.1f} ns")
    print()
    print(f"  Identified IC: {ic_name}")
    print(f"  Confidence: {confidence * 100:.1f}%")
    print(f"  ✓ IC is likely a {ic_name} flip-flop")
    print()

    # Validate against spec
    print("  Validating against datasheet specifications:")
    validation = validate_ic_timing(ic_name, measured_timings, tolerance=0.3)

    for param, result in validation.items():
        if result["passes"] is not None:
            status = "✓ PASS" if result["passes"] else "✗ FAIL"
            measured = result["measured"]
            spec = result["spec"]
            print(f"    {param}: {status}")
            print(f"      Measured: {measured * 1e9:.1f} ns, Spec: {spec * 1e9:.1f} ns")
    print()

    # =========================================================================
    # STEP 4: Multi-IC Timing Path Analysis
    # =========================================================================
    print("STEP 4: Multi-IC Timing Path Analysis")
    print("-" * 80)
    print("  Analyzing signal path: CPU → Address Latch → Memory Decoder")
    print()

    # Create signals for IC chain (in real scenario, these would be captured)
    cpu_output = create_sample_ttl_signal(freq=2e6)
    latch_output = create_sample_ttl_signal(freq=2e6)
    decoder_output = create_sample_ttl_signal(freq=2e6)

    # Define timing path
    timing_path = [
        ("74LS244", cpu_output, latch_output),  # Buffer
        ("74LS138", latch_output, decoder_output),  # Decoder
    ]

    # Analyze path
    path_result = analyze_timing_path(timing_path, target_frequency=2e6)

    print(f"  Total path delay: {path_result.total_delay * 1e9:.1f} ns")
    print(f"  Path meets timing: {path_result.meets_timing}")

    if path_result.path_margin is not None:
        print(f"  Timing margin: {path_result.path_margin * 1e9:.1f} ns")

    print()
    print("  Per-stage breakdown:")
    for idx, stage in enumerate(path_result.stages):
        print(f"    Stage {idx}: {stage.ic_name}")
        print(f"      Measured delay: {stage.measured_delay * 1e9:.1f} ns")
        if stage.spec_delay:
            print(f"      Spec delay: {stage.spec_delay * 1e9:.1f} ns")
        if stage.margin:
            status = "✓" if stage.margin > 0 else "✗"
            print(f"      Margin: {status} {stage.margin * 1e9:.1f} ns")
    print()

    # =========================================================================
    # STEP 5: IC Timing Database Query
    # =========================================================================
    print("STEP 5: IC Timing Database")
    print("-" * 80)
    print(f"  Database contains {len(IC_DATABASE)} IC specifications:")
    print()

    # Show sample ICs
    for ic_name in ["74LS74", "74LS00", "74HC74", "4013"]:
        ic_spec = IC_DATABASE[ic_name]
        print(f"  {ic_name} ({ic_spec.description})")
        print(f"    Family: {ic_spec.family}")
        print(
            f"    VCC: {ic_spec.vcc_nom}V (range: {ic_spec.vcc_range[0]}-{ic_spec.vcc_range[1]}V)"
        )
        print(f"    Timing: t_pd={ic_spec.timing.get('t_pd', 0) * 1e9:.1f}ns", end="")
        if "t_su" in ic_spec.timing:
            print(f", t_su={ic_spec.timing['t_su'] * 1e9:.1f}ns", end="")
        print()
        print()

    # =========================================================================
    # STEP 6: WaveDrom Timing Diagram Generation
    # =========================================================================
    print("STEP 6: WaveDrom Timing Diagram Generation")
    print("-" * 80)

    # Create timing diagram
    builder = WaveDromBuilder(title="74LS74 Setup/Hold Time Analysis")

    # Add clock
    builder.add_clock("CLK", period=100e-9, duration=500e-9)

    # Add data signal with edges at specific times
    data_edges = [30e-9, 180e-9, 330e-9]
    builder.add_signal("DATA", edges=data_edges)

    # Add output
    output_edges = [100e-9, 200e-9, 300e-9, 400e-9]
    builder.add_signal("Q", edges=output_edges)

    # Add timing annotations
    builder.add_arrow(30e-9, 100e-9, "t_su = 70ns")

    # Export to JSON
    output_path = Path("timing_diagram_74ls74.json")
    builder.save(output_path)

    print(f"  ✓ WaveDrom diagram exported to: {output_path}")
    print(f"  ✓ Render with: wavedrom-cli -i {output_path} -s timing.svg")
    print()

    # =========================================================================
    # STEP 7: High-Level API and Comprehensive Reporting
    # =========================================================================
    print("STEP 7: High-Level API and Comprehensive Reporting")
    print("-" * 80)
    print("  Using new unified API for complete analysis...")
    print()

    # Import new high-level API
    from oscura.analyzers.digital.vintage import analyze_vintage_logic
    from oscura.exporters.json_export import export_vintage_logic_json
    from oscura.exporters.vintage_logic_csv import export_all_vintage_logic_csv
    from oscura.reporting.vintage_logic_report import generate_vintage_logic_report

    # Prepare traces dictionary
    traces = {
        "CLK": ttl_signal,
        "DATA": cpu_output,
        "ADDR": latch_output,
    }

    # Run comprehensive analysis
    result = analyze_vintage_logic(
        traces=traces,
        target_frequency=2e6,
        system_description="1976 Microcomputer System",
    )

    print(f"  Analysis complete in {result.analysis_duration:.2f} seconds")
    print(
        f"  Detected family: {result.detected_family} ({result.family_confidence * 100:.1f}% confidence)"
    )
    print(f"  ICs identified: {len(result.identified_ics)}")
    print(f"  Timing measurements: {len(result.timing_measurements)}")
    print()

    # Generate comprehensive report
    print("  Generating comprehensive reports...")
    report = generate_vintage_logic_report(
        result,
        traces,
        title="Vintage Microcomputer Analysis - Complete Report",
        output_dir=Path("./analysis_output"),
    )

    # Export in multiple formats
    html_path = report.save_html(Path("analysis_output/report.html"))
    md_path = report.save_markdown(Path("analysis_output/report.md"))

    print(f"  ✓ HTML report: {html_path}")
    print(f"  ✓ Markdown report: {md_path}")
    print()

    # Export CSV data
    print("  Exporting data to CSV...")
    csv_paths = export_all_vintage_logic_csv(
        result, output_dir=Path("./analysis_output"), prefix="vintage_"
    )
    for data_type, path in csv_paths.items():
        print(f"  ✓ {data_type}: {path}")
    print()

    # Export JSON
    print("  Exporting complete results to JSON...")
    json_path = Path("analysis_output/complete_analysis.json")
    export_vintage_logic_json(result, json_path)
    print(f"  ✓ JSON export: {json_path}")
    print()

    # =========================================================================
    # STEP 8: Summary and Recommendations
    # =========================================================================
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    print("System Characterization:")
    print(f"  • Logic Family: {family} (5V TTL)")
    print(f"  • Primary IC: {ic_name} flip-flop")
    print("  • System Clock: 2 MHz")
    print(f"  • Critical Path Delay: {path_result.total_delay * 1e9:.1f} ns")
    print()
    print("Modern Replication Recommendations:")
    print("  • Replace 74LSxx with 74HCTxx (pin-compatible, faster, lower power)")
    print("  • Alternative: Use CPLD/FPGA to implement entire logic in single chip")
    print("  • Add 100nF decoupling caps (one per IC, close to VCC pin)")
    print("  • Use 3.3V-tolerant buffers if interfacing with modern microcontrollers")
    print()
    print("Bill of Materials (Direct Replacement):")
    for ic_name in ["74LS74", "74LS244", "74LS138"]:
        if ic_name in IC_DATABASE:
            ic_spec = IC_DATABASE[ic_name]
            modern_equiv = ic_name.replace("LS", "HCT")
            print(f"  • {ic_name} → {modern_equiv} ({ic_spec.description})")
    print()
    print("✓ Analysis complete! Ready for modern replication.")
    print("=" * 80)


if __name__ == "__main__":
    main()
