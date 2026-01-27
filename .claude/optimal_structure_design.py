#!/usr/bin/env python3
"""Design optimal consolidated example structure.

Combines demonstrations/ and demos/ into single optimal structure
with ZERO redundancy and 100% capability coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
AUDIT_FILE = ROOT / ".claude" / "examples_audit_comprehensive.json"


def load_audit() -> list[dict[str, Any]]:
    """Load audit results."""
    with open(AUDIT_FILE) as f:
        return json.load(f)


def design_optimal_structure(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """Design optimal consolidated structure.

    Returns:
        Dictionary mapping new structure to source files
    """

    # Split files by source
    demonstrations = [a for a in analyses if "demonstrations/" in a["path"]]
    demos = [a for a in analyses if "demos/" in a["path"]]

    # Optimal category structure (progressive learning order)
    optimal_structure = {
        "00_getting_started": {
            "description": "Introduction to Oscura",
            "from_demonstrations": [
                "00_hello_world.py",
                "01_core_types.py",
                "02_supported_formats.py",
            ],
            "from_demos": [],
        },
        "01_data_loading": {
            "description": "Loading all file formats",
            "from_demonstrations": [
                "01_oscilloscopes.py",
                "02_logic_analyzers.py",
                "03_automotive_formats.py",
                "04_scientific_formats.py",
                "05_custom_binary.py",
                "06_streaming_large_files.py",
                "07_multi_channel.py",
                "08_network_formats.py",
                "09_specialized_formats.py",
                "10_performance_loading.py",
            ],
            "from_demos": [
                # Additional unique demos content
                "demos/02_file_format_io/vcd_loader_demo.py → vcd_advanced_demo.py",
                "demos/03_custom_daq/simple_loader.py → custom_loader_simple.py",
                "demos/03_custom_daq/chunked_loader.py → custom_loader_chunked.py",
                "demos/03_custom_daq/optimal_streaming_loader.py → custom_loader_streaming.py",
            ],
        },
        "02_basic_analysis": {
            "description": "Core signal analysis",
            "from_demonstrations": [
                "01_waveform_measurements.py",
                "02_statistics.py",
                "03_spectral_analysis.py",
                "04_filtering.py",
                "05_triggering.py",
                "06_math_operations.py",
                "07_wavelet_analysis.py",
            ],
            "from_demos": [
                "demos/01_waveform_analysis/comprehensive_wfm_analysis.py → waveform_comprehensive.py",
                "demos/11_mixed_signal/comprehensive_mixed_signal_demo.py → mixed_signal.py",
                "demos/12_spectral_compliance/comprehensive_spectral_demo.py → spectral_compliance.py",
            ],
        },
        "03_protocol_decoding": {
            "description": "All protocol decoders",
            "from_demonstrations": [
                "01_serial_comprehensive.py",
                "02_automotive_protocols.py",
                "03_debug_protocols.py",
                "04_parallel_bus.py",  # Has SKIP - keep for structure
                "05_encoded_protocols.py",
                "06_auto_detection.py",
            ],
            "from_demos": [
                # Additional protocols from demos
                "demos/04_serial_protocols/i2s_demo.py → i2s.py",
                "demos/04_serial_protocols/jtag_demo.py → jtag.py",
                "demos/04_serial_protocols/manchester_demo.py → manchester.py",
                "demos/04_serial_protocols/onewire_demo.py → onewire.py",
                "demos/04_serial_protocols/swd_demo.py → swd.py",
                "demos/04_serial_protocols/usb_demo.py → usb.py",
                "demos/05_protocol_decoding/comprehensive_protocol_demo.py → protocol_comprehensive.py",
                "demos/06_udp_packet_analysis/comprehensive_udp_analysis.py → udp_analysis.py",
            ],
        },
        "04_advanced_analysis": {
            "description": "Advanced signal analysis",
            "from_demonstrations": [
                "01_jitter_analysis.py",
                "02_power_analysis.py",
                "03_signal_integrity.py",
                "04_eye_diagrams.py",
                "05_pattern_discovery.py",
                "06_quality_assessment.py",
                "07_component_characterization.py",
                "08_transmission_lines.py",
                "09_digital_timing.py",
            ],
            "from_demos": [
                # Specialized demos content
                "demos/13_jitter_analysis/ddj_dcd_demo.py → jitter_ddj_dcd.py",
                "demos/13_jitter_analysis/bathtub_curve_demo.py → jitter_bathtub.py",
                "demos/14_power_analysis/dcdc_efficiency_demo.py → power_dcdc.py",
                "demos/14_power_analysis/ripple_analysis_demo.py → power_ripple.py",
                "demos/15_signal_integrity/tdr_impedance_demo.py → signal_integrity_tdr.py",
                "demos/15_signal_integrity/setup_hold_timing_demo.py → signal_integrity_timing.py",
                "demos/15_signal_integrity/sparams_demo.py → signal_integrity_sparams.py",
            ],
        },
        "05_domain_specific": {
            "description": "Industry-specific analysis",
            "from_demonstrations": [
                "01_automotive_diagnostics.py",
                "02_emc_compliance.py",
                "03_vintage_logic.py",
                "04_side_channel.py",  # Has SKIP
            ],
            "from_demos": [
                "demos/08_automotive_protocols/lin_demo.py → automotive_lin.py",
                "demos/08_automotive_protocols/flexray_demo.py → automotive_flexray.py",
                "demos/09_automotive/comprehensive_automotive_demo.py → automotive_comprehensive.py",
                "demos/16_emc_compliance/comprehensive_emc_demo.py → emc_comprehensive.py",
                "demos/10_timing_measurements/ieee_181_pulse_demo.py → timing_ieee181.py",
            ],
        },
        "06_reverse_engineering": {
            "description": "Protocol reverse engineering",
            "from_demonstrations": [
                "01_unknown_protocol.py",
                "02_crc_recovery.py",
                "03_state_machines.py",
                "04_field_inference.py",
                "05_pattern_discovery.py",
                "06_wireshark_export.py",
                "07_entropy_analysis.py",
                "08_data_classification.py",
                "09_signal_classification.py",
                "10_anomaly_detection.py",
            ],
            "from_demos": [
                "demos/07_protocol_inference/crc_reverse_demo.py → crc_reverse.py",
                "demos/07_protocol_inference/state_machine_learning.py → state_machine_learning.py",
                "demos/07_protocol_inference/wireshark_dissector_demo.py → wireshark_dissector.py",
                "demos/17_signal_reverse_engineering/comprehensive_re.py → re_comprehensive.py",
                "demos/17_signal_reverse_engineering/exploratory_analysis.py → re_exploratory.py",
                "demos/17_signal_reverse_engineering/reverse_engineer_tool.py → re_tool.py",
                "demos/18_advanced_inference/bayesian_inference_demo.py → inference_bayesian.py",
                "demos/18_advanced_inference/protocol_dsl_demo.py → inference_dsl.py",
                "demos/18_advanced_inference/active_learning_demo.py → inference_active_learning.py",
            ],
        },
        "07_advanced_api": {
            "description": "Expert API features",
            "from_demonstrations": [
                "01_pipeline_api.py",
                "02_dsl_syntax.py",
                "03_operators.py",
                "04_composition.py",
                "05_optimization.py",
                "06_streaming_api.py",
                "07_parallel_processing.py",
                "08_gpu_acceleration.py",
            ],
            "from_demos": [],
        },
        "08_extensibility": {
            "description": "Plugins and customization",
            "from_demonstrations": [
                "01_plugin_basics.py",
                "02_custom_measurement.py",
                "03_custom_algorithm.py",
                "04_plugin_development.py",
                "05_measurement_registry.py",
                "06_plugin_templates.py",
            ],
            "from_demos": [],
        },
        "09_batch_processing": {
            "description": "Parallel and batch operations",
            "from_demonstrations": [
                "01_parallel_batch.py",
                "02_result_aggregation.py",
                "03_progress_tracking.py",
                "04_optimization.py",
            ],
            "from_demos": [],
        },
        "10_sessions": {
            "description": "Interactive analysis sessions",
            "from_demonstrations": [
                "01_analysis_session.py",
                "02_can_session.py",
                "03_blackbox_session.py",
                "04_session_persistence.py",
                "05_interactive_analysis.py",
            ],
            "from_demos": [],
        },
        "11_integration": {
            "description": "Tool integration",
            "from_demonstrations": [
                "01_cli_usage.py",  # Has SKIP
                "02_jupyter_notebooks.py",
                "03_llm_integration.py",
                "04_configuration_files.py",
                "05_hardware_integration.py",
            ],
            "from_demos": [],
        },
        "12_quality_tools": {
            "description": "Quality assessment tools",
            "from_demonstrations": [
                "01_ensemble_methods.py",
                "02_quality_scoring.py",
                "03_warning_system.py",
                "04_recommendations.py",
            ],
            "from_demos": [],
        },
        "13_guidance": {
            "description": "User guidance and wizards",
            "from_demonstrations": [
                "01_smart_recommendations.py",
                "02_analysis_wizards.py",
                "03_onboarding_helpers.py",
            ],
            "from_demos": [],
        },
        "14_exploratory": {
            "description": "Exploratory analysis",
            "from_demonstrations": [
                "01_unknown_signals.py",  # Has SKIP
                "02_fuzzy_matching.py",
                "03_signal_recovery.py",
                "04_exploratory_analysis.py",
                "05_advanced_search.py",
            ],
            "from_demos": [],
        },
        "15_export_visualization": {
            "description": "Export and visualization",
            "from_demonstrations": [
                "01_export_formats.py",  # Has SKIP
                "02_wavedrom_timing.py",
                "03_wireshark_dissectors.py",
                "04_report_generation.py",
                "05_visualization_gallery.py",
                "06_comprehensive_export.py",  # Has SKIP
            ],
            "from_demos": [
                "demos/01_waveform_analysis/all_output_formats.py → all_export_formats.py",
            ],
        },
        "16_complete_workflows": {
            "description": "End-to-end workflows",
            "from_demonstrations": [
                "01_unknown_device_re.py",
                "02_automotive_diagnostics.py",
                "03_emc_testing.py",
                "04_production_testing.py",
                "05_failure_analysis.py",
                "06_device_characterization.py",
            ],
            "from_demos": [
                "demos/19_complete_workflows/automotive_full_workflow.py → automotive_workflow.py",
                "demos/19_complete_workflows/network_analysis_workflow.py → network_workflow.py",
                "demos/19_complete_workflows/unknown_signal_workflow.py → unknown_signal_workflow.py",
            ],
        },
        "17_signal_generation": {
            "description": "Signal and protocol generation",
            "from_demonstrations": [
                "01_signal_builder_comprehensive.py",
                "02_protocol_generation.py",
                "03_impairment_simulation.py",
            ],
            "from_demos": [],
        },
        "18_comparison_testing": {
            "description": "Comparison and regression testing",
            "from_demonstrations": [
                "01_golden_reference.py",
                "02_limit_testing.py",
                "03_mask_testing.py",
                "04_regression_testing.py",
            ],
            "from_demos": [],
        },
        "19_standards_compliance": {
            "description": "IEEE standards validation",
            "from_demonstrations": [
                "01_ieee_181.py",
                "02_ieee_1241.py",
                "03_ieee_1459.py",
                "04_ieee_2414.py",
            ],
            "from_demos": [],
        },
    }

    return optimal_structure


def generate_implementation_plan(structure: dict[str, Any]) -> dict[str, Any]:
    """Generate detailed implementation plan.

    Returns:
        Implementation plan with actions
    """
    plan = {
        "phase_1_keep_demonstrations": [],
        "phase_2_migrate_from_demos": [],
        "phase_3_merge_or_rename": [],
        "phase_4_cleanup": [],
        "summary": {
            "total_categories": len(structure),
            "files_from_demonstrations": 0,
            "files_from_demos": 0,
            "total_final_files": 0,
        },
    }

    for category, content in structure.items():
        # Files to keep from demonstrations/
        for file in content["from_demonstrations"]:
            plan["phase_1_keep_demonstrations"].append(f"demonstrations/{category}/{file}")
            plan["summary"]["files_from_demonstrations"] += 1

        # Files to migrate/merge from demos/
        for file_spec in content["from_demos"]:
            if " → " in file_spec:
                source, target = file_spec.split(" → ")
                plan["phase_2_migrate_from_demos"].append(
                    {
                        "source": source,
                        "target": f"demonstrations/{category}/{target}",
                        "action": "copy_and_adapt",
                    }
                )
            else:
                plan["phase_2_migrate_from_demos"].append(
                    {
                        "source": file_spec,
                        "target": f"demonstrations/{category}/{Path(file_spec).name}",
                        "action": "copy",
                    }
                )
            plan["summary"]["files_from_demos"] += 1

    plan["summary"]["total_final_files"] = (
        plan["summary"]["files_from_demonstrations"] + plan["summary"]["files_from_demos"]
    )

    # Cleanup
    plan["phase_4_cleanup"] = [
        "Delete all content from demos/ directory",
        "Remove demos/common/",
        "Remove demos/data_generation/",
        "Update CONTRIBUTING.md to reference only demonstrations/",
        "Update demonstrations/README.md with new structure",
    ]

    return plan


def main() -> None:
    """Main entry point."""
    analyses = load_audit()

    print("=" * 80)
    print("OPTIMAL STRUCTURE DESIGN")
    print("=" * 80)
    print()

    structure = design_optimal_structure(analyses)
    plan = generate_implementation_plan(structure)

    print(f"Total categories: {plan['summary']['total_categories']}")
    print(f"Files from demonstrations/: {plan['summary']['files_from_demonstrations']}")
    print(f"Files from demos/: {plan['summary']['files_from_demos']}")
    print(f"TOTAL FINAL FILES: {plan['summary']['total_final_files']}")
    print()

    # Count files with SKIP_VALIDATION
    skip_count = sum(1 for a in analyses if "demonstrations/" in a["path"] and a["skip_validation"])
    print(f"Files with SKIP_VALIDATION in demonstrations/: {skip_count}")
    print()

    # Show structure summary
    print("CATEGORY BREAKDOWN:")
    print("-" * 80)
    for category, content in structure.items():
        demo_count = len(content["from_demonstrations"])
        demos_count = len(content["from_demos"])
        total = demo_count + demos_count
        print(f"\n{category}: {total} files")
        print(f"  Description: {content['description']}")
        print(f"  From demonstrations/: {demo_count}")
        print(f"  From demos/: {demos_count}")

    print()
    print("=" * 80)
    print()

    # Write structure to file
    output_file = ROOT / ".claude" / "optimal_structure.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "structure": structure,
                "implementation_plan": plan,
            },
            f,
            indent=2,
        )

    print(f"✓ Optimal structure saved: {output_file}")

    # Write human-readable plan
    plan_file = ROOT / ".claude" / "implementation_plan.md"
    with open(plan_file, "w") as f:
        f.write("# Example Consolidation Implementation Plan\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total categories:** {plan['summary']['total_categories']}\n")
        f.write(
            f"- **Files from demonstrations/:** {plan['summary']['files_from_demonstrations']}\n"
        )
        f.write(f"- **Files from demos/:** {plan['summary']['files_from_demos']}\n")
        f.write(f"- **Total final files:** {plan['summary']['total_final_files']}\n")
        f.write(f"- **Reduction:** {163 - plan['summary']['total_final_files']} files (from 163)\n")
        f.write("\n")

        f.write("## Phase 1: Keep demonstrations/ files (already good)\n\n")
        f.write(f"{len(plan['phase_1_keep_demonstrations'])} files - no action needed\n\n")

        f.write("## Phase 2: Migrate from demos/\n\n")
        f.write(f"{len(plan['phase_2_migrate_from_demos'])} files to migrate\n\n")
        for item in plan["phase_2_migrate_from_demos"]:
            f.write(f"- `{item['source']}` → `{item['target']}`\n")

        f.write("\n## Phase 3: Validation\n\n")
        f.write("1. Run `python demonstrations/validate_all.py`\n")
        f.write("2. Verify all examples pass\n")
        f.write("3. Check SKIP_VALIDATION markers are preserved\n\n")

        f.write("## Phase 4: Cleanup\n\n")
        for cleanup_item in plan["phase_4_cleanup"]:
            f.write(f"- {cleanup_item}\n")

    print(f"✓ Implementation plan saved: {plan_file}")
    print()


if __name__ == "__main__":
    main()
