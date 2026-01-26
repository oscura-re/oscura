# Example Consolidation Implementation Plan

## Summary

- **Total categories:** 20
- **Files from demonstrations/:** 112
- **Files from demos/:** 40
- **Total final files:** 152
- **Reduction:** 11 files (from 163)

## Phase 1: Keep demonstrations/ files (already good)

112 files - no action needed

## Phase 2: Migrate from demos/

40 files to migrate

- `demos/02_file_format_io/vcd_loader_demo.py` → `demonstrations/01_data_loading/vcd_advanced_demo.py`
- `demos/03_custom_daq/simple_loader.py` → `demonstrations/01_data_loading/custom_loader_simple.py`
- `demos/03_custom_daq/chunked_loader.py` → `demonstrations/01_data_loading/custom_loader_chunked.py`
- `demos/03_custom_daq/optimal_streaming_loader.py` → `demonstrations/01_data_loading/custom_loader_streaming.py`
- `demos/01_waveform_analysis/comprehensive_wfm_analysis.py` → `demonstrations/02_basic_analysis/waveform_comprehensive.py`
- `demos/11_mixed_signal/comprehensive_mixed_signal_demo.py` → `demonstrations/02_basic_analysis/mixed_signal.py`
- `demos/12_spectral_compliance/comprehensive_spectral_demo.py` → `demonstrations/02_basic_analysis/spectral_compliance.py`
- `demos/04_serial_protocols/i2s_demo.py` → `demonstrations/03_protocol_decoding/i2s.py`
- `demos/04_serial_protocols/jtag_demo.py` → `demonstrations/03_protocol_decoding/jtag.py`
- `demos/04_serial_protocols/manchester_demo.py` → `demonstrations/03_protocol_decoding/manchester.py`
- `demos/04_serial_protocols/onewire_demo.py` → `demonstrations/03_protocol_decoding/onewire.py`
- `demos/04_serial_protocols/swd_demo.py` → `demonstrations/03_protocol_decoding/swd.py`
- `demos/04_serial_protocols/usb_demo.py` → `demonstrations/03_protocol_decoding/usb.py`
- `demos/05_protocol_decoding/comprehensive_protocol_demo.py` → `demonstrations/03_protocol_decoding/protocol_comprehensive.py`
- `demos/06_udp_packet_analysis/comprehensive_udp_analysis.py` → `demonstrations/03_protocol_decoding/udp_analysis.py`
- `demos/13_jitter_analysis/ddj_dcd_demo.py` → `demonstrations/04_advanced_analysis/jitter_ddj_dcd.py`
- `demos/13_jitter_analysis/bathtub_curve_demo.py` → `demonstrations/04_advanced_analysis/jitter_bathtub.py`
- `demos/14_power_analysis/dcdc_efficiency_demo.py` → `demonstrations/04_advanced_analysis/power_dcdc.py`
- `demos/14_power_analysis/ripple_analysis_demo.py` → `demonstrations/04_advanced_analysis/power_ripple.py`
- `demos/15_signal_integrity/tdr_impedance_demo.py` → `demonstrations/04_advanced_analysis/signal_integrity_tdr.py`
- `demos/15_signal_integrity/setup_hold_timing_demo.py` → `demonstrations/04_advanced_analysis/signal_integrity_timing.py`
- `demos/15_signal_integrity/sparams_demo.py` → `demonstrations/04_advanced_analysis/signal_integrity_sparams.py`
- `demos/08_automotive_protocols/lin_demo.py` → `demonstrations/05_domain_specific/automotive_lin.py`
- `demos/08_automotive_protocols/flexray_demo.py` → `demonstrations/05_domain_specific/automotive_flexray.py`
- `demos/09_automotive/comprehensive_automotive_demo.py` → `demonstrations/05_domain_specific/automotive_comprehensive.py`
- `demos/16_emc_compliance/comprehensive_emc_demo.py` → `demonstrations/05_domain_specific/emc_comprehensive.py`
- `demos/10_timing_measurements/ieee_181_pulse_demo.py` → `demonstrations/05_domain_specific/timing_ieee181.py`
- `demos/07_protocol_inference/crc_reverse_demo.py` → `demonstrations/06_reverse_engineering/crc_reverse.py`
- `demos/07_protocol_inference/state_machine_learning.py` → `demonstrations/06_reverse_engineering/state_machine_learning.py`
- `demos/07_protocol_inference/wireshark_dissector_demo.py` → `demonstrations/06_reverse_engineering/wireshark_dissector.py`
- `demos/17_signal_reverse_engineering/comprehensive_re.py` → `demonstrations/06_reverse_engineering/re_comprehensive.py`
- `demos/17_signal_reverse_engineering/exploratory_analysis.py` → `demonstrations/06_reverse_engineering/re_exploratory.py`
- `demos/17_signal_reverse_engineering/reverse_engineer_tool.py` → `demonstrations/06_reverse_engineering/re_tool.py`
- `demos/18_advanced_inference/bayesian_inference_demo.py` → `demonstrations/06_reverse_engineering/inference_bayesian.py`
- `demos/18_advanced_inference/protocol_dsl_demo.py` → `demonstrations/06_reverse_engineering/inference_dsl.py`
- `demos/18_advanced_inference/active_learning_demo.py` → `demonstrations/06_reverse_engineering/inference_active_learning.py`
- `demos/01_waveform_analysis/all_output_formats.py` → `demonstrations/15_export_visualization/all_export_formats.py`
- `demos/19_complete_workflows/automotive_full_workflow.py` → `demonstrations/16_complete_workflows/automotive_workflow.py`
- `demos/19_complete_workflows/network_analysis_workflow.py` → `demonstrations/16_complete_workflows/network_workflow.py`
- `demos/19_complete_workflows/unknown_signal_workflow.py` → `demonstrations/16_complete_workflows/unknown_signal_workflow.py`

## Phase 3: Validation

1. Run `python demonstrations/validate_all.py`
2. Verify all examples pass
3. Check SKIP_VALIDATION markers are preserved

## Phase 4: Cleanup

- Delete all content from demos/ directory
- Remove demos/common/
- Remove demos/data_generation/
- Update CONTRIBUTING.md to reference only demonstrations/
- Update demonstrations/README.md with new structure
