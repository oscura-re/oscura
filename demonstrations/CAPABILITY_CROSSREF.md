# CAPABILITY-TO-DEMONSTRATION CROSS-REFERENCE

**Purpose:** Quick lookup to find which demonstration shows a specific Oscura capability

**Last Updated:** 2026-01-23

---

## HOW TO USE THIS REFERENCE

**Format:** `API Function/Class` → Demo file(s) that demonstrate it

**Symbols:**

- ✅ **Primary demo** - Main demonstration of capability
- 📝 **Mentioned** - Capability discussed but not primary focus
- ⚠️ **Minimal** - Partial/incomplete demonstration
- ❌ **Not demonstrated** - API exists but no demo

---

## DATA LOADING

### File Format Loaders

| Capability | Demo | Status |
|-----------|------|--------|
| `load_vcd()` | 02_logic_analyzers.py | ✅ Primary |
| `load_tdms()` | 01_oscilloscopes.py, 04_scientific_formats.py | ✅ Primary |
| `load_tektronix_wfm()` | 01_oscilloscopes.py | ✅ Primary |
| `load_rigol_wfm()` | 01_oscilloscopes.py | ✅ Primary |
| `load_wav()` | 04_scientific_formats.py | ✅ Primary |
| `load_hdf5()` | 04_scientific_formats.py | ✅ Primary |
| `load_npz()` | 04_scientific_formats.py | ✅ Primary |
| `load_can_log()` | 03_automotive_formats.py | 📝 Mentioned |
| `load_binary_packets()` | 05_custom_binary.py | ✅ Primary |
| `load_trace_lazy()` | 06_streaming_large_files.py | ⚠️ Minimal |
| `load_all_channels()` | 07_multi_channel.py | ✅ Primary |
| **`load_touchstone()`** | — | ❌ **Not demonstrated** |
| **`load_pcap()`** | — | ❌ **Not demonstrated** |
| **`load_chipwhisperer()`** | — | ❌ **Not demonstrated** |
| **`mmap_loader`** | — | ❌ **Not demonstrated** |
| **`load_sigrok()`** | 02_logic_analyzers.py | 📝 Mentioned only |
| **`load_csv()`, `load_json()`** | — | ❌ **Not demonstrated** |

---

## WAVEFORM MEASUREMENTS

### Timing Measurements

| Capability | Demo | Status |
|-----------|------|--------|
| `rise_time()` | 01_waveform_measurements.py, 03_signal_integrity.py | ✅ Primary |
| `fall_time()` | 01_waveform_measurements.py, 03_signal_integrity.py | ✅ Primary |
| `period()` | 01_waveform_measurements.py | ✅ Primary |
| `frequency()` | 01_waveform_measurements.py, 03_spectral_analysis.py | ✅ Primary |
| `pulse_width()` | 01_waveform_measurements.py, 05_triggering.py | ✅ Primary |
| `duty_cycle()` | 01_waveform_measurements.py | ✅ Primary |

### Amplitude Measurements

| Capability | Demo | Status |
|-----------|------|--------|
| `amplitude()` | 01_waveform_measurements.py | ✅ Primary |
| `mean()` | 01_waveform_measurements.py, 02_statistics.py | ✅ Primary |
| `rms()` | 01_waveform_measurements.py, 02_power_analysis.py | ✅ Primary |
| `overshoot()` | 01_waveform_measurements.py, 03_signal_integrity.py | ✅ Primary |
| `undershoot()` | 01_waveform_measurements.py, 03_signal_integrity.py | ✅ Primary |
| `preshoot()` | 01_waveform_measurements.py | ✅ Primary |
| `measure()` | 01_waveform_measurements.py | ✅ Primary |

---

## SPECTRAL ANALYSIS

### Frequency Domain

| Capability | Demo | Status |
|-----------|------|--------|
| `fft()` | 03_spectral_analysis.py | ✅ Primary |
| `psd()` | 03_spectral_analysis.py | ✅ Primary |
| `spectrogram()` | 03_spectral_analysis.py | 📝 Mentioned |
| `thd()` | 03_spectral_analysis.py, 02_dsl_syntax.py | ✅ Primary |
| `snr()` | 03_spectral_analysis.py, 06_quality_assessment.py | ✅ Primary |
| `sinad()` | 03_spectral_analysis.py | ✅ Primary |
| `sfdr()` | 03_spectral_analysis.py | ✅ Primary |
| `enob()` | 03_spectral_analysis.py | ✅ Primary |
| **`cwt()` (Continuous Wavelet)** | — | ❌ **Not demonstrated** |
| **`dwt()` (Discrete Wavelet)** | — | ❌ **Not demonstrated** |
| **`idwt()` (Inverse DWT)** | — | ❌ **Not demonstrated** |
| **`chunked_fft()`** | — | ❌ **Not demonstrated** |
| **`chunked_wavelet()`** | — | ❌ **Not demonstrated** |

---

## STATISTICAL ANALYSIS

### Basic Statistics

| Capability | Demo | Status |
|-----------|------|--------|
| `basic_stats()` | 02_statistics.py | ✅ Primary |
| `summary_stats()` | 02_statistics.py | ✅ Primary |
| `histogram()` | 02_statistics.py | ✅ Primary |
| `percentiles()` | 02_statistics.py | ✅ Primary |
| `quartiles()` | 02_statistics.py | ✅ Primary |
| `correlation_coefficient()` | 02_statistics.py | ✅ Primary |
| `autocorrelation()` | 02_statistics.py | ✅ Primary |

### Advanced Statistics

| Capability | Demo | Status |
|-----------|------|--------|
| **`shannon_entropy()`** | — | ❌ **Not demonstrated** |
| **`sliding_entropy()`** | — | ❌ **Not demonstrated** |
| **`entropy_transitions()`** | — | ❌ **Not demonstrated** |
| **`classify_data_type()`** | — | ❌ **Not demonstrated** |
| **`detect_encrypted_regions()`** | — | ❌ **Not demonstrated** |
| **`detect_compressed_regions()`** | — | ❌ **Not demonstrated** |
| **`extract_ngrams()`** | — | ❌ **Not demonstrated** |
| **`ngram_frequencies()`** | — | ❌ **Not demonstrated** |
| **`detect_checksum_fields()`** | 02_crc_recovery.py | ⚠️ Minimal |
| **`identify_checksum_algorithm()`** | 02_crc_recovery.py | ⚠️ Minimal |
| `detect_outliers()` | 02_statistics.py | ✅ Primary |

---

## PROTOCOL DECODERS

### Serial Protocols

| Capability | Demo | Status |
|-----------|------|--------|
| `decode_uart()` | 01_serial_comprehensive.py | ✅ Primary |
| `decode_spi()` | 01_serial_comprehensive.py | ✅ Primary |
| `decode_i2c()` | 01_serial_comprehensive.py | ✅ Primary |
| `decode_i2s()` | 05_encoded_protocols.py | ✅ Primary |
| `decode_onewire()` | 01_serial_comprehensive.py | ✅ Primary |

### Automotive Protocols

| Capability | Demo | Status |
|-----------|------|--------|
| `decode_can()` | 02_automotive_protocols.py, 01_automotive_diagnostics.py | ✅ Primary |
| `decode_can_fd()` | 02_automotive_protocols.py | ✅ Primary |
| `decode_lin()` | 02_automotive_protocols.py | ✅ Primary |
| `decode_flexray()` | 02_automotive_protocols.py | ✅ Primary |

### Debug Protocols

| Capability | Demo | Status |
|-----------|------|--------|
| `decode_jtag()` | 03_debug_protocols.py | ✅ Primary |
| `decode_swd()` | 03_debug_protocols.py | ✅ Primary |

### Other Protocols

| Capability | Demo | Status |
|-----------|------|--------|
| **`decode_usb()`** | 06_auto_detection.py | ⚠️ Auto-detect only |
| **`decode_hdlc()`** | — | ❌ **Not demonstrated** |
| `decode_manchester()` | 05_encoded_protocols.py | ✅ Primary |
| `decode_gpib()` | 04_parallel_bus.py | 📝 Mentioned |

---

## DIGITAL ANALYSIS

### Edge & Clock

| Capability | Demo | Status |
|-----------|------|--------|
| `detect_edges()` | 02_logic_analyzers.py, 01_jitter_analysis.py | ✅ Primary |
| `recover_clock()` | 03_debug_protocols.py | ⚠️ Minimal |
| `detect_baud_rate()` | 01_serial_comprehensive.py | ✅ Primary |
| `measure_clock_jitter()` | 01_jitter_analysis.py | ✅ Primary |
| `to_digital()` | 02_logic_analyzers.py | ✅ Primary |
| `detect_logic_family()` | 03_vintage_logic.py | ✅ Primary |
| **`detect_clock_frequency()`** | — | ❌ **Not demonstrated** |
| **`EdgeDetector` (advanced)** | — | ❌ **Not demonstrated** |

### Timing Analysis

| Capability | Demo | Status |
|-----------|------|--------|
| **`setup_time()`, `hold_time()`** | — | ❌ **Not demonstrated** |
| **`propagation_delay()`** | — | ❌ **Not demonstrated** |
| **`slew_rate()`** | 03_signal_integrity.py | ⚠️ Minimal |
| **`check_timing_constraints()`** | — | ❌ **Not demonstrated** |

---

## ADVANCED ANALYSIS

### Jitter & Eye Diagrams

| Capability | Demo | Status |
|-----------|------|--------|
| `period_jitter()` | 01_jitter_analysis.py | ✅ Primary |
| `cycle_to_cycle_jitter()` | 01_jitter_analysis.py | ✅ Primary |
| `tie_from_edges()` | 01_jitter_analysis.py | ✅ Primary |
| `decompose_jitter()` | 01_jitter_analysis.py | ✅ Primary |
| `generate_eye()` | 04_eye_diagrams.py | ✅ Primary |
| `eye_height()`, `eye_width()` | 04_eye_diagrams.py | ✅ Primary |
| `bathtub_curve()` | 04_eye_diagrams.py | ✅ Primary |
| `q_factor()` | 04_eye_diagrams.py | ✅ Primary |

### Power Analysis

| Capability | Demo | Status |
|-----------|------|--------|
| `average_power()` | 02_power_analysis.py | ✅ Primary |
| `apparent_power()` | 02_power_analysis.py | ✅ Primary |
| `power_factor()` | 02_power_analysis.py | ✅ Primary |
| `switching_loss()` | 02_power_analysis.py | ✅ Primary |
| `efficiency()` | 02_power_analysis.py | ✅ Primary |
| `ripple()` | 02_power_analysis.py | ✅ Primary |
| `soa_analysis()` | 02_power_analysis.py | ✅ Primary |

### Component Characterization

| Capability | Demo | Status |
|-----------|------|--------|
| **`extract_impedance()` (TDR)** | — | ❌ **Not demonstrated** |
| **`impedance_profile()`** | — | ❌ **Not demonstrated** |
| **`discontinuity_analysis()`** | — | ❌ **Not demonstrated** |
| **`measure_capacitance()`** | — | ❌ **Not demonstrated** |
| **`measure_inductance()`** | — | ❌ **Not demonstrated** |
| **`extract_parasitics()`** | — | ❌ **Not demonstrated** |
| **`characteristic_impedance()`** | — | ❌ **Not demonstrated** |
| **`velocity_factor()`** | — | ❌ **Not demonstrated** |
| **`transmission_line_analysis()`** | — | ❌ **Not demonstrated** |

---

## FILTERING

| Capability | Demo | Status |
|-----------|------|--------|
| `low_pass()` | 04_filtering.py | ✅ Primary |
| `high_pass()` | 04_filtering.py | ✅ Primary |
| `band_pass()` | 04_filtering.py | ✅ Primary |
| `band_stop()` | 04_filtering.py | ✅ Primary |
| `ButterworthFilter` | 04_filtering.py | ✅ Primary |
| `ChebyshevType1Filter` | 04_filtering.py | ✅ Primary |
| `notch_filter()` | 04_filtering.py | ✅ Primary |
| `savgol_filter()` | 04_filtering.py | ✅ Primary |

---

## TRIGGERING

| Capability | Demo | Status |
|-----------|------|--------|
| `EdgeTrigger` | 05_triggering.py | ✅ Primary |
| `PatternTrigger` | 05_triggering.py | ✅ Primary |
| `PulseWidthTrigger` | 05_triggering.py | ✅ Primary |
| `WindowTrigger` | 05_triggering.py | ✅ Primary |
| `find_glitches()` | 05_triggering.py | ✅ Primary |

---

## MATH OPERATIONS

| Capability | Demo | Status |
|-----------|------|--------|
| `add()`, `subtract()` | 06_math_operations.py | ✅ Primary |
| `multiply()`, `divide()` | 06_math_operations.py | ✅ Primary |
| `differentiate()` | 06_math_operations.py | ✅ Primary |
| `integrate()` | 06_math_operations.py | ✅ Primary |
| `interpolate()` | 06_math_operations.py | ✅ Primary |

---

## REVERSE ENGINEERING

### Protocol Inference

| Capability | Demo | Status |
|-----------|------|--------|
| `reverse_engineer_protocol()` | 01_unknown_protocol.py | ✅ Primary |
| `infer_fields()` | 04_field_inference.py | ✅ Primary |
| `detect_delimiter()` | 04_field_inference.py | ✅ Primary |
| `find_message_boundaries()` | 04_field_inference.py | ✅ Primary |
| `identify_checksum_algorithm()` | 02_crc_recovery.py | ✅ Primary |
| `find_repeating_sequences()` | 05_pattern_discovery.py | ✅ Primary |

### State Machine & Pattern

| Capability | Demo | Status |
|-----------|------|--------|
| `infer_state_machine()` | 03_state_machines.py | ✅ Primary |
| `detect_period()` | 05_pattern_discovery.py | ✅ Primary |
| `cluster_payloads()` | 05_pattern_discovery.py | ⚠️ Minimal |

### Signal Classification

| Capability | Demo | Status |
|-----------|------|--------|
| **`classify_signal()`** | — | ❌ **Not demonstrated** |
| **`characterize_unknown_signal()`** | 01_unknown_signals.py | ⚠️ Minimal |
| **`find_anomalies()`** | — | ❌ **Not demonstrated** |
| **`assess_data_quality()`** | 06_quality_assessment.py | ⚠️ Minimal |

---

## AUTOMOTIVE

| Capability | Demo | Status |
|-----------|------|--------|
| `CANSession` | 02_can_session.py, 01_automotive_diagnostics.py | ✅ Primary |
| `DTCDatabase.lookup()` | 01_automotive_diagnostics.py | ✅ Primary |
| `decode_j1939()` | 01_automotive_diagnostics.py | ✅ Primary |
| `decode_uds()` | 01_automotive_diagnostics.py | ✅ Primary |
| `plot_bus_timeline()` | 01_automotive_diagnostics.py | ✅ Primary |

---

## SIDE-CHANNEL

| Capability | Demo | Status |
|-----------|------|--------|
| `DPAAnalyzer` | 04_side_channel.py | ✅ Primary |
| `CPAAnalyzer` | 04_side_channel.py | ✅ Primary |
| `TimingAnalyzer` | 04_side_channel.py | ✅ Primary |
| `hamming_weight()` | 04_side_channel.py | ✅ Primary |

---

## SIGNAL GENERATION

| Capability | Demo | Status |
|-----------|------|--------|
| `SignalBuilder` | 01_signal_builder_comprehensive.py, 02_dsl_syntax.py | ✅ Primary |
| `.add_sine()` | 01_signal_builder_comprehensive.py | ✅ Primary |
| `.add_square()` | 01_signal_builder_comprehensive.py | ✅ Primary |
| `.add_uart()` | 02_protocol_generation.py | ✅ Primary |
| `.add_spi()` | 02_protocol_generation.py | ✅ Primary |
| `.add_noise()` | 01_signal_builder_comprehensive.py, 03_impairment_simulation.py | ✅ Primary |
| `.add_distortion()` | 03_impairment_simulation.py | ✅ Primary |

---

## BATCH PROCESSING

| Capability | Demo | Status |
|-----------|------|--------|
| `batch_analyze()` | 01_parallel_batch.py | ✅ Primary |
| `aggregate_results()` | 02_result_aggregation.py | ✅ Primary |
| `BatchLogger` | 03_progress_tracking.py | ✅ Primary |

---

## SESSIONS

| Capability | Demo | Status |
|-----------|------|--------|
| `AnalysisSession` | 01_analysis_session.py | ✅ Primary |
| `CANSession` | 02_can_session.py | ✅ Primary |
| `BlackBoxSession` | 03_blackbox_session.py | ✅ Primary |
| Session persistence | 04_session_persistence.py | ✅ Primary |

---

## PIPELINE & WORKFLOWS

| Capability | Demo | Status |
|-----------|------|--------|
| `Pipeline` | 01_pipeline_api.py | ✅ Primary |
| `compose()`, `pipe()` | 01_pipeline_api.py, 04_composition.py | ✅ Primary |
| `REPipeline` | 01_unknown_device_re.py | ✅ Primary |
| `reverse_engineer_signal()` | 01_unknown_device_re.py | ✅ Primary |
| `emc_compliance_test()` | 03_emc_testing.py | ✅ Primary |
| `power_analysis()` workflow | 02_power_analysis.py | ✅ Primary |

---

## STREAMING & PERFORMANCE

| Capability | Demo | Status |
|-----------|------|--------|
| `StreamingAnalyzer` | 06_streaming_api.py, 06_streaming_large_files.py | ✅ Primary |
| Parallel processing | 07_parallel_processing.py, 01_parallel_batch.py | ✅ Primary |
| **GPU acceleration (`gpu_backend`)** | — | ❌ **Not demonstrated** |

---

## VISUALIZATION

| Capability | Demo | Status |
|-----------|------|--------|
| `plot_waveform()` | 05_visualization_gallery.py | ✅ Primary |
| `plot_spectrum()` | 05_visualization_gallery.py, 03_spectral_analysis.py | ✅ Primary |
| `plot_eye()` | 04_eye_diagrams.py, 05_visualization_gallery.py | ✅ Primary |
| `plot_protocol_decode()` | 01_serial_comprehensive.py | ✅ Primary |
| `plot_logic_analyzer()` | 05_visualization_gallery.py | ✅ Primary |
| `plot_bathtub()` | 04_eye_diagrams.py | ✅ Primary |

---

## EXPORT

| Capability | Demo | Status |
|-----------|------|--------|
| `export_csv()` | 01_export_formats.py | ✅ Primary |
| `export_hdf5()` | 01_export_formats.py | ✅ Primary |
| `export_json()` | 01_export_formats.py | ✅ Primary |
| `export_mat()` | 01_export_formats.py | ✅ Primary |
| `export_wavedrom()` | 02_wavedrom_timing.py | ✅ Primary |
| `export_wireshark_dissector()` | 03_wireshark_dissectors.py, 06_wireshark_export.py | ✅ Primary |
| `generate_report()` | 04_report_generation.py | ✅ Primary |

---

## COMPARISON & TESTING

| Capability | Demo | Status |
|-----------|------|--------|
| `compare_to_golden()` | 01_golden_reference.py | ✅ Primary |
| `check_limits()` | 02_limit_testing.py | ✅ Primary |
| `mask_test()` | 03_mask_testing.py | ✅ Primary |
| `compare_traces()` | 04_regression_testing.py | ✅ Primary |

---

## QUALITY & COMPLIANCE

| Capability | Demo | Status |
|-----------|------|--------|
| `calculate_quality_score()` | 02_quality_scoring.py | ✅ Primary |
| `check_clipping()` | 03_warning_system.py | ✅ Primary |
| `EnsembleAggregator` | 01_ensemble_methods.py | ✅ Primary |
| `check_compliance()` (EMC) | 02_emc_compliance.py | ✅ Primary |
| IEEE 181 validation | 01_ieee_181.py | ✅ Primary |
| IEEE 1241 validation | 02_ieee_1241.py | ✅ Primary |
| IEEE 1459 validation | 03_ieee_1459.py | ✅ Primary |
| IEEE 2414 validation | 04_ieee_2414.py | ✅ Primary |

---

## INTEGRATION

| Capability | Demo | Status |
|-----------|------|--------|
| CLI usage | 01_cli_usage.py | ✅ Primary |
| Jupyter integration | 02_jupyter_notebooks.py | ✅ Primary |
| LLM integration | 03_llm_integration.py | ✅ Primary |
| Configuration files | 04_configuration_files.py | ✅ Primary |
| Hardware integration | 05_hardware_integration.py | ✅ Primary |

---

## GUIDANCE & RECOMMENDATIONS

| Capability | Demo | Status |
|-----------|------|--------|
| Smart recommendations | 01_smart_recommendations.py | ✅ Primary |
| Analysis wizards | 02_analysis_wizards.py | ✅ Primary |
| Onboarding helpers | 03_onboarding_helpers.py | ✅ Primary |
| Analysis recommendations | 04_recommendations.py | ✅ Primary |

---

## SUMMARY BY STATUS

### ✅ Well Demonstrated (78 capabilities)

- Core waveform measurements
- Spectral analysis (FFT, PSD, THD, SNR, SINAD, ENOB, SFDR)
- Protocol decoders (UART, SPI, I2C, CAN, LIN, JTAG, SWD)
- Filtering
- Triggering
- Power analysis (IEEE 1459)
- Jitter analysis (IEEE 2414)
- Eye diagrams
- Signal generation
- Reverse engineering workflows
- Export formats
- Visualization

### ⚠️ Minimally Demonstrated (15 capabilities)

- Lazy loading
- Auto-detect USB
- Clock recovery
- Signal characterization
- Data quality assessment
- Checksum detection
- Clustering

### ❌ Not Demonstrated (108 capabilities)

**Critical Gaps (requires Priority 0 demos):**

- Wavelet analysis (CWT, DWT)
- Entropy analysis
- Data classification
- TDR / component characterization
- Transmission line analysis

**Important Gaps (requires Priority 1 demos):**

- Specialized loaders (Touchstone, PCAP, ChipWhisperer)
- GPU acceleration
- Digital timing analysis
- Signal classification
- Anomaly detection
- Advanced search

---

## QUICK LOOKUP BY USE CASE

### "I want to reverse engineer a protocol"

→ `06_reverse_engineering/01_unknown_protocol.py`
→ `06_reverse_engineering/02_crc_recovery.py`
→ `06_reverse_engineering/04_field_inference.py`
→ `16_complete_workflows/01_unknown_device_re.py`

### "I want to analyze automotive diagnostics"

→ `05_domain_specific/01_automotive_diagnostics.py`
→ `16_complete_workflows/02_automotive_diagnostics.py`
→ `10_sessions/02_can_session.py`

### "I want to measure signal quality"

→ `02_basic_analysis/03_spectral_analysis.py` (THD, SNR, SINAD)
→ `04_advanced_analysis/06_quality_assessment.py`
→ `12_quality_tools/02_quality_scoring.py`

### "I want to analyze power consumption"

→ `04_advanced_analysis/02_power_analysis.py` (IEEE 1459)
→ `19_standards_compliance/03_ieee_1459.py`

### "I want to characterize high-speed digital signals"

→ `04_advanced_analysis/01_jitter_analysis.py` (IEEE 2414)
→ `04_advanced_analysis/04_eye_diagrams.py`
→ `19_standards_compliance/04_ieee_2414.py`

### "I want to test EMC compliance"

→ `05_domain_specific/02_emc_compliance.py`
→ `16_complete_workflows/03_emc_testing.py`

### "I want to perform side-channel attacks"

→ `05_domain_specific/04_side_channel.py` (DPA, CPA, timing)

### "I want to generate test signals"

→ `17_signal_generation/01_signal_builder_comprehensive.py`
→ `17_signal_generation/02_protocol_generation.py`

### "I want to load captures from my oscilloscope"

→ `01_data_loading/01_oscilloscopes.py` (Tektronix, Rigol, LeCroy, TDMS)
→ `01_data_loading/02_logic_analyzers.py` (Saleae, VCD)

### "I want production testing workflows"

→ `16_complete_workflows/04_production_testing.py`
→ `18_comparison_testing/01_golden_reference.py`
→ `18_comparison_testing/02_limit_testing.py`

---

**Last Updated:** 2026-01-23
**Total Capabilities Cataloged:** 201
**Demonstrated:** 78 (39%)
**Minimal/Mentioned:** 15 (7%)
**Not Demonstrated:** 108 (54%)
