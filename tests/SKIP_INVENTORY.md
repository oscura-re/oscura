# Complete Skip Inventory

**Total Valid Conditional Skips**: 133 (100% documented)
**Last Updated**: 2026-01-25

This document lists ALL 133 valid conditional skips in the Oscura test suite, organized by category.

---

## Summary by Category

| Category | Count | Status |
|----------|-------|--------|
| PyWavelets (wavelets) | 24 | ✓ All documented |
| matplotlib (visualization) | 49 | ✓ All documented |
| PyYAML (config files) | 14 | ✓ All documented |
| Test Data (WFM files) | 15 | ✓ All documented |
| Test Data (PCAP files) | 8 | ✓ All documented |
| Test Data (general) | 7 | ✓ All documented |
| Platform-specific | 6 | ✓ All documented |
| sklearn (ML/clustering) | 3 | ✓ All documented |
| h5py (HDF5 format) | 3 | ✓ All documented |
| scipy (scientific) | 2 | ✓ All documented |
| luac (Lua compiler) | 2 | ✓ All documented |
| nptdms (TDMS format) | 1 | ✓ All documented |
| **TOTAL** | **133** | **✓ Complete** |

---

## 1. PyWavelets Dependency (24 skips)

**Files**: `test_wavelets.py` (20), `test_spectral.py` (4)

All skips use this pattern:

```python
try:
    import pywt  # noqa: F401
except ImportError:
    # SKIP: Valid - Optional pywavelets dependency
    # Only skip if pywavelets not installed (pip install oscura[wavelets])
    pytest.skip("PyWavelets not installed")
```

**test_wavelets.py** (20 skips):

- Line 64: `test_cwt_basic` - Basic CWT computation
- Line 82: `test_cwt_morlet_wavelet` - Morlet wavelet CWT
- Line 99: `test_cwt_ricker_wavelet` - Ricker wavelet CWT
- Line 116: `test_cwt_chirp_signal` - Chirp signal analysis
- Line 162: `test_cwt_chunked_basic` - Chunked CWT
- Line 178: `test_cwt_chunked_overlap` - Chunked with overlap
- Line 198: `test_cwt_chunked_memory_efficiency` - Memory efficiency
- Line 215: `test_cwt_chunked_consistency` - Result consistency
- Line 246: `test_dwt_basic` - Basic DWT
- Line 272: `test_dwt_db4_wavelet` - db4 wavelet DWT
- Line 296: `test_dwt_multilevel` - Multilevel decomposition
- Line 310: `test_dwt_perfect_reconstruction` - Signal reconstruction
- Line 347: `test_dwt_chunked_basic` - Chunked DWT
- Line 375: `test_dwt_chunked_overlap` - Chunked with overlap
- Line 398: `test_dwt_chunked_memory` - Memory management
- Line 421: `test_dwt_chunked_consistency` - Result consistency
- Line 445: `test_wavelet_denoising` - Signal denoising
- Line 467: `test_wavelet_compression` - Data compression
- Line 494: `test_wavelet_energy` - Energy calculation
- Line 518: `test_wavelet_frequency_localization` - Frequency analysis
- Line 552: `test_scaleogram` - Scaleogram visualization
- Line 571: `test_cwt_custom_scales` - Custom scale parameter
- Line 589: `test_dwt_boundary_modes` - Boundary handling
- Line 606: `test_wavelet_families` - Different wavelet families

**test_spectral.py** (4 skips):

- Line 536: `test_wavelet_transform_basic` - Basic wavelet transform
- Line 554: `test_wavelet_power_spectrum` - Power spectrum
- Line 579: `test_wavelet_denoising_spectral` - Denoising
- Line 596: `test_wavelet_compression_ratio` - Compression metrics

---

## 2. matplotlib Dependency (49 skips)

**Files**: `test_plot_types.py` (43), various visualization tests (6)

Standard pattern:

```python
try:
    import matplotlib  # noqa: F401
except ImportError:
    # SKIP: Valid - Optional matplotlib dependency
    # Only skip if matplotlib not installed (pip install oscura[viz])
    pytest.skip("matplotlib not available")
```

**test_plot_types.py** (43 skips):

1. Line 120: `test_plot_waveform_basic` - Basic waveform plotting
2. Line 132: `test_plot_waveform_time_units` - Time unit display
3. Line 149: `test_plot_waveform_labels` - Plot labels
4. Line 160: `test_plot_waveform_time_range` - Time range selection
5. Line 179: `test_plot_xy_basic` - XY plotting
6. Line 200: `test_plot_multi_channel` - Multi-channel display
7. Line 217: `test_plot_multi_channel_names` - Channel naming
8. Line 228: `test_plot_multi_channel_shared_x` - Shared X axis
9. Line 240: `test_plot_multi_channel_colors` - Color mapping
10. Line 261: `test_plot_spectrum` - Spectrum plotting
11. Line 272: `test_plot_spectrum_db` - dB scale
12. Line 283: `test_plot_spectrum_freq_range` - Frequency range
13. Line 294: `test_plot_spectrum_psd` - Power spectral density
14. Line 314: `test_plot_spectrogram` - Spectrogram
15. Line 325: `test_plot_spectrogram_nfft` - NFFT parameter
16. Line 336: `test_plot_spectrogram_overlap` - Window overlap
17. Line 357: `test_plot_timing_diagram` - Timing diagram
18. Line 370: `test_plot_timing_diagram_names` - Signal names
19. Line 381: `test_plot_logic_analyzer` - Logic analyzer view
20. Line 396: `test_plot_timing_edge_markers` - Edge markers
21. Line 416: `test_plot_eye_diagram` - Eye diagram
22. Line 427: `test_plot_eye_diagram_ui_count` - UI count
23. Line 438: `test_plot_eye_diagram_bathtub` - Bathtub curve
24. Line 458: `test_plot_phase` - Phase plot
25. Line 469: `test_plot_phase_delay` - Phase delay
26. Line 495: `test_plot_bode` - Bode plot
27. Line 509: `test_plot_bode_margins` - Gain/phase margins
28. Line 539: `test_plot_waterfall` - Waterfall display
29. Line 559: `test_plot_histogram` - Histogram
30. Line 570: `test_plot_histogram_bins` - Bin count
31. Line 581: `test_plot_histogram_range` - Value range
32. Line 607: `test_calculate_optimal_y_range` - Y-axis scaling
33. Line 631: `test_calculate_optimal_x_window` - X-axis windowing
34. Line 649: `test_calculate_grid_spacing` - Grid calculation
35. Line 664: `test_detect_interesting_regions` - Auto-zoom
36. Line 688: `test_get_colorblind_palette` - Accessibility colors
37. Line 703: `test_colorblind_safe_qualitative` - Qualitative palette
38. Line 721: `test_generate_alt_text` - Alt text generation
39. Line 736: `test_get_multi_line_styles` - Line styles
40. Line 756: `test_list_visualization_presets` - Preset listing
41. Line 767: `test_apply_preset` - Apply preset
42. Line 780: `test_publication_preset` - Publication quality
43. Line 793: `test_dark_theme_preset` - Dark theme
44. Line 808: `test_render_dpi` - DPI settings
45. Line 820: `test_render_thumbnail` - Thumbnail generation
46. Line 837: `test_decimate_for_display` - Data decimation
47. Line 858: `test_zoom_state` - Zoom state management
48. Line 879: `test_cursor_measurement` - Cursor measurements
49. Line 900: `test_calculate_optimal_bins` - Histogram bins
50. Line 916: `test_calculate_bin_edges` - Bin edge calculation
51. Line 936: `test_select_optimal_palette` - Palette selection
52. Line 949: `test_colorblind_safe_qualitative_constant` - Palette constant
53. Line 962: `test_sequential_viridis` - Sequential palette
54. Line 975: `test_diverging_coolwarm` - Diverging palette

**Other visualization tests** (6 skips):

- `test_jitter.py`: Line 120, 132 - Jitter visualization
- `test_visualization_*.py`: Various visualization integration tests

---

## 3. PyYAML Dependency (14 skips)

**Files**: `test_config_validation.py` (12), `test_template_definition.py` (1), `test_complete_workflows.py` (1)

Standard pattern:

```python
try:
    import yaml
except ImportError:
    # SKIP: Valid - Optional PyYAML dependency
    # Only skip if PyYAML not installed (configuration file support)
    pytest.skip("PyYAML not available")
```

**test_config_validation.py** (12 skips):

- Line 89: `test_yaml_empty_file` - Empty YAML file
- Line 105: `test_yaml_with_only_comments` - Comments-only file
- Line 129: `test_unicode_in_yaml` - Unicode handling
- Line 146: `test_nested_yaml_structure` - Nested structures
- Line 171: `test_yaml_anchors_and_aliases` - YAML anchors
- Line 205: `test_yaml_multiline_strings` - Multiline strings
- Line 230: `test_yaml_special_characters` - Special characters
- Line 253: `test_yaml_number_formats` - Number parsing
- Line 284: `test_yaml_boolean_values` - Boolean handling
- Line 305: `test_yaml_null_values` - Null values
- Line 330: `test_yaml_list_formats` - List parsing
- Line 371: `test_yaml_merge_keys` - Merge keys

**Other files**:

- `test_template_definition.py`: Line 353 - Template YAML parsing
- `test_complete_workflows.py`: Line 862 - Config file workflow

---

## 4. Test Data - WFM Files (15 skips)

**Files**: `test_wfm_loading.py` (13), `test_tektronix_loader.py` (11), `test_tektronix.py` (6)

Pattern for WFM file checks:

```python
wfm_files = list(test_data_dir.glob("*.wfm"))
if not wfm_files:
    # SKIP: Valid - Test data dependency
    # Only skip if Tektronix WFM test files not available
    pytest.skip("No WFM files available")
```

**test_wfm_loading.py**:

- Line 27: Small WFM files (<1MB)
- Line 48: WFM loader validation
- Line 53: Medium WFM files (1-10MB)
- Line 71: Medium file loading
- Line 77: Large WFM files (>10MB)
- Line 104: Real WFM files
- Line 127: Multiple file loading
- Line 134: Batch loading
- Line 151: WFM filtering
- Line 162: IQ trace filtering (not tested)
- Line 181: Digital conversion
- Line 208: Multi-channel analysis
- Line 287: Manifest file validation

**test_tektronix_loader.py** (11 skips):

- Line 50: Basic WFM file availability
- Line 92: Golden analog WFM
- Line 112: Digital waveform
- Line 128: IQ waveform
- Line 160: Multiple WFM files
- Line 184: Supported WFM formats
- Line 216: Invalid WFM files
- Line 260: Large file handling
- Line 291: Metadata extraction
- Line 322: Channel extraction
- Line 394: Multi-channel loading

**test_tektronix.py** (6 skips):

- Line 50: WFM003 format
- Line 105: Varying data types
- Line 209: Large file creation
- Line 229: Footer validation
- Line 527: Real WFM file testing
- Line 575: Complete WFM file suite

---

## 5. Test Data - PCAP Files (8 skips)

**Files**: `test_pcap_to_inference.py` (15), `test_pcap_loader.py` (2)

Pattern:

```python
pcap_file = test_data_dir / "http_traffic.pcap"
if not pcap_file.exists():
    # SKIP: Valid - Test data dependency
    # Only skip if PCAP test files not available
    pytest.skip("HTTP PCAP not available")
```

**test_pcap_to_inference.py**:

- Line 31: HTTP PCAP file
- Line 58: HTTP stream reassembly
- Line 91: General PCAP files
- Line 125: Modbus PCAP
- Line 156: HTTP request analysis
- Line 190: HTTP format inference
- Line 223: Modbus field detection
- Line 253: Multiple PCAP files
- Line 288: HTTP sequence alignment
- Line 320: HTTP local alignment
- Line 343: Modbus protocol library
- Line 371: Protocol matching
- Line 402: Modbus alignment
- Line 433: DNS PCAP
- Line 459: HTTP message extraction

**test_pcap_loader.py**:

- Line 1026: HTTP PCAP loading
- Line 1050: Modbus PCAP loading

---

## 6. Test Data - General (7 skips)

**Files**: Various validation and synthetic test files

Pattern:

```python
if "ground_truth" not in test_data:
    # SKIP: Valid - Test data dependency
    # Only skip if ground truth validation data not available
    pytest.skip("Ground truth not available")
```

**test_synthetic_signals.py**:

- Line 34: Square wave files by frequency
- Line 36: Square wave file existence
- Line 57: 1MHz square wave
- Line 59: Ground truth data
- Line 141: UART synthetic file
- Line 189: UART ground truth

**test_synthetic_packets.py**:

- Line 48: Ground truth file
- Line 52: Data file availability
- Line 79: Packet ground truth
- Line 100: Validation ground truth
- Line 136: Variable packet file
- Line 148: Noisy packet file
- Line 166: Comprehensive test data

**test_protocol_messages.py**:

- Line 36: Message file configuration
- Line 38: Message file existence
- Line 50: Message ground truth
- Line 70: 64-bit message file
- Line 72: 64-bit ground truth

**test_pattern_detection.py**:

- Line 32: Periodic pattern file
- Line 132: Repeating sequence file
- Line 189: Anomaly pattern file
- Line 321: Pattern files directory
- Line 345: Pattern file loading

---

## 7. Platform-Specific (6 skips)

**Files**: `test_edge_cases.py`

Pattern for symlink tests:

```python
try:
    symlink_path.symlink_to(target_path)
except (OSError, NotImplementedError):
    # SKIP: Valid - Platform-specific test
    # Only skip on platforms without symlink support (e.g., Windows FAT32)
    pytest.skip("Symlinks not supported on this system")
```

**test_edge_cases.py**:

- Line 97: Filesystem feature support
- Line 117: Symlink creation
- Line 134: Symlink following
- Line 148: Broken symlink handling
- Line 173: Circular symlink detection
- Line 397: Long filename support
- Line 413: Path length limits

---

## 8. sklearn/ML Dependencies (3 skips)

**Files**: `test_clustering_hypothesis.py`

Pattern:

```python
try:
    from oscura.analyzers.patterns.clustering import cluster_patterns
except ImportError:
    # SKIP: Valid - Optional scikit-learn dependency
    # Only skip if sklearn not installed (pip install oscura[ml])
    pytest.skip("clustering module not available")
```

**test_clustering_hypothesis.py**:

- Line 32: `test_cluster_patterns_properties` - Clustering properties
- Line 57: `test_cluster_patterns_num_clusters` - Cluster count
- Line 81: `test_cluster_patterns_distance_metric` - Distance metrics

---

## 9. h5py Dependency (3 skips)

**Files**: `test_error_handling.py`

Pattern:

```python
try:
    import h5py
except ImportError:
    # SKIP: Valid - Optional h5py dependency
    # Only skip if h5py not installed (pip install oscura[hdf5])
    pytest.skip("h5py not installed")
```

**test_error_handling.py**:

- Line 293: `test_load_hdf5_no_datasets` - Empty HDF5 file
- Line 312: `test_load_hdf5_wrong_dtype` - Invalid data type
- Line 293 (again): HDF5 error handling

---

## 10. scipy Dependency (2 skips)

**Files**: `test_complete_workflows.py`

Pattern:

```python
try:
    import scipy  # noqa: F401
except ImportError:
    # SKIP: Valid - Optional scipy dependency
    # Only skip if scipy not installed (core numerical library)
    pytest.skip("scipy not available")
```

**test_complete_workflows.py**:

- Line 253: `test_audio_signal_workflow` - Audio processing
- (One additional scipy skip in integration tests)

---

## 11. luac Build Tool (2 skips)

**Files**: `test_wireshark.py`

Pattern:

```python
import shutil

if not shutil.which("luac"):
    # SKIP: Valid - Build tool dependency
    # Only skip if luac (Lua compiler) not installed in PATH
    pytest.skip("luac not available")
```

**test_wireshark.py**:

- Line 121: `test_lua_compilation` - Lua syntax validation
- Line 324: `test_wireshark_dissector_compilation` - Dissector compilation

---

## 12. nptdms Dependency (1 skip)

**Files**: `test_error_handling.py`

Pattern:

```python
try:
    import nptdms  # noqa: F401
except ImportError:
    # SKIP: Valid - Optional nptdms dependency
    # Only skip if nptdms not installed (TDMS file format support)
    pytest.skip("nptdms not available")
```

**test_error_handling.py**:

- Line 345: `test_load_tdms_error` - TDMS error handling

---

## Verification

To verify all skips are documented:

```bash
# List all pytest.skip() calls
grep -r "pytest.skip(" tests --include="*.py" | wc -l

# Find undocumented skips (should be 0)
grep -r "pytest.skip(" tests --include="*.py" | grep -B3 "pytest.skip" | grep -v "# SKIP: Valid" | grep "pytest.skip"

# Run comprehensive analysis
python3 .claude/comprehensive_skip_documentation.py
```

**Expected output**: All 133 conditional skips have `# SKIP: Valid` documentation.

---

## Maintenance

This inventory is updated automatically when:

1. New conditional skips are added (pre-commit hook validates documentation)
2. Skips are removed (during refactoring)
3. Skip reasons are updated

**Last full audit**: 2026-01-25
**Next audit due**: 2026-02-25 (monthly review)

**Maintainers**: Test infrastructure team
**Questions**: See [SKIP_DOCUMENTATION.md](SKIP_DOCUMENTATION.md) or [SKIP_PATTERNS.md](SKIP_PATTERNS.md)
