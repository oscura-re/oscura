# Oscura Test Data

**Version**: 2.0  
**Date**: 2026-01-10  
**Structure**: Three-Tier Organization

---

## Overview

This directory contains all test data for Oscura, organized into three categories for optimal testing, minimal size, and legal compliance.

**Total Size**: ~80MB (version controlled)  
**Files**: Curated subset of representative test data  
**Status**: Sanitized (no proprietary terminology or vendor-specific references)

---

## Directory Structure

```
test_data/
├── synthetic/               # Generated test data (PRIMARY for unit tests)
│   ├── waveforms/          # Synthetic waveforms (basic, edge_cases, etc.)
│   ├── waveforms_legacy/   # Legacy waveform fallback files
│   ├── binary/             # Binary packet test data
│   ├── power_analysis/     # Power supply waveforms
│   ├── entropy/            # Entropy test files
│   ├── patterns/           # Pattern detection test files
│   ├── ground_truth/       # Expected results for validation
│   └── validation/         # Comprehensive validation test data

├── real_captures/          # Curated real-world validation (ANONYMIZED)
│   ├── waveforms/
│   │   ├── extended/       # Long captures (5-24MB, >1M samples)
│   │   ├── multichannel/   # 8-channel captures with crosstalk
│   │   ├── power/          # Power supply voltage/current with ripple
│   │   └── timing/         # Sub-microsecond precision tests
│   ├── protocols/
│   │   ├── startup_sequences/  # System initialization with jitter
│   │   ├── steady_state/       # Normal operation with drift
│   │   ├── error_conditions/   # Malformed/unknown commands
│   │   └── edge_cases/         # Incomplete, noisy, boundary cases
│   └── sessions/
│       └── tektronix_sessions/ # .tss session files

└── formats/                # Format compliance testing
    ├── tektronix/          # Tektronix WFM files (valid/invalid)
    ├── sigrok/             # 45 representative .sr files (UART, SPI, I2C, CAN, USB, JTAG)
    ├── pcap/               # Network packet captures
    ├── csv/                # CSV waveforms
    ├── hdf5/               # HDF5 waveforms
    ├── vcd/                # Verilog VCD files
    ├── wav/                # WAV audio files
    └── touchstone/         # Touchstone S-parameter files
```

---

## Usage by Test Type

### Unit Tests

- **Primary source**: `synthetic/` directory
- **Purpose**: Fast, reproducible, edge-case coverage
- **Examples**: Synthetic waveforms, digital signals, protocol packets

### Integration Tests

- **Primary source**: `formats/` directory
- **Purpose**: Multi-format loading, parser validation
- **Examples**: Tektronix WFM, sigrok .sr, PCAP files

### Validation Tests

- **Primary source**: `real_captures/` directory
- **Purpose**: Real-world edge cases, stress testing
- **Examples**: Extended captures, multichannel crosstalk, startup sequences

---

## File Naming Conventions

All files follow generic, descriptive naming:

- **Waveforms**: `extended_24mb_01.wfm`, `multichannel_01.wfm`, `power_supply_01.wfm`
- **Protocols**: `startup_seq_01.tss`, `steady_state_01.tss`, `protocol_error_01.wfm`
- **Sessions**: `session_large_01.tss`, `session_multiiter_01.tss`
- **Edge cases**: `noise_test_01.tss`, `partial_channel_01.wfm`

**No proprietary terminology**: All names are sanitized and generic.

---

## Real Captures Summary

16 curated files (~54MB) selected for:

- **Timing diversity**: Jitter, clock drift, sub-microsecond precision
- **Channel diversity**: Single, multi-channel, crosstalk scenarios
- **Protocol diversity**: Startup, steady-state, error conditions
- **Size diversity**: 26KB to 24MB (stress testing)

All real captures are anonymized with no system-specific references.

---

## Sigrok Files

45 representative files (~17MB) covering:

- **UART**: 10 files (hello world, errors, MIDI, device init)
- **SPI**: 10 files (all modes, flash commands, NES gamepad)
- **I2C**: 10 files (EEPROM, RTC, potentiometer, Wii nunchuk)
- **CAN**: 5 files (bus loads, standard/extended messages)
- **USB**: 5 files (HID, CDC, serial over HID)
- **JTAG**: 5 files (cJTAG, ARM debugging, init sequences)

Reduced from 605 files (110MB) to 45 files (17MB) - 94% reduction.

---

## Generated Format Files

Minimal test files for format compliance:

- **CSV**: 2 files (single-channel, multi-channel waveforms)
- **HDF5**: 1 file (test waveform with metadata)
- **VCD**: 1 file (simple digital logic)
- **WAV**: 1 file (440Hz sine wave audio)

---

## Maintenance

### Adding New Test Data

1. **Synthetic data**: Add to `synthetic/` with descriptive names
2. **Real captures**: Must be anonymized, <5MB, curated for specific test case
3. **Format files**: Add minimal representative files to `formats/`

### Regenerating Synthetic Data

```bash
uv run python scripts/test-data/generate_comprehensive_test_data.py
```

### Validation

All test data must:

- Use generic naming (no proprietary terms)
- Be <100MB total size (tracked files)
- Include ground truth files where applicable
- Be documented in this README

---

## License and Attribution

All synthetic data is generated by Oscura and is unrestricted.

Real captures are sanitized excerpts from anonymous sources, curated for testing purposes only.

Sigrok captures are from the [sigrok project](https://sigrok.org/wiki/Example_dumps) (public domain test data).

---

## Version History

- **2.0** (2026-01-10): Three-tier reorganization, sanitization, size optimization
- **1.0** (2025-12-24): Initial test data organization
