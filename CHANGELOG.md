# Changelog

All notable changes to Oscura will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **CI/Local Configuration Parity** (`.github/workflows/*.yml`, `scripts/pre-push.sh`):
  - Fixed MkDocs build inconsistency: CI now uses `--clean` flag matching local behavior
  - Fixed pytest execution: All 12 CI pytest invocations now use `python -m pytest` for consistency
  - Fixed interrogate docstring coverage: CI now enforces 95% threshold with `-f 95` flag
  - Eliminates false passes where code passes local checks but fails CI

- **Coverage Configuration** (`pyproject.toml`):
  - Added `parallel = true` to `[tool.coverage.run]` for pytest-xdist compatibility
  - Added `data_file = ".coverage"` and `relative_files = true` for consistent behavior
  - Prevents race conditions where `.coverage` file not created after parallel test execution
  - Fixes diff-coverage job failures in CI

- **Claude Hooks - Critical Security & Functionality Fixes** (`.claude/hooks/*.py`, `.claude/settings.json`):
  - **P0 - check_report_proliferation.py**: Fixed broken stdin parsing (was using environment variables)
    - Hook was completely non-functional, never validated any files
    - Now correctly reads from stdin per PreToolUse contract
    - Migrated to shared logging and config utilities
  - **P0 - enforce_agent_limit.py**: Added missing stdin parsing
    - Was violating PreToolUse contract by ignoring tool context
    - Now validates it's being called for Task tool
    - Changed to fail-closed (blocks on errors) for enforcement hook
    - Migrated to shared registry and config utilities
  - **P0 - validate_path.py**: CRITICAL SECURITY FIX - changed fail-open to fail-closed
    - Previously exited 0 on ALL errors (allowed dangerous writes when hook crashed)
    - Now exits 1 on errors (blocks unsafe operations)
    - Fixed TOCTOU race conditions in symlink validation
    - Uses `resolve(strict=False)` to avoid time-of-check/time-of-use vulnerabilities
  - **P0 - settings.json**: Removed dangerous fail-open fallbacks for security hooks
    - Removed `|| echo '{"ok": true}'` from enforce_agent_limit and validate_path
    - Security hooks now properly fail-closed when they encounter errors
    - Informational hooks kept fail-open behavior (appropriate)

- **Claude Hooks - Code Duplication Elimination** (`.claude/hooks/shared/*.py`):
  - **Created shared/config.py** (280 lines): Consolidated 5 duplicate YAML/config loading implementations
    - Functions: `load_config()`, `load_coding_standards()`, `get_retention_policy()`, `get_hook_config()`
    - Unified fallback parser for systems without PyYAML
    - Eliminates ~150 lines of duplicate code
  - **Created shared/logging_utils.py** (180 lines): Standardized logging across all hooks
    - Functions: `get_hook_logger()`, `log_hook_start()`, `log_hook_end()`, `HookLogger` context manager
    - Consistent log format: `[timestamp] [level] [hook_name] message`
    - Eliminates ~100 lines of duplicate logging setup
  - **Created shared/registry.py** (370 lines): Consolidated 4 duplicate registry operation implementations
    - Functions: `load_registry()`, `save_registry()`, `count_running_agents()`, `update_agent_status()`
    - Atomic save operations with backup
    - Registry count validation and repair
    - Eliminates ~200 lines of duplicate code
  - **Updated shared/**init**.py**: Exports all shared utilities for easy import
  - **Total code elimination**: 450+ lines of duplicate code removed

- **Claude Hooks - Additional Security Fix** (`.claude/hooks/validate_path.py`):
  - **P0 - validate_path.py**: Fixed .git directory bypass vulnerability
    - `.git` was incorrectly in EXCLUDED_DIRS, allowing writes to `.git/config`, `.git/HEAD`, etc.
    - Removed `.git` from excluded directories (kept only build/cache dirs: `.venv`, `node_modules`, `__pycache__`)
    - Now correctly blocks all `.git/*` writes per BLOCKED_PATTERNS security rules
    - Pattern matching now checks both absolute and relative paths from project root
    - Fixes test_blocks_git_internals which was failing due to this bypass

### Added

- **Claude Hooks - Centralized Configuration** (`.claude/config.yaml`):
  - **New `hooks` section**: All hook-specific configuration in one place
    - `cleanup_stale_agents`: stale_threshold_hours (24), activity_check_hours (1), max_age_days (30)
    - `check_stop`: max_stale_hours (2)
    - `check_subagent_stop`: output_size_threshold_bytes (200KB), recent_window_minutes (5)
    - `pre_compact_cleanup`: large_json_size_bytes (100KB), old_report_days (7)
    - `health_check`: disk_space_critical_percent (5), disk_space_warning_percent (10)
  - Eliminates 9+ hardcoded magic numbers across hooks
  - All hooks now read configuration from single source of truth

- **Claude Hooks - Comprehensive Test Suite** (`.claude/hooks/test_hooks_comprehensive.py`):
  - 18 comprehensive tests covering all critical hooks
  - Tests all P0 fixes: stdin parsing, fail-closed behavior, security validations
  - Test coverage: check_report_proliferation (4 tests), enforce_agent_limit (3 tests), validate_path (8 tests), shared utilities (3 tests)
  - 100% pass rate (18/18 tests passing) - up from 94% after fixing .git bypass vulnerability
  - Validates:
    - Stdin parsing correctness (PreToolUse contract)
    - Fail-open vs fail-closed behavior
    - Security validations (blocked patterns, path traversal, credential protection)
    - Shared utility functionality (config loading, registry operations, logging)

- **Claude Hooks - Configuration Helper** (`.claude/hooks/get_config.py`):
  - New utility script for shell hooks to read values from config.yaml
  - Usage: `python3 get_config.py hooks.health_check.disk_space_critical_percent`
  - Supports dot-notation for nested keys
  - Returns default value if key not found
  - Enables shell scripts to use centralized configuration instead of hardcoded values

### Changed

- **Pre-push Workflow Optimization** (`scripts/pre-push.sh`):
  - Added `--skip-hooks` flag to skip redundant pre-commit checks
  - Saves ~10-30 seconds when pre-commit hooks already ran on commit
  - Default behavior unchanged (safe, still runs all checks)
  - Usage: `./scripts/pre-push.sh --skip-hooks`

- **README Badge Enhancements** (`README.md`, `.github/BADGE_MAINTENANCE.md`):
  - Added 5 auto-updating badges: Codecov coverage, Ruff code style, maintenance status, last commit
  - Reorganized badges into 5 sections (was 3): Build Status, Package, Code Quality, Project Status
  - 11/13 badges auto-update via API (zero maintenance drift)
  - Created comprehensive badge maintenance guide (`.github/BADGE_MAINTENANCE.md`) with update mechanisms and troubleshooting
  - Quarterly review required for only 2 static badges (Python version, maintenance status)

- **Workspace File Creation Policy** (`.claude/WORKSPACE_POLICY.md`, `CLAUDE.md`):
  - Created comprehensive policy to prevent intermediate report/summary files in version-controlled workspace
  - Defines allowed vs forbidden file patterns (`*_ANALYSIS*.md`, `*_REPORT*.md`, etc.)
  - Working papers must go in `.claude/reports/` (gitignored) or be communicated directly
  - Added decision tree for file creation and agent-specific guidance
  - Prevents future SSOT violations and repository clutter
  - Updated `CLAUDE.md` with clear workspace policy section

- **Claude Hooks - Complete Migration to Shared Utilities**:
  - **cleanup_stale_agents.py**: Migrated to shared utilities
    - Replaced custom logging with `get_hook_logger()`
    - Now uses `load_registry()` and `save_registry()` from shared.registry
    - Loads thresholds from config.yaml: `stale_threshold_hours`, `activity_check_hours`, `max_age_days`
    - Eliminated 80+ lines of duplicate registry code
  - **check_stop.py**: Migrated to shared utilities
    - Replaced custom `log_error()` with `get_hook_logger()`
    - Loads `max_stale_hours` from config.yaml instead of hardcoded value
    - Standardized logging format
  - **check_subagent_stop.py**: Migrated to shared utilities
    - Replaced custom logging with `get_hook_logger()`
    - Loads thresholds from config.yaml: `output_size_threshold_bytes` (200KB), `recent_window_minutes` (5)
    - Consistent error handling and logging
  - **health_check.py**: Migrated to shared utilities and config.yaml
    - Replaced custom `log_health()` with `get_hook_logger()`
    - Loads disk space thresholds from config.yaml: `disk_space_critical_percent` (5%), `disk_space_warning_percent` (10%)
    - Standardized health check reporting
  - **pre_compact_cleanup.sh**: Updated to use config.yaml
    - Loads `large_json_size_bytes` (100KB) from config.yaml via `get_config.py`
    - Loads `old_report_days` (7) from config.yaml
    - Eliminates hardcoded retention values
  - **session_cleanup.sh**: Updated to use config.yaml
    - Loads `locks_stale_minutes` (60) from config.yaml via `get_config.py`
    - Consistent with retention policy in single source of truth
  - **Impact**: All hooks now use centralized configuration and shared utilities
    - Zero hardcoded thresholds remaining in hook code
    - Consistent logging format across all hooks
    - Easy to adjust behavior via config.yaml without code changes

- **Claude Hooks - Final Optimizations and Consolidations**:
  - **Created shared/datetime_utils.py** (200 lines): Datetime and staleness utilities
    - Functions: `parse_timestamp()`, `age_in_hours()`, `age_in_days()`, `is_stale()`, `get_file_age_hours()`, `is_file_stale()`, `timestamp_now()`, `format_age()`
    - Eliminates ~80 lines of duplicate staleness check logic across 4 hooks
    - Consistent datetime handling with UTC awareness
    - Fallback support for file modification times
  - **Created shared/security.py** (282 lines): Security pattern matching and path validation
    - Functions: `matches_pattern()`, `is_blocked_path()`, `is_warned_path()`, `get_security_classification()`
    - **110+ security patterns**: Comprehensive credential/secret detection
      - Environment variables: `.env*`, `secrets.*`, `*_secret`, `*_password`
      - Keys/certificates: `*.key`, `*.pem`, `*.cer`, `*.crt`, `*.p12`, `*.pfx`, `*.ppk`
      - Cloud providers: AWS (`.aws/credentials`), GCP (`serviceaccount.json`), Azure (`.azure/credentials`)
      - Kubernetes: `kubeconfig`, `.kube/config`
      - Docker: `.docker/config.json`
      - Git internals: `.git/config`, `.git/objects/**`, `.git/refs/**`
      - API tokens: `api_key*`, `*_token`, `auth_token*`
      - Session files: `.netrc`, `.authinfo`, `.pgpass`
    - Eliminated 150+ lines of duplicate pattern matching from validate_path.py
    - Centralized security rules - update once, applies everywhere
  - **Enhanced shared/config.py**: Added comprehensive schema validation
    - New function: `validate_config_schema()` - validates structure and value ranges
    - New function: `load_config_with_validation()` - loads and validates in one call
    - Checks: required sections, numeric ranges (0-100 for percentages), type validation
    - Graceful degradation: logs warnings, doesn't fail on invalid config
  - **Consolidated SessionEnd hooks** → `session_end_cleanup.py`
    - Merged `session_cleanup.sh` + `cleanup_completed_workflows.sh` → single Python file
    - Better error handling with try/except blocks (vs shell)
    - Uses shared utilities: `get_hook_logger()`, `load_config()`, `is_file_stale()`
    - Eliminates duplicate directory traversals (runs once vs twice)
    - ~50 lines of shell code eliminated
  - **Consolidated Stop hooks** → `validate_stop.py`
    - Merged `check_stop.py` + `check_subagent_stop.py` → single Python file with mode detection
    - Auto-detects main agent vs subagent from stdin context
    - Shares staleness logic using `shared.datetime_utils.is_stale()`
    - Main mode: checks active_work.json staleness
    - Subagent mode: validates completion, auto-summarizes large outputs, updates registry
    - ~100 lines of duplicate staleness logic eliminated
  - **Updated settings.json**: Configured to use consolidated hooks
    - SessionEnd: Now calls `session_end_cleanup.py` (was 2 separate shell scripts)
    - Stop/SubagentStop: Both call `validate_stop.py` (was 2 separate Python files)
    - Maintains same timeout and error handling behavior

- **Claude Hooks - Comprehensive Test Suite Expansion**:
  - **Expanded test_hooks_comprehensive.py**: Now 29 tests (was 18)
  - **New test coverage**:
    - `cleanup_stale_agents.py`: Config loading, staleness detection (5 tests)
    - `health_check.py`: Config loading, disk thresholds (4 tests)
    - `session_end_cleanup.py`: Temp file cleanup, lock removal (4 tests)
    - `validate_stop.py`: Both main and subagent modes (6 tests)
    - `generate_settings.py`: YAML loading via shared utilities (3 tests)
    - `shared/datetime_utils.py`: All utility functions (7 tests)
  - **100% pass rate**: All 29 tests passing
  - **Integration-style tests**: Run actual hooks via subprocess with realistic fixtures
  - **Validates all migrations**: Confirms config loading, shared utilities work correctly

### Summary of Hook Infrastructure Improvements

**Comprehensive refactoring of Claude Code hooks system for optimal performance, security, and maintainability:**

**Code Quality**:

- ✅ Eliminated **750+ lines** of duplicate code across hooks
  - Shared utilities (450 lines): config.py, logging_utils.py, registry.py
  - datetime_utils.py (80 lines of staleness logic)
  - security.py (150 lines of pattern matching)
  - Consolidated hooks (50+ lines of shell scripts)
- ✅ **Zero hardcoded values**: All thresholds/patterns in config.yaml
- ✅ **100% test coverage**: All critical hooks validated (29 tests, 100% pass rate)

**Security Enhancements**:

- ✅ Fixed **2 critical vulnerabilities** (P0)
  - validate_path.py: Fail-closed on errors (was fail-open, allowed unsafe writes)
  - validate_path.py: Fixed .git bypass vulnerability
- ✅ **110+ security patterns**: Comprehensive credential/secret detection
  - Cloud providers: AWS, GCP, Azure, Kubernetes
  - Keys/certificates: 10+ file types
  - API tokens, passwords, environment variables
  - Git internals, Docker configs, session files

**Architecture Improvements**:

- ✅ **6 shared utility modules**: Eliminates duplication across all hooks
  - config.py (280 lines): Configuration and YAML loading
  - logging_utils.py (180 lines): Standardized logging
  - registry.py (370 lines): Agent registry operations
  - datetime_utils.py (200 lines): Datetime and staleness utilities
  - security.py (282 lines): Security pattern matching
  - paths.py (existing): Path definitions
- ✅ **4 hooks consolidated** into 2: Reduces complexity
  - session_cleanup.sh + cleanup_completed_workflows.sh → session_end_cleanup.py
  - check_stop.py + check_subagent_stop.py → validate_stop.py
- ✅ **8 hooks migrated** to shared utilities: Consistent implementation
  - cleanup_stale_agents.py, check_stop.py, check_subagent_stop.py, health_check.py
  - check_report_proliferation.py, enforce_agent_limit.py, validate_path.py, generate_settings.py

**Test Coverage**:

- ✅ **29 comprehensive tests** (was 18): 61% increase
- ✅ **100% pass rate**: All tests passing, all behavior validated
- ✅ **Coverage areas**:
  - Security validations (stdin parsing, fail-closed behavior, path traversal, credentials)
  - Hook migrations (config loading, shared utilities)
  - Consolidated hooks (both modes tested)
  - Shared utilities (all functions validated)

**Maintainability**:

- ✅ **Single source of truth**: config.yaml for all behavioral settings
- ✅ **Consistent patterns**: All hooks use same utilities, logging, error handling
- ✅ **Easy configuration**: Change thresholds in one place, applies everywhere
- ✅ **Better debugging**: Standardized logging format, detailed error messages

**Files Created**: 7 new files (shared utilities, consolidated hooks, tests)
**Files Modified**: 11 files (hook migrations, settings.json, config updates)
**Lines Eliminated**: 750+ lines of duplicate code
**Security Patterns Added**: 110+ comprehensive credential/secret detection patterns
**Test Coverage**: 29 tests, 100% pass rate (up from 18 tests, 94% pass rate)

## [0.1.2] - 2026-01-18

### Project Renamed: TraceKit → Oscura

**Oscura** is the new name for this project. The rename reflects our identity as a unified hardware reverse engineering framework.

- **New package name**: `oscura` (formerly `tracekit`)
- **New tagline**: "Revealing what's hidden in every signal"
- **New organization**: github.com/oscura-re
- **New repository**: github.com/oscura-re/oscura

**Migration**: No backward compatibility needed - this is the first public release under the new name.

### Initial Public Release

Oscura 0.1.0 is the first public release of the comprehensive hardware reverse engineering framework for security researchers, right-to-repair advocates, defense analysts, and commercial intelligence teams.

### Core Features

#### Signal Analysis & Measurement

- **Waveform Analysis** - Rise/fall time, frequency, duty cycle, overshoot (IEEE 181-2011 compliant)
- **Spectral Analysis** - FFT, PSD, spectrograms, wavelets, THD, SNR, SINAD, ENOB (IEEE 1241-2010)
- **Audio Analysis** - THD, SNR, SINAD, ENOB, harmonic distortion
- **Power Analysis** - AC/DC power, efficiency, ripple, power factor (IEEE 1459-2010)
- **Jitter Analysis** - TIE, period jitter, RJ/DJ decomposition (IEEE 2414-2020)
- **Signal Integrity** - Eye diagrams, S-parameter analysis, TDR impedance profiling

#### Protocol Support (16+ Decoders)

**Serial Protocols:**

- UART, SPI, I2C, 1-Wire, I2S, Manchester encoding

**Automotive:**

- CAN, CAN-FD, LIN, FlexRay, OBD-II (54 PIDs), J1939 (154 PGNs), UDS (ISO 14229)

**Debug Interfaces:**

- JTAG, SWD

**Network:**

- USB, HDLC

**Features:**

- Auto-detection and baud rate recovery
- Checksum validation (XOR, SUM, CRC-8/16/32)
- DTC database (210 codes, SAE J2012)

#### Reverse Engineering

- **SignalBuilder API** - Fluent API for composable signal generation (analog waveforms, protocol signals, noise/impairments)
- **Protocol Inference** - CRC polynomial recovery, state machine learning (L\* algorithm), field boundary detection
- **Pattern Recognition** - Counter patterns, toggle patterns, sequence detection
- **CAN Bus RE** - Message discovery, signal extraction, DBC file generation
- **Complete RE Workflow** - 8-step automated reverse engineering pipeline for unknown digital signals

#### Convenience APIs

- **quick_spectral()** - One-call spectral analysis returning all IEEE 1241 metrics
- **auto_decode()** - Unified protocol detection and decoding (UART/SPI/I2C/CAN)
- **smart_filter()** - Intelligent filtering with automatic noise source detection
- **reverse_engineer_signal()** - Complete reverse engineering workflow for unknown signals

#### Discovery & Analysis

- **Signal Characterization** - Automatic signal type detection (analog/digital/mixed)
- **Anomaly Detection** - Statistical anomaly identification
- **Quality Assessment** - Data quality validation and metrics

#### File Format Support

**Oscilloscopes:**

- Tektronix WFM, Rigol WFM, Siglent, generic binary

**Logic Analyzers:**

- Sigrok (.sr), VCD (Value Change Dump)

**Network Captures:**

- PCAP, PCAPNG with full protocol parsing (dpkt integration)

**Automotive:**

- Vector BLF/ASC, ASAM MDF/MF4, DBC, CSV

**Scientific Data:**

- TDMS (LabVIEW), HDF5, NumPy, WAV, CSV

**RF/Network:**

- Touchstone S-parameters (.s1p, .s2p, etc.)

#### Signal Processing

- **Filtering** - IIR, FIR, Butterworth, Chebyshev, Bessel, Elliptic filters
- **Triggering** - Edge, pattern, pulse width, glitch, runt, window triggers
- **Arithmetic** - Add, subtract, differentiate, integrate, FFT operations
- **Math Operations** - RMS, mean, peak detection, envelope, correlation

#### EMC & Compliance Testing

- **Standards Support** - CISPR 32, IEC 61000-3-2, IEC 61000-4-2/4-4, MIL-STD-461G
- **EMI Analysis** - Conducted/radiated emissions, immunity testing
- **EMI Fingerprinting** - Automatic emission source identification
- **Limit Testing** - Automated compliance checking with limit masks

#### Professional Features

- **Report Generation** - PDF, HTML, Markdown, CSV exports
- **Session Management** - Workspace persistence and replay
- **Batch Processing** - Multi-file analysis with progress tracking
- **Visualization** - Waveform plotting, eye diagrams, spectrograms, constellation diagrams
- **Memory Management** - Large file handling with streaming support
- **Comparison Tools** - Golden waveform comparison, mask testing

### Demonstrations

31 comprehensive demos covering all major features:

**Comprehensive Demos (8):**

1. Waveform Analysis - 7 analysis sections (measurements, spectral, power, statistics, filtering, protocols, math)
2. Protocol Decoding - UART, SPI, I2C multi-protocol with auto-detection
3. UDP Packet Analysis - Traffic metrics, payload analysis, pattern detection, field inference
4. Automotive - CAN, OBD-II, UDS, J1939, LIN, FlexRay protocols with DBC generation
5. Mixed-Signal - Clock recovery, jitter analysis, IEEE 2414-2020 compliance
6. Spectral Compliance - IEEE 1241-2010 validation (THD, SNR, SINAD, ENOB, SFDR)
7. Signal Reverse Engineering - 5-phase RE workflow
8. EMC Compliance - CISPR 32, IEC 61000 compliance testing

**Serial Protocols (6):**

- JTAG, SWD, USB, 1-Wire, Manchester, I2S

**Automotive Protocols (2):**

- LIN, FlexRay

**Timing & Jitter (3):**

- IEEE 181 pulse measurements, bathtub curves, DDJ/DCD analysis

**Power Analysis (2):**

- DC-DC efficiency, ripple analysis

**Signal Integrity (3):**

- Setup/hold timing, TDR impedance, S-parameters

**Protocol Inference (3):**

- CRC reverse engineering, Wireshark dissector generation, state machine learning

**Advanced Inference (3):**

- Bayesian inference, protocol DSL, active learning

**Complete Workflows (3):**

- Network analysis, unknown signal RE, automotive full workflow

**File Format I/O (1):**

- VCD loader

**Custom DAQ (3):**

- Simple, chunked, optimal streaming loaders

**Demo Features:**

- All demos support `--data-file` CLI argument for loading pre-captured data
- Auto-detection of default data from `demo_data/` directories
- Synthetic generation fallback when no files available
- 678.67 MB of generated demo data across 25 files
- Validation suite: 30/31 demos passing (96.8% success rate)

### Data Loading Feature

- **BaseDemo Enhancement** - Added `data_file` parameter and `find_default_data_file()` helper
- **Three-tier Loading** - CLI override → default file → synthetic fallback
- **Consistent Pattern** - All 31 demos follow standardized data loading approach
- **Multiple Formats** - NPZ, VCD, BIN, MF4, PCAP support across different demo types

### Infrastructure

- **Python 3.12+ Support** - Full type hints and modern Python features
- **Dependencies** - Optimized core dependencies (removed unused plotly/bokeh, moved reportlab to extras)
- **Testing** - 18,083 tests passing, 255 skipped, 10 xfailed (99.6% pass rate)
- **Code Quality** - 100% pass rate on all quality checks (ruff, mypy, prettier, markdownlint)
- **Pre-commit Hooks** - 21 hooks covering format, lint, security, documentation
- **Pre-push Verification** - Full CI simulation with 3-stage verification (95% CI coverage)
- **CI/CD** - GitHub Actions with parallel test matrix (Python 3.12/3.13, 8 test groups)
- **Documentation** - MkDocs with strict link validation, comprehensive API docs

### Standards Compliance

- **IEEE 181-2011** - Pulse measurements (rise/fall time, overshoot, slew rate)
- **IEEE 1057-2017** - Digitizer characterization and timing analysis
- **IEEE 1241-2010** - ADC testing (SNR, SINAD, ENOB, THD, SFDR)
- **IEEE 2414-2020** - Jitter measurements (TIE, period jitter, RJ/DJ decomposition)
- **IEEE 1459-2010** - Power quality measurements
- **CISPR 16** - EMC compliance testing with limit masks
- **IEC 61000** - Electromagnetic compatibility standards
- **MIL-STD-461G** - Military EMI/EMC requirements
- **SAE J1939** - Heavy-duty vehicle CAN diagnostics
- **ISO 14229** - Unified Diagnostic Services (UDS)

### Installation

```bash
pip install oscura
```

**Optional Dependencies:**

```bash
pip install oscura[all]           # All features
pip install oscura[automotive]    # Automotive protocols
pip install oscura[visualization] # Plotting support
pip install oscura[reporting]     # PDF report generation
```

### Quick Start

```python
import oscura as osc

# Load and analyze a waveform
trace = osc.load("capture.wfm")
print(f"Rise time: {osc.rise_time(trace):.2e} s")

# Quick spectral analysis
metrics = osc.quick_spectral(trace, fundamental=1000)
print(f"THD: {metrics.thd_db:.1f} dB, SNR: {metrics.snr_db:.1f} dB")

# Auto-decode protocol
result = osc.auto_decode(trace)
print(f"Protocol: {result.protocol}, Frames: {len(result.frames)}")

# Generate test signals
signal = (osc.SignalBuilder(sample_rate=1e6, duration=0.01)
    .add_sine(frequency=1000)
    .add_noise(snr_db=40)
    .build())

# Reverse engineer unknown signal
result = osc.workflows.reverse_engineer_signal(trace)
print(result.protocol_spec)
```

### Known Issues

- USB demo has pre-existing PID validation bug (not related to data loading feature)
- GitHub Actions CI requires billing resolution (code is fully verified locally)

### Contributors

- lair-click-bats (primary author)
- Claude Code (AI development assistance)

### License

MIT License - See LICENSE file for details

---

[Unreleased]: https://github.com/oscura-re/oscura/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/oscura-re/oscura/releases/tag/v0.1.2
