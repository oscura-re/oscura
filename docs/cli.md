# CLI Reference

> **Auto-generated** from actual CLI commands

## Overview

```
Usage: cli [OPTIONS] COMMAND [ARGS]...

  Oscura - Signal Analysis Framework for Oscilloscope Data.

  Command-line tools for characterizing buffers, decoding protocols, analyzing
  spectra, and comparing signals.

  Args:     ctx: Click context object.     verbose: Verbosity level (0=WARNING,
  1=INFO, 2+=DEBUG).

  Examples:     oscura characterize signal.wfm     oscura decode uart.wfm
  --protocol auto     oscura batch '*.wfm' --analysis characterize     oscura
  compare before.wfm after.wfm     oscura shell  # Interactive REPL

Options:
  -v, --verbose  Increase verbosity (-v for INFO, -vv for DEBUG).
  --version      Show the version and exit.
  --help         Show this message and exit.

Commands:
  batch         Batch process multiple files.
  characterize  Characterize buffer, signal, or power measurements.
  compare       Compare two signal captures.
  decode        Decode serial protocol data.
  shell         Start an interactive Oscura shell.
  tutorial      Run an interactive tutorial.

```

## `oscura batch`

```
Usage: cli batch [OPTIONS] PATTERN

  Batch process multiple files.

  Processes all files matching the given pattern with the specified analysis.
  Supports parallel processing for faster execution on multi-core systems.

  Args:     ctx: Click context object.     pattern: Glob pattern to match files.
  analysis: Type of analysis (characterize, decode, spectrum).     parallel:
  Number of parallel workers.     output: Output format (json, csv, html,
  table).     save_summary: Path to save CSV summary file.
  continue_on_error: Continue processing if individual files fail.

  Raises:     Exception: If batch processing fails or no files found.

  Examples:

          # Process all WFM files with characterization
          $ oscura batch '*.wfm' --analysis characterize

          # Parallel processing with 4 workers
          $ oscura batch 'test_run_*/*.wfm' \
              --analysis characterize \
              --parallel 4 \
              --save-summary results.csv

          # Decode all captures, continue on errors
          $ oscura batch 'captures/*.wfm' \
              --analysis decode \
              --continue-on-error

Options:
  --analysis [characterize|decode|spectrum]
                                  Type of analysis to perform on each file.
                                  [required]
  --parallel INTEGER              Number of files to process concurrently
                                  (default: 1).
  --output [json|csv|html|table]  Output format (default: table).
  --save-summary PATH             Save aggregated results to file (CSV format).
  --continue-on-error             Continue processing even if individual files
                                  fail.
  --help                          Show this message and exit.

```

## `oscura characterize`

```
Usage: cli characterize [OPTIONS] FILE

  Characterize buffer, signal, or power measurements.

  Analyzes a waveform file and extracts timing, quality, and performance
  characteristics. Supports automatic logic family detection and optional
  comparison to a reference signal.

  Args:     ctx: Click context object.     file: Path to waveform file to
  characterize.     analysis_type: Type of characterization (buffer, signal,
  power).     logic_family: Logic family for buffer characterization.
  compare: Path to reference file for comparison analysis.     output: Output
  format (json, csv, html, table).     save_report: Path to save HTML report
  file.

  Raises:     Exception: If characterization fails or file cannot be loaded.

  Examples:

          # Simple buffer characterization
          $ oscura characterize 74hc04_output.wfm

          # Full characterization with reference
          $ oscura characterize signal.wfm \
              --logic-family CMOS_3V3 \
              --compare golden_reference.wfm \
              --save-report report.html

          # Power analysis
          $ oscura characterize power_rail.wfm --type power --output json

Options:
  --type [buffer|signal|power]    Type of characterization to perform.
  --logic-family [ttl|cmos|cmos_3v3|cmos_5v|lvttl|lvcmos|auto]
                                  Logic family for buffer characterization
                                  (default: auto-detect).
  --compare PATH                  Reference file for comparison analysis.
  --output [json|csv|html|table]  Output format (default: table).
  --save-report PATH              Save HTML report to file.
  --help                          Show this message and exit.

```

## `oscura compare`

```
Usage: cli compare [OPTIONS] FILE1 FILE2

  Compare two signal captures.

  Analyzes differences between two waveforms including timing drift, amplitude
  changes, noise variations, and spectral differences.

  Args:     ctx: Click context object.     file1: Path to first waveform file.
  file2: Path to second waveform file.     threshold: Percentage threshold for
  reporting differences.     output: Output format (json, csv, html, table).
  save_report: Path to save HTML comparison report.     align: Align signals
  using cross-correlation before comparison.

  Raises:     Exception: If comparison fails or files cannot be loaded.

  Examples:

          # Simple comparison
          $ oscura compare before.wfm after.wfm

          # Report only significant differences (>10%)
          $ oscura compare golden.wfm measured.wfm --threshold 10

          # Full comparison with alignment and HTML report
          $ oscura compare reference.wfm test.wfm \
              --align \
              --save-report comparison.html

          # JSON output for automation
          $ oscura compare before.wfm after.wfm --output json

Options:
  --threshold FLOAT               Report differences greater than this
                                  percentage (default: 5%).
  --output [json|csv|html|table]  Output format (default: table).
  --save-report PATH              Save detailed HTML comparison report.
  --align                         Align signals using cross-correlation before
                                  comparison.
  --help                          Show this message and exit.

```

## `oscura decode`

```
Usage: cli decode [OPTIONS] FILE

  Decode serial protocol data.

  Automatically detects and decodes common serial protocols (UART, SPI, I2C,
  CAN). Can highlight errors with surrounding context for debugging.

  Args:     ctx: Click context object.     file: Path to waveform file to
  decode.     protocol: Protocol type (uart, spi, i2c, can, auto).
  baud_rate: Baud rate for UART (None for auto-detect).     parity: Parity
  setting for UART (none, even, odd).     stop_bits: Number of stop bits for
  UART (1 or 2).     show_errors: Show only packets with errors.     output:
  Output format (json, csv, html, table).

  Raises:     Exception: If decoding fails or file cannot be loaded.

  Examples:

          # Auto-detect and decode protocol
          $ oscura decode serial_capture.wfm

          # Decode specific protocol with parameters
          $ oscura decode uart.wfm \
              --protocol UART \
              --baud-rate 9600 \
              --parity even \
              --stop-bits 2

          # Show only errors for debugging
          $ oscura decode problematic.wfm --show-errors

          # Generate JSON output
          $ oscura decode i2c.wfm --protocol I2C --output json

Options:
  --protocol [uart|spi|i2c|can|auto]
                                  Protocol type (default: auto-detect).
  --baud-rate INTEGER             Baud rate for UART (auto-detect if not
                                  specified).
  --parity [none|even|odd]        Parity for UART (default: none).
  --stop-bits [1|2]               Stop bits for UART (default: 1).
  --show-errors                   Show only errors with context.
  --output [json|csv|html|table]  Output format (default: table).
  --help                          Show this message and exit.

```

## `oscura shell`

```
Usage: cli shell [OPTIONS]

  Start an interactive Oscura shell.

  Opens a Python REPL with Oscura pre-imported and ready to use. Features tab
  completion, persistent history, and helpful shortcuts.

  Example:     $ oscura shell     Oscura Shell v0.1.0     >>> trace =
  load("signal.wfm")     >>> rise_time(trace)

Options:
  --help  Show this message and exit.

```

## `oscura tutorial`

```
Usage: cli tutorial [OPTIONS] [TUTORIAL_ID]

  Run an interactive tutorial.

  Provides step-by-step guidance for learning Oscura.

  Args:     tutorial_id: ID of the tutorial to run (or None to list).
  list_tutorials: If True, list available tutorials.

  Examples:     oscura tutorial --list           # List available tutorials
  oscura tutorial getting_started  # Run the getting started tutorial

Options:
  --list  List available tutorials
  --help  Show this message and exit.

```
