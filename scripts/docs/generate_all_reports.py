#!/usr/bin/env python3
"""Generate comprehensive analysis reports for all test data files.

Runs ALL applicable analysis domains for each file type.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oscura.reporting.analyze import UnsupportedFormatError, analyze
from oscura.reporting.config import AnalysisConfig, AnalysisDomain

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_DIR = PROJECT_ROOT / "generated_reports"

SUPPORTED_EXTENSIONS = {".npz", ".bin", ".vcd"}

# Domains applicable to each file type
# Note: PATTERNS and advanced STATISTICS excluded due to slow execution (200+ sec)
WAVEFORM_DOMAINS = [
    AnalysisDomain.WAVEFORM,
    AnalysisDomain.SPECTRAL,
    AnalysisDomain.TIMING,
]

DIGITAL_DOMAINS = [
    AnalysisDomain.DIGITAL,
    AnalysisDomain.TIMING,
    AnalysisDomain.PROTOCOLS,
]

# PACKET domain excluded due to slow robust_packet_parse (400+ sec)
BINARY_DOMAINS = [
    AnalysisDomain.ENTROPY,
]


def find_all_test_files() -> list[Path]:
    """Find all supported test data files."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(TEST_DATA_DIR.rglob(f"*{ext}"))
    waveform = [f for f in files if f.suffix == ".npz"]
    digital = [f for f in files if f.suffix == ".vcd"]
    binary = [f for f in files if f.suffix == ".bin"]
    return waveform + digital + binary


def get_domains_for_file(file_path: Path) -> list[AnalysisDomain]:
    """Get applicable domains for a file based on its extension."""
    ext = file_path.suffix.lower()
    if ext == ".npz":
        return WAVEFORM_DOMAINS
    elif ext == ".vcd":
        return DIGITAL_DOMAINS
    elif ext == ".bin":
        return BINARY_DOMAINS
    return [AnalysisDomain.WAVEFORM]  # Default fallback


def generate_report(file_path: Path, output_base: Path) -> dict:
    """Generate a comprehensive report for a single file."""
    relative_path = file_path.relative_to(TEST_DATA_DIR)
    output_subdir = output_base / relative_path.parent / relative_path.stem

    result = {"input_file": str(file_path), "status": "pending", "duration": 0}
    start_time = time.time()

    try:
        # Use ALL applicable domains for this file type
        domains = get_domains_for_file(file_path)
        config = AnalysisConfig(
            domains=domains,
            generate_plots=False,
            output_formats=["json"],
            index_formats=["html", "md"],
            continue_on_error=True,
            timeout_per_analysis=5.0,
        )

        analysis_result = analyze(
            input_path=file_path,
            output_dir=output_subdir,
            config=config,
        )

        result["status"] = "success"
        result["output_dir"] = str(analysis_result.output_dir)
        result["analyses"] = analysis_result.successful_analyses

    except UnsupportedFormatError as e:
        result["status"] = "skipped"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)[:60]}"

    result["duration"] = time.time() - start_time
    return result


def main():
    logger.info("=" * 50)
    logger.info("Oscura Comprehensive Report Generation")
    logger.info("=" * 50)

    test_files = find_all_test_files()
    logger.info(f"Found {len(test_files)} files")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = OUTPUT_DIR / f"batch_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_base}")

    results = []
    success = 0

    for i, file_path in enumerate(test_files, 1):
        relative = file_path.relative_to(TEST_DATA_DIR)
        logger.info(f"[{i}/{len(test_files)}] {relative}")

        result = generate_report(file_path, output_base)
        results.append(result)

        if result["status"] == "success":
            success += 1
            logger.info(f"  ✓ {result['duration']:.1f}s")
        else:
            logger.warning(f"  ✗ {result.get('error', 'unknown')[:40]}")

    logger.info("=" * 50)
    logger.info(f"Done: {success}/{len(test_files)} successful")
    logger.info(f"Output: {output_base}")

    import json

    summary = {
        "timestamp": timestamp,
        "total": len(test_files),
        "successful": success,
        "output_dir": str(output_base),
        "results": results,
    }
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
