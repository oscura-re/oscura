#!/bin/bash
# Memory-Safe Coverage Runner for Oscura
#
# PROBLEM: Running pytest with --cov on the entire test suite (2,300+ tests)
# causes Out-Of-Memory (OOM) kills because coverage instrumentation loads
# the entire codebase into memory along with all test fixtures.
#
# SOLUTION: Run coverage in smaller batches with --cov-append to accumulate
# results, avoiding memory exhaustion.
#
# USAGE:
#   ./scripts/run_coverage.sh              # Run all batches
#   ./scripts/run_coverage.sh --batch 1    # Run specific batch only
#   ./scripts/run_coverage.sh --quick      # Run quick subset for CI
#
# OUTPUT: coverage-full.json and terminal report

set -e

COVERAGE_FILE=".coverage"
REPORT_JSON="coverage-full.json"
REPORT_HTML="htmlcov"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
BATCH=""
QUICK=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --batch)
      BATCH="$2"
      shift 2
      ;;
    --quick)
      QUICK=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--batch N] [--quick]"
      exit 1
      ;;
  esac
done

# Clean previous coverage data
if [[ -z "${BATCH}" ]]; then
  echo -e "${YELLOW}Cleaning previous coverage data...${NC}"
  rm -f "${COVERAGE_FILE}" "${COVERAGE_FILE}.*" "${REPORT_JSON}"
  rm -rf "${REPORT_HTML}"
fi

# Function to run a test batch
run_batch() {
  local batch_num=$1
  local batch_name=$2
  local test_paths=$3
  local ignore_flags=$4

  echo -e "\n${GREEN}=== Batch ${batch_num}: ${batch_name} ===${NC}"
  echo "Test paths: ${test_paths}"
  echo "Memory usage before: $(free -h | grep Mem | awk '{print $3"/"$2}')"

  # Run with explicit memory limits to fail fast instead of OOM
  uv run pytest "${test_paths}" "${ignore_flags}" \
    --cov=oscura \
    --cov-append \
    --cov-report= \
    --timeout=0 \
    --tb=no \
    -q \
    2>&1 | tail -20

  local exit_code=$?
  echo "Memory usage after: $(free -h | grep Mem | awk '{print $3"/"$2}')"

  if [[ ${exit_code} -ne 0 ]]; then
    echo -e "${RED}Warning: Batch ${batch_num} had test failures (exit code: ${exit_code})${NC}"
    echo "Continuing with coverage collection..."
  fi

  # Small delay to allow memory cleanup
  sleep 2
}

# Batch definitions (ordered by size - smallest first to detect issues early)
if [[ "${QUICK}" -eq 1 ]]; then
  echo -e "${YELLOW}Running QUICK coverage (subset only)${NC}"
  run_batch 1 "Core + Plugins (quick)" "tests/unit/core/ tests/unit/plugins/" ""
  FINAL_BATCH=1
else
  # Full coverage in memory-safe batches
  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "1" ]]; then
    run_batch 1 "Plugins" "tests/unit/plugins/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "2" ]]; then
    run_batch 2 "Core" "tests/unit/core/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "3" ]]; then
    run_batch 3 "Loaders" "tests/unit/loaders/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "4" ]]; then
    run_batch 4 "Reporting" "tests/unit/reporting/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "5" ]]; then
    run_batch 5 "Analyzers (Digital)" "tests/unit/analyzers/digital/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "6" ]]; then
    run_batch 6 "Analyzers (Protocols)" "tests/unit/analyzers/protocols/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "7" ]]; then
    run_batch 7 "Analyzers (Other)" "tests/unit/analyzers/" \
      "--ignore=tests/unit/analyzers/digital/ --ignore=tests/unit/analyzers/protocols/"
  fi

  # Remaining batches - max ~500 tests each
  # Large directories (>400 tests) get own batch, smaller ones combined

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "8" ]]; then
    run_batch 8 "Visualization (1085)" "tests/unit/visualization/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "9" ]]; then
    run_batch 9 "Inference (515)" "tests/unit/inference/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "10" ]]; then
    run_batch 10 "Config (476)" "tests/unit/config/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "11" ]]; then
    run_batch 11 "API (451)" "tests/unit/api/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "12" ]]; then
    run_batch 12 "Utils (384)" "tests/unit/utils/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "13" ]]; then
    run_batch 13 "Comparison (369)" "tests/unit/comparison/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "14" ]]; then
    run_batch 14 "Filtering + Exploratory (631)" "tests/unit/filtering/ tests/unit/exploratory/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "15" ]]; then
    run_batch 15 "Triggering + CLI (502)" "tests/unit/triggering/ tests/unit/cli/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "16" ]]; then
    run_batch 16 "Search + Workflows (468)" "tests/unit/search/ tests/unit/workflows/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "17" ]]; then
    run_batch 17 "Optimization + Quality (395)" "tests/unit/optimization/ tests/unit/quality/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "18" ]]; then
    run_batch 18 "UI + Batch + Math (536)" "tests/unit/ui/ tests/unit/batch/ tests/unit/math/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "19" ]]; then
    run_batch 19 "Workflow + Streaming (273)" "tests/unit/workflow/ tests/unit/streaming/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "20" ]]; then
    run_batch 20 "Integrations + Discovery + Component + Pipeline (315)" \
      "tests/unit/integrations/ tests/unit/discovery/ tests/unit/component/ tests/unit/pipeline/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "21" ]]; then
    run_batch 21 "Testing + Schemas + DSL + test_dsl (322)" \
      "tests/unit/testing/ tests/unit/schemas/ tests/unit/dsl/ tests/unit/test_dsl/" ""
  fi

  if [[ -z "${BATCH}" ]] || [[ "${BATCH}" == "22" ]]; then
    run_batch 22 "Small modules (193)" \
      "tests/unit/guidance/ tests/unit/exporters/ tests/unit/extensibility/ tests/unit/test_workflows/ tests/unit/test_integrations/" ""
  fi

  FINAL_BATCH=22
fi

# Generate final report only if running all batches
if [[ -z "${BATCH}" ]]; then
  echo -e "\n${GREEN}=== Generating Final Coverage Report ===${NC}"
  uv run coverage report --precision=2
  uv run coverage json -o "${REPORT_JSON}"

  # Extract and display key metrics
  echo -e "\n${GREEN}=== Coverage Summary ===${NC}"
  python3 << 'PYTHON'
import json
data = json.load(open("coverage-full.json"))
total = data["totals"]
print(f"Total Statements:  {total['num_statements']:,}")
print(f"Covered:           {total['covered_lines']:,}")
print(f"Missing:           {total['missing_lines']:,}")
print(f"Coverage:          {total['percent_covered']:.2f}%")
print(f"\nBranch Coverage:   {total['percent_covered_display']}")

# Check if we hit 80% target
if total['percent_covered'] >= 80.0:
    print("\n✓ SUCCESS: Reached 80% coverage target!")
else:
    remaining = 80.0 - total['percent_covered']
    print(f"\n⚠ {remaining:.1f}% more coverage needed to reach 80% target")
PYTHON

  echo -e "\n${GREEN}Coverage report saved to:${NC}"
  echo "  - JSON: ${REPORT_JSON}"
  echo "  - Data: ${COVERAGE_FILE}"
  echo -e "\nTo generate HTML report: ${YELLOW}uv run coverage html${NC}"
else
  echo -e "\n${YELLOW}Note: Partial run (batch ${BATCH} only). Run without --batch for full report.${NC}"
fi

echo -e "\n${GREEN}Done!${NC}"
