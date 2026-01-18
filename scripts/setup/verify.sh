#!/usr/bin/env bash
# =============================================================================
# verify.sh - Quick Development Verification
# =============================================================================
# A lightweight verification script for use during development.
# Faster than pre-push.sh, catches the most common issues.
#
# Usage: ./scripts/verify.sh [OPTIONS]
#
# This is the script to run frequently during development.
# Use pre-push.sh for comprehensive verification before pushing.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# =============================================================================
# Configuration
# =============================================================================

FIX_MODE=false
INCLUDE_TESTS=false
VERBOSE=false

# Track results
reset_counters # Use common.sh counter management
START_TIME=$(date +%s)

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fix | -f)
      FIX_MODE=true
      shift
      ;;
    --test | -t)
      INCLUDE_TESTS=true
      shift
      ;;
    --verbose | -v)
      VERBOSE=true
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Quick development verification"
      echo ""
      echo "Options:"
      echo "  --fix, -f     Auto-fix issues where possible"
      echo "  --test, -t    Include quick test run"
      echo "  --verbose, -v Verbose output"
      echo "  -h, --help    Show this help message"
      echo ""
      echo "For comprehensive verification, use: ./scripts/pre-push.sh"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

# =============================================================================
# Check Functions
# =============================================================================

run_check() {
  local name="$1"
  shift
  local cmd=("$@")

  echo -n -e "  ${CYAN}*${NC} ${name}..."

  local output_file
  output_file=$(mktemp)

  if "${cmd[@]}" > "${output_file}" 2>&1; then
    echo -e " ${GREEN}[OK]${NC}"
    PASSED=$((PASSED + 1))
    rm -f "${output_file}"
    return 0
  else
    echo -e " ${RED}[FAIL]${NC}"
    FAILED=$((FAILED + 1))

    if [[ "${VERBOSE}" == "true" ]] && [[ -s "${output_file}" ]]; then
      head -20 "${output_file}" | sed 's/^/      /'
    fi

    rm -f "${output_file}"
    return 1
  fi
}

# =============================================================================
# Main
# =============================================================================

cd "${PROJECT_ROOT}"

print_header "QUICK VERIFY"
echo ""
echo -e "  ${DIM}Fix mode:${NC} ${FIX_MODE}"
echo -e "  ${DIM}Include tests:${NC} ${INCLUDE_TESTS}"

# Auto-fix if requested
if [[ "${FIX_MODE}" == "true" ]]; then
  echo ""
  echo -e "  ${CYAN}Running auto-fix...${NC}"
  uv run ruff check src/ tests/ --fix --quiet 2> /dev/null || true
  uv run ruff format src/ tests/ --quiet 2> /dev/null || true
fi

print_section "Lint & Format"

run_check "Ruff lint" uv run ruff check src/ tests/ || true
run_check "Ruff format" uv run ruff format --check src/ tests/ || true
run_check "MyPy" uv run mypy src/ --no-error-summary || true

if [[ "${INCLUDE_TESTS}" == "true" ]]; then
  print_section "Quick Tests"
  run_check "Smoke test" uv run pytest tests/unit/core/test_types.py -x -q --tb=no || true
fi

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

print_section "Result"
echo ""
if [[ ${FAILED} -eq 0 ]]; then
  echo -e "  ${GREEN}${BOLD}All ${PASSED} checks passed${NC} (${DURATION}s)"
  echo ""
  echo -e "  ${DIM}Ready to commit. For full verification: ./scripts/pre-push.sh${NC}"
  exit 0
else
  echo -e "  ${RED}${BOLD}${FAILED} of $((PASSED + FAILED)) checks failed${NC} (${DURATION}s)"
  echo ""
  echo -e "  ${DIM}Run with --fix to auto-fix some issues${NC}"
  exit 1
fi
