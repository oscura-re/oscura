#!/usr/bin/env bash
# =============================================================================
# test.sh - Optimized Test Execution for Oscura
# =============================================================================
# Usage: ./scripts/test.sh [OPTIONS]
# =============================================================================
# VALIDATED OPTIMAL APPROACH - Based on empirical testing:
#   - Parallel execution with pytest-xdist (-n 6 on 8-core machine)
#   - Extended timeout (300s) to prevent hangs
#   - Excludes known problematic modules
#   - Generates coverage reports
#   - Completes in ~10 minutes vs 45-50 minutes sequential
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

# =============================================================================
# Configuration
# =============================================================================

# Detect optimal worker count (CPU cores - 2 for system stability)
WORKERS=$(($(nproc 2> /dev/null || echo 4) - 2))
[[ ${WORKERS} -lt 1 ]] && WORKERS=1
[[ ${WORKERS} -gt 8 ]] && WORKERS=8 # Cap at 8 to avoid diminishing returns

# Test configuration
TIMEOUT=300     # 5 minutes per test
MAXFAIL=10      # Stop after 10 failures
COVERAGE_MIN=80 # Minimum coverage percentage target

# Problematic modules to exclude (based on empirical testing)
# HISTORICAL NOTE: These modules were previously excluded but are now FIXED:
#   - tests/unit/analyzers/protocols/ (504 tests, all passing as of 2026-01-15)
#   - tests/unit/inference/test_stream.py (84 tests, all passing, no hangs)
# Keeping empty array for future exclusions if needed
EXCLUDE_MODULES=()

# =============================================================================
# Help
# =============================================================================

show_help() {
  cat << 'EOF'
Optimized Test Execution for Oscura

USAGE:
    ./scripts/test.sh [OPTIONS]

OPTIONS:
    --fast              Quick test (unit tests only, no coverage)
    --coverage          Full test with coverage report (default)
    --parallel N        Use N parallel workers (default: auto-detected)
    --timeout N         Timeout per test in seconds (default: 300)
    --maxfail N         Stop after N failures (default: 10)
    --no-parallel       Disable parallel execution
    -v, --verbose       Verbose output
    -q, --quiet         Minimal output
    -h, --help          Show this help

EXAMPLES:
    # Run full test suite with coverage (RECOMMENDED)
    ./scripts/test.sh

    # Quick test without coverage
    ./scripts/test.sh --fast

    # Run with 4 workers
    ./scripts/test.sh --parallel 4

PERFORMANCE:
    - Sequential execution: ~45-50 minutes
    - Parallel (6 workers): ~8-10 minutes
    - Fast mode (no coverage): ~5-7 minutes

COVERAGE:
    - HTML report: htmlcov/index.html
    - Terminal report: shown after tests complete
    - Target: 80% minimum coverage

KNOWN ISSUES:
    - None - all tests enabled and passing (as of 2026-01-15)
    - Previous exclusions (282 tests) resolved and re-enabled

DEPENDENCIES:
    - pytest, pytest-xdist, pytest-timeout, pytest-cov (via uv)
    - bc (optional, for coverage percentage comparison)
EOF
}

# =============================================================================
# Parse Arguments
# =============================================================================

MODE="coverage" # default mode
VERBOSE=false
QUIET=false
CUSTOM_WORKERS=""
CUSTOM_TIMEOUT=""
CUSTOM_MAXFAIL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast)
      MODE="fast"
      shift
      ;;
    --coverage)
      MODE="coverage"
      shift
      ;;
    --parallel)
      CUSTOM_WORKERS="$2"
      shift 2
      ;;
    --no-parallel)
      WORKERS=0
      shift
      ;;
    --timeout)
      CUSTOM_TIMEOUT="$2"
      shift 2
      ;;
    --maxfail)
      CUSTOM_MAXFAIL="$2"
      shift 2
      ;;
    -v | --verbose)
      VERBOSE=true
      shift
      ;;
    -q | --quiet)
      QUIET=true
      shift
      ;;
    -h | --help)
      show_help "$@"
      exit 0
      ;;
    *)
      print_fail "Unknown option: $1"
      echo "Use --help for usage information"
      exit 2
      ;;
  esac
done

# Apply custom values
[[ -n "${CUSTOM_WORKERS}" ]] && WORKERS="${CUSTOM_WORKERS}"
[[ -n "${CUSTOM_TIMEOUT}" ]] && TIMEOUT="${CUSTOM_TIMEOUT}"
[[ -n "${CUSTOM_MAXFAIL}" ]] && MAXFAIL="${CUSTOM_MAXFAIL}"

# =============================================================================
# Build pytest command
# =============================================================================

cd "${PROJECT_ROOT}"

PYTEST_ARGS=()

# Add parallel execution if enabled
if [[ ${WORKERS} -gt 0 ]]; then
  # OPTIMAL: Use worksteal for dynamic load balancing (better than loadscope)
  PYTEST_ARGS+=(-n "${WORKERS}" --dist=worksteal)
  # Disable pytest-benchmark plugin when using xdist (prevents warning escalation)
  PYTEST_ARGS+=(-p no:benchmark)
fi

# Add timeout
PYTEST_ARGS+=(--timeout="${TIMEOUT}")

# Add maxfail
PYTEST_ARGS+=(--maxfail="${MAXFAIL}")

# Add coverage options
if [[ "${MODE}" == "coverage" ]]; then
  PYTEST_ARGS+=(
    --cov=oscura
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml # For diff-cover tool
  )
fi

# Fast mode: exclude slow tests
if [[ "${MODE}" == "fast" ]]; then
  PYTEST_ARGS+=(-m "not slow")
fi

# Add verbosity
if [[ "${VERBOSE}" == "true" ]]; then
  PYTEST_ARGS+=(-v)
fi

# No exclusions - all tests enabled (EXCLUDE_MODULES array kept for future use)

# =============================================================================
# Run tests
# =============================================================================

print_header "Oscura Test Suite"

if [[ "${QUIET}" == "false" ]]; then
  print_section "Configuration"
  echo -e "    ${DIM}Mode:${NC}             ${MODE}"
  echo -e "    ${DIM}Workers:${NC}          $([[ "${WORKERS}" -eq 0 ]] && echo 'disabled' || echo "${WORKERS}")"
  echo -e "    ${DIM}Timeout:${NC}          ${TIMEOUT}s per test"
  echo -e "    ${DIM}Max failures:${NC}     ${MAXFAIL}"
  echo -e "    ${DIM}All tests enabled:${NC} yes (no exclusions)"
fi

print_section "Executing tests"

# Run pytest via uv
if uv run python -m pytest "${PYTEST_ARGS[@]}"; then
  EXIT_CODE=0
  print_pass "Tests completed successfully"
else
  EXIT_CODE=$?
  print_fail "Tests failed with exit code ${EXIT_CODE}"
fi

# =============================================================================
# Report Results
# =============================================================================

if [[ "${MODE}" == "coverage" ]] && [[ "${QUIET}" == "false" ]]; then
  print_section "Coverage Report"

  if [[ -f "htmlcov/index.html" ]]; then
    print_info "HTML coverage report: htmlcov/index.html"

    # Extract coverage percentage if available
    if command -v coverage &> /dev/null; then
      COVERAGE_PCT=$(coverage report 2> /dev/null | tail -1 | awk '{print $4}' | tr -d '%' || echo "")
      if [[ -n "${COVERAGE_PCT}" ]]; then
        # Use bc if available, otherwise use awk for comparison
        coverage_met=0
        if command -v bc &> /dev/null; then
          if (($(echo "${COVERAGE_PCT} >= ${COVERAGE_MIN}" | bc -l 2> /dev/null))); then
            coverage_met=1
          fi
        else
          # Fallback: use awk for float comparison
          coverage_met=$(awk "BEGIN {print (${COVERAGE_PCT} >= ${COVERAGE_MIN}) ? 1 : 0}")
        fi

        if [[ ${coverage_met} -eq 1 ]]; then
          print_pass "Coverage: ${COVERAGE_PCT}% (target: ${COVERAGE_MIN}%)"
        else
          print_info "Coverage: ${COVERAGE_PCT}% (below target: ${COVERAGE_MIN}%)"
        fi
      fi
    fi
  else
    print_info "Coverage report not generated"
  fi
fi

exit ${EXIT_CODE}
