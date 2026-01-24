#!/usr/bin/env bash
# =============================================================================
# test-serial.sh - Run ALL tests without xdist parallelization
# =============================================================================
# Usage: ./scripts/test-serial.sh [pytest args...]
# =============================================================================
# This script runs the full test suite in serial mode (no parallel execution).
# Useful for:
#   - Running isolation tests (resource limits conflict with xdist)
#   - Debugging timing-sensitive tests
#   - Investigating flaky tests
#   - CI environments with limited resources
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

# =============================================================================
# Configuration
# =============================================================================

TIMEOUT=300 # 5 minutes per test
MAXFAIL=10  # Stop after 10 failures

# =============================================================================
# Help
# =============================================================================

show_help() {
  cat << 'EOF'
Run ALL Tests in Serial Mode (No Parallelization)

USAGE:
    ./scripts/test-serial.sh [OPTIONS] [-- PYTEST_ARGS...]

OPTIONS:
    --coverage          Generate coverage report (default: off)
    --slow              Include slow/performance tests (default: excluded)
    --timeout N         Timeout per test in seconds (default: 300)
    --maxfail N         Stop after N failures (default: 10)
    -v, --verbose       Verbose output
    -q, --quiet         Minimal output
    -h, --help          Show this help

EXAMPLES:
    # Run all tests serially
    ./scripts/test-serial.sh

    # Run with coverage
    ./scripts/test-serial.sh --coverage

    # Run including slow tests
    ./scripts/test-serial.sh --slow

    # Run specific tests
    ./scripts/test-serial.sh tests/unit/plugins/test_isolation.py

    # Pass custom pytest args
    ./scripts/test-serial.sh -- -k test_sandbox

WHY SERIAL MODE:
    - Isolation tests apply resource limits that crash xdist workers
    - Some timing-sensitive tests are more reliable in serial mode
    - Easier debugging (no worker process confusion)
    - Lower memory usage

PERFORMANCE:
    - Serial execution: ~30-45 minutes
    - Parallel (-n 6): ~5-10 minutes
    - Trade-off: Speed vs. Compatibility

CRITICAL TESTS THAT REQUIRE SERIAL MODE:
    - tests/unit/plugins/test_isolation.py (94 tests)
      These apply CPU/memory limits that interfere with xdist workers
EOF
}

# =============================================================================
# Parse Arguments
# =============================================================================

COVERAGE=false
INCLUDE_SLOW=false
VERBOSE=false
QUIET=false
CUSTOM_TIMEOUT=""
CUSTOM_MAXFAIL=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coverage)
      COVERAGE=true
      shift
      ;;
    --slow)
      INCLUDE_SLOW=true
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
      show_help
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Apply custom values
[[ -n "${CUSTOM_TIMEOUT}" ]] && TIMEOUT="${CUSTOM_TIMEOUT}"
[[ -n "${CUSTOM_MAXFAIL}" ]] && MAXFAIL="${CUSTOM_MAXFAIL}"

# =============================================================================
# Build pytest command
# =============================================================================

cd "${PROJECT_ROOT}"

PYTEST_ARGS=(
  "-p" "no:benchmark" # Disable benchmark plugin
  "--timeout=${TIMEOUT}"
  "--maxfail=${MAXFAIL}"
)

# Markers
if [[ "${INCLUDE_SLOW}" == "false" ]]; then
  PYTEST_ARGS+=("-m" "not slow and not performance")
fi

# Coverage
if [[ "${COVERAGE}" == "true" ]]; then
  PYTEST_ARGS+=(
    "--cov=src/oscura"
    "--cov-report=term-missing"
    "--cov-report=html"
    "--cov-report=xml"
  )
fi

# Verbosity
if [[ "${VERBOSE}" == "true" ]]; then
  PYTEST_ARGS+=("-v")
elif [[ "${QUIET}" == "true" ]]; then
  PYTEST_ARGS+=("-q")
fi

# Extra args (paths or other pytest options)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  PYTEST_ARGS+=("${EXTRA_ARGS[@]}")
else
  # Default: run all tests
  PYTEST_ARGS+=("tests/")
fi

# =============================================================================
# Run tests
# =============================================================================

print_header "Serial Test Execution (No Parallelization)"

if [[ "${QUIET}" == "false" ]]; then
  print_section "Configuration"
  echo -e "    ${DIM}Mode:${NC}             serial (no xdist)"
  echo -e "    ${DIM}Timeout:${NC}          ${TIMEOUT}s per test"
  echo -e "    ${DIM}Max failures:${NC}     ${MAXFAIL}"
  echo -e "    ${DIM}Include slow:${NC}     ${INCLUDE_SLOW}"
  echo -e "    ${DIM}Coverage:${NC}         ${COVERAGE}"
  echo ""
  echo -e "    ${YELLOW}⚠️  Serial mode takes 30-45 minutes${NC}"
  echo -e "    ${YELLOW}⚠️  Use ./scripts/test.sh for faster parallel execution${NC}"
  echo -e "    ${DIM}Serial mode is required for isolation tests${NC}"
fi

print_section "Executing tests"

# Run pytest via uv (NO -n flag = serial execution)
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

if [[ "${COVERAGE}" == "true" ]] && [[ "${QUIET}" == "false" ]]; then
  print_section "Coverage Report"

  if [[ -f "htmlcov/index.html" ]]; then
    print_info "HTML coverage report: htmlcov/index.html"

    if command -v coverage &> /dev/null; then
      COVERAGE_PCT=$(coverage report 2> /dev/null | tail -1 | awk '{print $4}' | tr -d '%' || echo "")
      if [[ -n "${COVERAGE_PCT}" ]]; then
        print_info "Coverage: ${COVERAGE_PCT}%"
      fi
    fi
  fi
fi

exit ${EXIT_CODE}
