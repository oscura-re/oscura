#!/usr/bin/env bash
# =============================================================================
# pre-push.sh - Comprehensive Local CI Verification
# =============================================================================
# Run this script before pushing to catch ALL CI/CD issues locally.
# This mirrors the GitHub Actions CI pipeline to ensure no failures on push.
#
# Usage: ./scripts/pre-push.sh [OPTIONS]
#
# Options:
#   --quick       Skip slow checks (tests, docs build)
#   --full        Run ALL checks including integration tests (default)
#   --fix         Auto-fix issues where possible before checking
#   --parallel N  Use N parallel workers for tests (default: auto)
#   --no-tests    Skip test execution (lint/format only)
#   --verbose     Show detailed output
#   -h, --help    Show this help message
#
# Exit codes:
#   0 - All checks passed, safe to push
#   1 - Checks failed, do NOT push
#   2 - Script/configuration error
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

# =============================================================================
# Configuration
# =============================================================================

# Default settings
MODE="full"
FIX_MODE=false
VERBOSE=false
RUN_TESTS=true
PARALLEL_WORKERS=""
PROFILE_TESTS=false

# Track results
declare -A CHECK_RESULTS
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
SKIPPED_CHECKS=0

# Timing
START_TIME=$(date +%s)

# =============================================================================
# Help
# =============================================================================

show_help() {
  cat <<'EOF'
Comprehensive Local CI Verification for Oscura

This script mirrors the GitHub Actions CI pipeline to catch all issues
before pushing. Run this before every push to avoid CI failures.

USAGE:
    ./scripts/pre-push.sh [OPTIONS]

OPTIONS:
    --quick       Skip slow checks (tests, docs build) - ~2 minutes
    --full        Run ALL checks including tests - ~10-15 minutes (default)
    --fix         Auto-fix issues where possible before checking
    --parallel N  Use N parallel workers for tests (default: auto-detected)
    --no-tests    Skip test execution (lint/format only) - ~3 minutes
    --profile     Enable test profiling (shows 20 slowest tests)
    --verbose     Show detailed output from each check
    -h, --help    Show this help message

MODES:
    quick    - Pre-commit hooks + basic lint (fastest, catches most issues)
    full     - All CI checks: lint, type check, tests, docs, config validation

WHAT IT CHECKS (matching CI workflows):
    Stage 1 - Fast Checks (parallel):
      * Pre-commit hooks (ruff, format, yaml, markdown, etc.)
      * Ruff lint & format check
      * MyPy type checking
      * Config validation (SSOT, orchestration)

    Stage 2 - Tests:
      * Unit tests (parallelized)
      * Integration tests
      * Compliance tests

    Stage 3 - Build Verification:
      * MkDocs build (--strict)
      * Package build (uv build)

EXIT CODES:
    0 - All checks passed, safe to push
    1 - Checks failed, fix issues before pushing
    2 - Script/configuration error

EXAMPLES:
    # Full verification before push (RECOMMENDED)
    ./scripts/pre-push.sh

    # Quick check during development
    ./scripts/pre-push.sh --quick

    # Auto-fix then verify
    ./scripts/pre-push.sh --fix

    # Skip tests, just check lint/format
    ./scripts/pre-push.sh --no-tests

TIPS:
    1. Run with --fix first to auto-correct formatting issues
    2. Use --quick during development, --full before final push
    3. If tests fail, run: uv run python -m pytest <failing_test> -v --tb=long
    4. Check CI logs for exact error messages if local passes but CI fails

EOF
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
  case "$1" in
  --quick)
    MODE="quick"
    shift
    ;;
  --full)
    MODE="full"
    shift
    ;;
  --fix)
    FIX_MODE=true
    shift
    ;;
  --parallel)
    PARALLEL_WORKERS="$2"
    shift 2
    ;;
  --no-tests)
    RUN_TESTS=false
    shift
    ;;
  --profile)
    PROFILE_TESTS=true
    shift
    ;;
  --verbose | -v)
    VERBOSE=true
    shift
    ;;
  -h | --help)
    show_help
    exit 0
    ;;
  *)
    echo "Unknown option: $1" >&2
    echo "Use --help for usage information" >&2
    exit 2
    ;;
  esac
done

# =============================================================================
# Utility Functions
# =============================================================================

log_check_start() {
  local check_name="$1"
  TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
  if [[ "${VERBOSE}" == "true" ]]; then
    echo ""
    echo -e "  ${CYAN}>>>${NC} ${BOLD}${check_name}${NC}"
  else
    echo -n -e "  ${CYAN}*${NC} ${check_name}..."
  fi
}

log_check_pass() {
  local check_name="$1"
  local duration="${2:-}"
  PASSED_CHECKS=$((PASSED_CHECKS + 1))
  CHECK_RESULTS["${check_name}"]="pass"

  if [[ "${VERBOSE}" == "true" ]]; then
    echo -e "    ${GREEN}[PASS]${NC} ${check_name}${duration:+ (${duration}s)}"
  else
    echo -e " ${GREEN}[PASS]${NC}${duration:+ (${duration}s)}"
  fi
}

log_check_fail() {
  local check_name="$1"
  local duration="${2:-}"
  FAILED_CHECKS=$((FAILED_CHECKS + 1))
  CHECK_RESULTS["${check_name}"]="fail"

  if [[ "${VERBOSE}" == "true" ]]; then
    echo -e "    ${RED}[FAIL]${NC} ${check_name}${duration:+ (${duration}s)}"
  else
    echo -e " ${RED}[FAIL]${NC}${duration:+ (${duration}s)}"
  fi
}

log_check_skip() {
  local check_name="$1"
  local reason="${2:-}"
  SKIPPED_CHECKS=$((SKIPPED_CHECKS + 1))
  CHECK_RESULTS["${check_name}"]="skip"

  if [[ "${VERBOSE}" == "true" ]]; then
    echo -e "    ${YELLOW}[SKIP]${NC} ${check_name}${reason:+ - ${reason}}"
  else
    echo -e " ${YELLOW}[SKIP]${NC}${reason:+ (${reason})}"
  fi
}

run_check() {
  local check_name="$1"
  shift
  local cmd=("$@")

  log_check_start "${check_name}"

  local check_start
  check_start=$(date +%s)
  local output_file
  output_file=$(mktemp)

  if "${cmd[@]}" >"${output_file}" 2>&1; then
    local check_end
    check_end=$(date +%s)
    local duration=$((check_end - check_start))
    log_check_pass "${check_name}" "${duration}"
    rm -f "${output_file}"
    return 0
  else
    local exit_code=$?
    local check_end
    check_end=$(date +%s)
    local duration=$((check_end - check_start))
    log_check_fail "${check_name}" "${duration}"

    # Show output on failure
    if [[ -s "${output_file}" ]]; then
      echo ""
      echo -e "    ${DIM}--- Output ---${NC}"
      head -50 "${output_file}" | sed 's/^/    /'
      local lines
      lines=$(wc -l <"${output_file}")
      if [[ ${lines} -gt 50 ]]; then
        echo -e "    ${DIM}... (${lines} total lines, showing first 50)${NC}"
      fi
      echo -e "    ${DIM}--- End Output ---${NC}"
    fi

    rm -f "${output_file}"
    return ${exit_code}
  fi
}

# =============================================================================
# Check Functions
# =============================================================================

check_pre_commit() {
  run_check "Pre-commit hooks" pre-commit run --all-files
}

check_ruff_lint() {
  run_check "Ruff lint" uv run ruff check src/ tests/
}

check_ruff_format() {
  run_check "Ruff format" uv run ruff format --check src/ tests/
}

check_mypy() {
  run_check "MyPy type check" uv run mypy src/
}

check_config_consistency() {
  run_check "Config consistency" uv run python .claude/hooks/validate_config_consistency.py
}

check_ssot() {
  run_check "SSOT validation" uv run python .claude/hooks/validate_ssot.py
}

check_hook_tests() {
  # Only run if test file exists
  if [[ -f ".claude/hooks/test_hooks.py" ]]; then
    run_check "Hook unit tests" uv run python .claude/hooks/test_hooks.py
  else
    log_check_skip "Hook unit tests" "Test file not found (optional)"
  fi
}

check_test_markers() {
  run_check "Test markers" uv run python scripts/testing/validate_test_markers.py --strict
}

check_unit_tests() {
  local pytest_args=(
    "tests/unit/"
    # Config-agnostic: Inherits --tb, --capture, --ff, --nf from pyproject.toml
    # This allows pyproject.toml to be the single source of truth
    "-m" "not slow and not performance"
    "--maxfail=10"
    "--benchmark-disable"
  )

  # Add profiling if requested
  if [[ "${PROFILE_TESTS}" == "true" ]]; then
    pytest_args+=("--durations=20" "--durations-min=0.1")
    echo ""
    echo -e "  ${CYAN}[PROFILING ENABLED]${NC} Will show 20 slowest tests (>0.1s)"
    echo ""
  fi

  # Add parallel workers (AGGRESSIVE: No cap for maximum performance)
  if [[ -n "${PARALLEL_WORKERS}" ]]; then
    pytest_args+=("-n" "${PARALLEL_WORKERS}" "--dist=worksteal")
  else
    # AGGRESSIVE: Use ALL available cores (no cap) for sub-2-min target
    local workers
    workers=$(nproc 2>/dev/null || echo 4)
    # No cap - use all cores for maximum parallelization
    pytest_args+=("-n" "${workers}" "--dist=worksteal")
  fi

  run_check "Unit tests" uv run python -m pytest "${pytest_args[@]}"
}

check_integration_tests() {
  # OPTIMIZED: Enable parallelization for integration tests
  # Integration tests are typically I/O bound, so modest parallelization helps
  # Using loadgroup distribution to keep test classes together
  local pytest_args=(
    "tests/integration/"
    # Config-agnostic: Inherits from pyproject.toml
    "-m" "integration"
    "--maxfail=5"
  )

  # Use modest parallelization (half of available cores, min 2)
  # Integration tests may share resources, so we're conservative
  local workers
  workers=$(nproc 2>/dev/null || echo 4)
  workers=$((workers / 2))
  if [[ ${workers} -lt 2 ]]; then
    workers=2
  fi

  pytest_args+=("-n" "${workers}" "--dist=loadgroup")

  run_check "Integration tests" uv run python -m pytest "${pytest_args[@]}"
}

check_compliance_tests() {
  local pytest_args=(
    "tests/compliance/"
    # Config-agnostic: Inherits from pyproject.toml
    "-m" "compliance"
    "--maxfail=5"
  )

  run_check "Compliance tests" uv run python -m pytest "${pytest_args[@]}"
}

check_mkdocs_build() {
  # OPTIMIZED: Cache MkDocs build based on docs/ directory hash
  # Only rebuild if docs/ or mkdocs.yml have changed
  # Saves ~3-4 seconds on unchanged runs

  local cache_dir="${PROJECT_ROOT}/.cache/pre-push"
  local cache_file="${cache_dir}/mkdocs-hash.txt"
  mkdir -p "${cache_dir}"

  # Compute hash of docs/ directory and mkdocs.yml
  local current_hash
  if command -v sha256sum >/dev/null 2>&1; then
    current_hash=$(find docs/ mkdocs.yml -type f 2>/dev/null |
      sort | xargs sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1)
  else
    # Fallback for macOS (uses shasum)
    current_hash=$(find docs/ mkdocs.yml -type f 2>/dev/null |
      sort | xargs shasum -a 256 2>/dev/null | shasum -a 256 | cut -d' ' -f1)
  fi

  # Check if hash matches cached version
  if [[ -f "${cache_file}" ]]; then
    local cached_hash
    cached_hash=$(cat "${cache_file}")

    if [[ "${current_hash}" == "${cached_hash}" ]]; then
      # Hash matches - skip rebuild
      log_check_start "MkDocs build"
      log_check_pass "MkDocs build (cached)" "0"
      return 0
    fi
  fi

  # Hash changed or no cache - rebuild
  if run_check "MkDocs build" uv run python -m mkdocs build --strict --clean; then
    # Save new hash on success
    echo "${current_hash}" >"${cache_file}"
    return 0
  else
    return 1
  fi
}

check_package_build() {
  # OPTIMIZED: Cache package build based on source file hash
  # Only rebuild if pyproject.toml or src/ have changed
  # Saves ~6-7 seconds on unchanged runs

  local cache_dir="${PROJECT_ROOT}/.cache/pre-push"
  local cache_file="${cache_dir}/build-hash.txt"
  mkdir -p "${cache_dir}"

  # Compute hash of pyproject.toml and src/ directory
  local current_hash
  if command -v sha256sum >/dev/null 2>&1; then
    current_hash=$(find pyproject.toml src/ -type f -name "*.py" -o -name "*.toml" 2>/dev/null |
      sort | xargs sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1)
  else
    # Fallback for macOS (uses shasum)
    current_hash=$(find pyproject.toml src/ -type f -name "*.py" -o -name "*.toml" 2>/dev/null |
      sort | xargs shasum -a 256 2>/dev/null | shasum -a 256 | cut -d' ' -f1)
  fi

  # Check if hash matches cached version
  if [[ -f "${cache_file}" ]]; then
    local cached_hash
    cached_hash=$(cat "${cache_file}")

    if [[ "${current_hash}" == "${cached_hash}" ]]; then
      # Hash matches - skip rebuild
      log_check_start "Package build"
      log_check_pass "Package build (cached)" "0"
      return 0
    fi
  fi

  # Hash changed or no cache - rebuild
  if run_check "Package build" uv build; then
    # Save new hash on success
    echo "${current_hash}" >"${cache_file}"
    return 0
  else
    return 1
  fi
}

check_cli() {
  # OPTIMIZED: Simple validation that CLI is importable and has entry points
  # Old approach: 2x `uv run` with separate process overhead (~12s)
  # New approach: Direct import validation (~1s) - 12x faster!
  local check_cmd='
import sys

# Test 1: Module imports successfully
try:
    from oscura.__main__ import main
except ImportError as e:
    print(f"❌ Failed to import oscura: {e}", file=sys.stderr)
    sys.exit(1)

# Test 2: Version is accessible
try:
    import oscura
    version = oscura.__version__
    if not version:
        print("❌ Version not set", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"❌ Failed to get version: {e}", file=sys.stderr)
    sys.exit(1)

# Success
print(f"✅ CLI module loads successfully (version {version})")
sys.exit(0)
'

  run_check "CLI commands" uv run python -c "${check_cmd}"
}

check_docstring_coverage() {
  # CI uses -f 95 threshold
  run_check "Docstring coverage" uv run python -m interrogate src/oscura -vv -f 95
}

# =============================================================================
# Auto-fix Mode
# =============================================================================

run_auto_fix() {
  print_header "AUTO-FIX MODE"
  echo -e "  ${DIM}Attempting to auto-fix issues before verification...${NC}"
  echo ""

  echo -e "  ${CYAN}*${NC} Running ruff --fix..."
  uv run ruff check src/ tests/ --fix || true

  echo -e "  ${CYAN}*${NC} Running ruff format..."
  uv run ruff format src/ tests/ || true

  echo -e "  ${CYAN}*${NC} Running pre-commit autofixes..."
  pre-commit run --all-files || true

  echo ""
  echo -e "  ${GREEN}Auto-fix complete.${NC} Proceeding to verification..."
}

# =============================================================================
# Main Execution
# =============================================================================

cd "${PROJECT_ROOT}"

print_header "PRE-PUSH VERIFICATION"
echo ""
echo -e "  ${DIM}Repository:${NC} ${PROJECT_ROOT}"
echo -e "  ${DIM}Mode:${NC}       ${MODE}"
echo -e "  ${DIM}Fix mode:${NC}   ${FIX_MODE}"
echo -e "  ${DIM}Run tests:${NC}  ${RUN_TESTS}"
echo -e "  ${DIM}Timestamp:${NC}  $(date '+%Y-%m-%d %H:%M:%S')"

# Run auto-fix if requested
if [[ "${FIX_MODE}" == "true" ]]; then
  run_auto_fix
fi

# =============================================================================
# Stage 1: Fast Checks (matches CI Stage 1)
# =============================================================================

print_header "STAGE 1: Fast Checks"

# Pre-commit (comprehensive check)
check_pre_commit || true

# Python linting
check_ruff_lint || true
check_ruff_format || true

# Type checking
check_mypy || true

# Config validation (matches CI config-validation job)
check_config_consistency || true
check_ssot || true
check_hook_tests || true

# =============================================================================
# Stage 2: Tests (matches CI Stage 2)
# =============================================================================

if [[ "${RUN_TESTS}" == "true" ]] && [[ "${MODE}" != "quick" ]]; then
  print_header "STAGE 2: Tests"

  # Test marker validation (matches test-quality.yml)
  check_test_markers || true

  # Unit tests
  check_unit_tests || true

  # Integration tests (for full mode)
  if [[ "${MODE}" == "full" ]]; then
    check_integration_tests || true
    check_compliance_tests || true
  fi
else
  print_header "STAGE 2: Tests"
  log_check_start "Tests"
  log_check_skip "Tests" "skipped (--quick or --no-tests)"
fi

# =============================================================================
# Stage 3: Build Verification (matches CI Stage 3-4)
# =============================================================================

if [[ "${MODE}" != "quick" ]]; then
  print_header "STAGE 3: Build Verification"

  # Documentation build
  check_mkdocs_build || true

  # Package build
  check_package_build || true

  # CLI verification
  check_cli || true

  # Docstring coverage (matches code-quality.yml)
  check_docstring_coverage || true
else
  print_header "STAGE 3: Build Verification"
  log_check_start "Build checks"
  log_check_skip "Build checks" "skipped (--quick mode)"
fi

# =============================================================================
# Summary
# =============================================================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

print_header "SUMMARY"
echo ""
echo -e "  ${GREEN}Passed:${NC}  ${PASSED_CHECKS}"
echo -e "  ${RED}Failed:${NC}  ${FAILED_CHECKS}"
echo -e "  ${YELLOW}Skipped:${NC} ${SKIPPED_CHECKS}"
echo -e "  ${DIM}Total:${NC}   ${TOTAL_CHECKS}"
echo ""
echo -e "  ${DIM}Duration:${NC} ${TOTAL_DURATION}s ($((TOTAL_DURATION / 60))m $((TOTAL_DURATION % 60))s)"
echo ""

# Show failed checks
if [[ ${FAILED_CHECKS} -gt 0 ]]; then
  echo -e "  ${RED}${BOLD}Failed checks:${NC}"
  for check in "${!CHECK_RESULTS[@]}"; do
    if [[ "${CHECK_RESULTS[${check}]}" == "fail" ]]; then
      echo -e "    ${RED}*${NC} ${check}"
    fi
  done
  echo ""
fi

# Final verdict
if [[ ${FAILED_CHECKS} -eq 0 ]]; then
  echo -e "  ${GREEN}${BOLD}All checks passed! Safe to push.${NC}"
  echo ""
  echo -e "  ${DIM}Tip: Run 'git push' to push your changes.${NC}"
  exit 0
else
  echo -e "  ${RED}${BOLD}${FAILED_CHECKS} check(s) failed. DO NOT PUSH.${NC}"
  echo ""
  echo -e "  ${DIM}Tip: Run './scripts/pre-push.sh --fix' to auto-fix some issues.${NC}"
  echo -e "  ${DIM}     Then re-run this script to verify.${NC}"
  exit 1
fi
