#!/usr/bin/env bash
# =============================================================================
# common.sh - Shared Functions for Oscura Scripts
# =============================================================================
# Source this file: source "$(dirname "$0")/lib/common.sh"
# Or from tools/: source "${SCRIPT_DIR}/../lib/common.sh"
# =============================================================================

# Strict mode
set -euo pipefail

# =============================================================================
# Colors (detect terminal capability)
# =============================================================================

if [[ -t 1 ]] && [[ -n "${TERM:-}" ]] && [[ "${TERM}" != "dumb" ]]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'
  CYAN='\033[0;36m'
  BOLD='\033[1m'
  DIM='\033[2m'
  NC='\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BLUE=''
  CYAN=''
  BOLD=''
  DIM=''
  NC=''
fi

# =============================================================================
# Output Functions
# =============================================================================

print_header() {
  echo ""
  echo -e "${BLUE}======================================================================${NC}"
  echo -e "${BLUE}  ${BOLD}$1${NC}"
  echo -e "${BLUE}======================================================================${NC}"
}

print_section() {
  echo ""
  echo -e "  ${CYAN}---${NC} ${BOLD}$1${NC}"
}

print_tool() {
  echo -e "  ${CYAN}*${NC} ${BOLD}$1${NC}"
}

print_tool_detail() {
  echo -e "    ${DIM}Tool:${NC} $1"
}

print_pass() {
  echo -e "    ${GREEN}[PASS]${NC} $1"
}

print_fail() {
  echo -e "    ${RED}[FAIL]${NC} $1"
}

print_skip() {
  echo -e "    ${YELLOW}[SKIP]${NC} ${DIM}$1${NC}"
}

print_info() {
  echo -e "    ${DIM}$1${NC}"
}

print_formatted() {
  echo -e "    ${GREEN}[PASS]${NC} Formatted"
}

print_unchanged() {
  echo -e "    ${DIM}[----]${NC} No changes needed"
}

print_would_format() {
  echo -e "    ${YELLOW}[WARN]${NC} Would format"
}

# =============================================================================
# Tool Detection
# =============================================================================

has_tool() {
  command -v "$1" &> /dev/null
}

require_tool() {
  local tool="$1"
  local install_hint="${2:-}"
  local is_py_tool="${3:-false}"

  # For Python tools, check if available via uv run
  if [[ "${is_py_tool}" == "true" ]] && has_uv; then
    if uv run "${tool}" --version &> /dev/null; then
      return 0
    fi
  fi

  # Otherwise check if tool is in PATH
  if ! has_tool "${tool}"; then
    print_skip "${tool} not installed${install_hint:+ (${install_hint})}"
    return 1
  fi
  return 0
}

# Check if uv is available for Python tools
has_uv() {
  has_tool "uv"
}

# Run a Python tool via uv if available, otherwise directly
run_py_tool() {
  local tool="$1"
  shift
  if has_uv; then
    uv run "${tool}" "$@"
  else
    "${tool}" "$@"
  fi
}

# =============================================================================
# File Detection
# =============================================================================

has_files() {
  local pattern="$1"
  local path="${2:-.}"

  # Use find to check if any files match
  [[ -n "$(find "${path}" -type f -name "${pattern}" -print -quit 2> /dev/null)" ]]
}

has_files_glob() {
  local pattern="$1"
  local path="${2:-.}"

  # Use glob pattern matching
  shopt -s nullglob globstar
  # shellcheck disable=SC2206
  local files=("${path}"/**/"${pattern}")
  shopt -u nullglob globstar
  [[ ${#files[@]} -gt 0 ]]
}

find_files() {
  local pattern="$1"
  local path="${2:-.}"
  local exclude="${3:-.git,.venv,node_modules,__pycache__,*.egg-info}"

  local exclude_args=()
  IFS=',' read -ra excludes <<< "${exclude}"
  for ex in "${excludes[@]}"; do
    exclude_args+=(-not -path "*/${ex}/*")
  done

  find "${path}" -type f -name "${pattern}" "${exclude_args[@]}" 2> /dev/null
}

# =============================================================================
# JSON Output (for Claude/CI)
# =============================================================================

JSON_OUTPUT=false

enable_json() {
  JSON_OUTPUT=true
}

is_json_mode() {
  ${JSON_OUTPUT}
}

json_result() {
  local tool="$1"
  local status="$2" # pass, fail, skip
  local message="${3:-}"

  if ${JSON_OUTPUT}; then
    printf '{"tool":"%s","status":"%s","message":"%s"}\n' "${tool}" "${status}" "${message}"
  fi
}

# =============================================================================
# Repository Root Detection
# =============================================================================

get_repo_root() {
  git rev-parse --show-toplevel 2> /dev/null || pwd
}

# Set REPO_ROOT if not already set
: "${REPO_ROOT:=$(get_repo_root)}"

# =============================================================================
# Config File Detection
# =============================================================================

find_config() {
  local config_name="$1"
  local search_path="${2:-${REPO_ROOT}}"

  # Check current directory first, then repo root
  if [[ -f "./${config_name}" ]]; then
    echo "./${config_name}"
  elif [[ -f "${search_path}/${config_name}" ]]; then
    echo "${search_path}/${config_name}"
  fi
}

# =============================================================================
# Counter Management
# =============================================================================

# Global counters - scripts can use these for summaries
PASSED=0
FAILED=0
SKIPPED=0
FORMATTED=0
UNCHANGED=0

reset_counters() {
  PASSED=0
  FAILED=0
  SKIPPED=0
  FORMATTED=0
  UNCHANGED=0
}

increment_passed() { ((PASSED++)) || true; }
increment_failed() { ((FAILED++)) || true; }
increment_skipped() { ((SKIPPED++)) || true; }
increment_formatted() { ((FORMATTED++)) || true; }
increment_unchanged() { ((UNCHANGED++)) || true; }

# Print summary of check/lint/format results
print_summary() {
  local check_type="${1:-check}" # check, lint, format

  print_header "SUMMARY"
  echo ""

  case "${check_type}" in
    format-check)
      echo -e "  ${YELLOW}Would format:${NC} ${FORMATTED}"
      echo -e "  ${DIM}Unchanged:${NC}    ${UNCHANGED}"
      ;;
    format)
      echo -e "  ${GREEN}Formatted:${NC}   ${FORMATTED}"
      echo -e "  ${DIM}Unchanged:${NC}    ${UNCHANGED}"
      ;;
    *)
      echo -e "  ${GREEN}Passed:${NC}  ${PASSED}"
      echo -e "  ${RED}Failed:${NC}  ${FAILED}"
      echo -e "  ${YELLOW}Skipped:${NC} ${SKIPPED}"
      ;;
  esac

  echo ""

  # Calculate and display pass rate if applicable
  if [[ "${check_type}" != "format"* ]]; then
    local total=$((PASSED + FAILED))
    if [[ ${total} -gt 0 ]]; then
      local pass_rate=$((PASSED * 100 / total))
      echo -e "  Pass rate: ${pass_rate}%"
      echo ""
    fi
  fi

  # Return appropriate exit code
  if [[ ${FAILED} -gt 0 ]]; then
    return 1
  fi
  return 0
}

# =============================================================================
# Exit Code Handling
# =============================================================================

# Exit codes:
# 0 - Success (or gracefully skipped)
# 1 - Failures found
# 2 - Tool/config error

# shellcheck disable=SC2034
EXIT_SUCCESS=0
# shellcheck disable=SC2034
EXIT_FAILURE=1
# shellcheck disable=SC2034
EXIT_ERROR=2

# =============================================================================
# Argument Parsing Helpers
# =============================================================================

show_help() {
  local script_name="$1"
  local tool_name="$2"
  local extra_options="${3:-}"

  echo "Usage: ${script_name} [OPTIONS] [paths...]"
  echo ""
  echo "${tool_name} wrapper script"
  echo ""
  echo "Options:"
  echo "  --check     Check only, don't modify files (default)"
  echo "  --fix       Fix issues automatically"
  echo "  --format    Format code"
  echo "  --json      Output machine-readable JSON"
  echo "  -v          Verbose output"
  echo "  -h, --help  Show this help message"
  if [[ -n "${extra_options}" ]]; then
    echo ""
    echo "${extra_options}"
  fi
  echo ""
  echo "If no paths given, searches current directory."
}
