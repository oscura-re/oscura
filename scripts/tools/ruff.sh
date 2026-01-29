#!/usr/bin/env bash
# =============================================================================
# ruff.sh - Python Linting and Formatting with Ruff
# =============================================================================
# Usage: ./scripts/tools/ruff.sh [--check|--fix|--format] [--json] [-v] [paths...]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Configuration
TOOL_NAME="Ruff"
TOOL_CMD="ruff"
FILE_PATTERN="*.py"
INSTALL_HINT="uv pip install ruff"

# Defaults
MODE="check"
VERBOSE=false
PATHS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --check)
      MODE="check"
      shift
      ;;
    --fix)
      MODE="fix"
      shift
      ;;
    --format)
      MODE="format"
      shift
      ;;
    --json)
      enable_json
      shift
      ;;
    -v)
      VERBOSE=true
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [OPTIONS] [paths...]"
      echo ""
      echo "Python linting and formatting with Ruff"
      echo ""
      echo "Options:"
      echo "  --check     Check for lint and format issues (default)"
      echo "  --fix       Fix auto-fixable lint issues"
      echo "  --format    Format code"
      echo "  --json      Output machine-readable JSON"
      echo "  -v          Verbose output"
      echo "  -h, --help  Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --check src/"
      echo "  $0 --fix src/"
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      PATHS+=("$1")
      shift
      ;;
  esac
done

# Default to current directory
[[ ${#PATHS[@]} -eq 0 ]] && PATHS=(".")

# Main
print_tool "${TOOL_NAME} (${MODE})"

# Check tool installed
if ! require_tool "${TOOL_CMD}" "${INSTALL_HINT}"; then
  json_result "${TOOL_CMD}" "skip" "Tool not installed"
  exit 0
fi

# Check for Python files
has_python_files=false
for path in "${PATHS[@]}"; do
  if [[ -f "${path}" && "${path}" == *.py ]] || has_files "${FILE_PATTERN}" "${path}"; then
    has_python_files=true
    break
  fi
done

if ! ${has_python_files}; then
  print_skip "No Python files found"
  json_result "${TOOL_CMD}" "skip" "No files found"
  exit 0
fi

# Run tool based on mode
case ${MODE} in
  check)
    lint_ok=true
    format_ok=true

    if ${VERBOSE}; then
      run_py_tool ruff check "${PATHS[@]}" || lint_ok=false
      run_py_tool ruff format --check "${PATHS[@]}" || format_ok=false
    else
      run_py_tool ruff check "${PATHS[@]}" &> /dev/null || lint_ok=false
      run_py_tool ruff format --check "${PATHS[@]}" &> /dev/null || format_ok=false
    fi

    if ${lint_ok} && ${format_ok}; then
      print_pass "All checks passed"
      json_result "${TOOL_CMD}" "pass" ""
      exit 0
    else
      print_fail "Issues found"
      json_result "${TOOL_CMD}" "fail" "Linting issues found"
      exit 1
    fi
    ;;
  fix)
    if ${VERBOSE}; then
      if run_py_tool ruff check --fix "${PATHS[@]}"; then
        print_pass "Fixed lint issues"
        json_result "${TOOL_CMD}" "pass" "Fixed"
        exit 0
      else
        print_fail "Could not fix all issues"
        json_result "${TOOL_CMD}" "fail" "Some issues unfixable"
        exit 1
      fi
    else
      if run_py_tool ruff check --fix "${PATHS[@]}" &> /dev/null; then
        print_pass "Fixed lint issues"
        json_result "${TOOL_CMD}" "pass" "Fixed"
        exit 0
      else
        print_fail "Could not fix all issues"
        json_result "${TOOL_CMD}" "fail" "Some issues unfixable"
        exit 1
      fi
    fi
    ;;
  format)
    if ${VERBOSE}; then
      if run_py_tool ruff format "${PATHS[@]}"; then
        print_pass "Formatted"
        json_result "${TOOL_CMD}" "pass" "Formatted"
        exit 0
      else
        print_fail "Format failed"
        json_result "${TOOL_CMD}" "fail" "Format error"
        exit 1
      fi
    else
      if run_py_tool ruff format "${PATHS[@]}" &> /dev/null; then
        print_pass "Formatted"
        json_result "${TOOL_CMD}" "pass" "Formatted"
        exit 0
      else
        print_fail "Format failed"
        json_result "${TOOL_CMD}" "fail" "Format error"
        exit 1
      fi
    fi
    ;;
esac
