#!/usr/bin/env bash
# =============================================================================
# mypy.sh - Python Type Checking with Mypy
# =============================================================================
# Usage: ./scripts/tools/mypy.sh [--json] [-v] [paths...]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Configuration
TOOL_NAME="Mypy"
TOOL_CMD="mypy"
FILE_PATTERN="*.py"
INSTALL_HINT="uv pip install mypy"

# Defaults
VERBOSE=false
PATHS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
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
      echo "Python type checking with Mypy"
      echo ""
      echo "Options:"
      echo "  --json      Output machine-readable JSON"
      echo "  -v          Verbose output"
      echo "  -h, --help  Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 src/"
      echo "  $0 src/oscura tests/"
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

# Default to src/ directory (avoid demo duplicate modules)
[[ ${#PATHS[@]} -eq 0 ]] && PATHS=("src/")

# Main
print_tool "${TOOL_NAME}"

# Check tool installed
if ! require_tool "${TOOL_CMD}" "${INSTALL_HINT}" "true"; then
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

# Run mypy
if ${VERBOSE}; then
  if run_py_tool mypy "${PATHS[@]}"; then
    print_pass "Type check passed"
    json_result "${TOOL_CMD}" "pass" ""
    exit 0
  else
    print_fail "Type errors found"
    json_result "${TOOL_CMD}" "fail" "Type errors"
    exit 1
  fi
else
  if run_py_tool mypy "${PATHS[@]}" &> /dev/null; then
    print_pass "Type check passed"
    json_result "${TOOL_CMD}" "pass" ""
    exit 0
  else
    print_fail "Type errors found"
    print_info "Run with -v for details"
    json_result "${TOOL_CMD}" "fail" "Type errors"
    exit 1
  fi
fi
