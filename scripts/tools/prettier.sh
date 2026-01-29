#!/usr/bin/env bash
# =============================================================================
# prettier.sh - Multi-format Code Formatting with Prettier
# =============================================================================
# Usage: ./scripts/tools/prettier.sh [--check|--fix] [--json] [-v] [paths...]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Cleanup temporary files on exit (reliability fix)
STDERR_FILE=""
cleanup() {
  [[ -n "${STDERR_FILE}" && -f "${STDERR_FILE}" ]] && rm -f "${STDERR_FILE}"
  return 0 # Don't affect exit code
}
trap cleanup EXIT

# Configuration
TOOL_NAME="Prettier"
TOOL_CMD="prettier"
INSTALL_HINT="npm install -g prettier"

# Defaults
MODE="check"
VERBOSE=false
PATHS=()
FILE_TYPES="json,yaml,yml,md" # Default file types

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
    --json-only)
      FILE_TYPES="json"
      shift
      ;;
    --yaml-only)
      FILE_TYPES="yaml,yml"
      shift
      ;;
    --md-only)
      FILE_TYPES="md"
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
      echo "Code formatting with Prettier"
      echo ""
      echo "Options:"
      echo "  --check       Check only (default)"
      echo "  --fix         Fix formatting"
      echo "  --json-only   Only format JSON files"
      echo "  --yaml-only   Only format YAML files"
      echo "  --md-only     Only format Markdown files"
      echo "  --json        Output machine-readable JSON"
      echo "  -v            Verbose output"
      echo "  -h, --help    Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --check ."
      echo "  $0 --fix docs/"
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

# Check for npx (preferred) or prettier directly
if has_tool "npx"; then
  PRETTIER_CMD="npx prettier"
elif has_tool "${TOOL_CMD}"; then
  PRETTIER_CMD="prettier"
else
  print_skip "prettier/npx not installed (${INSTALL_HINT})"
  json_result "${TOOL_CMD}" "skip" "Tool not installed"
  exit 0
fi

# Build glob patterns
globs=()
IFS=',' read -ra types <<< "${FILE_TYPES}"
for path in "${PATHS[@]}"; do
  for type in "${types[@]}"; do
    if [[ -f "${path}" ]]; then
      globs+=("${path}")
    elif [[ -d "${path}" ]]; then
      globs+=("${path}/**/*.${type}")
    fi
  done
done

if [[ ${#globs[@]} -eq 0 ]]; then
  print_skip "No files to format"
  json_result "${TOOL_CMD}" "skip" "No files"
  exit 0
fi

# Check for ignore file
ignore_args=()
if [[ -f ".prettierignore" ]]; then
  ignore_args+=("--ignore-path" ".prettierignore")
fi

# Run prettier
case ${MODE} in
  check)
    # Capture stderr to check for "no files found" case
    STDERR_FILE=$(mktemp)
    if ${VERBOSE}; then
      if ${PRETTIER_CMD} --check "${ignore_args[@]}" "${globs[@]}" 2> "${STDERR_FILE}"; then
        print_pass "All files formatted"
        json_result "${TOOL_CMD}" "pass" ""
        exit 0
      else
        # Check if failure is due to no matching files
        if grep -q "No files matching the pattern were found" "${STDERR_FILE}"; then
          print_pass "No files to check"
          json_result "${TOOL_CMD}" "pass" "No files"
          exit 0
        else
          print_fail "Formatting issues found"
          json_result "${TOOL_CMD}" "fail" "Format issues"
          exit 1
        fi
      fi
    else
      if ${PRETTIER_CMD} --check "${ignore_args[@]}" "${globs[@]}" 2> "${STDERR_FILE}" > /dev/null; then
        print_pass "All files formatted"
        json_result "${TOOL_CMD}" "pass" ""
        exit 0
      else
        # Check if failure is due to no matching files
        if grep -q "No files matching the pattern were found" "${STDERR_FILE}"; then
          print_pass "No files to check"
          json_result "${TOOL_CMD}" "pass" "No files"
          exit 0
        else
          print_fail "Formatting issues found"
          print_info "Run with --fix to format"
          json_result "${TOOL_CMD}" "fail" "Format issues"
          exit 1
        fi
      fi
    fi
    ;;
  fix)
    # Capture stderr to check for "no files found" case
    STDERR_FILE=$(mktemp)
    if ${VERBOSE}; then
      if ${PRETTIER_CMD} --write "${ignore_args[@]}" "${globs[@]}" 2> "${STDERR_FILE}"; then
        print_pass "Formatted files"
        json_result "${TOOL_CMD}" "pass" "Formatted"
        exit 0
      else
        # Check if failure is due to no matching files
        if grep -q "No files matching the pattern were found" "${STDERR_FILE}"; then
          print_pass "No files to format"
          json_result "${TOOL_CMD}" "pass" "No files"
          exit 0
        else
          print_fail "Format failed"
          json_result "${TOOL_CMD}" "fail" "Format error"
          exit 1
        fi
      fi
    else
      if ${PRETTIER_CMD} --write "${ignore_args[@]}" "${globs[@]}" 2> "${STDERR_FILE}" > /dev/null; then
        print_pass "Formatted files"
        json_result "${TOOL_CMD}" "pass" "Formatted"
        exit 0
      else
        # Check if failure is due to no matching files
        if grep -q "No files matching the pattern were found" "${STDERR_FILE}"; then
          print_pass "No files to format"
          json_result "${TOOL_CMD}" "pass" "No files"
          exit 0
        else
          print_fail "Format failed"
          json_result "${TOOL_CMD}" "fail" "Format error"
          exit 1
        fi
      fi
    fi
    ;;
esac
