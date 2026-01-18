#!/usr/bin/env bash
# =============================================================================
# markdownlint.sh - Markdown Linting and Formatting
# =============================================================================
# Usage: ./scripts/tools/markdownlint.sh [--check|--fix] [--json] [-v] [paths...]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Configuration
TOOL_NAME="markdownlint"
TOOL_CMD="markdownlint"
INSTALL_HINT="npm install -g markdownlint-cli"

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
      echo "Markdown linting with markdownlint"
      echo ""
      echo "Options:"
      echo "  --check     Check only (default)"
      echo "  --fix       Fix issues automatically"
      echo "  --json      Output machine-readable JSON"
      echo "  -v          Verbose output"
      echo "  -h, --help  Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --check ."
      echo "  $0 --fix README.md"
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

# Build glob patterns for markdownlint
md_globs=()
for path in "${PATHS[@]}"; do
  if [[ -f "${path}" ]]; then
    md_globs+=("${path}")
  elif [[ -d "${path}" ]]; then
    md_globs+=("${path}/**/*.md")
  fi
done

if [[ ${#md_globs[@]} -eq 0 ]]; then
  print_skip "No Markdown paths specified"
  json_result "${TOOL_CMD}" "skip" "No paths"
  exit 0
fi

# Build config argument if config exists
config_args=()
config_file=$(find_config ".markdownlint.yaml")
if [[ -n "${config_file}" ]]; then
  config_args+=("--config" "${config_file}")
fi

# Common ignore patterns
ignore_args=("--ignore" "node_modules" "--ignore" ".venv" "--ignore" ".git")

# Run markdownlint
case ${MODE} in
  check)
    if ${VERBOSE}; then
      if markdownlint "${config_args[@]}" "${ignore_args[@]}" "${md_globs[@]}"; then
        print_pass "All Markdown valid"
        json_result "${TOOL_CMD}" "pass" ""
        exit 0
      else
        print_fail "Issues found"
        json_result "${TOOL_CMD}" "fail" "Markdown issues"
        exit 1
      fi
    else
      if markdownlint "${config_args[@]}" "${ignore_args[@]}" "${md_globs[@]}" &> /dev/null; then
        print_pass "All Markdown valid"
        json_result "${TOOL_CMD}" "pass" ""
        exit 0
      else
        print_fail "Issues found"
        print_info "Run with -v for details"
        json_result "${TOOL_CMD}" "fail" "Markdown issues"
        exit 1
      fi
    fi
    ;;
  fix)
    if ${VERBOSE}; then
      if markdownlint "${config_args[@]}" "${ignore_args[@]}" --fix "${md_globs[@]}"; then
        print_pass "Fixed Markdown issues"
        json_result "${TOOL_CMD}" "pass" "Fixed"
        exit 0
      else
        print_fail "Could not fix all issues"
        json_result "${TOOL_CMD}" "fail" "Some unfixable"
        exit 1
      fi
    else
      if markdownlint "${config_args[@]}" "${ignore_args[@]}" --fix "${md_globs[@]}" &> /dev/null; then
        print_pass "Fixed Markdown issues"
        json_result "${TOOL_CMD}" "pass" "Fixed"
        exit 0
      else
        print_fail "Could not fix all issues"
        json_result "${TOOL_CMD}" "fail" "Some unfixable"
        exit 1
      fi
    fi
    ;;
esac
