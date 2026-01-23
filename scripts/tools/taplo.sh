#!/usr/bin/env bash
# =============================================================================
# taplo.sh - TOML Formatting and Linting with Taplo
# =============================================================================
# Usage: ./scripts/tools/taplo.sh [--check|--fix] [--json] [-v] [paths...]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Configuration
TOOL_NAME="Taplo (TOML)"
TOOL_CMD="taplo"
INSTALL_HINT="cargo install taplo-cli --locked"

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
      echo "TOML formatting and linting with Taplo"
      echo ""
      echo "Options:"
      echo "  --check     Check only (default)"
      echo "  --fix       Fix formatting"
      echo "  --json      Output machine-readable JSON"
      echo "  -v          Verbose output"
      echo "  -h, --help  Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --check pyproject.toml"
      echo "  $0 --fix ."
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

# Find TOML files
toml_files=()
for path in "${PATHS[@]}"; do
  if [[ -f "${path}" && "${path}" == *.toml ]]; then
    toml_files+=("${path}")
  elif [[ -d "${path}" ]]; then
    while IFS= read -r -d '' file; do
      toml_files+=("${file}")
    done < <(find "${path}" -type f -name "*.toml" \
      -not -path "*/.git/*" \
      -not -path "*/.venv/*" \
      -not -path "*/node_modules/*" \
      -not -path "*/build/*" \
      -not -path "*/dist/*" \
      -print0 2> /dev/null)
  fi
done

if [[ ${#toml_files[@]} -eq 0 ]]; then
  print_skip "No TOML files found"
  json_result "${TOOL_CMD}" "skip" "No files found"
  exit 0
fi

file_count=${#toml_files[@]}
${VERBOSE} && print_info "Found ${file_count} TOML file(s)"

# Run taplo
case ${MODE} in
  check)
    if ${VERBOSE}; then
      if taplo format --check "${toml_files[@]}"; then
        print_pass "All ${file_count} files formatted correctly"
        json_result "${TOOL_CMD}" "pass" ""
        exit 0
      else
        print_fail "Formatting issues found"
        json_result "${TOOL_CMD}" "fail" "Format issues"
        exit 1
      fi
    else
      if taplo format --check "${toml_files[@]}" &> /dev/null; then
        print_pass "All ${file_count} files formatted correctly"
        json_result "${TOOL_CMD}" "pass" ""
        exit 0
      else
        print_fail "Formatting issues found"
        print_info "Run with --fix to format"
        json_result "${TOOL_CMD}" "fail" "Format issues"
        exit 1
      fi
    fi
    ;;
  fix)
    if ${VERBOSE}; then
      if taplo format "${toml_files[@]}"; then
        print_pass "Formatted ${file_count} file(s)"
        json_result "${TOOL_CMD}" "pass" "Formatted"
        exit 0
      else
        print_fail "Format failed"
        json_result "${TOOL_CMD}" "fail" "Format error"
        exit 1
      fi
    else
      if taplo format "${toml_files[@]}" &> /dev/null; then
        print_pass "Formatted ${file_count} file(s)"
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
