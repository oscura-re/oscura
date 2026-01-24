#!/usr/bin/env bash
# =============================================================================
# yamllint.sh - YAML Linting with yamllint
# =============================================================================
# Usage: ./scripts/tools/yamllint.sh [--json] [-v] [paths...]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Configuration
TOOL_NAME="yamllint"
TOOL_CMD="yamllint"
INSTALL_HINT="uv pip install yamllint"

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
    echo "YAML linting with yamllint"
    echo ""
    echo "Options:"
    echo "  --json      Output machine-readable JSON"
    echo "  -v          Verbose output"
    echo "  -h, --help  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 ."
    echo "  $0 demos/05_protocol_decoding/"
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
print_tool "${TOOL_NAME}"

# Check tool installed
if ! require_tool "${TOOL_CMD}" "${INSTALL_HINT}" "true"; then
  json_result "${TOOL_CMD}" "skip" "Tool not installed"
  exit 0
fi

# Find YAML files (only git-tracked files to avoid build artifacts)
yaml_files=()
for path in "${PATHS[@]}"; do
  if [[ -f "${path}" ]]; then
    yaml_files+=("${path}")
  elif [[ -d "${path}" ]]; then
    # Use git ls-files if in a git repo, otherwise fall back to find
    if git rev-parse --git-dir >/dev/null 2>&1; then
      while IFS= read -r file; do
        [[ -f "${file}" ]] && yaml_files+=("${file}")
      done < <(git ls-files "${path}" | grep -E "\.(yaml|yml)$")
    else
      # Fallback to find if not in a git repo
      while IFS= read -r -d "" file; do
        yaml_files+=("${file}")
      done < <(find "${path}" -type f \( -name "*.yaml" -o -name "*.yml" \) \
        -not -path "*/.git/*" \
        -not -path "*/.venv/*" \
        -not -path "*/node_modules/*" \
        -print0 2>/dev/null)
    fi
  fi
done

if [[ ${#yaml_files[@]} -eq 0 ]]; then
  print_skip "No YAML files found"
  json_result "${TOOL_CMD}" "skip" "No files found"
  exit 0
fi

# Build config argument if config exists
config_args=()
config_file=$(find_config ".yamllint.yaml")
if [[ -n "${config_file}" ]]; then
  config_args+=("-c" "${config_file}")
fi

# Run yamllint
if ${VERBOSE}; then
  if run_py_tool yamllint "${config_args[@]}" "${yaml_files[@]}"; then
    print_pass "All YAML files valid"
    json_result "${TOOL_CMD}" "pass" ""
    exit 0
  else
    print_fail "Issues found"
    json_result "${TOOL_CMD}" "fail" "YAML issues"
    exit 1
  fi
else
  if run_py_tool yamllint "${config_args[@]}" "${yaml_files[@]}" &>/dev/null; then
    print_pass "All YAML files valid"
    json_result "${TOOL_CMD}" "pass" ""
    exit 0
  else
    print_fail "Issues found"
    print_info "Run with -v for details"
    json_result "${TOOL_CMD}" "fail" "YAML issues"
    exit 1
  fi
fi
