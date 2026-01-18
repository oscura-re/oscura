#!/usr/bin/env bash
# =============================================================================
# check_links.sh - Link Validation with lychee
# =============================================================================
# Usage: ./scripts/tools/check_links.sh [--check|--online] [--json] [-v] [paths...]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Configuration
TOOL_NAME="lychee"
TOOL_CMD="lychee"
INSTALL_HINT="brew install lychee (macOS) or cargo install lychee (Linux)"

# Defaults
MODE="offline" # Default to offline (internal links only)
VERBOSE=false
PATHS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --check)
      MODE="offline"
      shift
      ;;
    --online)
      MODE="online"
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
      echo "Link validation with lychee"
      echo ""
      echo "Options:"
      echo "  --check     Check internal links only (offline mode, default)"
      echo "  --online    Check all links including external URLs"
      echo "  --json      Output machine-readable JSON"
      echo "  -v          Verbose output"
      echo "  -h, --help  Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                  # Check all markdown in current directory (offline)"
      echo "  $0 --online docs/   # Check docs/ including external URLs"
      echo "  $0 README.md        # Check specific file"
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
print_tool "${TOOL_NAME} (${MODE} mode)"

# Check tool installed
if ! require_tool "${TOOL_CMD}" "${INSTALL_HINT}"; then
  json_result "${TOOL_CMD}" "skip" "Tool not installed"
  exit 0
fi

# Build file list for lychee
files_to_check=()
for path in "${PATHS[@]}"; do
  if [[ -f "${path}" ]]; then
    files_to_check+=("${path}")
  elif [[ -d "${path}" ]]; then
    # Find all markdown files recursively
    while IFS= read -r -d '' file; do
      files_to_check+=("${file}")
    done < <(find "${path}" -type f -name "*.md" -print0 2> /dev/null)
  fi
done

if [[ ${#files_to_check[@]} -eq 0 ]]; then
  print_skip "No markdown files found"
  json_result "${TOOL_CMD}" "skip" "No files"
  exit 0
fi

# Check for config file
config_args=()
if [[ -f "${REPO_ROOT}/lychee.toml" ]]; then
  config_args+=("--config" "${REPO_ROOT}/lychee.toml")
fi

# Common exclusion patterns
exclude_patterns=(
  "^mailto:"                    # Email addresses
  "^file:"                      # Local file URLs
  ".*node_modules.*"            # Dependencies
  ".*\\.venv.*"                 # Python virtual env
  ".*\\.git.*"                  # Git directory
  ".*__pycache__.*"             # Python cache
  ".*\\.coordination.*"         # Coordination files
  ".*\\.claude/agent-outputs.*" # Agent outputs
)

# Build exclude arguments
exclude_args=()
for pattern in "${exclude_patterns[@]}"; do
  exclude_args+=("--exclude" "${pattern}")
done

# Mode-specific arguments
mode_args=()
case ${MODE} in
  offline)
    mode_args+=("--offline")
    ;;
  online)
    mode_args+=("--timeout" "30")
    mode_args+=("--max-redirects" "5")
    mode_args+=("--retry-wait-time" "2")
    ;;
esac

# Verbose arguments
verbose_args=()
if ${VERBOSE}; then
  verbose_args+=("--verbose")
else
  verbose_args+=("--no-progress")
fi

# Run lychee
if lychee \
  "${config_args[@]}" \
  "${mode_args[@]}" \
  "${exclude_args[@]}" \
  "${verbose_args[@]}" \
  --format detailed \
  "${files_to_check[@]}"; then

  print_pass "All links valid"
  json_result "${TOOL_CMD}" "pass" ""
  exit 0
else
  exit_code=$?

  if [[ ${exit_code} -eq 2 ]]; then
    # Exit code 2 means broken links found
    print_fail "Broken links found"
    if ! ${VERBOSE}; then
      print_info "Run with -v for details"
    fi
    json_result "${TOOL_CMD}" "fail" "Broken links"
    exit 1
  else
    # Other error (tool error, config error, etc.)
    print_fail "Link check failed"
    json_result "${TOOL_CMD}" "fail" "Check error"
    exit 1
  fi
fi
