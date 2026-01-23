#!/usr/bin/env bash
# =============================================================================
# check.sh - Quick Quality Check (lint + format --check)
# =============================================================================
# Usage: ./scripts/check.sh [--json] [-v] [-h|--help]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

# Defaults
JSON_ARGS=""
VERBOSE_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --json)
      enable_json
      JSON_ARGS="--json"
      shift
      ;;
    -v)
      VERBOSE_ARGS="-v"
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Quick quality check: lint + format --check"
      echo ""
      echo "Options:"
      echo "  --json      Output machine-readable JSON"
      echo "  -v          Verbose output"
      echo "  -h, --help  Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

print_header "QUALITY CHECK"
echo ""
echo -e "  ${DIM}Repository:${NC} ${REPO_ROOT}"
echo -e "  ${DIM}Timestamp:${NC}  $(date '+%Y-%m-%d %H:%M:%S')"

reset_counters

run_check() {
  local script="$1"
  local args="${2:-}"

  # Check if script exists - skip silently if not
  if [[ ! -f "${SCRIPT_DIR}/tools/${script}" ]]; then
    return 0
  fi

  # shellcheck disable=SC2086
  if "${SCRIPT_DIR}/tools/${script}" ${args} ${JSON_ARGS} ${VERBOSE_ARGS}; then
    increment_passed
  else
    increment_failed
  fi
}

# Python (core tools - always run)
print_header "Python"
run_check "ruff.sh" "--check"
run_check "mypy.sh"

# Shell (core tools - always run)
print_header "Shell"
run_check "shellcheck.sh"
run_check "shfmt.sh" "--check"

# Markup & Data (core tools - always run)
print_header "Markup & Data"
run_check "yamllint.sh"
run_check "jsonc.sh" "--check"
run_check "markdownlint.sh" "--check"
run_check "prettier.sh" "--check"
run_check "taplo.sh" "--check"
run_check "xmllint.sh" "--check"

# Diagrams (optional - only run if script exists)
print_header "Diagrams"
run_check "plantuml.sh" "--check"
run_check "mermaid.sh" "--check"

# VS Code validation (optional)
print_header "VS Code"
run_check "vscode_extensions.sh" "--check"

# Summary
if print_summary "check"; then
  echo -e "  ${GREEN}${BOLD}All checks passed!${NC}"
  echo -e "  ${DIM}Ready to commit.${NC}"
  exit 0
else
  echo -e "  ${RED}${BOLD}${FAILED} check(s) failed${NC}"
  echo -e "  ${DIM}Run ./scripts/fix.sh to auto-fix issues${NC}"
  exit 1
fi
