#!/usr/bin/env bash
# =============================================================================
# lint.sh - Run All Linters
# =============================================================================
# Usage: ./scripts/lint.sh [--json] [-v] [--python|--shell|--yaml|--markdown|--json-files]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Defaults
JSON_ARGS=""
VERBOSE_ARGS=""

# Filter flags
RUN_ALL=true
RUN_PYTHON=false
RUN_SHELL=false
RUN_YAML=false
RUN_JSON=false
RUN_MARKDOWN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --json)
      enable_json
      JSON_ARGS="--json"
      shift
      ;;
    -v | --verbose)
      VERBOSE_ARGS="-v"
      shift
      ;;
    --python)
      RUN_ALL=false
      RUN_PYTHON=true
      shift
      ;;
    --shell)
      RUN_ALL=false
      RUN_SHELL=true
      shift
      ;;
    --yaml)
      RUN_ALL=false
      RUN_YAML=true
      shift
      ;;
    --json-files)
      RUN_ALL=false
      RUN_JSON=true
      shift
      ;;
    --markdown)
      RUN_ALL=false
      RUN_MARKDOWN=true
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Run all linters across the codebase"
      echo ""
      echo "Options:"
      echo "  --json        Output machine-readable JSON"
      echo "  -v, --verbose Verbose output"
      echo "  --python      Lint only Python files"
      echo "  --shell       Lint only shell scripts"
      echo "  --yaml        Lint only YAML files"
      echo "  --json-files  Lint only JSON/JSONC files"
      echo "  --markdown    Lint only Markdown files"
      echo "  -h, --help    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

print_header "OSCURA - LINT ALL"
echo ""
echo -e "  ${DIM}Repository:${NC} ${REPO_ROOT}"
echo -e "  ${DIM}Timestamp:${NC}  $(date '+%Y-%m-%d %H:%M:%S')"

reset_counters

run_linter() {
  local script="$1"
  local args="${2:-}"

  # Check if script exists
  if [[ ! -f "${SCRIPT_DIR}/tools/${script}" ]]; then
    increment_skipped
    return
  fi

  # shellcheck disable=SC2086
  if "${SCRIPT_DIR}/tools/${script}" ${args} ${JSON_ARGS} ${VERBOSE_ARGS}; then
    increment_passed
  else
    increment_failed
  fi
}

# Python
if ${RUN_ALL} || ${RUN_PYTHON}; then
  print_header "Python"
  run_linter "ruff.sh" "--check"
  run_linter "mypy.sh"
fi

# Shell
if ${RUN_ALL} || ${RUN_SHELL}; then
  print_header "Shell"
  run_linter "shellcheck.sh"
fi

# Markup & Data
if ${RUN_ALL} || ${RUN_YAML}; then
  print_header "YAML"
  run_linter "yamllint.sh"
fi

# JSON/JSONC
if ${RUN_ALL} || ${RUN_JSON}; then
  print_header "JSON/JSONC"
  run_linter "jsonc.sh" "--check"
fi

# Markdown
if ${RUN_ALL} || ${RUN_MARKDOWN}; then
  print_header "Markdown"
  run_linter "markdownlint.sh" "--check"
fi

# Summary
if print_summary "lint"; then
  echo -e "  ${GREEN}${BOLD}All linters passed!${NC}"
  exit 0
else
  echo -e "  ${RED}${BOLD}${FAILED} linter(s) failed${NC}"
  echo -e "  ${DIM}Run with -v for details${NC}"
  exit 1
fi
