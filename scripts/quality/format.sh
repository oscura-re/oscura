#!/usr/bin/env bash
# =============================================================================
# format.sh - Run All Formatters
# =============================================================================
# Usage: ./scripts/format.sh [--check] [--json] [-v] [--python|--markdown|...]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

# Defaults
CHECK_ONLY=false
JSON_ARGS=""
VERBOSE_ARGS=""

# Filter flags
RUN_ALL=true
RUN_PYTHON=false
RUN_MARKDOWN=false
RUN_YAML=false
RUN_JSON=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -c | --check)
      CHECK_ONLY=true
      shift
      ;;
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
    --markdown)
      RUN_ALL=false
      RUN_MARKDOWN=true
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
    -h | --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Run all formatters across the codebase"
      echo ""
      echo "Options:"
      echo "  -c, --check   Check only, don't modify files"
      echo "  --json        Output machine-readable JSON"
      echo "  -v, --verbose Verbose output"
      echo "  --python      Format only Python files"
      echo "  --markdown    Format only Markdown files"
      echo "  --yaml        Format only YAML files"
      echo "  --json-files  Format only JSON files"
      echo "  -h, --help    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if ${CHECK_ONLY}; then
  print_header "OSCURA - FORMAT CHECK"
else
  print_header "OSCURA - FORMAT ALL"
fi
echo ""
echo -e "  ${DIM}Repository:${NC} ${REPO_ROOT}"
echo -e "  ${DIM}Timestamp:${NC}  $(date '+%Y-%m-%d %H:%M:%S')"
if ${CHECK_ONLY}; then
  echo -e "  ${DIM}Mode:${NC}       Check only (no changes)"
fi

reset_counters

# Determine mode argument
MODE_ARG="--fix"
${CHECK_ONLY} && MODE_ARG="--check"

run_formatter() {
  local script="$1"
  local mode="${2:-${MODE_ARG}}"

  # Check if script exists
  if [[ ! -f "${SCRIPT_DIR}/tools/${script}" ]]; then
    increment_skipped
    return
  fi

  # shellcheck disable=SC2086
  if "${SCRIPT_DIR}/tools/${script}" ${mode} ${JSON_ARGS} ${VERBOSE_ARGS}; then
    # Exit code 0 means success
    if ${CHECK_ONLY}; then
      increment_unchanged
    else
      increment_formatted
    fi
  else
    # Non-zero exit code
    if ${CHECK_ONLY}; then
      increment_formatted # Would format
    else
      increment_failed
    fi
  fi
}

# Python
if ${RUN_ALL} || ${RUN_PYTHON}; then
  print_header "Python"
  run_formatter "ruff.sh" "--format"
  if ! ${CHECK_ONLY}; then
    run_formatter "ruff.sh" "--fix"
  fi
fi

# Markdown
if ${RUN_ALL} || ${RUN_MARKDOWN}; then
  print_header "Markdown"
  run_formatter "markdownlint.sh"
  run_formatter "prettier.sh" "${MODE_ARG} --md-only"
fi

# YAML
if ${RUN_ALL} || ${RUN_YAML}; then
  print_header "YAML"
  run_formatter "prettier.sh" "${MODE_ARG} --yaml-only"
fi

# JSON
if ${RUN_ALL} || ${RUN_JSON}; then
  print_header "JSON"
  run_formatter "prettier.sh" "${MODE_ARG} --json-only"
fi

# Summary
print_header "SUMMARY"
echo ""
if ${CHECK_ONLY}; then
  echo -e "  ${YELLOW}Would format:${NC} ${FORMATTED}"
else
  echo -e "  ${GREEN}Formatted:${NC}   ${FORMATTED}"
fi
echo -e "  ${DIM}Unchanged:${NC}   ${UNCHANGED}"
echo -e "  ${RED}Failed:${NC}      ${FAILED}"
echo -e "  ${YELLOW}Skipped:${NC}     ${SKIPPED}"
echo ""

if [[ ${FAILED} -eq 0 ]]; then
  if ${CHECK_ONLY}; then
    if [[ ${FORMATTED} -eq 0 ]]; then
      echo -e "  ${GREEN}${BOLD}All files are properly formatted!${NC}"
    else
      echo -e "  ${YELLOW}${BOLD}${FORMATTED} file(s) would be formatted${NC}"
      echo -e "  ${DIM}Run without --check to apply changes${NC}"
      exit 1
    fi
  else
    if [[ ${FORMATTED} -eq 0 ]]; then
      echo -e "  ${GREEN}${BOLD}All files are properly formatted!${NC}"
    else
      echo -e "  ${GREEN}${BOLD}Formatted ${FORMATTED} file(s) successfully!${NC}"
    fi
  fi
  exit 0
else
  echo -e "  ${RED}${BOLD}Some formatting operations failed${NC}"
  exit 1
fi
