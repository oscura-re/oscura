#!/usr/bin/env bash
# =============================================================================
# fix.sh - Quick Fix (format + lint --fix)
# =============================================================================
# Usage: ./scripts/fix.sh [--json] [-v] [-h|--help]
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
      echo "Quick fix: format + lint --fix"
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

print_header "AUTO-FIX"
echo ""
echo -e "  ${DIM}Repository:${NC} ${REPO_ROOT}"
echo -e "  ${DIM}Timestamp:${NC}  $(date '+%Y-%m-%d %H:%M:%S')"

fixed=0
failed=0

run_fix() {
  local script="$1"
  local args="${2:---fix}"

  # Check if script exists
  if [[ ! -f "${SCRIPT_DIR}/tools/${script}" ]]; then
    return 0
  fi

  # shellcheck disable=SC2086
  if "${SCRIPT_DIR}/tools/${script}" ${args} ${JSON_ARGS} ${VERBOSE_ARGS}; then
    ((fixed++)) || true
  else
    ((failed++)) || true
  fi
}

# Python - format first, then fix lint issues
print_header "Python"
run_fix "ruff.sh" "--format"
run_fix "ruff.sh" "--fix"

# Shell - format with shfmt
print_header "Shell"
run_fix "shfmt.sh" "--fix"

# Markup/Data - formatting (YAML, JSON, Markdown)
print_header "Markup & Data"
run_fix "markdownlint.sh" "--fix"
run_fix "prettier.sh" "--fix"

# Cleanup - remove backup and generated files
print_header "Cleanup"
print_section "Removing temporary files"
cleanup_count=0

# Clean up common backup files (*.bak)
while IFS= read -r -d '' file; do
  rm -f "${file}"
  ((cleanup_count++)) || true
done < <(find "${REPO_ROOT}" -type f -name "*.bak" \
  -not -path "*/.git/*" -not -path "*/.venv/*" -not -path "*/node_modules/*" \
  -print0 2> /dev/null)

if [[ ${cleanup_count} -gt 0 ]]; then
  print_pass "Removed ${cleanup_count} temporary file(s)"
else
  print_skip "No temporary files to remove"
fi

# Summary
print_header "SUMMARY"
echo ""
echo -e "  ${GREEN}Fixed:${NC}  ${fixed} tool(s)"
echo -e "  ${RED}Failed:${NC} ${failed} tool(s)"
echo ""

if [[ ${failed} -eq 0 ]]; then
  echo -e "  ${GREEN}${BOLD}All auto-fixes applied!${NC}"
  echo -e "  ${DIM}Run ./scripts/check.sh to verify${NC}"
  exit 0
else
  echo -e "  ${YELLOW}${BOLD}Some fixes could not be applied${NC}"
  echo -e "  ${DIM}Review issues manually${NC}"
  exit 1
fi
