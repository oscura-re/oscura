#!/usr/bin/env bash
# =============================================================================
# clean.sh - Clean Build Artifacts and Caches
# =============================================================================
# Usage: ./scripts/clean.sh [--dry-run] [--all] [-h|--help]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

# Defaults
DRY_RUN=false
CLEAN_ALL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --all)
      CLEAN_ALL=true
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Clean build artifacts and caches"
      echo ""
      echo "Options:"
      echo "  --dry-run   Show what would be deleted"
      echo "  --all       Also clean venv and node_modules"
      echo "  -h, --help  Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

print_header "CLEAN WORKSPACE"
echo ""
echo -e "  ${DIM}Repository:${NC} ${REPO_ROOT}"
if ${DRY_RUN}; then
  echo -e "  ${YELLOW}Mode:${NC}       Dry run (no deletions)"
fi

cd "${REPO_ROOT}"

# Define cleanup patterns
ALWAYS_CLEAN=(
  # Python
  "__pycache__"
  "*.pyc"
  "*.pyo"
  ".pytest_cache"
  ".mypy_cache"
  ".ruff_cache"
  "*.egg-info"
  "dist"
  "build"
  # Editors
  "*~"
  "*.swp"
  "*.swo"
  ".DS_Store"
  # Coverage
  ".coverage"
  "htmlcov"
  "coverage.xml"
  # Backup files
  "*.bak"
)

OPTIONAL_CLEAN=(
  ".venv"
  "venv"
  "node_modules"
  ".uv"
)

clean_pattern() {
  local pattern="$1"
  local found=0

  # Use find for directories and files
  if [[ "${pattern}" == *"*"* ]]; then
    # Glob pattern - use find with -name
    while IFS= read -r -d '' item; do
      ((found++)) || true
      if ${DRY_RUN}; then
        print_info "Would delete: ${item}"
      else
        rm -rf "${item}"
      fi
    done < <(find . -name "${pattern}" -not -path "./.git/*" -print0 2> /dev/null)
  else
    # Exact name - use find with -name
    while IFS= read -r -d '' item; do
      ((found++)) || true
      if ${DRY_RUN}; then
        print_info "Would delete: ${item}"
      else
        rm -rf "${item}"
      fi
    done < <(find . -name "${pattern}" -not -path "./.git/*" -print0 2> /dev/null)
  fi

  return ${found}
}

# Clean standard artifacts
print_header "Build Artifacts & Caches"

total_cleaned=0
for pattern in "${ALWAYS_CLEAN[@]}"; do
  if clean_pattern "${pattern}"; then
    count=$?
    ((total_cleaned += count)) || true
  fi
done

if [[ ${total_cleaned} -eq 0 ]]; then
  print_skip "No artifacts found"
else
  if ${DRY_RUN}; then
    print_info "Would clean ${total_cleaned} item(s)"
  else
    print_pass "Cleaned ${total_cleaned} item(s)"
  fi
fi

# Clean optional items
if ${CLEAN_ALL}; then
  print_header "Virtual Environments & Dependencies"

  optional_cleaned=0
  for pattern in "${OPTIONAL_CLEAN[@]}"; do
    if clean_pattern "${pattern}"; then
      count=$?
      ((optional_cleaned += count)) || true
    fi
  done

  if [[ ${optional_cleaned} -eq 0 ]]; then
    print_skip "No environments found"
  else
    if ${DRY_RUN}; then
      print_info "Would clean ${optional_cleaned} item(s)"
    else
      print_pass "Cleaned ${optional_cleaned} item(s)"
    fi
  fi
fi

# Summary
print_header "SUMMARY"
echo ""
if ${DRY_RUN}; then
  echo -e "  ${YELLOW}${BOLD}Dry run complete - no files were deleted${NC}"
  echo -e "  ${DIM}Run without --dry-run to actually clean${NC}"
else
  echo -e "  ${GREEN}${BOLD}Cleanup complete!${NC}"
fi

if ! ${CLEAN_ALL}; then
  echo -e "  ${DIM}Run with --all to also clean .venv and node_modules${NC}"
fi
