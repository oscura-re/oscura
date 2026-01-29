#!/usr/bin/env bash
# =============================================================================
# check_version.sh - Version Number Consistency Checker (SSOT Enforcement)
# =============================================================================
# Ensures version numbers are consistent across all files in the repository.
# This prevents version drift and ensures SSOT compliance.
#
# Usage:
#   ./scripts/check_version.sh          # Check for version consistency
#   ./scripts/check_version.sh --fix    # Auto-fix version mismatches
#   ./scripts/check_version.sh --update <version>  # Update all to new version
#
# SSOT: Version is defined ONCE in pyproject.toml
# All other files derive from this source of truth.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=../lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"

# =============================================================================
# Configuration
# =============================================================================

# SSOT: pyproject.toml is the single source of truth for version
SSOT_FILE="${REPO_ROOT}/pyproject.toml"

# Files that must match the SSOT version
VERSION_FILES=(
  "src/oscura/__init__.py"
  "CITATION.cff"
)

# Files that contain version references but don't require updates
# (e.g., CHANGELOG.md documents version history)
INFORMATIONAL_FILES=(
  "CHANGELOG.md"
  "RELEASE_CHECKLIST.md"
)

# =============================================================================
# Functions
# =============================================================================

get_ssot_version() {
  grep '^version = ' "${SSOT_FILE}" | sed 's/version = "\(.*\)"/\1/'
}

get_file_version() {
  local file="$1"
  case "${file}" in
    */__init__.py)
      grep '^__version__ = ' "${REPO_ROOT}/${file}" | sed 's/__version__ = "\(.*\)"/\1/'
      ;;
    *CITATION.cff)
      grep '^version: ' "${REPO_ROOT}/${file}" | sed 's/version: //; s/"//g' | tr -d ' '
      ;;
    *)
      echo "ERROR: Unknown file type: ${file}" >&2
      return 1
      ;;
  esac
}

update_file_version() {
  local file="$1"
  local new_version="$2"

  case "${file}" in
    */__init__.py)
      sed -i "s/^__version__ = \".*\"/__version__ = \"${new_version}\"/" "${REPO_ROOT}/${file}"
      ;;
    *CITATION.cff)
      sed -i "s/^version: .*/version: ${new_version}/" "${REPO_ROOT}/${file}"
      ;;
    *)
      echo "ERROR: Unknown file type: ${file}" >&2
      return 1
      ;;
  esac
}

# =============================================================================
# Main Logic
# =============================================================================

MODE="check"
NEW_VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fix)
      MODE="fix"
      shift
      ;;
    --update)
      MODE="update"
      NEW_VERSION="$2"
      shift 2
      ;;
    -h | --help)
      cat << 'EOF'
Usage: ./scripts/check_version.sh [OPTIONS]

Check and enforce version number consistency across repository.

Options:
  --fix              Auto-fix version mismatches to match SSOT
  --update <version> Update SSOT and all files to new version
  -h, --help         Show this help message

SSOT: pyproject.toml is the single source of truth for version.

Exit codes:
  0 - All versions consistent
  1 - Version mismatches found (check mode)
  2 - Error in execution
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

print_header "VERSION CONSISTENCY CHECK"
echo ""

# Get SSOT version
if [[ ! -f "${SSOT_FILE}" ]]; then
  echo -e "${RED}ERROR: SSOT file not found: ${SSOT_FILE}${NC}"
  exit 2
fi

SSOT_VERSION=$(get_ssot_version)
echo -e "  ${CYAN}SSOT Version (pyproject.toml):${NC} ${BOLD}${SSOT_VERSION}${NC}"
echo ""

# Update mode: change SSOT first
if [[ "${MODE}" == "update" ]]; then
  if [[ -z "${NEW_VERSION}" ]]; then
    echo -e "${RED}ERROR: No version specified for --update${NC}"
    exit 2
  fi

  echo -e "${YELLOW}Updating SSOT to version ${NEW_VERSION}...${NC}"
  sed -i "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" "${SSOT_FILE}"
  SSOT_VERSION="${NEW_VERSION}"
  echo -e "${GREEN}✓ Updated ${SSOT_FILE}${NC}"
  echo ""
fi

# Check all version files
MISMATCHES=0
echo -e "${CYAN}Checking version consistency:${NC}"
echo ""

for file in "${VERSION_FILES[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${file}" ]]; then
    echo -e "  ${YELLOW}[SKIP]${NC} ${file} (file not found)"
    continue
  fi

  FILE_VERSION=$(get_file_version "${file}")

  if [[ "${FILE_VERSION}" == "${SSOT_VERSION}" ]]; then
    echo -e "  ${GREEN}[OK]${NC}   ${file}: ${FILE_VERSION}"
  else
    echo -e "  ${RED}[FAIL]${NC} ${file}: ${FILE_VERSION} (expected ${SSOT_VERSION})"
    MISMATCHES=$((MISMATCHES + 1))

    if [[ "${MODE}" == "fix" ]] || [[ "${MODE}" == "update" ]]; then
      update_file_version "${file}" "${SSOT_VERSION}"
      echo -e "         ${GREEN}✓ Fixed to ${SSOT_VERSION}${NC}"
    fi
  fi
done

echo ""

# Summary
if [[ ${MISMATCHES} -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}✓ All versions consistent!${NC}"
  echo -e "${DIM}All files match SSOT version ${SSOT_VERSION}${NC}"
  exit 0
else
  if [[ "${MODE}" == "check" ]]; then
    echo -e "${RED}${BOLD}✗ Found ${MISMATCHES} version mismatch(es)${NC}"
    echo -e "${DIM}Run with --fix to auto-correct${NC}"
    exit 1
  else
    echo -e "${GREEN}${BOLD}✓ Fixed ${MISMATCHES} version mismatch(es)${NC}"
    echo -e "${DIM}All files now match SSOT version ${SSOT_VERSION}${NC}"
    exit 0
  fi
fi
