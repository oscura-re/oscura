#!/usr/bin/env bash
# =============================================================================
# setup-git-hooks.sh - Install Git Hooks for Oscura
# =============================================================================
# This script installs custom git hooks that enforce CI verification before push.
#
# Usage: ./scripts/setup-git-hooks.sh [OPTIONS]
#
# Options:
#   --uninstall   Remove the custom hooks
#   --status      Show current hook installation status
#   -h, --help    Show this help message
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOOKS_SOURCE="${REPO_ROOT}/scripts/hooks"
HOOKS_TARGET="${REPO_ROOT}/.git/hooks"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# =============================================================================
# Functions
# =============================================================================

show_help() {
  cat << 'EOF'
Install Git Hooks for Oscura

This script installs custom git hooks that enforce CI verification before push.

USAGE:
    ./scripts/setup-git-hooks.sh [OPTIONS]

OPTIONS:
    --uninstall   Remove the custom hooks
    --status      Show current hook installation status
    -h, --help    Show this help message

HOOKS INSTALLED:
    pre-push      Runs pre-push.sh verification before allowing push

WHAT IT DOES:
    - For pushes to main/develop: Runs FULL verification
    - For pushes to other branches: Runs QUICK verification
    - Blocks push if verification fails

BRANCH PROTECTION:
    The main branch has GitHub branch protection enabled.
    All CI checks must pass before merge can occur.
    Pre-push hooks help catch issues BEFORE pushing.

NOTE:
    This is separate from pre-commit hooks (which run on commit).
    Pre-commit hooks are installed via: pre-commit install

EOF
}

show_status() {
  echo ""
  echo -e "${CYAN}${BOLD}Git Hooks Status${NC}"
  echo ""

  # Check pre-push hook
  if [[ -f "${HOOKS_TARGET}/pre-push" ]]; then
    if grep -q "pre-push.sh" "${HOOKS_TARGET}/pre-push" 2> /dev/null; then
      echo -e "  ${GREEN}[INSTALLED]${NC} pre-push (Oscura verification)"
    else
      echo -e "  ${YELLOW}[CUSTOM]${NC} pre-push (not Oscura hook)"
    fi
  else
    echo -e "  ${RED}[NOT INSTALLED]${NC} pre-push"
  fi

  # Check pre-commit hook (managed by pre-commit)
  if [[ -f "${HOOKS_TARGET}/pre-commit" ]]; then
    if grep -q "pre-commit" "${HOOKS_TARGET}/pre-commit" 2> /dev/null; then
      echo -e "  ${GREEN}[INSTALLED]${NC} pre-commit (pre-commit framework)"
    else
      echo -e "  ${YELLOW}[CUSTOM]${NC} pre-commit (not pre-commit framework)"
    fi
  else
    echo -e "  ${RED}[NOT INSTALLED]${NC} pre-commit"
    echo -e "       ${DIM}Install with: pre-commit install${NC}"
  fi

  echo ""
}

install_hooks() {
  echo ""
  echo -e "${CYAN}${BOLD}Installing Git Hooks${NC}"
  echo ""

  # Ensure hooks directory exists
  mkdir -p "${HOOKS_TARGET}"

  # Install pre-push hook
  local hook_name="pre-push"
  local source_hook="${HOOKS_SOURCE}/${hook_name}"
  local target_hook="${HOOKS_TARGET}/${hook_name}"

  if [[ ! -f "${source_hook}" ]]; then
    echo -e "  ${RED}[ERROR]${NC} Source hook not found: ${source_hook}"
    exit 2
  fi

  if [[ -f "${target_hook}" ]]; then
    # Check if it's our hook
    if grep -q "pre-push.sh" "${target_hook}" 2> /dev/null; then
      echo -e "  ${YELLOW}[UPDATING]${NC} ${hook_name} (already installed, updating)"
    else
      # Backup existing hook
      local backup="${target_hook}.backup.$(date +%Y%m%d%H%M%S)"
      cp "${target_hook}" "${backup}"
      echo -e "  ${YELLOW}[BACKUP]${NC} Existing ${hook_name} backed up to: ${backup}"
    fi
  fi

  # Copy and make executable
  cp "${source_hook}" "${target_hook}"
  chmod +x "${target_hook}"
  echo -e "  ${GREEN}[INSTALLED]${NC} ${hook_name}"

  echo ""
  echo -e "${GREEN}${BOLD}Installation complete!${NC}"
  echo ""
  echo -e "  The pre-push hook will now:"
  echo -e "    - Run FULL verification for pushes to main/develop"
  echo -e "    - Run QUICK verification for pushes to other branches"
  echo -e "    - Block push if verification fails"
  echo ""
  echo -e "  To bypass (use sparingly): ${YELLOW}git push --no-verify${NC}"
  echo ""
}

uninstall_hooks() {
  echo ""
  echo -e "${CYAN}${BOLD}Uninstalling Git Hooks${NC}"
  echo ""

  local hook_name="pre-push"
  local target_hook="${HOOKS_TARGET}/${hook_name}"

  if [[ -f "${target_hook}" ]]; then
    if grep -q "pre-push.sh" "${target_hook}" 2> /dev/null; then
      rm "${target_hook}"
      echo -e "  ${GREEN}[REMOVED]${NC} ${hook_name}"
    else
      echo -e "  ${YELLOW}[SKIPPED]${NC} ${hook_name} (not a Oscura hook)"
    fi
  else
    echo -e "  ${YELLOW}[SKIPPED]${NC} ${hook_name} (not installed)"
  fi

  echo ""
  echo -e "${GREEN}${BOLD}Uninstallation complete!${NC}"
  echo ""
}

# =============================================================================
# Main
# =============================================================================

cd "${REPO_ROOT}"

# Parse arguments
case "${1:-}" in
  --status)
    show_status
    ;;
  --uninstall)
    uninstall_hooks
    ;;
  -h | --help)
    show_help
    ;;
  "")
    install_hooks
    ;;
  *)
    echo "Unknown option: $1" >&2
    echo "Use --help for usage information" >&2
    exit 2
    ;;
esac
