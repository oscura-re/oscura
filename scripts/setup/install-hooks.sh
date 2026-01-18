#!/usr/bin/env bash
# =============================================================================
# install-hooks.sh - Automated Git Hooks Installation
# =============================================================================
# This script is automatically called during development setup to ensure
# ALL developers have the required git hooks installed.
#
# Called by: uv sync (via postinstall script)
# Manual: ./scripts/install-hooks.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

cd "${REPO_ROOT}"

echo ""
echo -e "${CYAN}${BOLD}Oscura Git Hooks Installation${NC}"
echo ""

# =============================================================================
# Install Pre-Commit Hooks (via pre-commit framework)
# =============================================================================

if command -v pre-commit >/dev/null 2>&1; then
  echo -e "  ${CYAN}[1/2]${NC} Installing pre-commit hooks..."
  if pre-commit install >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Pre-commit hooks installed"
  else
    echo -e "  ${YELLOW}⚠${NC} Pre-commit hooks already installed"
  fi
else
  echo -e "  ${YELLOW}⚠${NC} pre-commit not found, skipping framework hooks"
  echo -e "       ${DIM}Run: pip install pre-commit${NC}"
fi

echo ""

# =============================================================================
# Install Pre-Push Hook (custom Oscura verification)
# =============================================================================

echo -e "  ${CYAN}[2/2]${NC} Installing pre-push verification hook..."

if "${SCRIPT_DIR}/setup-git-hooks.sh" >/dev/null 2>&1; then
  echo -e "  ${GREEN}✓${NC} Pre-push hook installed"
else
  echo -e "  ${YELLOW}⚠${NC} Pre-push hook installation had issues"
fi

echo ""
echo -e "${GREEN}${BOLD}✓ Git hooks installation complete!${NC}"
echo ""
echo -e "  ${DIM}Hooks active:${NC}"
echo -e "    • Pre-commit:  Run quality checks on every commit"
echo -e "    • Pre-push:    Run full CI verification before push"
echo ""
echo -e "  ${YELLOW}⚠  Branch protection enabled on main${NC}"
echo -e "     All CI checks must pass before merge can occur"
echo -e "     Pre-push hooks help catch issues before pushing"
echo ""
