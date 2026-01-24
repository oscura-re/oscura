#!/usr/bin/env bash
# =============================================================================
# verify-setup.sh - Verify Development Environment Setup
# =============================================================================
# This script checks that all required setup steps have been completed.
# Usage: ./scripts/verify-setup.sh
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

issues=0

echo ""
echo -e "${CYAN}${BOLD}Development Environment Verification${NC}"
echo ""

# Check 1: Git hooks
if [[ -f ".git/hooks/pre-push" ]] && grep -q "pre-push.sh" ".git/hooks/pre-push" 2> /dev/null; then
  echo -e "  ${GREEN}✓${NC} Pre-push hook installed"
else
  echo -e "  ${RED}✗${NC} Pre-push hook NOT installed"
  echo -e "      ${YELLOW}Fix:${NC} ./scripts/setup/install-hooks.sh"
  ((issues++))
fi

if [[ -f ".git/hooks/pre-commit" ]] && grep -q "pre-commit" ".git/hooks/pre-commit" 2> /dev/null; then
  echo -e "  ${GREEN}✓${NC} Pre-commit hook installed"
else
  echo -e "  ${RED}✗${NC} Pre-commit hook NOT installed"
  echo -e "      ${YELLOW}Fix:${NC} ./scripts/setup/install-hooks.sh"
  ((issues++))
fi

# Check 2: Dependencies
if command -v uv > /dev/null 2>&1; then
  echo -e "  ${GREEN}✓${NC} uv package manager installed"
else
  echo -e "  ${RED}✗${NC} uv NOT installed"
  echo -e "      ${YELLOW}Fix:${NC} curl -LsSf https://astral.sh/uv/install.sh | sh"
  ((issues++))
fi

# Check 3: Virtual environment
if [[ -d ".venv" ]]; then
  echo -e "  ${GREEN}✓${NC} Virtual environment exists"
else
  echo -e "  ${YELLOW}⚠${NC} Virtual environment not found"
  echo -e "      ${YELLOW}Fix:${NC} uv sync --all-extras"
  ((issues++))
fi

# Check 4: Branch protection
if gh api repos/oscura-re/oscura/branches/main/protection > /dev/null 2>&1; then
  echo -e "  ${GREEN}✓${NC} Branch protection enabled"
else
  echo -e "  ${RED}✗${NC} Branch protection DISABLED"
  echo -e "      ${YELLOW}WARNING:${NC} Main branch is not protected!"
  ((issues++))
fi

echo ""

if [[ $issues -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}✓ All checks passed!${NC}"
  echo ""
  exit 0
else
  echo -e "${RED}${BOLD}✗ Found $issues issue(s)${NC}"
  echo ""
  echo -e "  ${BOLD}Quick fix:${NC}"
  echo -e "    ./scripts/setup.sh"
  echo ""
  exit 1
fi
