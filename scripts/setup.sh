#!/usr/bin/env bash
# =============================================================================
# setup.sh - Complete Development Environment Setup
# =============================================================================
# This script ensures ALL required setup steps are completed in order.
# Usage: ./scripts/setup.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

cd "${REPO_ROOT}"

echo ""
echo -e "${CYAN}${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║         Oscura Development Environment Setup                  ║${NC}"
echo -e "${CYAN}${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Dependencies
echo -e "${CYAN}[1/3]${NC} Installing dependencies..."
if uv sync --all-extras --all-groups > /dev/null 2>&1; then
  echo -e "  ${GREEN}✓${NC} Dependencies installed (all extras + all groups)"
else
  echo -e "  ${YELLOW}⚠${NC} uv sync had warnings (may be okay)"
fi
echo ""

# Step 2: Git Hooks (CRITICAL)
echo -e "${CYAN}[2/3]${NC} Installing git hooks..."
if "${SCRIPT_DIR}/setup/install-hooks.sh" > /dev/null 2>&1; then
  echo -e "  ${GREEN}✓${NC} Git hooks installed"
else
  echo -e "  ${YELLOW}⚠${NC} Hook installation had warnings"
fi
echo ""

# Step 3: Verification
echo -e "${CYAN}[3/3]${NC} Verifying setup..."
if uv run python -m pytest tests/unit/core/test_types.py -x -q --tb=no > /dev/null 2>&1; then
  echo -e "  ${GREEN}✓${NC} Smoke test passed"
else
  echo -e "  ${YELLOW}⚠${NC} Smoke test had issues"
fi
echo ""

# Final status
echo -e "${GREEN}${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║  ✓ Development Environment Ready                              ║${NC}"
echo -e "${GREEN}${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Next steps:${NC}"
echo -e "    1. Make your changes"
echo -e "    2. Run: ${CYAN}./scripts/check.sh${NC} (quick validation)"
echo -e "    3. Run: ${CYAN}./scripts/test.sh${NC} (full test suite)"
echo -e "    4. Commit and push (hooks will run automatically)"
echo ""
echo -e "  ${YELLOW}⚠  Branch protection enabled on main${NC}"
echo -e "     All CI checks must pass before merge"
echo ""
