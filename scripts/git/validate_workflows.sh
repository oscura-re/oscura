#!/bin/bash
# Workflow Validation Script
# Validates GitHub Actions workflows for consistency and correctness
# Version: 1.0.0

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

echo "==================================="
echo "GitHub Actions Workflow Validation"
echo "==================================="
echo ""

# Check 1: Validate YAML syntax
echo "📋 Checking YAML syntax..."
for workflow in .github/workflows/*.yml; do
  if ! python3 -c "import yaml; yaml.safe_load(open('$workflow'))" 2> /dev/null; then
    echo -e "${RED}✗${NC} Invalid YAML: $workflow"
    ((ERRORS++))
  else
    echo -e "${GREEN}✓${NC} Valid YAML: $(basename $workflow)"
  fi
done
echo ""

# Check 2: Verify Python setup consistency
echo "🐍 Checking Python setup consistency..."
SETUP_PYTHON_COUNT=$(grep -r "uses: actions/setup-python@" .github/workflows/ 2> /dev/null | wc -l | tr -d ' ' || echo "0")
if [ "${SETUP_PYTHON_COUNT:-0}" -gt 0 ]; then
  echo -e "${YELLOW}⚠${NC} Found $SETUP_PYTHON_COUNT instances of actions/setup-python@ (should use uv)"
  ((WARNINGS++))
else
  echo -e "${GREEN}✓${NC} All workflows use uv for Python setup"
fi
echo ""

# Check 3: Check for duplicate test triggers
echo "🔄 Checking for duplicate test triggers..."
MAIN_PUSH_COUNT=$(grep -A 3 "on:" .github/workflows/tests-chunked.yml 2> /dev/null | grep "push:" | wc -l | tr -d ' ' || echo "0")
if [ "${MAIN_PUSH_COUNT:-0}" -gt 0 ]; then
  echo -e "${RED}✗${NC} tests-chunked.yml has push trigger (should be nightly only)"
  ((ERRORS++))
else
  echo -e "${GREEN}✓${NC} tests-chunked.yml is nightly-only"
fi
echo ""

# Check 4: Verify artifact retention consistency
echo "📦 Checking artifact retention consistency..."
INCONSISTENT_RETENTION=$(grep -r "retention-days:" .github/workflows/ 2> /dev/null | grep -v "env.RETENTION" | grep -v "#" | wc -l | tr -d ' ' || echo "0")
if [ "${INCONSISTENT_RETENTION:-0}" -gt 30 ]; then
  echo -e "${YELLOW}⚠${NC} Found $INCONSISTENT_RETENTION hardcoded retention values (informational)"
  # Don't count as warning - some workflows don't need env vars
else
  echo -e "${GREEN}✓${NC} Artifact retention is reasonable ($INCONSISTENT_RETENTION instances)"
fi
echo ""

# Check 5: Verify reusable action exists
echo "♻️  Checking reusable actions..."
if [ -f ".github/actions/setup-python-env/action.yml" ]; then
  echo -e "${GREEN}✓${NC} Reusable setup-python-env action exists"
else
  echo -e "${RED}✗${NC} Missing reusable setup-python-env action"
  ((ERRORS++))
fi
echo ""

# Check 6: Verify .env.example exists
echo "📄 Checking .env.example..."
if [ -f ".env.example" ]; then
  echo -e "${GREEN}✓${NC} .env.example exists"
else
  echo -e "${RED}✗${NC} Missing .env.example file"
  ((ERRORS++))
fi
echo ""

# Check 7: Count workflows
echo "📊 Workflow Statistics:"
WORKFLOW_COUNT=$(ls -1 .github/workflows/*.yml | wc -l)
TOTAL_LINES=$(wc -l .github/workflows/*.yml | tail -1 | awk '{print $1}')
echo "   Total workflows: $WORKFLOW_COUNT"
echo "   Total lines: $TOTAL_LINES"
echo ""

# Summary
echo "==================================="
echo "Validation Summary"
echo "==================================="
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
  echo -e "${GREEN}✅ All checks passed!${NC}"
  exit 0
elif [ $ERRORS -eq 0 ]; then
  echo -e "${YELLOW}⚠️  $WARNINGS warnings found${NC}"
  exit 0
else
  echo -e "${RED}❌ $ERRORS errors, $WARNINGS warnings${NC}"
  exit 1
fi
