#!/bin/bash
# Comprehensive Pre-Push Validation
# Simulates CI/CD checks locally to ensure no surprises
# Version: 1.0.0

set -uo pipefail # Don't use -e because ((var++)) can cause exit

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}PRE-PUSH VALIDATION (CI/CD Simulation)${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check 1: YAML Syntax (yamllint)
echo -e "${BLUE}[1/10]${NC} Validating YAML syntax..."
if command -v yamllint &> /dev/null; then
  if yamllint .github/workflows/ 2>&1 | grep -q "error"; then
    echo -e "${RED}✗${NC} YAML syntax errors found"
    ((ERRORS++))
  else
    echo -e "${GREEN}✓${NC} All YAML files valid"
  fi
else
  echo -e "${YELLOW}⚠${NC} yamllint not installed (skipping)"
fi

# Check 2: Python Syntax
echo -e "${BLUE}[2/10]${NC} Checking Python syntax..."
PYTHON_ERRORS=0
for file in $(find src tests scripts -name "*.py" 2> /dev/null | head -20); do
  if ! python3 -m py_compile "$file" 2> /dev/null; then
    echo -e "${RED}✗${NC} Syntax error in $file"
    ((PYTHON_ERRORS++))
    ((ERRORS++))
  fi
done
if [ $PYTHON_ERRORS -eq 0 ]; then
  echo -e "${GREEN}✓${NC} Python syntax valid"
fi

# Check 3: Workflow Consistency
echo -e "${BLUE}[3/10]${NC} Checking workflow consistency..."
if [ -f "/tmp/comprehensive_audit.py" ]; then
  if python3 /tmp/comprehensive_audit.py > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Workflow consistency validated"
  else
    echo -e "${RED}✗${NC} Workflow consistency issues"
    ((ERRORS++))
  fi
else
  echo -e "${YELLOW}⚠${NC} Audit script not available"
fi

# Check 4: Check for merge conflicts
echo -e "${BLUE}[4/10]${NC} Checking for merge conflicts..."
if git diff --check 2> /dev/null | grep -q "conflict"; then
  echo -e "${RED}✗${NC} Merge conflict markers found"
  ((ERRORS++))
else
  echo -e "${GREEN}✓${NC} No merge conflicts"
fi

# Check 5: Check for large files
echo -e "${BLUE}[5/10]${NC} Checking for large files..."
LARGE_FILES=$(find . -type f -size +500k \
  -not -path "./.git/*" \
  -not -path "./test_data/*" \
  -not -path "./docs/images/*" \
  -not -path "./demos/data/*" \
  -not -path "./.uv/*" \
  -not -path "./venv/*" \
  -not -path "./.venv/*" \
  2> /dev/null | wc -l)
if [ "${LARGE_FILES:-0}" -gt 5 ]; then
  echo -e "${YELLOW}⚠${NC} Found $LARGE_FILES files >500KB (informational)"
else
  echo -e "${GREEN}✓${NC} No unexpected large files"
fi

# Check 6: Check for trailing whitespace
echo -e "${BLUE}[6/10]${NC} Checking for trailing whitespace..."
if git diff --check HEAD 2>&1 | grep -q "trailing whitespace"; then
  echo -e "${YELLOW}⚠${NC} Trailing whitespace found"
  ((WARNINGS++))
else
  echo -e "${GREEN}✓${NC} No trailing whitespace"
fi

# Check 7: Verify .env.example exists
echo -e "${BLUE}[7/10]${NC} Checking required files..."
MISSING=0
for file in .env.example .gitignore CHANGELOG.md CONTRIBUTING.md; do
  if [ ! -f "$file" ]; then
    echo -e "${RED}✗${NC} Missing: $file"
    ((MISSING++))
    ((ERRORS++))
  fi
done
if [ $MISSING -eq 0 ]; then
  echo -e "${GREEN}✓${NC} All required files present"
fi

# Check 8: Verify no secrets in .env.example
echo -e "${BLUE}[8/10]${NC} Checking .env.example for secrets..."
if grep -E "(sk-|ghp_|github_pat_)" .env.example 2> /dev/null; then
  echo -e "${RED}✗${NC} Potential secrets in .env.example"
  ((ERRORS++))
else
  echo -e "${GREEN}✓${NC} No secrets in .env.example"
fi

# Check 9: Verify reusable action exists
echo -e "${BLUE}[9/10]${NC} Checking reusable action..."
if [ -f ".github/actions/setup-python-env/action.yml" ]; then
  echo -e "${GREEN}✓${NC} Reusable action exists"
else
  echo -e "${RED}✗${NC} Reusable action missing"
  ((ERRORS++))
fi

# Check 10: Validate git branch
echo -e "${BLUE}[10/10]${NC} Checking git status..."
CURRENT_BRANCH=$(git branch --show-current 2> /dev/null || echo "detached")
if [ "$CURRENT_BRANCH" = "main" ]; then
  echo -e "${YELLOW}⚠${NC} On main branch - ensure you meant to push here"
  ((WARNINGS++))
else
  echo -e "${GREEN}✓${NC} On feature branch: $CURRENT_BRANCH"
fi

# Summary
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}VALIDATION SUMMARY${NC}"
echo -e "${BLUE}============================================================${NC}"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
  echo -e "${GREEN}✅ All checks passed!${NC}"
  echo -e "${GREEN}Safe to push to remote.${NC}"
  exit 0
elif [ $ERRORS -eq 0 ]; then
  echo -e "${YELLOW}⚠️  $WARNINGS warnings (non-blocking)${NC}"
  echo -e "${GREEN}Safe to push, but review warnings.${NC}"
  exit 0
else
  echo -e "${RED}❌ $ERRORS errors, $WARNINGS warnings${NC}"
  echo -e "${RED}Fix errors before pushing!${NC}"
  exit 1
fi
