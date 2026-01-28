#!/usr/bin/env bash
# ==============================================================================
# CI Test Coverage Validation Script
# ==============================================================================
# Purpose: Ensures all test directories under tests/unit/ are included in
#          .github/workflows/ci.yml test group configurations.
#
# This prevents situations where test directories are created but not added
# to CI configuration, causing tests to be silently skipped.
#
# Exit codes:
#   0 - All test directories are covered in CI
#   1 - One or more test directories are missing from CI config
#
# Usage: ./scripts/ci/validate_test_coverage.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CI_CONFIG=".github/workflows/ci.yml"
TEST_DIR="tests/unit"

echo "=========================================="
echo "CI Test Coverage Validation"
echo "=========================================="
echo ""

# Check if CI config exists
if [[ ! -f "$CI_CONFIG" ]]; then
  echo -e "${RED}❌ CI configuration not found: $CI_CONFIG${NC}"
  exit 1
fi

# Check if test directory exists
if [[ ! -d "$TEST_DIR" ]]; then
  echo -e "${RED}❌ Test directory not found: $TEST_DIR${NC}"
  exit 1
fi

# Get all test directories (excluding __pycache__ and hidden directories)
echo "Scanning test directories in $TEST_DIR..."
test_dirs=()
while IFS= read -r -d '' dir; do
  dirname=$(basename "$dir")
  # Skip __pycache__ and hidden directories
  if [[ "$dirname" != "__pycache__" && "$dirname" != .* ]]; then
    test_dirs+=("$dirname")
  fi
done < <(find "$TEST_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

echo "Found ${#test_dirs[@]} test directories"
echo ""

# Check each directory is mentioned in CI config
missing=()
covered=()
empty=()

for dir in "${test_dirs[@]}"; do
  # Check if directory has any test files
  test_file_count=$(find "$TEST_DIR/$dir" -name "test_*.py" -o -name "*_test.py" 2>/dev/null | wc -l)

  if [[ $test_file_count -eq 0 ]]; then
    # Skip empty directories - they don't need CI coverage
    empty+=("$dir")
    echo -e "${YELLOW}○${NC} $dir (empty, skipped)"
  elif grep -q "tests/unit/$dir/" "$CI_CONFIG"; then
    covered+=("$dir")
    echo -e "${GREEN}✓${NC} $dir"
  else
    missing+=("$dir")
    echo -e "${RED}✗${NC} $dir"
  fi
done

echo ""
echo "=========================================="
echo "Results"
echo "=========================================="
echo "Total directories: ${#test_dirs[@]}"
echo -e "${GREEN}Covered: ${#covered[@]}${NC}"
echo -e "${YELLOW}Empty (skipped): ${#empty[@]}${NC}"
echo -e "${RED}Missing: ${#missing[@]}${NC}"
echo ""

if [ ${#missing[@]} -gt 0 ]; then
  echo -e "${RED}❌ VALIDATION FAILED${NC}"
  echo ""
  echo "The following test directories are NOT included in CI configuration:"
  echo ""
  for dir in "${missing[@]}"; do
    echo -e "  ${RED}✗${NC} tests/unit/$dir/"
  done
  echo ""
  echo "These directories need to be added to one of the test groups in:"
  echo "  $CI_CONFIG"
  echo ""
  echo "Suggested test groups:"
  echo "  - analyzers: For analyzer tests"
  echo "  - core-protocols-loaders: For core functionality, protocols, loaders"
  echo "  - unit-workflows: For sessions, workflows, pipeline tests"
  echo "  - unit-exploratory: For inference, discovery, guidance tests"
  echo "  - unit-utils: For utilities, config, plugins"
  echo ""
  exit 1
fi

echo -e "${GREEN}✅ VALIDATION PASSED${NC}"
echo ""
echo "All test directories are properly included in CI configuration."
exit 0
