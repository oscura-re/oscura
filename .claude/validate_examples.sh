#!/usr/bin/env bash
# Validate all examples execute successfully

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/.claude/analysis/example_validation"

mkdir -p "${LOG_DIR}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
total=0
passed=0
failed=0
skipped=0

# Find all example files
mapfile -t examples < <(find "${PROJECT_ROOT}/examples" "${PROJECT_ROOT}/demonstrations" "${PROJECT_ROOT}/demos" \
  -name "*.py" \
  -not -name "__init__.py" \
  -not -path "*/common/*" \
  -not -path "*/data_generation/*" \
  2>/dev/null | sort)

echo "Found ${#examples[@]} example files to validate"
echo "=========================================="
echo ""

for example in "${examples[@]}"; do
  total=$((total + 1))
  rel_path="${example#${PROJECT_ROOT}/}"
  log_file="${LOG_DIR}/$(basename "${example}").log"

  echo -n "Testing: ${rel_path} ... "

  # Check if example should be skipped (requires external dependencies)
  if grep -q "# SKIP_VALIDATION" "${example}" 2>/dev/null; then
    echo -e "${YELLOW}SKIP${NC} (marked for manual testing)"
    skipped=$((skipped + 1))
    continue
  fi

  # Run example with timeout using uv for package access
  if timeout 30s uv run python "${example}" >"${log_file}" 2>&1; then
    echo -e "${GREEN}PASS${NC}"
    passed=$((passed + 1))
  else
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
      echo -e "${RED}FAIL${NC} (timeout)"
    else
      echo -e "${RED}FAIL${NC} (exit code: $exit_code)"
    fi
    failed=$((failed + 1))

    # Show last 10 lines of error
    echo "  Error output:"
    tail -n 10 "${log_file}" | sed 's/^/    /'
  fi
done

echo ""
echo "=========================================="
echo "Results:"
echo "  Total:   ${total}"
echo -e "  ${GREEN}Passed:  ${passed}${NC}"
echo -e "  ${RED}Failed:  ${failed}${NC}"
echo -e "  ${YELLOW}Skipped: ${skipped}${NC}"
echo ""

if [ ${failed} -eq 0 ]; then
  echo -e "${GREEN}✓ All examples validated successfully${NC}"
  exit 0
else
  echo -e "${RED}✗ ${failed} examples failed validation${NC}"
  echo "Check logs in: ${LOG_DIR}"
  exit 1
fi
