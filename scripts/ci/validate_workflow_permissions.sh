#!/usr/bin/env bash
# ==============================================================================
# Workflow Permissions Validation Script
# ==============================================================================
# Purpose: Ensures all GitHub Actions workflows have explicit permissions
#          blocks following the principle of least privilege.
#
# This prevents security issues from workflows inheriting default repository
# permissions, which may be overly permissive.
#
# Exit codes:
#   0 - All workflows have explicit permissions
#   1 - One or more workflows are missing explicit permissions
#
# Usage: ./scripts/ci/validate_workflow_permissions.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
WORKFLOWS_DIR=".github/workflows"

echo "=========================================="
echo "Workflow Permissions Validation"
echo "=========================================="
echo ""

# Check if workflows directory exists
if [[ ! -d "$WORKFLOWS_DIR" ]]; then
  echo -e "${RED}❌ Workflows directory not found: $WORKFLOWS_DIR${NC}"
  exit 1
fi

# Get all workflow files
echo "Scanning workflows in $WORKFLOWS_DIR..."
workflows=()
while IFS= read -r -d '' file; do
  workflows+=("$file")
done < <(find "$WORKFLOWS_DIR" \( -name "*.yml" -o -name "*.yaml" \) -print0 | sort -z)

echo "Found ${#workflows[@]} workflow files"
echo ""

# Check each workflow has explicit permissions
missing=()
has_permissions=()

for workflow in "${workflows[@]}"; do
  filename=$(basename "$workflow")

  # Check if workflow has top-level permissions block
  # We look for "permissions:" at the start of a line (allowing whitespace)
  # This should be at the top level, not nested in jobs
  if grep -q "^permissions:" "$workflow"; then
    has_permissions+=("$filename")
    echo -e "${GREEN}✓${NC} $filename"
  else
    missing+=("$filename")
    echo -e "${RED}✗${NC} $filename"
  fi
done

echo ""
echo "=========================================="
echo "Results"
echo "=========================================="
echo "Total workflows: ${#workflows[@]}"
echo -e "${GREEN}With permissions: ${#has_permissions[@]}${NC}"
echo -e "${RED}Missing permissions: ${#missing[@]}${NC}"
echo ""

if [ ${#missing[@]} -gt 0 ]; then
  echo -e "${RED}❌ VALIDATION FAILED${NC}"
  echo ""
  echo "The following workflows are missing explicit permissions blocks:"
  echo ""
  for workflow in "${missing[@]}"; do
    echo -e "  ${RED}✗${NC} $workflow"
  done
  echo ""
  echo "Add an explicit permissions block at the top level of each workflow:"
  echo ""
  echo "Example:"
  echo "  name: My Workflow"
  echo "  on: [push, pull_request]"
  echo ""
  echo "  # Explicit permissions for security"
  echo "  permissions:"
  echo "    contents: read        # Read repository contents"
  echo "    pull-requests: read   # Read PR information"
  echo "    checks: write         # Write check results"
  echo ""
  echo "  jobs:"
  echo "    ..."
  echo ""
  echo "Common permission combinations:"
  echo "  - Read-only: contents: read"
  echo "  - CI: contents: read, checks: write"
  echo "  - PR comments: contents: read, pull-requests: write"
  echo "  - Release: contents: write, id-token: write"
  echo ""
  exit 1
fi

echo -e "${GREEN}✅ VALIDATION PASSED${NC}"
echo ""
echo "All workflows have explicit permissions blocks."
echo ""
echo "Security best practice: Principle of least privilege ✓"
exit 0
