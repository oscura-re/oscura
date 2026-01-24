#!/usr/bin/env bash
# =============================================================================
# validate_required_checks.sh - Validate Branch Protection Required Checks
# =============================================================================
# Ensures that required status checks in branch protection match actual
# CI workflow job names. Prevents configuration drift and ensures all
# required checks actually exist.
#
# Usage: ./scripts/ci/validate_required_checks.sh
#
# Exit codes:
#   0 - All required checks are valid
#   1 - One or more required checks are invalid or missing
#   2 - Script error (gh CLI not available, API error, etc.)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKFLOWS_DIR="${REPO_ROOT}/.github/workflows"

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

# Check if gh CLI is available
check_gh_cli() {
  if ! command -v gh &> /dev/null; then
    echo -e "${RED}ERROR: gh CLI not found${NC}" >&2
    echo "Install: https://cli.github.com/" >&2
    exit 2
  fi

  # Check if authenticated
  if ! gh auth status &> /dev/null; then
    echo -e "${RED}ERROR: Not authenticated with GitHub${NC}" >&2
    echo "Run: gh auth login" >&2
    exit 2
  fi
}

# Get required status checks from branch protection
get_required_checks() {
  local ruleset_id="11977857"

  gh api "repos/oscura-re/oscura/rulesets/${ruleset_id}" \
    --jq '.rules[] | select(.type == "required_status_checks") | .parameters.required_status_checks[].context' \
    2> /dev/null || {
    echo -e "${RED}ERROR: Failed to fetch required status checks${NC}" >&2
    exit 2
  }
}

# Get all status check names from CI workflows
get_workflow_checks() {
  local workflows=(
    "${WORKFLOWS_DIR}/ci.yml"
    "${WORKFLOWS_DIR}/test-quality.yml"
    "${WORKFLOWS_DIR}/code-quality.yml"
    "${WORKFLOWS_DIR}/codeql.yml"
  )

  for workflow in "${workflows[@]}"; do
    if [[ -f "${workflow}" ]]; then
      # Extract job names (lines that start with "  <name>:" in jobs section)
      # This is the job name that becomes the status check name
      yq eval '.jobs | keys[]' "${workflow}" 2> /dev/null || true
    fi
  done
}

# Map job names to their display names (if different)
get_job_display_name() {
  local job_key="$1"
  local workflow_file="$2"

  # Get the "name:" field from the job if it exists
  yq eval ".jobs.${job_key}.name" "${workflow_file}" 2> /dev/null || echo "${job_key}"
}

# Get all possible status check names from workflows
get_all_status_checks() {
  local workflows=(
    "${WORKFLOWS_DIR}/ci.yml"
    "${WORKFLOWS_DIR}/test-quality.yml"
    "${WORKFLOWS_DIR}/code-quality.yml"
    "${WORKFLOWS_DIR}/codeql.yml"
    "${WORKFLOWS_DIR}/docs.yml"
  )

  # Use Python if yq not available (more reliable)
  if ! command -v yq &> /dev/null && command -v python3 &> /dev/null; then
    python3 - "${WORKFLOWS_DIR}" << 'PYTHON_EOF'
import yaml
import sys
from pathlib import Path

workflows_dir = Path(sys.argv[1])
workflow_files = [
    "ci.yml",
    "test-quality.yml",
    "code-quality.yml",
    "codeql.yml",
    "docs.yml",
]

checks = set()

for workflow_file in workflow_files:
    workflow_path = workflows_dir / workflow_file
    if not workflow_path.exists():
        continue

    try:
        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)

        if not workflow or 'jobs' not in workflow:
            continue

        for job_key, job_config in workflow['jobs'].items():
            # Use job name if available, otherwise use job key
            if isinstance(job_config, dict) and 'name' in job_config:
                checks.add(job_config['name'])
            else:
                checks.add(job_key)
    except Exception:
        continue

for check in sorted(checks):
    print(check)
PYTHON_EOF
    return
  fi

  # Use yq if available
  declare -A checks

  for workflow in "${workflows[@]}"; do
    if [[ -f "${workflow}" ]]; then
      # Get all job keys
      local job_keys
      job_keys=$(yq eval '.jobs | keys[]' "${workflow}" 2> /dev/null || true)

      while IFS= read -r job_key; do
        [[ -z "${job_key}" ]] && continue

        # Get job display name
        local display_name
        display_name=$(yq eval ".jobs.${job_key}.name" "${workflow}" 2> /dev/null || echo "")

        # Use display name if available, otherwise use job key
        if [[ -n "${display_name}" && "${display_name}" != "null" ]]; then
          checks["${display_name}"]=1
        else
          checks["${job_key}"]=1
        fi
      done <<< "${job_keys}"
    fi
  done

  # Print all unique check names
  for check in "${!checks[@]}"; do
    echo "${check}"
  done
}

# =============================================================================
# Main
# =============================================================================

cd "${REPO_ROOT}"

echo ""
echo -e "${CYAN}${BOLD}=========================================${NC}"
echo -e "${CYAN}${BOLD}Required Status Checks Validation${NC}"
echo -e "${CYAN}${BOLD}=========================================${NC}"
echo ""

# Check prerequisites
check_gh_cli

# Check if yq is available (optional but recommended)
if ! command -v yq &> /dev/null; then
  echo -e "${YELLOW}WARNING: yq not found, using fallback parsing${NC}"
  echo -e "${DIM}For better parsing, install yq: https://github.com/mikefarah/yq${NC}"
  echo ""
fi

# Get required checks from branch protection
echo -e "${DIM}Fetching required status checks from branch protection...${NC}"
mapfile -t required_checks < <(get_required_checks | sort)

if [[ ${#required_checks[@]} -eq 0 ]]; then
  echo -e "${YELLOW}WARNING: No required status checks found${NC}"
  echo ""
  exit 0
fi

echo -e "${DIM}Found ${#required_checks[@]} required status checks${NC}"
echo ""

# Get all available status checks from workflows
echo -e "${DIM}Scanning CI workflows for available status checks...${NC}"
mapfile -t available_checks < <(get_all_status_checks | sort -u)

echo -e "${DIM}Found ${#available_checks[@]} available status checks in workflows${NC}"
echo ""

# Validate each required check
echo -e "${BOLD}Validating required checks:${NC}"
echo ""

declare -a valid_checks=()
declare -a invalid_checks=()

for check in "${required_checks[@]}"; do
  # Check if this required check exists in available checks
  if printf '%s\n' "${available_checks[@]}" | grep -qx "${check}"; then
    echo -e "  ${GREEN}✓${NC} ${check}"
    valid_checks+=("${check}")
  else
    echo -e "  ${RED}✗${NC} ${check} ${DIM}(not found in CI workflows)${NC}"
    invalid_checks+=("${check}")
  fi
done

echo ""
echo -e "${CYAN}${BOLD}=========================================${NC}"
echo -e "${CYAN}${BOLD}Results${NC}"
echo -e "${CYAN}${BOLD}=========================================${NC}"
echo -e "${DIM}Total required checks:${NC} ${#required_checks[@]}"
echo -e "${GREEN}Valid:${NC}   ${#valid_checks[@]}"
echo -e "${RED}Invalid:${NC} ${#invalid_checks[@]}"
echo ""

if [[ ${#invalid_checks[@]} -gt 0 ]]; then
  echo -e "${RED}${BOLD}❌ VALIDATION FAILED${NC}"
  echo ""
  echo "The following required checks do NOT exist in CI workflows:"
  echo ""
  for check in "${invalid_checks[@]}"; do
    echo -e "  ${RED}✗${NC} ${check}"
  done
  echo ""
  echo "This means:"
  echo "  1. These checks will NEVER pass (blocking merges forever)"
  echo "  2. Branch protection configuration needs to be updated"
  echo ""
  echo "Actions:"
  echo "  1. Review .github/workflows/ to find correct check names"
  echo "  2. Update branch protection ruleset (ID: 11977857)"
  echo "  3. Run this script again to verify"
  echo ""
  exit 1
else
  echo -e "${GREEN}${BOLD}✅ VALIDATION PASSED${NC}"
  echo ""
  echo "All required status checks exist in CI workflows."
  echo ""
fi

# Show available checks that are NOT required (informational)
echo -e "${DIM}${BOLD}Available but not required:${NC}"
echo ""
for check in "${available_checks[@]}"; do
  if ! printf '%s\n' "${required_checks[@]}" | grep -qx "${check}"; then
    echo -e "  ${DIM}○${NC} ${check}"
  fi
done
echo ""
