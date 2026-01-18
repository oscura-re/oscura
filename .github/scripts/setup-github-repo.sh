#!/bin/bash
# GitHub Repository Setup Script
# Configures all repository settings using gh CLI
# Usage: ./setup-github-repo.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Get repository information
REPO_OWNER=$(gh repo view --json owner -q '.owner.login')
REPO_NAME=$(gh repo view --json name -q '.name')
REPO_FULL="${REPO_OWNER}/${REPO_NAME}"

log_info "Configuring repository: ${REPO_FULL}"
echo ""

# ============================================================================
# 1. Configure Branch Protection for main
# ============================================================================

log_info "Step 1: Configuring branch protection for 'main'..."

gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO_FULL}/branches/main/protection" \
  -f "required_status_checks[strict]=true" \
  -f "required_status_checks[contexts][]=ci / pre-commit" \
  -f "required_status_checks[contexts][]=ci / lint" \
  -f "required_status_checks[contexts][]=ci / type-check" \
  -f "required_status_checks[contexts][]=ci / test (3.12)" \
  -f "required_status_checks[contexts][]=ci / test (3.13)" \
  -f "enforce_admins=true" \
  -f "required_pull_request_reviews[dismiss_stale_reviews]=true" \
  -f "required_pull_request_reviews[require_code_owner_reviews]=true" \
  -f "required_pull_request_reviews[required_approving_review_count]=1" \
  -f "required_pull_request_reviews[require_last_push_approval]=false" \
  -f "required_conversation_resolution=true" \
  -f "required_linear_history=true" \
  -f "allow_force_pushes=false" \
  -f "allow_deletions=false" \
  -f "block_creations=false" \
  -f "required_signatures=false" \
  -f "lock_branch=false" \
  -f "allow_fork_syncing=true" \
  > /dev/null 2>&1
# shellcheck disable=SC2181
if [[ $? -eq 0 ]]; then
  log_success "Branch protection configured"
else
  log_warning "Branch protection may need manual adjustment (some status checks may not exist yet)"
fi

# ============================================================================
# 2. Configure Repository Settings
# ============================================================================

log_info "Step 2: Configuring repository settings..."

gh api \
  --method PATCH \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO_FULL}" \
  -f "description=Python library for loading, analyzing, and reporting on high-speed digital waveforms and signal integrity measurements" \
  -F "has_issues=true" \
  -F "has_projects=true" \
  -F "has_wiki=false" \
  -F "has_discussions=true" \
  -F "allow_squash_merge=true" \
  -F "allow_merge_commit=false" \
  -F "allow_rebase_merge=true" \
  -F "allow_auto_merge=true" \
  -F "delete_branch_on_merge=true" \
  -F "allow_update_branch=true" \
  -F "squash_merge_commit_title=PR_TITLE" \
  -F "squash_merge_commit_message=PR_BODY" \
  > /dev/null && log_success "Repository settings configured"

# ============================================================================
# 3. Enable Security Features
# ============================================================================

log_info "Step 3: Enabling security features..."

# Enable Dependabot alerts
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO_FULL}/vulnerability-alerts" \
  > /dev/null 2>&1
# shellcheck disable=SC2181
if [[ $? -eq 0 ]]; then
  log_success "Dependabot alerts enabled"
else
  log_warning "Dependabot alerts may already be enabled"
fi

# Enable automated security fixes
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO_FULL}/automated-security-fixes" \
  > /dev/null 2>&1
# shellcheck disable=SC2181
if [[ $? -eq 0 ]]; then
  log_success "Automated security fixes enabled"
else
  log_warning "Automated security fixes may already be enabled"
fi

# Enable secret scanning
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO_FULL}/secret-scanning" \
  > /dev/null 2>&1
# shellcheck disable=SC2181
if [[ $? -eq 0 ]]; then
  log_success "Secret scanning enabled"
else
  log_warning "Secret scanning may require GitHub Advanced Security"
fi

# Enable secret scanning push protection
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO_FULL}/secret-scanning/push-protection" \
  > /dev/null 2>&1
# shellcheck disable=SC2181
if [[ $? -eq 0 ]]; then
  log_success "Secret scanning push protection enabled"
else
  log_warning "Push protection may require GitHub Advanced Security"
fi

# Enable private vulnerability reporting
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO_FULL}/private-vulnerability-reporting" \
  > /dev/null 2>&1
# shellcheck disable=SC2181
if [[ $? -eq 0 ]]; then
  log_success "Private vulnerability reporting enabled"
else
  log_warning "Private vulnerability reporting may already be enabled"
fi

# ============================================================================
# 4. Add Repository Topics
# ============================================================================

log_info "Step 4: Adding repository topics..."

gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO_FULL}/topics" \
  -f "names[]=python" \
  -f "names[]=signal-integrity" \
  -f "names[]=waveform-analysis" \
  -f "names[]=test-automation" \
  -f "names[]=hardware-testing" \
  -f "names[]=oscilloscope" \
  -f "names[]=measurement" \
  -f "names[]=debugging" \
  -f "names[]=developer-tools" \
  -f "names[]=observability" \
  -f "names[]=ieee-181" \
  -f "names[]=jedec" \
  -f "names[]=high-speed-digital" \
  > /dev/null && log_success "Repository topics added"

# ============================================================================
# 5. Create PyPI Environment
# ============================================================================

log_info "Step 5: Creating 'pypi' environment..."

# Create environment
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO_FULL}/environments/pypi" \
  -f "wait_timer=0" \
  -f "prevent_self_review=false" \
  -f "deployment_branch_policy[protected_branches]=false" \
  -f "deployment_branch_policy[custom_branch_policies]=true" \
  > /dev/null && log_success "'pypi' environment created"

# Add deployment branch policy for main
gh api \
  --method POST \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO_FULL}/environments/pypi/deployment-branch-policies" \
  -f "name=main" \
  -f "type=branch" \
  > /dev/null 2>&1 && log_success "Added 'main' branch to pypi deployment policy"

# Add deployment branch policy for version tags
gh api \
  --method POST \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO_FULL}/environments/pypi/deployment-branch-policies" \
  -f "name=v*" \
  -f "type=tag" \
  > /dev/null 2>&1 && log_success "Added 'v*' tags to pypi deployment policy"

# Add required reviewers to environment
REVIEWER_ID=$(gh api /users/"${REPO_OWNER}" -q '.id')
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO_FULL}/environments/pypi" \
  -f "reviewers[0][type]=User" \
  -F "reviewers[0][id]=${REVIEWER_ID}" \
  > /dev/null && log_success "Added required reviewer to pypi environment"

# ============================================================================
# 6. Configure Tag Protection
# ============================================================================

log_info "Step 6: Configuring tag protection..."

gh api \
  --method POST \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/${REPO_FULL}/tags/protection" \
  -f "pattern=v*" \
  > /dev/null 2>&1
# shellcheck disable=SC2181
if [[ $? -eq 0 ]]; then
  log_success "Tag protection configured for 'v*'"
else
  log_warning "Tag protection may already exist"
fi

# ============================================================================
# 7. Configure Labels
# ============================================================================

log_info "Step 7: Configuring repository labels..."

# Function to create or update label
create_or_update_label() {
  local name=$1
  local color=$2
  local description=$3

  gh label create "${name}" --color "${color}" --description "${description}" --force 2> /dev/null \
    && log_success "Label created/updated: ${name}"
}

# Bug tracking labels
create_or_update_label "bug" "d73a4a" "Something isn't working"
create_or_update_label "needs-triage" "ffffff" "Needs initial review"
create_or_update_label "confirmed" "0e8a16" "Bug confirmed and reproducible"
create_or_update_label "wontfix" "666666" "Will not be fixed"

# Features & enhancements
create_or_update_label "enhancement" "a2eeef" "New feature or request"
create_or_update_label "good first issue" "7057ff" "Good for newcomers"
create_or_update_label "help wanted" "008672" "Extra attention needed"
create_or_update_label "question" "d876e3" "Further information requested"

# Documentation
create_or_update_label "documentation" "0075ca" "Documentation improvements"
create_or_update_label "examples" "0075ca" "Example code or tutorials"

# Dependencies & CI
create_or_update_label "dependencies" "0366d6" "Dependency updates"
create_or_update_label "python" "3572A5" "Python dependency updates"
create_or_update_label "github-actions" "000000" "GitHub Actions updates"
create_or_update_label "ci" "000000" "CI/CD pipeline changes"

# Security
create_or_update_label "security" "b60205" "Security-related issues"
create_or_update_label "vulnerability" "b60205" "Security vulnerability"

# Priority
create_or_update_label "priority: critical" "b60205" "Critical priority"
create_or_update_label "priority: high" "d93f0b" "High priority"
create_or_update_label "priority: medium" "fbca04" "Medium priority"
create_or_update_label "priority: low" "0e8a16" "Low priority"

# Breaking changes
create_or_update_label "breaking" "b60205" "Breaking changes (semver major)"

# ============================================================================
# 8. PyPI API Token (will prompt user)
# ============================================================================

log_info "Step 8: Adding PyPI API token..."
echo ""
log_warning "You will now be prompted to enter your PyPI API token."
log_info "The token will be securely stored as a GitHub secret in the 'pypi' environment."
echo ""

# Set the PyPI API token as an environment secret
gh secret set PYPI_API_TOKEN --env pypi && log_success "PyPI API token added to 'pypi' environment"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================================================"
log_success "GitHub repository setup complete!"
echo "========================================================================"
echo ""
log_info "Configured:"
echo "  ✓ Branch protection for 'main' (1 approval required)"
echo "  ✓ Repository settings (squash merge, auto-delete branches)"
echo "  ✓ Security features (Dependabot, secret scanning, vulnerability reporting)"
echo "  ✓ Repository topics for discoverability"
echo "  ✓ 'pypi' environment with deployment restrictions"
echo "  ✓ Tag protection for version tags (v*)"
echo "  ✓ Repository labels for issue/PR management"
echo "  ✓ PyPI API token secret"
echo ""
log_info "Next steps:"
echo "  1. Verify settings: gh repo view --web"
echo "  2. Check branch protection: gh api /repos/${REPO_FULL}/branches/main/protection"
echo "  3. Test the setup with a test PR"
echo "  4. Consider enabling CodeQL (may require manual setup)"
echo ""
log_warning "Note: Some status checks in branch protection may show as not found until CI runs at least once."
echo ""
