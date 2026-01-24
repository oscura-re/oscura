#!/bin/bash
# Export GitHub Repository Configuration
# Exports current repository settings for backup/reference
# Usage: ./export-github-config.sh

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

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Get repository information
REPO_OWNER=$(gh repo view --json owner -q '.owner.login')
REPO_NAME=$(gh repo view --json name -q '.name')
REPO_FULL="${REPO_OWNER}/${REPO_NAME}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../config"

# Ensure config directory exists
mkdir -p "${CONFIG_DIR}"

log_info "Exporting configuration for: ${REPO_FULL}"
echo ""

# ============================================================================
# Export Repository Rulesets
# ============================================================================

log_info "Exporting repository rulesets..."

RULESETS=$(gh api "/repos/${REPO_FULL}/rulesets" --jq '.')
RULESET_COUNT=$(echo "${RULESETS}" | jq '. | length')

if [[ "${RULESET_COUNT}" -gt 0 ]]; then
  log_info "Found ${RULESET_COUNT} ruleset(s)"

  # Export all rulesets summary
  echo "${RULESETS}" | jq >"${CONFIG_DIR}/rulesets-all.json"
  log_success "All rulesets summary → ${CONFIG_DIR}/rulesets-all.json"

  # Export each ruleset individually
  echo "${RULESETS}" | jq -c '.[]' | while read -r ruleset; do
    RULESET_ID=$(echo "${ruleset}" | jq -r '.id')
    RULESET_NAME=$(echo "${ruleset}" | jq -r '.name')
    RULESET_SLUG=$(echo "${RULESET_NAME}" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')

    log_info "Exporting: ${RULESET_NAME} (ID: ${RULESET_ID})"

    # Export full ruleset details
    gh api "/repos/${REPO_FULL}/rulesets/${RULESET_ID}" >"${CONFIG_DIR}/${RULESET_SLUG}.json"
    log_success "${RULESET_NAME} → ${CONFIG_DIR}/${RULESET_SLUG}.json"
  done
else
  log_error "No rulesets found"
fi

# ============================================================================
# Export Repository Settings
# ============================================================================

log_info "Exporting repository settings..."

gh api "/repos/${REPO_FULL}" >"${CONFIG_DIR}/repository-settings.json"
log_success "Repository settings → ${CONFIG_DIR}/repository-settings.json"

# ============================================================================
# Export Environments
# ============================================================================

log_info "Exporting deployment environments..."

ENVIRONMENTS=$(gh api "/repos/${REPO_FULL}/environments" --jq '.environments // []')
ENV_COUNT=$(echo "${ENVIRONMENTS}" | jq '. | length')

if [[ "${ENV_COUNT}" -gt 0 ]]; then
  echo "${ENVIRONMENTS}" | jq >"${CONFIG_DIR}/environments.json"
  log_success "Environments → ${CONFIG_DIR}/environments.json"
else
  log_info "No environments found"
fi

# ============================================================================
# Export Tag Protection
# ============================================================================

log_info "Exporting tag protection rules..."

TAG_PROTECTION=$(gh api "/repos/${REPO_FULL}/tags/protection" --jq '.' 2>/dev/null || echo '[]')
TAG_COUNT=$(echo "${TAG_PROTECTION}" | jq '. | length')

if [[ "${TAG_COUNT}" -gt 0 ]]; then
  echo "${TAG_PROTECTION}" | jq >"${CONFIG_DIR}/tag-protection.json"
  log_success "Tag protection → ${CONFIG_DIR}/tag-protection.json"
else
  log_info "No tag protection rules found"
fi

# ============================================================================
# Export Labels
# ============================================================================

log_info "Exporting repository labels..."

LABELS=$(gh api "/repos/${REPO_FULL}/labels?per_page=100" --jq '.')
LABEL_COUNT=$(echo "${LABELS}" | jq '. | length')

if [[ "${LABEL_COUNT}" -gt 0 ]]; then
  echo "${LABELS}" | jq >"${CONFIG_DIR}/labels.json"
  log_success "Labels (${LABEL_COUNT}) → ${CONFIG_DIR}/labels.json"
else
  log_info "No labels found"
fi

# ============================================================================
# Export Topics
# ============================================================================

log_info "Exporting repository topics..."

TOPICS=$(gh api "/repos/${REPO_FULL}/topics" -H "Accept: application/vnd.github+json" --jq '.names')
TOPIC_COUNT=$(echo "${TOPICS}" | jq '. | length')

if [[ "${TOPIC_COUNT}" -gt 0 ]]; then
  echo "${TOPICS}" | jq >"${CONFIG_DIR}/topics.json"
  log_success "Topics (${TOPIC_COUNT}) → ${CONFIG_DIR}/topics.json"
else
  log_info "No topics found"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================================================"
log_success "Configuration export complete!"
echo "========================================================================"
echo ""
log_info "Exported to: ${CONFIG_DIR}"
echo ""
log_info "Files created:"
find "${CONFIG_DIR}" -maxdepth 1 -name "*.json" -type f -exec ls -lh {} \; | awk '{print "  " $9 " (" $5 ")"}'
echo ""
log_info "To apply this configuration to another repository:"
echo "  1. Copy .github/config/ directory to target repo"
echo "  2. Run: .github/scripts/setup-github-repo.sh"
echo ""
log_info "Note: Secrets are NOT exported (for security)"
echo ""
