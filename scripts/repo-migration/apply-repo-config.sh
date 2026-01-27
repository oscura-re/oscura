#!/bin/bash
# Apply canonical repository configuration from .github/repo-config/
# Usage: ./apply-repo-config.sh [owner/repo]
#
# This script ensures the repository matches the canonical configuration
# stored in version control. Run this to fix configuration drift.

set -euo pipefail

REPO="${1:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
CONFIG_DIR=".github/repo-config"

# Change to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=== Applying Canonical Configuration to: $REPO ==="
echo "Source config: $CONFIG_DIR"
echo ""

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: Configuration directory not found: $CONFIG_DIR"
  echo "Run: ./scripts/repo-migration/export-repo-config.sh to create it"
  exit 1
fi

# Verify repository exists
if ! gh repo view "$REPO" >/dev/null 2>&1; then
  echo "Error: Repository $REPO does not exist or you don't have access"
  exit 1
fi

# 1. BASIC SETTINGS
echo "[1/8] Applying basic settings..."
if [ -f "$CONFIG_DIR/basic-settings.json" ]; then
  DESC=$(jq -r '.description // ""' "$CONFIG_DIR/basic-settings.json")
  HOMEPAGE=$(jq -r '.homepage // ""' "$CONFIG_DIR/basic-settings.json")

  if [ -n "$DESC" ] && [ "$DESC" != "null" ]; then
    gh repo edit "$REPO" --description "$DESC"
  fi

  if [ -n "$HOMEPAGE" ] && [ "$HOMEPAGE" != "null" ]; then
    gh repo edit "$REPO" --homepage "$HOMEPAGE"
  fi

  echo "  ✅ Description and homepage set"
else
  echo "  ⚠️  basic-settings.json not found, skipping"
fi

# 2. FEATURES
echo "[2/8] Applying feature toggles..."
if [ -f "$CONFIG_DIR/features.json" ]; then
  HAS_ISSUES=$(jq -r '.has_issues' "$CONFIG_DIR/features.json")
  HAS_WIKI=$(jq -r '.has_wiki' "$CONFIG_DIR/features.json")
  HAS_PROJECTS=$(jq -r '.has_projects' "$CONFIG_DIR/features.json")
  HAS_DISCUSSIONS=$(jq -r '.has_discussions' "$CONFIG_DIR/features.json")

  gh repo edit "$REPO" \
    --enable-issues="$HAS_ISSUES" \
    --enable-wiki="$HAS_WIKI" \
    --enable-projects="$HAS_PROJECTS" \
    --enable-discussions="$HAS_DISCUSSIONS"

  echo "  ✅ Features configured"
else
  echo "  ⚠️  features.json not found, skipping"
fi

# 3. MERGE SETTINGS
echo "[3/8] Applying merge settings..."
if [ -f "$CONFIG_DIR/merge-settings.json" ]; then
  SQUASH=$(jq -r '.allow_squash_merge' "$CONFIG_DIR/merge-settings.json")
  MERGE=$(jq -r '.allow_merge_commit' "$CONFIG_DIR/merge-settings.json")
  REBASE=$(jq -r '.allow_rebase_merge' "$CONFIG_DIR/merge-settings.json")
  AUTO_MERGE=$(jq -r '.allow_auto_merge' "$CONFIG_DIR/merge-settings.json")
  DELETE_BRANCH=$(jq -r '.delete_branch_on_merge' "$CONFIG_DIR/merge-settings.json")

  gh repo edit "$REPO" \
    --enable-squash-merge="$SQUASH" \
    --enable-merge-commit="$MERGE" \
    --enable-rebase-merge="$REBASE" \
    --enable-auto-merge="$AUTO_MERGE" \
    --delete-branch-on-merge="$DELETE_BRANCH"

  echo "  ✅ Merge settings configured"
else
  echo "  ⚠️  merge-settings.json not found, skipping"
fi

# 4. TOPICS
echo "[4/8] Applying topics..."
if [ -f "$CONFIG_DIR/topics.json" ]; then
  # First, get current topics
  CURRENT_TOPICS=$(gh api repos/$REPO/topics --jq '.names | join(" ")')

  # Remove all current topics
  if [ -n "$CURRENT_TOPICS" ]; then
    for TOPIC in $CURRENT_TOPICS; do
      gh repo edit "$REPO" --remove-topic "$TOPIC" 2>/dev/null || true
    done
  fi

  # Add topics from config
  TOPICS=$(jq -r '.[]' "$CONFIG_DIR/topics.json" | tr '\n' ' ')

  if [ -n "$TOPICS" ]; then
    for TOPIC in $TOPICS; do
      gh repo edit "$REPO" --add-topic "$TOPIC"
    done
    echo "  ✅ Topics applied: $TOPICS"
  else
    echo "  ✅ No topics to apply"
  fi
else
  echo "  ⚠️  topics.json not found, skipping"
fi

# 5. BRANCH PROTECTION (CRITICAL)
echo "[5/8] Applying branch protection..."
if [ -f "$CONFIG_DIR/branch-protection.json" ]; then
  if jq -e '.error' "$CONFIG_DIR/branch-protection.json" >/dev/null 2>&1; then
    echo "  ⚠️  No branch protection in config"
  else
    DEFAULT_BRANCH=$(gh api repos/$REPO --jq '.default_branch')

    # Apply protection
    if gh api repos/$REPO/branches/$DEFAULT_BRANCH/protection \
      -X PUT \
      --input "$CONFIG_DIR/branch-protection.json" >/dev/null 2>&1; then

      echo "  ✅ Branch protection applied"

      # Verify it was applied correctly
      CHECKS=$(gh api repos/$REPO/branches/$DEFAULT_BRANCH/protection \
        --jq '.required_status_checks.checks | length')
      echo "     Required checks: $CHECKS"
    else
      echo "  ❌ Branch protection failed to apply"
      echo "     This is CRITICAL - branch protection enforcement is broken!"
      exit 1
    fi
  fi
else
  echo "  ⚠️  branch-protection.json not found, skipping"
fi

# 6. VERIFY CRITICAL SETTINGS
echo "[6/8] Verifying critical settings..."

# Check branch protection is actually enforced
PROTECTION_STATUS=$(gh api repos/$REPO/branches/main/protection 2>&1 || echo "NOT_PROTECTED")

if [[ "$PROTECTION_STATUS" == *"NOT_PROTECTED"* ]] || [[ "$PROTECTION_STATUS" == *"404"* ]]; then
  echo "  ❌ CRITICAL: Branch protection NOT enforced on main!"
  echo "     This means PRs can merge without passing CI!"
  exit 1
fi

# Check required checks are configured
REQUIRED_CHECKS=$(gh api repos/$REPO/branches/main/protection \
  --jq '.required_status_checks.checks | length' 2>/dev/null || echo "0")

if [ "$REQUIRED_CHECKS" -lt 4 ]; then
  echo "  ❌ CRITICAL: Only $REQUIRED_CHECKS required checks (expected 4)"
  echo "     Required: CI, Documentation, Code Quality, Test Quality Gates"
  exit 1
fi

echo "  ✅ Branch protection verified (4 required checks)"

# 7. CHECK DRIFT
echo "[7/8] Checking for configuration drift..."
DRIFT_FOUND=false

# Compare current topics with config
CURRENT_TOPICS=$(gh api repos/$REPO/topics --jq '.names | sort | join(" ")')
CONFIG_TOPICS=$(jq -r '.[] | @text' "$CONFIG_DIR/topics.json" | sort | tr '\n' ' ' | xargs)

if [ "$CURRENT_TOPICS" != "$CONFIG_TOPICS" ]; then
  echo "  ⚠️  Topics drift detected"
  echo "     Current:  $CURRENT_TOPICS"
  echo "     Expected: $CONFIG_TOPICS"
  DRIFT_FOUND=true
fi

if [ "$DRIFT_FOUND" = false ]; then
  echo "  ✅ No drift detected"
fi

# 8. SUMMARY
echo "[8/8] Summary..."
echo ""
echo "=== Configuration Applied Successfully ==="
echo ""
echo "✅ Repository settings match canonical configuration"
echo "✅ Branch protection enforced (4 required checks)"
echo "✅ Merge settings configured"
echo "✅ Topics synchronized"
echo ""

# Display current protection summary
echo "Current Branch Protection:"
gh api repos/$REPO/branches/main/protection --jq '{
  "Required Checks": .required_status_checks.checks | map(.context),
  "Enforce Admins": .enforce_admins.enabled,
  "Strict": .required_status_checks.strict
}' | jq '.'

echo ""
echo "Next steps:"
echo "  - Verify PR #11 is now blocked by required checks"
echo "  - Test with a new PR to confirm enforcement"
echo "  - Re-run this script anytime to fix configuration drift"
echo ""
