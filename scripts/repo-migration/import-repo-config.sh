#!/bin/bash
# Import repository configuration from export
# Usage: ./import-repo-config.sh <new-repo-name> [config-dir]
#
# This script recreates ALL exported settings in a new repository.
# Must be run AFTER creating the new repository.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <owner/new-repo-name> [config-dir]"
  echo "Example: $0 lair-click-bats/oscura-fresh repo-config-export"
  exit 1
fi

NEW_REPO="$1"
CONFIG_DIR="${2:-repo-config-export}"

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: Configuration directory not found: $CONFIG_DIR"
  echo "Run export-repo-config.sh first to export configuration"
  exit 1
fi

echo "=== Importing Configuration to: $NEW_REPO ==="
echo "Source config: $CONFIG_DIR"
echo ""

# Verify repository exists
if ! gh repo view "$NEW_REPO" > /dev/null 2>&1; then
  echo "Error: Repository $NEW_REPO does not exist"
  echo "Create it first with: gh repo create $NEW_REPO --public"
  exit 1
fi

# 1. BASIC SETTINGS
echo "[1/10] Applying basic settings..."
if [ -f "$CONFIG_DIR/basic-settings.json" ]; then
  DESC=$(jq -r '.description // ""' "$CONFIG_DIR/basic-settings.json")
  HOMEPAGE=$(jq -r '.homepage // ""' "$CONFIG_DIR/basic-settings.json")

  if [ -n "$DESC" ] && [ "$DESC" != "null" ]; then
    gh repo edit "$NEW_REPO" --description "$DESC"
  fi

  if [ -n "$HOMEPAGE" ] && [ "$HOMEPAGE" != "null" ]; then
    gh repo edit "$NEW_REPO" --homepage "$HOMEPAGE"
  fi

  echo "  ✅ Description and homepage set"
else
  echo "  ⚠️  basic-settings.json not found, skipping"
fi

# 2. FEATURES
echo "[2/10] Applying feature toggles..."
if [ -f "$CONFIG_DIR/features.json" ]; then
  HAS_ISSUES=$(jq -r '.has_issues' "$CONFIG_DIR/features.json")
  HAS_WIKI=$(jq -r '.has_wiki' "$CONFIG_DIR/features.json")
  HAS_PROJECTS=$(jq -r '.has_projects' "$CONFIG_DIR/features.json")
  HAS_DISCUSSIONS=$(jq -r '.has_discussions' "$CONFIG_DIR/features.json")

  gh repo edit "$NEW_REPO" \
    --enable-issues="$HAS_ISSUES" \
    --enable-wiki="$HAS_WIKI" \
    --enable-projects="$HAS_PROJECTS" \
    --enable-discussions="$HAS_DISCUSSIONS"

  echo "  ✅ Features configured"
else
  echo "  ⚠️  features.json not found, skipping"
fi

# 3. MERGE SETTINGS
echo "[3/10] Applying merge settings..."
if [ -f "$CONFIG_DIR/merge-settings.json" ]; then
  SQUASH=$(jq -r '.allow_squash_merge' "$CONFIG_DIR/merge-settings.json")
  MERGE=$(jq -r '.allow_merge_commit' "$CONFIG_DIR/merge-settings.json")
  REBASE=$(jq -r '.allow_rebase_merge' "$CONFIG_DIR/merge-settings.json")
  AUTO_MERGE=$(jq -r '.allow_auto_merge' "$CONFIG_DIR/merge-settings.json")
  DELETE_BRANCH=$(jq -r '.delete_branch_on_merge' "$CONFIG_DIR/merge-settings.json")

  gh repo edit "$NEW_REPO" \
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
echo "[4/10] Applying topics..."
if [ -f "$CONFIG_DIR/topics.json" ]; then
  TOPICS=$(jq -r '.[]' "$CONFIG_DIR/topics.json" | tr '\n' ' ')

  if [ -n "$TOPICS" ]; then
    for TOPIC in $TOPICS; do
      gh repo edit "$NEW_REPO" --add-topic "$TOPIC"
    done
    echo "  ✅ Topics added: $TOPICS"
  else
    echo "  ⚠️  No topics to add"
  fi
else
  echo "  ⚠️  topics.json not found, skipping"
fi

# 5. BRANCH PROTECTION
echo "[5/10] Applying branch protection..."
if [ -f "$CONFIG_DIR/branch-protection.json" ]; then
  if jq -e '.error' "$CONFIG_DIR/branch-protection.json" > /dev/null 2>&1; then
    echo "  ⚠️  No branch protection to apply"
  else
    DEFAULT_BRANCH=$(gh api repos/$NEW_REPO --jq '.default_branch')

    # Apply protection (complex, requires full JSON)
    gh api repos/$NEW_REPO/branches/$DEFAULT_BRANCH/protection \
      -X PUT \
      --input "$CONFIG_DIR/branch-protection.json" \
      2> /dev/null && echo "  ✅ Branch protection applied" \
      || echo "  ⚠️  Branch protection failed (may need manual configuration)"
  fi
else
  echo "  ⚠️  branch-protection.json not found, skipping"
fi

# 6. RULESETS
echo "[6/10] Applying rulesets..."
if [ -f "$CONFIG_DIR/rulesets.json" ]; then
  RULESET_COUNT=$(jq '. | length' "$CONFIG_DIR/rulesets.json")

  if [ "$RULESET_COUNT" -gt 0 ]; then
    echo "  ⚠️  Found $RULESET_COUNT rulesets - manual recreation recommended"
    echo "      Rulesets are complex and may require organization-level permissions"
  else
    echo "  ✅ No rulesets to apply"
  fi
else
  echo "  ⚠️  rulesets.json not found, skipping"
fi

# 7. COLLABORATORS
echo "[7/10] Applying collaborators..."
if [ -f "$CONFIG_DIR/collaborators.json" ]; then
  if jq -e '.error' "$CONFIG_DIR/collaborators.json" > /dev/null 2>&1; then
    echo "  ⚠️  No collaborator data available"
  else
    COLLAB_COUNT=$(jq '. | length' "$CONFIG_DIR/collaborators.json")
    echo "  ⚠️  Found $COLLAB_COUNT collaborators - add manually:"
    jq -r '.[] | "      gh api repos/'$NEW_REPO'/collaborators/\(.login) -X PUT -f permission=\(.permissions.push // .permissions.admin // "read")"' "$CONFIG_DIR/collaborators.json"
  fi
else
  echo "  ⚠️  collaborators.json not found, skipping"
fi

# 8. WEBHOOKS
echo "[8/10] Applying webhooks..."
if [ -f "$CONFIG_DIR/webhooks.json" ]; then
  if jq -e '.error' "$CONFIG_DIR/webhooks.json" > /dev/null 2>&1; then
    echo "  ⚠️  No webhook data available"
  else
    WEBHOOK_COUNT=$(jq '. | length' "$CONFIG_DIR/webhooks.json")

    if [ "$WEBHOOK_COUNT" -gt 0 ]; then
      echo "  ⚠️  Found $WEBHOOK_COUNT webhooks - recreation not automated"
      echo "      Review webhooks.json and recreate manually"
    else
      echo "  ✅ No webhooks to apply"
    fi
  fi
else
  echo "  ⚠️  webhooks.json not found, skipping"
fi

# 9. SECRETS
echo "[9/10] Secrets (MANUAL ACTION REQUIRED)..."
if [ -f "$CONFIG_DIR/secret-names.txt" ]; then
  SECRET_COUNT=$(grep -v '^#' "$CONFIG_DIR/secret-names.txt" | wc -l)

  if [ "$SECRET_COUNT" -gt 0 ]; then
    echo "  ⚠️  Found $SECRET_COUNT secrets to recreate:"
    while IFS= read -r SECRET_NAME; do
      if [ -n "$SECRET_NAME" ] && [ "${SECRET_NAME:0:1}" != "#" ]; then
        echo "      gh secret set $SECRET_NAME -R $NEW_REPO"
      fi
    done < "$CONFIG_DIR/secret-names.txt"
    echo ""
    echo "  ℹ️  Run these commands manually to set secret values"
  else
    echo "  ✅ No secrets to recreate"
  fi
else
  echo "  ⚠️  secret-names.txt not found, skipping"
fi

# 10. VARIABLES
echo "[10/10] Applying variables..."
if [ -f "$CONFIG_DIR/variables.json" ]; then
  VAR_COUNT=$(jq '. | length' "$CONFIG_DIR/variables.json")

  if [ "$VAR_COUNT" -gt 0 ]; then
    while IFS= read -r VAR; do
      NAME=$(echo "$VAR" | jq -r '.name')
      VALUE=$(echo "$VAR" | jq -r '.value')
      gh variable set "$NAME" --body "$VALUE" -R "$NEW_REPO"
    done < <(jq -c '.[]' "$CONFIG_DIR/variables.json")

    echo "  ✅ $VAR_COUNT variables applied"
  else
    echo "  ✅ No variables to apply"
  fi
else
  echo "  ⚠️  variables.json not found, skipping"
fi

echo ""
echo "=== Import Complete ==="
echo ""
echo "✅ Automated configuration applied"
echo ""
echo "⚠️  MANUAL STEPS REQUIRED:"
echo ""
echo "1. Secrets: Set values using commands shown above"
echo "2. Social Preview: Upload image via Settings > Social Preview"
echo "3. Deploy Keys: Add keys if needed via Settings > Deploy keys"
echo "4. Branch Protection: Verify rules applied correctly"
echo "5. Webhooks: Recreate if needed (see webhooks.json)"
echo ""
echo "Verify configuration:"
echo "  gh repo view $NEW_REPO"
