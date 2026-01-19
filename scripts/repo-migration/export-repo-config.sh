#!/bin/bash
# Export complete repository configuration for migration
# Usage: ./export-repo-config.sh [owner/repo]
#
# This script exports ALL repository settings that can be recreated
# via CLI/API when creating a fresh repository.

set -euo pipefail

REPO="${1:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
OUTPUT_DIR="repo-config-export"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Exporting Configuration for: $REPO ==="
echo "Output directory: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

mkdir -p "$OUTPUT_DIR"

# 1. BASIC SETTINGS (fully exportable)
echo "[1/10] Exporting basic settings..."
gh api repos/$REPO --jq '{
  name: .name,
  description: .description,
  homepage: .homepage,
  visibility: .visibility,
  default_branch: .default_branch,
  is_template: .is_template,
  license: .license.spdx_id
}' > "$OUTPUT_DIR/basic-settings.json"

# 2. FEATURES (fully exportable)
echo "[2/10] Exporting feature toggles..."
gh api repos/$REPO --jq '{
  has_issues: .has_issues,
  has_projects: .has_projects,
  has_wiki: .has_wiki,
  has_discussions: .has_discussions,
  has_downloads: .has_downloads,
  allow_forking: .allow_forking
}' > "$OUTPUT_DIR/features.json"

# 3. MERGE SETTINGS (fully exportable)
echo "[3/10] Exporting merge settings..."
gh api repos/$REPO --jq '{
  allow_squash_merge: .allow_squash_merge,
  allow_merge_commit: .allow_merge_commit,
  allow_rebase_merge: .allow_rebase_merge,
  allow_auto_merge: .allow_auto_merge,
  delete_branch_on_merge: .delete_branch_on_merge,
  allow_update_branch: .allow_update_branch,
  squash_merge_commit_title: .squash_merge_commit_title,
  squash_merge_commit_message: .squash_merge_commit_message,
  merge_commit_title: .merge_commit_title,
  merge_commit_message: .merge_commit_message
}' > "$OUTPUT_DIR/merge-settings.json"

# 4. TOPICS (fully exportable)
echo "[4/10] Exporting topics..."
gh api repos/$REPO/topics --jq '.names' > "$OUTPUT_DIR/topics.json"

# 5. BRANCH PROTECTION (fully exportable)
echo "[5/10] Exporting branch protection..."
DEFAULT_BRANCH=$(gh api repos/$REPO --jq '.default_branch')
gh api repos/$REPO/branches/$DEFAULT_BRANCH/protection 2> /dev/null \
  > "$OUTPUT_DIR/branch-protection.json" \
  || echo '{"error": "No protection or insufficient permissions"}' > "$OUTPUT_DIR/branch-protection.json"

# 6. RULESETS (fully exportable, if using)
echo "[6/10] Exporting rulesets..."
gh api repos/$REPO/rulesets 2> /dev/null \
  > "$OUTPUT_DIR/rulesets.json" \
  || echo '[]' > "$OUTPUT_DIR/rulesets.json"

# 7. COLLABORATORS (fully exportable)
echo "[7/10] Exporting collaborators..."
gh api repos/$REPO/collaborators --jq '[.[] | {login: .login, role: .role_name, permissions: .permissions}]' 2> /dev/null \
  > "$OUTPUT_DIR/collaborators.json" \
  || echo '{"error": "Insufficient permissions"}' > "$OUTPUT_DIR/collaborators.json"

# 8. WEBHOOKS (fully exportable)
echo "[8/10] Exporting webhooks..."
gh api repos/$REPO/hooks 2> /dev/null \
  > "$OUTPUT_DIR/webhooks.json" \
  || echo '{"error": "Insufficient permissions"}' > "$OUTPUT_DIR/webhooks.json"

# 9. SECRETS (NAMES ONLY - values cannot be exported)
echo "[9/10] Exporting secret names (values CANNOT be exported)..."
gh secret list -R $REPO 2> /dev/null | awk '{print $1}' | grep -v "^NAME$" \
  > "$OUTPUT_DIR/secret-names.txt" \
  || echo "# No secrets or insufficient permissions" > "$OUTPUT_DIR/secret-names.txt"

# 10. VARIABLES (fully exportable)
echo "[10/10] Exporting variables..."
gh variable list -R $REPO --json name,value 2> /dev/null \
  > "$OUTPUT_DIR/variables.json" \
  || echo '[]' > "$OUTPUT_DIR/variables.json"

# BONUS: GitHub Pages (if enabled)
echo "[BONUS] Exporting GitHub Pages config..."
gh api repos/$REPO/pages 2> /dev/null \
  > "$OUTPUT_DIR/pages-config.json" \
  || echo '{"error": "Pages not enabled"}' > "$OUTPUT_DIR/pages-config.json"

# BONUS: Deploy keys (metadata only, private keys cannot be exported)
echo "[BONUS] Exporting deploy key metadata..."
gh api repos/$REPO/keys --jq '[.[] | {id: .id, title: .title, read_only: .read_only, created_at: .created_at}]' 2> /dev/null \
  > "$OUTPUT_DIR/deploy-keys.json" \
  || echo '[]' > "$OUTPUT_DIR/deploy-keys.json"

# Create summary
cat > "$OUTPUT_DIR/README.md" << EOF
# Repository Configuration Export

**Source Repository**: $REPO
**Export Date**: $TIMESTAMP
**Script Version**: 1.0.0

## Exported Files

### Fully Recreatable via CLI/API
- \`basic-settings.json\` - Name, description, visibility, etc.
- \`features.json\` - Issues, Wiki, Projects, Discussions toggles
- \`merge-settings.json\` - Squash/merge/rebase settings
- \`topics.json\` - Repository topics/tags
- \`branch-protection.json\` - Branch protection rules
- \`rulesets.json\` - Repository rulesets (if any)
- \`collaborators.json\` - Team/user access permissions
- \`webhooks.json\` - Webhook configurations
- \`variables.json\` - Actions variables (names AND values)
- \`pages-config.json\` - GitHub Pages configuration

### Partially Exportable
- \`secret-names.txt\` - Secret names only (VALUES CANNOT BE EXPORTED)
- \`deploy-keys.json\` - Metadata only (PRIVATE KEYS CANNOT BE EXPORTED)

### NOT Exportable via API
- Social preview image (must re-upload via web UI)
- Some security scanning settings (web UI only)
- Actions workflow permissions (some require web UI)
- Stars/Watchers/Forks counts (cannot be transferred)

## Using This Export

See \`import-repo-config.sh\` to recreate these settings in a new repository.

## Warnings

1. **Secrets**: You must manually recreate secrets - values cannot be exported
2. **Deploy Keys**: You must have the original private keys to recreate
3. **Social Preview**: Must re-upload image via Settings > Social Preview
4. **History**: This export does NOT include commit history, issues, PRs, or discussions
EOF

echo ""
echo "=== Export Complete ==="
echo "Configuration saved to: $OUTPUT_DIR/"
echo ""
echo "Summary:"
echo "  - Fully exportable settings: ✅ Saved"
echo "  - Secret NAMES: ✅ Saved (values require manual recreation)"
echo "  - Deploy key metadata: ✅ Saved (private keys require manual recreation)"
echo ""
echo "Next steps:"
echo "  1. Review files in $OUTPUT_DIR/"
echo "  2. Use import-repo-config.sh to recreate in new repository"
echo "  3. Manually set secrets via 'gh secret set'"
echo "  4. Manually upload social preview image via web UI"
