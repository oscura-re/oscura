# GitHub Configuration Files

This directory contains Infrastructure-as-Code (IaC) configuration files for GitHub repository settings.

## Files

### `main-branch-ruleset.json`

Complete export of the current active ruleset configuration, including metadata (IDs, timestamps, links).

**Purpose**: Documentation and backup of the live configuration.

**Usage**: Reference only. Do not use for imports (contains repo-specific IDs).

### `main-branch-ruleset-template.json`

Clean template for creating/updating the main branch ruleset.

**Purpose**: Replicable configuration for forks and new instances.

**Usage**: Applied by `.github/scripts/setup-github-repo.sh` during repository setup.

## Configuration Details

### Main Branch Ruleset

**Protection mechanisms**:

- **Pull Request Rule**: Requires squash merge only (no required reviews for maintainers)
- **Merge Queue Rule**: ALLGREEN strategy ensures all workflow checks pass before merging

**Key settings**:

```json
{
  "grouping_strategy": "ALLGREEN",      // All checks must pass
  "merge_method": "SQUASH",              // Squash commits on merge
  "check_response_timeout_minutes": 60,  // 1-hour timeout
  "max_entries_to_build": 5,             // Batch up to 5 PRs
  "min_entries_to_merge": 1              // Minimum 1 PR per batch
}
```

**Why no required_status_checks rule?**

The merge queue's ALLGREEN strategy already ensures all workflow checks pass. Adding explicit required status checks causes conflicts because:

- PR workflows run on `pull_request` events
- Merge queue creates `merge_group` events
- Required checks defined for PR events won't run on merge_group events
- Queue gets stuck waiting for checks that can't run

**Solution**: Let ALLGREEN handle validation. It automatically requires all checks to pass without naming them explicitly.

## Applying Configuration

### Initial Setup

```bash
.github/scripts/setup-github-repo.sh
```

This script:

1. Reads `main-branch-ruleset-template.json`
2. Checks if ruleset exists
3. Creates new ruleset OR updates existing one
4. Idempotent (safe to run multiple times)

### Manual Application

```bash
# Create new ruleset
gh api -X POST repos/OWNER/REPO/rulesets \
  --input .github/config/main-branch-ruleset-template.json

# Update existing ruleset (replace ID with your ruleset ID)
gh api -X PUT repos/OWNER/REPO/rulesets/12055878 \
  --input .github/config/main-branch-ruleset-template.json
```

### Exporting Current Config

```bash
# List all rulesets and find ID
gh api repos/OWNER/REPO/rulesets --jq '.[] | {id, name}'

# Export specific ruleset
gh api repos/OWNER/REPO/rulesets/ID > .github/config/main-branch-ruleset.json
```

## For Fork Maintainers

When setting up a fork:

1. **Option A - Use setup script** (recommended):

   ```bash
   .github/scripts/setup-github-repo.sh
   ```

2. **Option B - Manual setup**:

   ```bash
   # Set your repository
   export REPO_FULL="your-org/your-repo"

   # Create ruleset
   gh api -X POST repos/${REPO_FULL}/rulesets \
     --input .github/config/main-branch-ruleset-template.json
   ```

3. **Verify**:

   ```bash
   gh api repos/${REPO_FULL}/rulesets --jq '.[] | {name, enforcement}'
   ```

## Troubleshooting

### Merge Queue Stuck in AWAITING_CHECKS

**Symptoms**: PR enters merge queue but stays stuck, never progresses

**Cause**: Ruleset has `required_status_checks` that only run on `pull_request` events, not `merge_group` events

**Fix**: Remove `required_status_checks` from ruleset, rely on ALLGREEN strategy instead:

```bash
gh api -X PUT repos/OWNER/REPO/rulesets/ID \
  --input .github/config/main-branch-ruleset-template.json
```

### Ruleset Already Exists

The setup script is idempotent - it updates existing rulesets by name. Safe to run multiple times.

### Permission Denied

Requires admin permissions on the repository. Check:

```bash
gh auth status
gh repo view OWNER/REPO --json viewerPermission
```

## Version History

- **2026-01-24**: Removed `required_status_checks` rule after merge queue stuck incident
- **2026-01-22**: Initial ruleset created with merge queue enabled
