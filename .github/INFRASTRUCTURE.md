# Infrastructure as Code

This repository uses Infrastructure-as-Code (IaC) principles to make GitHub repository configuration replicable and version-controlled.

## Philosophy

**All repository configuration should be:**

- ✅ Version controlled in git
- ✅ Replicable on forks
- ✅ Documented and understandable
- ✅ Automated via scripts
- ✅ Idempotent (safe to run multiple times)

## What's Covered

### Fully Automated

- Repository rulesets (branch protection, merge queue)
- Repository settings (merge methods, auto-merge)
- Security features (Dependabot, secret scanning)
- Labels and topics
- Tag protection
- Environment configuration

### Requires Manual Setup

- Secrets (PYPI_API_TOKEN, etc.) - for security
- Team permissions - requires org admin
- OAuth apps - requires org settings

## File Structure

```
.github/
├── config/                                # Configuration templates
│   ├── main-branch-ruleset-template.json  # Ruleset for main branch
│   ├── main-branch-ruleset.json           # Current live config (reference)
│   └── README.md                          # Config documentation
├── scripts/
│   └── setup-github-repo.sh               # Main setup script
├── workflows/                             # GitHub Actions
│   ├── ci.yml                             # PR validation
│   ├── merge-queue.yml                    # Merge queue validation
│   └── ...
└── INFRASTRUCTURE.md                      # This file
```

## Usage

### Initial Repository Setup

```bash
# Complete setup (recommended)
.github/scripts/setup-github-repo.sh
```

This configures:

1. Repository ruleset with merge queue
2. Repository settings
3. Security features
4. Labels and topics
5. Environments
6. Tag protection
7. Prompts for PyPI token

### Updating Existing Configuration

```bash
# Same command - script is idempotent
.github/scripts/setup-github-repo.sh
```

The script:

- Updates existing rulesets instead of failing
- Skips already-configured settings
- Warns about conflicts instead of erroring

### Fork Setup

When forking this repository:

1. **Clone your fork**:

   ```bash
   git clone https://github.com/YOUR-ORG/oscura.git
   cd oscura
   ```

2. **Run setup script**:

   ```bash
   .github/scripts/setup-github-repo.sh
   ```

3. **Verify settings**:

   ```bash
   gh repo view --web
   # Navigate to Settings → Rules to verify ruleset
   ```

## Configuration Details

### Repository Ruleset

**File**: `.github/config/main-branch-ruleset-template.json`

**Applied to**: `refs/heads/main`

**Rules**:

- **Pull Request**: Squash merge only, no required reviews for maintainers
- **Merge Queue**: ALLGREEN strategy (all checks must pass)

**Key insight**: We don't use explicit `required_status_checks` because:

- They only work on `pull_request` events
- Merge queue creates `merge_group` events
- This mismatch causes queue to get stuck
- ALLGREEN strategy handles this automatically

**Validation**:

```bash
# List all rulesets
gh api repos/OWNER/REPO/rulesets --jq '.[] | {id, name, enforcement}'

# View specific ruleset
gh api repos/OWNER/REPO/rulesets/ID | jq
```

### Merge Queue Strategy

**Strategy**: ALLGREEN

**How it works**:

1. PR enters queue when developer clicks "Merge when ready"
2. GitHub creates temporary merge commit
3. Merge queue workflow runs (`.github/workflows/merge-queue.yml`)
4. ALL workflow checks must pass
5. If all pass → merge completes
6. If any fail → merge blocked, PR exits queue

**Benefits**:

- Prevents untested commits on main
- Catches merge conflicts before merging
- No configuration drift (auto-adapts to CI changes)
- Works with any number of CI jobs

**Configuration**:

```json
{
  "grouping_strategy": "ALLGREEN",
  "merge_method": "SQUASH",
  "max_entries_to_build": 5,
  "min_entries_to_merge": 1,
  "check_response_timeout_minutes": 60
}
```

## Adding New Configuration

### Adding a New Ruleset

1. **Create template**:

   ```bash
   # Create new JSON file
   vim .github/config/new-ruleset-template.json
   ```

2. **Update setup script**:

   ```bash
   # Edit .github/scripts/setup-github-repo.sh
   # Add new step to create ruleset
   ```

3. **Test**:

   ```bash
   # Dry run (if script supports it)
   .github/scripts/setup-github-repo.sh --dry-run

   # Or test on fork first
   ```

4. **Document**:
   - Update `.github/config/README.md`
   - Update this file (INFRASTRUCTURE.md)
   - Add to CHANGELOG.md

### Exporting Current Configuration

```bash
# Export all rulesets
gh api repos/OWNER/REPO/rulesets | jq > .github/config/rulesets-export.json

# Export specific ruleset
gh api repos/OWNER/REPO/rulesets/ID > .github/config/ruleset-ID.json

# Export repository settings
gh api repos/OWNER/REPO > .github/config/repo-settings.json
```

## Best Practices

### DO

- ✅ Keep templates clean (no IDs, timestamps, repo-specific data)
- ✅ Document why configuration exists
- ✅ Test on fork before applying to main repo
- ✅ Version control all configuration
- ✅ Make scripts idempotent

### DON'T

- ❌ Hardcode repository names in config files
- ❌ Store secrets in config files (use GitHub Secrets)
- ❌ Commit repo-specific exports (use templates instead)
- ❌ Modify production config without testing
- ❌ Skip documentation updates

## Troubleshooting

### "Resource not accessible by integration"

**Cause**: Insufficient permissions

**Solution**:

```bash
# Check auth status
gh auth status

# Re-authenticate with all scopes
gh auth login --scopes repo,admin:repo_hook,admin:org
```

### "Ruleset already exists"

**Solution**: The setup script handles this automatically by detecting and updating existing rulesets.

### Changes not taking effect

1. **Verify application**:

   ```bash
   gh api repos/OWNER/REPO/rulesets
   ```

2. **Check enforcement**:
   - Rulesets can be `active`, `evaluate`, or `disabled`
   - Only `active` rulesets enforce rules

3. **Clear cache**:
   - GitHub may cache ruleset evaluation
   - Wait 1-2 minutes or create new test PR

## Migration from Old Branch Protection

If you have old branch protection rules:

1. **Export old config** (for reference):

   ```bash
   gh api repos/OWNER/REPO/branches/main/protection > old-protection.json
   ```

2. **Apply new ruleset**:

   ```bash
   .github/scripts/setup-github-repo.sh
   ```

3. **Delete old protection** (optional):

   ```bash
   # Rulesets override branch protection, so old rules are ignored
   # Can safely delete old protection rules via web UI
   ```

## Contributing

When adding new infrastructure configuration:

1. Create configuration template in `.github/config/`
2. Update `.github/scripts/setup-github-repo.sh`
3. Document in `.github/config/README.md`
4. Add section to this file
5. Update CHANGELOG.md
6. Test on fork
7. Create PR with all changes

## Resources

- [GitHub Repository Rulesets Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)
- [Merge Queue Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue)
- [GitHub CLI API Documentation](https://cli.github.com/manual/gh_api)
- [GitHub REST API Documentation](https://docs.github.com/en/rest)
