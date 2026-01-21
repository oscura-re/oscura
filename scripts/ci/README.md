# CI Validation Scripts

This directory contains validation scripts that ensure repository configuration remains correct and consistent.

## Scripts

### `validate_test_coverage.sh`

**Purpose**: Ensures all test directories under `tests/unit/` are included in CI test group configurations.

**Why This Matters**: Prevents situations where test directories are created but not added to CI configuration, causing tests to be silently skipped. This happened with `tests/unit/automotive/` which was missing from CI for an extended period.

**Usage**:

```bash
./scripts/ci/validate_test_coverage.sh
```

**Exit Codes**:

- `0`: All test directories are covered in CI
- `1`: One or more test directories are missing from CI config

**What It Checks**:

- Scans all directories in `tests/unit/`
- Checks if each directory is mentioned in `.github/workflows/ci.yml`
- Reports any missing directories with suggestions for which test group to add them to

**Example Output** (when passing):

```
==========================================
CI Test Coverage Validation
==========================================

Scanning test directories in tests/unit...
Found 47 test directories

✓ acquisition
✓ analyzers
✓ automotive
...
✓ workflows

==========================================
Results
==========================================
Total directories: 47
Covered: 47
Missing: 0

✅ VALIDATION PASSED

All test directories are properly included in CI configuration.
```

**Example Output** (when failing):

```
==========================================
CI Test Coverage Validation
==========================================

Scanning test directories in tests/unit...
Found 47 test directories

✓ acquisition
✗ new_feature
...

==========================================
Results
==========================================
Total directories: 47
Covered: 46
Missing: 1

❌ VALIDATION FAILED

The following test directories are NOT included in CI configuration:

  ✗ tests/unit/new_feature/

These directories need to be added to one of the test groups in:
  .github/workflows/ci.yml

Suggested test groups:
  - analyzers: For analyzer tests
  - core-protocols-loaders: For core functionality, protocols, loaders
  - unit-workflows: For sessions, workflows, pipeline tests
  - unit-exploratory: For inference, discovery, guidance tests
  - unit-utils: For utilities, config, plugins
```

**How to Fix Issues**:

1. Identify the missing directory
2. Determine the appropriate test group based on the content
3. Edit `.github/workflows/ci.yml` and add the path to the chosen test group
4. Run the validation script again to verify

**Pre-commit Hook**: This script runs automatically via pre-commit hooks when:

- Test files are modified
- `.github/workflows/ci.yml` is modified

---

### `validate_required_checks.sh`

**Purpose**: Ensures that required status checks in branch protection match actual CI workflow job names, preventing configuration drift.

**Why This Matters**: Branch protection can be configured with required status checks that don't exist in CI workflows. This causes PRs to be blocked forever since the non-existent checks will never pass. This validation ensures all required checks actually exist.

**Usage**:

```bash
./scripts/ci/validate_required_checks.sh
```

**Exit Codes**:

- `0`: All required status checks are valid
- `1`: One or more required checks don't exist in CI workflows
- `2`: Script error (gh CLI not available, etc.)

**What It Checks**:

- Fetches required status checks from GitHub branch protection API
- Scans all workflow files in `.github/workflows/`
- Extracts all possible status check names (job names and display names)
- Reports any required checks that don't exist in workflows

**Example Output** (when passing):

```
=========================================
Required Status Checks Validation
=========================================

Fetching required status checks from branch protection...
Found 10 required status checks

Scanning CI workflows for available status checks...
Found 45 available status checks in workflows

Validating required checks:

✓ CI
✓ Pre-Commit Hooks
✓ Lint
✓ Type Check
✓ Config Validation
✓ Integration Tests
✓ Build Package
✓ Check Test Isolation
✓ Validate Test Markers
✓ Build Documentation

=========================================
Results
=========================================
Total required checks: 10
Valid:   10
Invalid: 0

✅ VALIDATION PASSED

All required status checks exist in CI workflows.
```

**Example Output** (when failing):

```
=========================================
Required Status Checks Validation
=========================================

Fetching required status checks from branch protection...
Found 11 required status checks

Scanning CI workflows for available status checks...
Found 45 available status checks in workflows

Validating required checks:

✓ CI
✓ Pre-Commit Hooks
✗ NonExistent Check (not found in CI workflows)
...

=========================================
Results
=========================================
Total required checks: 11
Valid:   10
Invalid: 1

❌ VALIDATION FAILED

The following required checks do NOT exist in CI workflows:

  ✗ NonExistent Check

This means:
  1. These checks will NEVER pass (blocking merges forever)
  2. Branch protection configuration needs to be updated

Actions:
  1. Review .github/workflows/ to find correct check names
  2. Update branch protection ruleset (ID: 11977857)
  3. Run this script again to verify
```

**How to Fix Issues**:

1. Identify the invalid required check name
2. Check `.github/workflows/` to find the correct job name
3. Update branch protection via GitHub API or web UI
4. Run the validation script again to verify

**Requirements**:

- `gh` CLI tool (GitHub CLI) installed and authenticated
- `yq` (optional, for better parsing - will use fallback if not available)

**Manual Validation**:

```bash
# Install prerequisites (if needed)
# gh CLI: https://cli.github.com/
# yq: https://github.com/mikefarah/yq

# Authenticate (if needed)
gh auth login

# Run validation
./scripts/ci/validate_required_checks.sh
```

---

### `validate_workflow_permissions.sh`

**Purpose**: Ensures all GitHub Actions workflows have explicit permissions blocks following the principle of least privilege.

**Why This Matters**: Workflows without explicit permissions inherit repository defaults, which may be overly permissive. This creates security vulnerabilities and makes it harder to audit what permissions each workflow needs.

**Usage**:

```bash
./scripts/ci/validate_workflow_permissions.sh
```

**Exit Codes**:

- `0`: All workflows have explicit permissions
- `1`: One or more workflows are missing explicit permissions

**What It Checks**:

- Scans all workflow files in `.github/workflows/`
- Checks if each workflow has a top-level `permissions:` block
- Reports any workflows missing explicit permissions

**Example Output** (when passing):

```
==========================================
Workflow Permissions Validation
==========================================

Scanning workflows in .github/workflows...
Found 10 workflow files

✓ ci.yml
✓ codeql.yml
✓ code-quality.yml
...
✓ test-quality.yml

==========================================
Results
==========================================
Total workflows: 10
With permissions: 10
Missing permissions: 0

✅ VALIDATION PASSED

All workflows have explicit permissions blocks.

Security best practice: Principle of least privilege ✓
```

**Example Output** (when failing):

```
==========================================
Workflow Permissions Validation
==========================================

Scanning workflows in .github/workflows...
Found 10 workflow files

✓ ci.yml
✗ new-workflow.yml
...

==========================================
Results
==========================================
Total workflows: 10
With permissions: 9
Missing permissions: 1

❌ VALIDATION FAILED

The following workflows are missing explicit permissions blocks:

  ✗ new-workflow.yml

Add an explicit permissions block at the top level of each workflow:

Example:
  name: My Workflow
  on: [push, pull_request]

  # Explicit permissions for security
  permissions:
    contents: read        # Read repository contents
    pull-requests: read   # Read PR information
    checks: write         # Write check results

  jobs:
    ...

Common permission combinations:
  - Read-only: contents: read
  - CI: contents: read, checks: write
  - PR comments: contents: read, pull-requests: write
  - Release: contents: write, id-token: write
```

**How to Fix Issues**:

1. Identify the workflow missing permissions
2. Determine what permissions the workflow actually needs
3. Add an explicit `permissions:` block at the top level of the workflow
4. Run the validation script again to verify

**Pre-commit Hook**: This script runs automatically via pre-commit hooks when:

- Workflow files (`.github/workflows/*.yml`) are modified

---

## Integration with Pre-commit Hooks

Both scripts are integrated with pre-commit hooks to catch issues early:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: validate-test-coverage
      name: Validate test directories in CI
      entry: ./scripts/ci/validate_test_coverage.sh
      language: script
      files: '^(tests/unit/.*|\.github/workflows/ci\.yml)$'
      pass_filenames: false

    - id: validate-workflow-permissions
      name: Validate workflow permissions
      entry: ./scripts/ci/validate_workflow_permissions.sh
      language: script
      files: '^\.github/workflows/.*\.yml$'
      pass_filenames: false
```

## Manual Testing

Run all validations manually:

```bash
# Test coverage validation
./scripts/ci/validate_test_coverage.sh

# Workflow permissions validation
./scripts/ci/validate_workflow_permissions.sh

# Required checks validation (requires gh CLI)
./scripts/ci/validate_required_checks.sh

# Or run via pre-commit (test coverage & workflow permissions)
pre-commit run validate-test-coverage --all-files
pre-commit run validate-workflow-permissions --all-files
```

## CI Integration

These scripts should also be run in CI to catch configuration issues that might slip past pre-commit hooks.

Recommended: Add to the fast-checks job in CI workflow:

```yaml
- name: Validate test coverage
  run: ./scripts/ci/validate_test_coverage.sh

- name: Validate workflow permissions
  run: ./scripts/ci/validate_workflow_permissions.sh

- name: Validate required checks
  run: ./scripts/ci/validate_required_checks.sh
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Troubleshooting

### Script Not Executable

If you get "Permission denied":

```bash
chmod +x scripts/ci/*.sh
```

### Script Not Found in Pre-commit

Pre-commit runs in its own environment. Ensure scripts are executable and committed to the repository.

### False Positives

If a test directory is intentionally excluded from CI (e.g., disabled tests):

- Document the exclusion reason in CI configuration comments
- Consider moving disabled tests to a separate location outside `tests/unit/`

## History

These validation scripts were created after discovering that `tests/unit/automotive/` was completely excluded from CI test runs, causing all automotive tests to be silently skipped. This led to coverage failures and missed test executions.

**Lessons Learned**:

1. Test directory changes require CI configuration updates
2. Automated validation prevents silent configuration drift
3. Pre-commit hooks catch issues early in development
4. Explicit is better than implicit (for both tests and permissions)

## See Also

- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Git workflow and commit guidelines
- [CLAUDE.md](../../CLAUDE.md) - Project instructions for development
- [.pre-commit-config.yaml](../../.pre-commit-config.yaml) - Pre-commit hooks configuration
- [.github/workflows/ci.yml](../../.github/workflows/ci.yml) - CI test configuration
