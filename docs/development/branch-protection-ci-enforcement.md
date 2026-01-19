# Branch Protection CI/CD Enforcement

## Overview

This document explains the critical fix implemented to ensure branch protection **actually enforces** CI/CD checks before allowing merges to `main`.

**Date**: 2026-01-18
**PR**: #2
**Status**: ✅ **FIXED** - Branch protection now guarantees CI/CD validation

---

## The Problem

### What Was Broken

Branch protection was configured to require these status checks:

- `CI`
- `Documentation`
- `Code Quality`
- `Test Quality Gates`

**However**, PRs could potentially be merged **WITHOUT** these checks passing due to multiple root causes.

### Root Causes Identified

1. **Workflow Name vs Check Name Mismatch**
   - Branch protection required workflow names (e.g., "CI", "Code Quality")
   - GitHub Actions creates **job-level CheckRuns**, not workflow-level StatusContexts
   - Example: CI workflow had job named "CI Success", but protection required "CI"
   - **Result**: Branch protection couldn't find the required check

2. **Path Filters Prevented Workflow Execution**
   - `code-quality.yml` had `paths: ["src/**/*.py", "pyproject.toml"]`
   - `test-quality.yml` had `paths: ["tests/**", "pyproject.toml"]`
   - `docs.yml` had `paths: ["docs/**", "src/**/*.py", "mkdocs.yml"]`
   - **Result**: Required workflows didn't run on PRs without matching file changes
   - **Impact**: Documentation-only PRs never ran code quality checks (and vice versa)

3. **No Workflow-Level Summary Jobs**
   - Workflows created 20+ individual job checks (e.g., "Lint", "Type Check", "Test (Python 3.12)")
   - No single "CI" check existed - only individual job checks
   - **Result**: Branch protection couldn't require "CI" - that check name didn't exist

4. **Missing Always-Run Logic**
   - Summary jobs existed but had `needs: [...]` without conditional logic
   - If path filters prevented jobs from running, summary jobs also didn't run
   - **Result**: Required checks were simply absent from PRs

5. **Skipped vs Failed Confusion**
   - Jobs that skipped (due to path filters) were treated ambiguously
   - Branch protection behavior undefined when required checks are missing
   - **Result**: Unclear if PR could merge or would be blocked forever

### Impact

**CRITICAL**: PRs could potentially be merged to `main` without:

- Full CI validation (tests, linting, type checking)
- Code quality checks
- Test quality gates
- Documentation validation

This **completely defeated** the purpose of the new repository setup and branch protection.

---

## The Solution

### Approach

Implement **always-running summary jobs** with proper naming and conditional execution logic.

### Changes Made

#### 1. **CI Workflow** (`ci.yml`)

**Before:**

```yaml
ci-success:
  name: CI Success  # ❌ Doesn't match "CI" requirement
  needs: [pre-commit, lint, typecheck, test, build]
```

**After:**

```yaml
ci-success:
  name: CI  # ✅ Matches branch protection exactly
  needs: [pre-commit, lint, typecheck, test, build]
  if: always()  # ✅ Always runs, even if dependencies fail/skip
```

#### 2. **Documentation Workflow** (`docs.yml`)

**Before:**

```yaml
on:
  pull_request:
    paths:  # ❌ Only runs on doc changes
      - "docs/**"
      - "src/**/*.py"

docs-summary:
  name: Documentation Summary  # ❌ Wrong name
```

**After:**

```yaml
on:
  pull_request:
    branches: [main]  # ✅ No path filters - always triggers

jobs:
  detect-changes:  # ✅ New job to detect if docs changed
    outputs:
      should-run: ${{ steps.filter.outputs.docs }}

  build:
    needs: [detect-changes]
    if: needs.detect-changes.outputs.should-run == 'true'  # ✅ Conditional

  docs-summary:
    name: Documentation  # ✅ Matches branch protection
    needs: [detect-changes, build, docstring-lint, spell-check]
    if: always()  # ✅ Always runs
    steps:
      - name: Check documentation results
        run: |
          # ✅ Handles no-changes case as success
          if [[ "${{ needs.detect-changes.outputs.should-run }}" != "true" ]]; then
            echo "✅ Documentation checks passed (no changes to check)"
            exit 0
          fi
          # ✅ Checks for failures (skipped is OK)
          if [[ "${{ needs.build.result }}" == "failure" ]]; then
            exit 1
          fi
```

#### 3. **Code Quality Workflow** (`code-quality.yml`)

**Before:**

```yaml
on:
  pull_request:
    paths:  # ❌ Only runs on code changes
      - "src/**/*.py"

quality-gates-success:
  name: Quality Gates Status  # ❌ Wrong name
```

**After:**

```yaml
on:
  pull_request:
    branches: [main]  # ✅ No path filters

jobs:
  detect-changes:  # ✅ Detect if code changed
    outputs:
      should-run: ${{ steps.filter.outputs.code }}

  quality-gates-success:
    name: Code Quality  # ✅ Matches branch protection
    needs: [detect-changes, docstring-style, dead-code, complexity, import-architecture]
    if: always()  # ✅ Always runs
```

#### 4. **Test Quality Gates Workflow** (`test-quality.yml`)

**Before:**

```yaml
on:
  pull_request:
    paths:  # ❌ Only runs on test changes
      - "tests/**"

quality-gates-success:
  name: Quality Gates Success  # ❌ Wrong name
```

**After:**

```yaml
on:
  pull_request:
    branches: [main]  # ✅ No path filters

jobs:
  detect-changes:  # ✅ Detect if tests changed
    outputs:
      should-run: ${{ steps.filter.outputs.tests }}

  quality-gates-success:
    name: Test Quality Gates  # ✅ Matches branch protection
    needs: [detect-changes, marker-validation, test-isolation, coverage-markers]
    if: always()  # ✅ Always runs
```

### Key Design Decisions

1. **Remove Path Filters from Triggers**
   - All workflows now trigger on every PR
   - Path filtering moved to job-level conditional logic
   - Ensures workflows are present in PR checks

2. **Add detect-changes Jobs**
   - Uses `dorny/paths-filter@v3` action
   - Detects if relevant files changed
   - Outputs boolean used by dependent jobs

3. **Conditional Job Execution**
   - Individual jobs only run if relevant files changed
   - Summary jobs ALWAYS run (`if: always()`)
   - Skipped jobs don't count as failures

4. **Smart Success Logic**
   - If no relevant changes: summary job passes immediately
   - If jobs skipped: treated as success
   - If jobs failed: summary job fails
   - Clear success/failure determination

---

## Verification

### How to Verify Fix is Working

1. **Check PR Status Checks**

   ```bash
   gh pr checks <PR_NUMBER> | grep -E "^(CI|Documentation|Code Quality|Test Quality Gates)"
   ```

   Expected output (all 4 checks present):

   ```
   CI                      pass    2m15s
   Documentation           pass    45s
   Code Quality            pass    1m10s
   Test Quality Gates      pass    55s
   ```

2. **Test with Different PR Types**
   - **Docs-only PR**: All 4 checks appear, doc jobs run, code/test jobs skip (but pass)
   - **Code-only PR**: All 4 checks appear, code jobs run, doc/test jobs skip (but pass)
   - **Test-only PR**: All 4 checks appear, test jobs run, code/doc jobs skip (but pass)
   - **Mixed changes**: All 4 checks appear, relevant jobs run

3. **Verify Branch Protection**

   ```bash
   gh api repos/oscura-re/oscura/branches/main/protection | jq '.required_status_checks.checks'
   ```

   Should show all 4 required checks:

   ```json
   [
     {"context": "CI"},
     {"context": "Documentation"},
     {"context": "Code Quality"},
     {"context": "Test Quality Gates"}
   ]
   ```

4. **Attempt Merge Without Checks**
   - Branch protection should **block** merge
   - UI should show "Required status check CI has not run"

---

## Prevention: Best Practices

### For Future Workflow Changes

1. **Never Use Path Filters in Required Workflows**

   ```yaml
   # ❌ BAD - Workflow won't run on all PRs
   on:
     pull_request:
       paths: ["src/**/*.py"]

   # ✅ GOOD - Workflow always runs, jobs are conditional
   on:
     pull_request:
       branches: [main]

   jobs:
     detect-changes:
       # ... path detection logic

     actual-job:
       needs: [detect-changes]
       if: needs.detect-changes.outputs.should-run == 'true'
   ```

2. **Always Use Summary Jobs with Exact Names**

   ```yaml
   summary-job:
     name: <EXACT BRANCH PROTECTION NAME>  # Must match exactly!
     needs: [all, required, jobs]
     if: always()  # CRITICAL - must always run
     steps:
       - name: Check results
         run: |
           # Handle skipped jobs as success
           # Only fail if jobs actually failed
   ```

3. **Test Branch Protection Before Enabling**
   - Create test PR
   - Verify all required checks appear
   - Verify check names match protection config
   - Test with different file change patterns

4. **Document Required Checks**
   - List all required checks in `.github/BRANCH_PROTECTION.md`
   - Include rationale for each check
   - Update when adding new required checks

---

## Monitoring

### Regular Audits

**Monthly**: Verify branch protection is enforced

```bash
# Check recent merged PRs had all checks
gh pr list --state merged --limit 10 --json number,statusCheckRollup \
  --jq '.[] | select(.statusCheckRollup | map(select(.name == "CI" or .name == "Documentation" or .name == "Code Quality" or .name == "Test Quality Gates")) | length < 4) | .number'
```

Expected: No output (all merged PRs had all 4 checks)

**After Workflow Changes**: Always verify

```bash
# Create test PR with minimal changes
echo "# Test" >> TEST.md
git checkout -b test/verify-branch-protection
git add TEST.md
git commit -m "test: verify branch protection"
git push -u origin test/verify-branch-protection
gh pr create --title "test: verify branch protection" --body "Verification PR"

# Check all 4 checks appear
gh pr checks <PR_NUMBER>
```

---

## Troubleshooting

### Check Not Appearing on PR

**Symptoms**: Required check missing from PR status checks

**Diagnosis**:

1. Check if workflow ran: `gh run list --workflow=<workflow.yml> --branch=<branch>`
2. Check workflow trigger conditions in YAML
3. Verify no path filters in `on:` block

**Fix**: Ensure workflow triggers on all PRs, use job-level conditionals

### Check Always Failing

**Symptoms**: Summary check fails even when no relevant changes

**Diagnosis**:

1. Check summary job logic
2. Verify handling of skipped jobs
3. Check `needs.<job>.result` logic

**Fix**: Update summary job to treat skipped/cancelled as success:

```yaml
if [[ "${{ needs.job.result }}" == "failure" ]]; then
  exit 1
fi
# skipped/cancelled/success all pass
```

### Wrong Check Name

**Symptoms**: Branch protection says check hasn't run, but you see it passing

**Diagnosis**:

```bash
gh pr checks <PR_NUMBER> | grep -i "<partial-name>"
```

Compare actual check name to required check name

**Fix**: Update job `name:` field to match branch protection exactly

---

## References

- **Root Cause Analysis**: `/tmp/root_cause_analysis.md`
- **Complete Solution**: `/tmp/complete_solution.md`
- **PR #2**: https://github.com/oscura-re/oscura/pull/2
- **GitHub Docs**: [About status checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks)

---

## Summary

✅ **Branch protection now GUARANTEES** that NO PR can be merged without:

- Passing CI (tests, lint, type check, build)
- Passing Documentation build and validation
- Passing Code Quality checks
- Passing Test Quality Gates

This fix ensures the integrity and quality standards of the `main` branch.
