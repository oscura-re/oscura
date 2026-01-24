# CRITICAL: Merge Commit CI Bypass Problem

**Status**: ACTIVE ISSUE - Commits merged to main without passing CI
**Occurrences**: 2 incidents (commits 6f9e2c0, 4533316)
**Impact**: Failed CI checks on main branch, broken build states

## Root Cause

GitHub branch protection checks the **PR branch commit** CI status, NOT the **merge commit** CI status.

### How It Happens

```
1. Developer creates PR from feature branch
2. CI runs on feature branch commit ✅ PASS
3. GitHub branch protection: "All checks passed, merge allowed"
4. Developer merges (squash/merge commit)
5. **NEW commit created on main** (never tested before!)
6. CI runs on merge commit ❌ FAIL
7. Main branch now has failing commit!
```

### Why Branch Protection Doesn't Catch This

- **Squash merges** create a NEW commit with different SHA
- **Merge commits** combine multiple changes in ways not tested on PR branch
- **GitHub limitation**: Can't test merge commit BEFORE merge completes
- **Random test sampling**: Pre-commit hooks may sample different tests between runs

## Incidents

### Incident 1: Commit 6f9e2c0 (Release 0.3.0)

- **PR branch**: FFT clock recovery test passed
- **Main branch**: Same test failed on merge commit
- **Cause**: Test flakiness + random behavior differences

### Incident 2: Commit 4533316 (pytest-benchmark fix)

- **PR branch**: Test isolation check passed (sampled different files)
- **Main branch**: Test isolation check failed (sampled `test_performance.py`)
- **Cause**: Random sampling + file with only performance tests

## Permanent Solutions (Choose One)

### Option 1: Merge Queue (RECOMMENDED)

**What**: GitHub merge queue feature
**How**: Test merge commit BEFORE merging to main

```yaml
# .github/workflows/merge-queue.yml
on:
  merge_group:
    types: [checks_requested]

jobs:
  # Run ALL CI checks on merge commit
  # Only merge if ALL pass
```

**.github/branch-protection.yaml** (if using ruleset):

```yaml
branches:
  main:
    protection:
      required_status_checks:
        strict: true # Require branch up-to-date
      merge_queue:
        enabled: true
        grouping_strategy: HEADGREEN
        merge_method: SQUASH
```

**Pros**:

- Tests actual merge commit before merge
- Native GitHub feature
- Prevents ALL merge-time failures

**Cons**:

- Requires GitHub Enterprise or public repo
- Adds ~5-10min merge latency

### Option 2: Require Linear History (Fast-Forward Only)

**What**: Only allow fast-forward merges (no squash, no merge commits)
**How**: Require branches to be rebased on main before merge

**Branch Protection**:

- Enable "Require linear history"
- Disable "Allow squash merging"
- Disable "Allow merge commits"
- Enable "Require branches to be up to date before merging"

**Workflow**:

```bash
# Before merge
git fetch origin
git rebase origin/main
git push --force-with-lease
# Now PR branch commit IS the future main commit
# CI on PR = CI on future main
```

**Pros**:

- PR branch commit = merge commit (identical)
- No "untested commit" problem
- Clean linear history

**Cons**:

- Requires developer discipline (rebase workflow)
- Force pushes can be risky
- Lose squash convenience

### Option 3: Post-Merge CI with Auto-Revert

**What**: If main CI fails, automatically revert
**How**: GitHub Action watches main CI, reverts on failure

```yaml
# .github/workflows/auto-revert.yml
on:
  workflow_run:
    workflows: ['CI']
    branches: [main]
    types: [completed]

jobs:
  revert-on-failure:
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2

      - name: Revert failed commit
        run: |
          git revert HEAD --no-edit
          git push origin main

      - name: Create issue
        uses: actions/github-script@v7
        with:
          script: |
            await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: 'Auto-revert: Main branch CI failed',
              body: 'Commit ${{ github.sha }} was automatically reverted due to CI failure.'
            })
```

**Pros**:

- Automatic recovery
- Main stays mostly healthy

**Cons**:

- Brief period with broken main
- Reverts can cause confusion
- Doesn't prevent the problem

### Option 4: Eliminate Random Sampling

**What**: Make all checks deterministic
**How**: Remove randomization from test isolation check

**Change**:

```python
# Before: Random sample
sampled_files = random.sample(test_files, sample_size)

# After: Deterministic (e.g., alphabetical first N)
sampled_files = sorted(test_files)[:sample_size]
```

**Pros**:

- Same files tested every time
- Predictable behavior

**Cons**:

- Only fixes THIS specific issue
- Doesn't solve general merge commit problem
- Reduces test coverage diversity

## Recommended Approach

**SHORT TERM** (Immediate):

1. ✅ Fix test isolation exclusions (add `test_performance.py`)
2. Commit and test locally
3. Create PR, wait for FULL CI pass
4. Manually verify ALL checks green before merge

**MEDIUM TERM** (Next Sprint):

1. Implement **Merge Queue** (Option 1) - best long-term solution
2. Document new merge workflow
3. Update CONTRIBUTING.md

**LONG TERM** (Continuous):

1. Eliminate random sampling where possible
2. Improve test reliability (reduce flakiness)
3. Add more pre-merge validations

## Implementation Plan

### Phase 1: Immediate Fix (Today)

- [x] Fix test isolation script
- [ ] Test fix locally
- [ ] Create PR
- [ ] **MANUAL VERIFICATION**: Check ALL CI jobs before merge
- [ ] Merge only after triple-checking

### Phase 2: Enable Merge Queue (This Week)

- [ ] Create `.github/workflows/merge-queue.yml`
- [ ] Test on non-protected branch
- [ ] Enable in repository settings
- [ ] Document new workflow
- [ ] Train team

### Phase 3: Process Improvements (Ongoing)

- [ ] Add deterministic testing where possible
- [ ] Improve flaky test detection
- [ ] Monitor main branch health metrics

## Monitoring

Add alerts for main branch CI failures:

```yaml
# .github/workflows/alert-main-failure.yml
on:
  workflow_run:
    workflows: ['CI']
    branches: [main]
    types: [completed]

jobs:
  alert:
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    runs-on: ubuntu-latest
    steps:
      - name: Send alert
        # Slack/email/etc
```

## Decision

**DECISION NEEDED**: Which solution to implement?

Vote with comments:

- 👍 Option 1: Merge Queue
- 🎯 Option 2: Linear History
- 🔄 Option 3: Auto-Revert
- 📊 Option 4: Remove Randomness

**Recommended**: Option 1 (Merge Queue) for comprehensive protection
