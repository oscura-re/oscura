# Branch Protection Improvement Recommendations

## Current Situation

**Date**: 2026-01-20
**Issue**: Flaky test passed on PR branch but failed on merge commit
**PR**: #17 (Release 0.3.0)
**Affected Test**: `tests/unit/visualization/test_eye.py::TestPlotEye::test_auto_clock_recovery_fft`

## Root Cause

GitHub branch protection checks status of the **PR branch commit**, not the **merge commit**. When tests pass on the branch but fail on merge, the merge is allowed.

This is expected GitHub behavior, not a configuration error.

## Immediate Fix

PR #18: Make flaky test more robust

- Longer signal (100 → 500 bits)
- Deterministic pattern for FFT
- Reduced noise
- More reruns (2 → 3)

## Long-Term Recommendations

### Option 1: Enable Merge Queue (Recommended)

**What it does**: Tests run on the merge commit BEFORE merging

**How to enable**:

```bash
gh api -X PATCH "/repos/oscura-re/oscura/branches/main/protection" \
  -f "required_status_checks[strict]=true" \
  -f "required_status_checks[contexts][]=CI" \
  -f "required_status_checks[contexts][]=Code Quality" \
  -f "required_status_checks[contexts][]=Documentation" \
  -f "required_status_checks[contexts][]=Test Quality Gates" \
  -f "required_pull_request_reviews[required_approving_review_count]=0" \
  -f "enforce_admins=true" \
  -f "required_linear_history=true"
```

Then enable merge queue in repository settings.

**Pros**:

- Catches flaky tests before merge
- Prevents broken main branch
- GitHub's recommended solution

**Cons**:

- Slightly slower merges (wait for CI on merge commit)
- Requires GitHub Enterprise or GitHub Teams

### Option 2: Post-Merge CI Gate

Add a workflow that reverts commits if CI fails:

```yaml
# .github/workflows/post-merge-validation.yml
name: Post-Merge Validation

on:
  push:
    branches: [main]

jobs:
  revert-on-failure:
    runs-on: ubuntu-latest
    if: failure()
    needs: [ci, quality, docs]
    steps:
      - uses: actions/checkout@v4
      - name: Revert failed commit
        run: |
          git revert HEAD --no-edit
          git push
      - name: Create issue
        uses: actions/github-script@v7
        with:
          script: |
            await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: 'CI Failed on Main - Commit Reverted',
              body: 'Commit ${{ github.sha }} failed CI and was automatically reverted.'
            })
```

**Pros**:

- Works with free GitHub
- Automatic remediation

**Cons**:

- Main is briefly broken
- Reverts can cause confusion

### Option 3: Stricter Flaky Test Policy

**Policy**: No flaky tests allowed in main

- Require flaky tests to pass 100% of time on PR branch (use `reruns=5+`)
- Move unreliable tests to optional/exploratory suite
- Use test quarantine for known flaky tests

**Implementation**:

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "flaky: Tests with known instability (not run in required CI)",
    "quarantine: Tests under investigation (skipped)",
]

# .github/workflows/ci.yml
- name: Run required tests
  run: pytest -m "not flaky and not quarantine"
```

**Pros**:

- Forces test reliability improvements
- Clear signal when tests aren't production-ready

**Cons**:

- May require significant test refactoring
- Some legitimate flaky tests (external dependencies)

## Recommendation

**Implement Option 3 immediately** (stricter flaky test policy):

1. Move `test_auto_clock_recovery_fft` to optional suite OR
2. Fix it properly (PR #18 in progress) and remove `@pytest.mark.flaky`
3. Require future flaky tests to be marked and excluded from required CI

**Consider Option 1** (Merge Queue) if:

- More flaky test issues occur
- Team has GitHub Teams/Enterprise
- Want stronger guarantees for main branch

## Action Items

- [ ] Merge PR #18 (test fix)
- [ ] Monitor CI for 1 week to verify fix
- [ ] If fixed: Remove `@pytest.mark.flaky` decorator
- [ ] If not fixed: Move to exploratory test suite
- [ ] Document flaky test policy in CONTRIBUTING.md
- [ ] Consider merge queue if pattern repeats

## Related

- PR #17: Release 0.3.0 (merged with flaky test)
- PR #18: Fix flaky FFT clock recovery test
- Issue: (create if needed) "Establish flaky test policy"
