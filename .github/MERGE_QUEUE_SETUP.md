# Merge Queue Setup Guide

**Purpose**: Prevent untested commits from landing on main by testing the ACTUAL merge commit before merging.

**Status**: ✅ Workflow created (`.github/workflows/merge-queue.yml`)
**Next Step**: Enable in repository settings

---

## The Problem We're Solving

**Current behavior** (causes CI failures on main):

```
1. PR created from feature branch
2. CI runs on PR branch commit → ✅ PASS
3. GitHub checks PR branch CI status → Merge allowed
4. Squash merge creates NEW commit on main
5. CI runs on NEW merge commit → ❌ FAIL (too late!)
```

**With Merge Queue enabled**:

```
1. PR created from feature branch
2. CI runs on PR branch commit → ✅ PASS
3. Developer clicks "Merge"
4. GitHub creates TEMPORARY merge commit
5. Merge queue CI runs on temporary commit
6. If ✅ PASS → Merge completes
7. If ❌ FAIL → Merge BLOCKED
```

---

## Step 1: Enable Merge Queue in Repository Settings

### Via Web UI

1. Go to https://github.com/oscura-re/oscura/settings/branches
2. Find the `main` branch rule (or create one)
3. Scroll to "Merge queue"
4. Click ✅ "Require merge queue"
5. Configure settings:
   - **Merge method**: Squash (recommended)
   - **Build concurrency**: 1 (prevent race conditions)
   - **Merge queue grouping**: HEADGREEN
   - **Minimum time in queue**: 0 minutes
   - **Maximum time in queue**: 60 minutes
   - **Only merge non-failing PRs**: ✅ Enabled
6. Click "Save changes"

### Via API (Alternative)

```bash
# Requires GitHub CLI with admin permissions
gh api -X PUT repos/oscura-re/oscura/branches/main/protection \
  --field required_status_checks='{"strict":true,"contexts":["merge-queue-summary"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"dismiss_stale_reviews":true,"require_code_owner_reviews":false}' \
  --field restrictions=null \
  --field allow_force_pushes=false \
  --field allow_deletions=false \
  --field required_conversation_resolution=true \
  --field lock_branch=false \
  --field allow_fork_syncing=false
```

---

## Step 2: Update Branch Protection Rules

**Required status checks**:

- `Pre-Commit Hooks`
- `CI`
- `Code Quality`
- `Documentation`
- `Test Quality Gates`
- `CodeQL`
- **`merge-queue-summary`** ← NEW: Merge queue check

**Settings to enable**:

- ✅ Require branches to be up to date before merging
- ✅ Require status checks to pass before merging
- ✅ Require conversation resolution before merging
- ✅ Require linear history (optional but recommended)

---

## Step 3: Test the Merge Queue

### Test PR Workflow

1. Create a test PR with a trivial change
2. Wait for PR CI to complete
3. Click "Merge when ready" button
4. Observe:
   - PR enters merge queue
   - Temporary merge commit created
   - Merge queue CI runs (see `.github/workflows/merge-queue.yml`)
   - If all checks pass → Auto-merges
   - If any check fails → Merge blocked

### Expected Behavior

**When merge queue CI passes**:

```
✅ All checks have passed
✅ Merge queue: Ready to merge
→ [Merge when ready] button enabled
→ Click → Auto-merges
```

**When merge queue CI fails**:

```
❌ Merge blocked
❌ Merge queue: Checks failed
→ [Merge when ready] button disabled
→ Fix issues, push new commit
→ Re-enters queue automatically
```

---

## Step 4: Update Developer Workflow Documentation

Add to `CONTRIBUTING.md`:

```markdown
## Merging Pull Requests

This repository uses a **merge queue** to ensure all commits on `main` pass CI.

### How It Works

1. Create your PR as normal
2. Wait for PR CI to pass
3. Click **"Merge when ready"** button
4. Your PR enters the merge queue
5. GitHub creates a temporary merge commit
6. Merge queue CI runs (same checks as PR CI)
7. If all pass → Auto-merges
8. If any fail → Merge blocked, fix and retry

### Why We Use Merge Queue

- **Prevents untested commits**: Squash merges create new commits that aren't tested
- **Catches merge conflicts**: Tests the actual code that will land on `main`
- **Eliminates CI failures on main**: No more broken builds

### Troubleshooting

**"Merge blocked by merge queue"**:

- Check the merge queue CI run for failures
- Fix the issues in your branch
- Push a new commit
- Queue automatically restarts

**"Waiting in merge queue"**:

- Queue processes PRs one at a time
- Your turn will come (usually <10 minutes)
- You can continue working on other PRs
```

---

## Step 5: Monitor Merge Queue Health

### Metrics to Track

```bash
# Check merge queue status
gh api repos/oscura-re/oscura/pulls --jq '.[] | select(.state == "open") | {number, title, mergeable_state}'

# Check recent merge queue runs
gh run list --workflow=merge-queue.yml --limit 10

# Check merge queue failure rate
gh run list --workflow=merge-queue.yml --limit 100 --json conclusion | \
  jq '[.[] | select(.conclusion == "failure")] | length'
```

### Success Metrics

- **Merge queue pass rate**: >95% (shows PRs are well-tested before queue)
- **Queue time**: <10 minutes average (shows queue is efficient)
- **Main branch CI failures**: 0 (the goal!)

---

## Rollback Plan

If merge queue causes problems, disable it:

1. Go to repository settings → Branches
2. Edit `main` branch protection
3. Uncheck "Require merge queue"
4. Save changes

PRs will immediately revert to direct merging.

**Note**: Keep the workflow file (`.github/workflows/merge-queue.yml`) - it's harmless when merge queue is disabled.

---

## Cost/Performance Impact

**CI Runs Per PR**:

- **Before**: 1 CI run (PR branch)
- **After**: 2 CI runs (PR branch + merge commit)

**Merge Latency**:

- **Before**: Instant (click merge → done)
- **After**: ~5-10 minutes (queue CI must complete)

**Trade-off**: Worth it to eliminate main branch failures!

---

## Advanced: Merge Queue Grouping

For high-traffic repositories, enable **grouping** to batch multiple PRs:

```yaml
# .github/merge-queue.yml (optional)
queue:
  merge_method: squash
  grouping_strategy: HEADGREEN # Group PRs targeting same SHA
  max_entries_to_merge: 5 # Batch up to 5 PRs
  min_entries_to_merge: 1 # Minimum 1 PR
  merge_commit_message_regex: '' # No filtering
```

**Benefits**:

- Faster throughput for multiple PRs
- Reduced CI costs (one run for 5 PRs)

**Risks**:

- If batched CI fails, all PRs blocked
- Harder to identify which PR caused failure

**Recommendation**: Start with 1 PR at a time, optimize later if needed.

---

## Troubleshooting

### Problem: Merge queue CI fails but PR CI passed

**Cause**: Merge commit differs from PR branch (e.g., conflicts, base branch changed)

**Solution**:

1. Update your branch: `git fetch origin && git rebase origin/main`
2. Resolve any conflicts
3. Push (may need `--force-with-lease`)
4. Queue automatically retries

### Problem: Merge queue stuck "in progress"

**Cause**: CI timeout or infrastructure issue

**Solution**:

1. Check workflow runs for errors
2. Cancel stuck run: `gh run cancel <run-id>`
3. Remove from queue and re-add

### Problem: Too many PRs waiting in queue

**Cause**: Low throughput (CI takes too long)

**Solutions**:

1. Optimize CI (parallel tests, caching)
2. Enable grouping (batch multiple PRs)
3. Increase concurrency (allow 2-3 simultaneous queue runs)

---

## Next Steps

- [ ] Enable merge queue in repository settings
- [ ] Update branch protection rules
- [ ] Test with a dummy PR
- [ ] Update CONTRIBUTING.md
- [ ] Announce to team
- [ ] Monitor for 1 week
- [ ] Optimize if needed

**Expected Outcome**: ZERO CI failures on main branch! 🎉
