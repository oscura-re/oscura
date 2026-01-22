# Phantom Agents Root Cause Analysis and Solution

**Date**: 2026-01-22
**Status**: ✅ Implemented and Tested

## Root Cause

Phantom agents occur when the agent registry (`.claude/agent-registry.json`) marks agents as "running" but they are no longer executing. This happens due to:

### 1. **Output File Location Mismatch**

- Agents store output in `/tmp/claude/-<project-path>/tasks/*.output`
- Existing cleanup hook only checks `.claude/agent-outputs/` for activity
- When task files are missing/stale, hook doesn't detect phantom state

### 2. **Session Interruption**

- Sessions can end abruptly (crashes, Ctrl+C, system restart)
- Agent status remains "running" in persistent registry
- No automatic cleanup on session abnormal termination

### 3. **Temp Directory Cleanup**

- System temp cleanup may delete `/tmp/claude/` files
- Registry not updated when external cleanup occurs
- Orphaned registry entries remain indefinitely

### 4. **Registry Persistence Without Validation**

- Registry persists across sessions (by design)
- No validation that "running" agents still have active output files
- No age-based staleness detection for missing output files

## Impact

**Observed Symptoms**:

- 2 agents from 2+ days ago marked "running" in registry
- No corresponding output files in `/tmp/claude/`
- Agent limit enforcement prevents new agents from launching
- 56 task output files accumulated (47 empty)

**Actual Impact**:

- Agent orchestration blocked due to false "max concurrent" limit
- Context pollution from tracking non-existent agents
- No automatic recovery mechanism

## Solution (4-Layer Defense)

### Layer 1: Immediate Fix Tool ✅

**File**: `.claude/hooks/fix_phantom_agents.py`

Standalone tool that:

- Detects agents marked "running" without valid output files
- Checks for stale agents (>24h old)
- Marks phantom agents as "stale" (preserves history)
- Cleans up old empty task files
- Can run manually or as emergency recovery

**Usage**:

```bash
python3 .claude/hooks/fix_phantom_agents.py         # Fix issues
python3 .claude/hooks/fix_phantom_agents.py --dry-run  # Preview
```

**Results** (Initial Run):

- ✅ Fixed 2 phantom agents from 2-day-old session
- ✅ Updated registry with cleanup metadata
- ✅ Validated registry structure

### Layer 2: Enhanced Cleanup Hook 🔄

**File**: `.claude/hooks/cleanup_stale_agents.py` (enhancement pending)

**Current Issue**: Only checks `.claude/agent-outputs/` for activity, misses `/tmp/claude/` task files

**Proposed Enhancement**:

1. Check `agent["output_file"]` path directly (in `/tmp/claude/`)
2. Validate output file exists and has recent activity
3. Mark agents with missing output files as phantom
4. Cross-check both locations for comprehensive detection

**Implementation Status**: Identified but not yet implemented (existing hook works but incomplete)

### Layer 3: Session Lifecycle Hooks ⏳

**Missing Hooks to Add**:

1. **SessionEnd Hook** (not yet implemented)
   - Mark all "running" agents as "interrupted" on session end
   - Provide recovery mechanism on next session start
   - Prevent orphaned "running" states

2. **Task Output Validator** (not yet implemented)
   - Periodic validation that "running" agents have active output files
   - Auto-mark as phantom when output file deleted externally
   - Could run every N minutes during long sessions

### Layer 4: Registry Schema Enhancement ⏳

**Proposed Fields** (not yet implemented):

```json
{
  "agents": {
    "agent_id": {
      "status": "running",
      "session_id": "uuid",           // Track which session launched
      "last_validated": "timestamp",  // Last time we checked it's alive
      "last_activity": "timestamp",   // Last output file modification
      "heartbeat_file": "path"        // File that must exist while running
    }
  }
}
```

**Benefits**:

- Detect cross-session orphans
- Track actual agent activity
- Enable proactive validation

## Configuration Changes

**Current** (`.claude/config.yaml`):

```yaml
hooks:
  cleanup_stale_agents:
    stale_threshold_hours: 24
    activity_check_hours: 1
    max_age_days: 30
```

**Recommended Addition**:

```yaml
hooks:
  cleanup_stale_agents:
    stale_threshold_hours: 24
    activity_check_hours: 1
    max_age_days: 30
    check_task_output_files: true      # NEW: Validate /tmp/claude/ files
    mark_missing_as_phantom: true      # NEW: Auto-mark when file missing
    cleanup_empty_task_files_hours: 24 # NEW: Clean old empty files
```

## Prevention Strategy

### Immediate (Implemented)

1. ✅ Manual cleanup tool available
2. ✅ Dry-run mode for safety
3. ✅ Detailed logging of cleanup actions

### Short-term (Next Steps)

1. 🔄 Enhance existing cleanup hook to check `/tmp/claude/`
2. 🔄 Add SessionEnd hook to mark running agents
3. 🔄 Add periodic validation during sessions

### Long-term (Architectural)

1. ⏳ Add session tracking to agent registry
2. ⏳ Implement heartbeat mechanism
3. ⏳ Add registry schema validation on load
4. ⏳ Create agent lifecycle state machine

## Testing

**Verification Steps**:

```bash
# 1. Check current registry status
cat .claude/agent-registry.json | jq '.agents[] | select(.status=="running")'

# 2. Run phantom detection (dry-run)
python3 .claude/hooks/fix_phantom_agents.py --dry-run

# 3. Clean if needed
python3 .claude/hooks/fix_phantom_agents.py

# 4. Verify fix
cat .claude/agent-registry.json | jq '.agents[] | select(.status=="running")'

# 5. Check running agent count
cat .claude/agent-registry.json | jq '[.agents[] | select(.status=="running")] | length'
```

**Expected Results**:

- No agents marked "running" without active output files
- Old empty task files cleaned up
- Agent limit enforcement works correctly

## Monitoring

**Health Check Indicators**:

```bash
# Running agents (should be 0-2)
jq '[.agents[] | select(.status=="running")] | length' .claude/agent-registry.json

# Stale agents (should be cleaned periodically)
jq '[.agents[] | select(.status=="stale")] | length' .claude/agent-registry.json

# Task files (shouldn't accumulate indefinitely)
find /tmp/claude/-home-lair-click-bats-development-oscura/tasks -name "*.output" | wc -l

# Empty task files (should be cleaned)
find /tmp/claude/-home-lair-click-bats-development-oscura/tasks -name "*.output" -size 0 | wc -l
```

**Alerting Thresholds**:

- ⚠️ Warning: >10 empty task files
- 🚨 Critical: >5 agents marked "running" for >24h
- 🚨 Critical: >100 task files total

## Recovery Procedures

### If Agent Limit Reached

```bash
# 1. Check for phantoms
python3 .claude/hooks/fix_phantom_agents.py --dry-run

# 2. Clean if found
python3 .claude/hooks/fix_phantom_agents.py

# 3. Verify
jq '[.agents[] | select(.status=="running")] | length' .claude/agent-registry.json
```

### If Registry Corrupted

```bash
# 1. Backup current registry
cp .claude/agent-registry.json .claude/agent-registry.json.backup

# 2. Validate structure
python3 .claude/hooks/fix_phantom_agents.py

# 3. If validation fails, reset
echo '{"agents": {}}' > .claude/agent-registry.json
```

### If Task Files Accumulate

```bash
# Clean old empty files manually
find /tmp/claude/-home-lair-click-bats-development-oscura/tasks \
  -name "*.output" -size 0 -mtime +1 -delete
```

## Lessons Learned

1. **Persistence Requires Validation**: Any persistent state needs periodic validation
2. **Temporary Files Need Tracking**: External cleanup of temp files must update registry
3. **Session Boundaries Matter**: Need explicit session start/end handling
4. **Multiple Sources of Truth**: Output files and registry must stay in sync
5. **Defensive Programming**: Always assume registry might be stale

## Future Enhancements

1. **Distributed Lock**: Use file-based lock for concurrent registry access
2. **Audit Trail**: Log all registry modifications
3. **Metrics**: Track agent launch/completion rates, failure patterns
4. **Dashboard**: Visual monitoring of agent health
5. **Auto-recovery**: Automatic restart of failed agents
6. **Resource Limits**: Enforce disk space limits for task outputs

## References

- Agent Registry: `.claude/agent-registry.json`
- Config: `.claude/config.yaml` (lines 200-204)
- Cleanup Hook: `.claude/hooks/cleanup_stale_agents.py`
- Fix Tool: `.claude/hooks/fix_phantom_agents.py`
- Task Output: `/tmp/claude/-<project>/tasks/*.output`

## Status

- ✅ Root cause identified
- ✅ Immediate fix implemented and tested
- ✅ Documentation complete
- 🔄 Hook enhancement design complete (implementation pending)
- ⏳ Session lifecycle hooks (future work)
- ⏳ Registry schema enhancement (future work)

**Next Action**: Enhance `cleanup_stale_agents.py` to check `/tmp/claude/` task files in addition to `.claude/agent-outputs/`.
