# Phantom Agent Prevention & Recovery

## Quick Start

### Check for Phantom Agents

```bash
# Show running agents
jq '[.agents[] | select(.status=="running")] | length' .claude/agent-registry.json

# List all running agents with details
jq '.agents[] | select(.status=="running") | {id: .task, file: .output_file}' .claude/agent-registry.json
```

### Fix Phantom Agents

```bash
# Preview what would be fixed
python3 .claude/hooks/fix_phantom_agents.py --dry-run

# Fix issues
python3 .claude/hooks/fix_phantom_agents.py
```

### Test Cleanup Hook

```bash
# Run enhanced cleanup hook
python3 .claude/hooks/cleanup_stale_agents.py --dry-run
```

## What Are Phantom Agents?

**Phantom agents** are entries in `.claude/agent-registry.json` marked as "running" but no longer actively executing. They:

- Block new agents from launching (false "max concurrent" limit)
- Pollute the agent registry with stale data
- Indicate session interruption or external temp file cleanup

## Root Causes

1. **Missing Output Files**: Task output in `/tmp/claude/` deleted externally
2. **Session Interruption**: Crashes, Ctrl+C, system restarts
3. **Stale Detection Gap**: Old cleanup hook didn't check `/tmp/claude/` files

## Tools

### 1. Manual Fix Tool (`fix_phantom_agents.py`) ✅

**Use when**: Agent limit reached, registry appears stale, manual intervention needed

**Features**:

- Detects agents with missing/stale output files
- Marks phantom agents as "stale" (preserves history)
- Cleans up old empty task files
- Validates registry structure
- Dry-run mode for safety

**Example**:

```bash
$ python3 .claude/hooks/fix_phantom_agents.py --dry-run
======================================================================
PHANTOM AGENT CLEANUP
======================================================================
Step 2: Cleaning phantom agents...
  Total agents in registry: 4
  Phantom agents found: 2
    - Missing/stale output files: 2
    - Stale (>24h): 2

  Fixed agents:
    - ab505f2: Documentation completeness audit (missing or stale output file, stale (>24h old))
    - aa38d4a: Testing strategy audit (missing or stale output file, stale (>24h old))
```

### 2. Enhanced Cleanup Hook (`cleanup_stale_agents.py`) ✅

**Use when**: Automatic cleanup on session start

**Enhancements** (as of 2026-01-22):

- ✅ Checks actual task output files in `/tmp/claude/`
- ✅ Detects missing output files (phantom agents)
- ✅ Detects old empty files (stale agents)
- ✅ Preserves agents with recent activity
- ✅ Returns detailed cleanup statistics

**Automatic Trigger**: Runs on every session start (configured in `.claude/config.yaml`)

**Example Output**:

```json
{
  "ok": true,
  "phantom_marked_failed": 2,
  "stale_marked_failed": 0,
  "old_removed": 1,
  "active_preserved": 0
}
```

## Prevention Strategy

### Automatic (Configured)

1. ✅ Session-start cleanup hook (checks phantoms)
2. ✅ Stale threshold: 24 hours
3. ✅ Activity check: 1 hour window
4. ✅ Old agent removal: 30 days

### Manual (As Needed)

1. Run `fix_phantom_agents.py` when limit reached
2. Clean old task files periodically
3. Monitor agent count trends

## Monitoring

### Health Check Commands

```bash
# Count running agents (should be 0-2)
jq '[.agents[] | select(.status=="running")] | length' .claude/agent-registry.json

# Count stale agents
jq '[.agents[] | select(.status=="stale")] | length' .claude/agent-registry.json

# Count task files
find /tmp/claude/-home-lair-click-bats-development-oscura/tasks -name "*.output" | wc -l

# Count empty task files
find /tmp/claude/-home-lair-click-bats-development-oscura/tasks -name "*.output" -size 0 | wc -l

# List running agents with details
jq '.agents[] | select(.status=="running") | {task, launched_at, output_file}' .claude/agent-registry.json
```

### Alert Thresholds

- ⚠️ **Warning**: >5 running agents for >1 hour
- ⚠️ **Warning**: >10 empty task files
- 🚨 **Critical**: >10 running agents
- 🚨 **Critical**: Any agent running >24 hours
- 🚨 **Critical**: >100 task files total

## Recovery Procedures

### Scenario 1: Agent Limit Reached

```bash
# 1. Check current state
jq '[.agents[] | select(.status=="running")] | length' .claude/agent-registry.json

# 2. Look for phantoms
python3 .claude/hooks/fix_phantom_agents.py --dry-run

# 3. Fix if found
python3 .claude/hooks/fix_phantom_agents.py

# 4. Verify
jq '[.agents[] | select(.status=="running")] | length' .claude/agent-registry.json
```

### Scenario 2: Registry Corrupted

```bash
# 1. Backup
cp .claude/agent-registry.json .claude/agent-registry.json.backup.$(date +%s)

# 2. Validate and fix
python3 .claude/hooks/fix_phantom_agents.py

# 3. If still broken, reset (LAST RESORT)
echo '{"agents": {}}' > .claude/agent-registry.json
```

### Scenario 3: Task Files Accumulating

```bash
# Clean old empty files (>24h)
find /tmp/claude/-home-lair-click-bats-development-oscura/tasks \
  -name "*.output" -size 0 -mtime +1 -delete

# Or use the fix tool
python3 .claude/hooks/fix_phantom_agents.py
```

## Configuration

**File**: `.claude/config.yaml`

**Relevant Settings**:

```yaml
orchestration:
  agents:
    max_concurrent: 2                    # Agent limit
    max_batch_size: 2
    polling_interval_seconds: 10

hooks:
  cleanup_stale_agents:
    stale_threshold_hours: 24            # Mark as stale after
    activity_check_hours: 1              # Recent activity window
    max_age_days: 30                     # Remove completed agents after

cleanup:
  session_start:
    - stale_agents                       # Runs cleanup hook
    - old_locks
    - health_check
```

## Technical Details

### Agent Lifecycle States

```
running → { completed | failed | stale }
         ↓
      deleted (after max_age_days)
```

### Detection Logic (Enhanced)

```python
# 1. Check if task output file exists
if agent["status"] == "running" and "output_file" in agent:
    task_output = Path(agent["output_file"])
    if not task_output.exists():
        → PHANTOM AGENT

# 2. Check if task output is old and empty
    elif task_output.size == 0 and age > 1 hour:
        → PHANTOM AGENT

# 3. Check for stale agents (old with no activity)
    elif age > 24 hours and no_recent_activity:
        → STALE AGENT
```

### Files Checked for Activity

1. **Primary**: `agent["output_file"]` (in `/tmp/claude/...`)
2. **Secondary**: `.claude/agent-outputs/*.json`
3. **Tertiary**: `.claude/summaries/{agent_id}.md`

## Maintenance Schedule

### Daily

- Automatic cleanup on session start (handled by hook)

### Weekly

- Manual review of agent counts
- Clean old task files if >50
- Check for accumulating empty files

### Monthly

- Review cleanup logs
- Adjust thresholds if needed
- Archive old agent registry backups

## Troubleshooting

### Q: Why are there agents marked "running" for days?

**A**: These are phantom agents. The task files were deleted or the session ended abnormally. Run `fix_phantom_agents.py`.

### Q: Why can't I launch new agents?

**A**: You've hit the max concurrent limit (2). Check for phantom agents with `fix_phantom_agents.py --dry-run`.

### Q: Why are task files accumulating?

**A**: Empty task files don't get cleaned automatically. The fix tool cleans files >24h old.

### Q: What if the cleanup hook doesn't run?

**A**: Check `.claude/config.yaml` line 111 includes `- stale_agents`. Manually run the hook to test.

### Q: Can I delete the agent registry?

**A**: Yes, but you'll lose all agent history. Better to use `fix_phantom_agents.py` to clean it up properly.

## Recent Improvements (2026-01-22)

1. ✅ Created `fix_phantom_agents.py` standalone tool
2. ✅ Enhanced `cleanup_stale_agents.py` to check `/tmp/claude/` files
3. ✅ Added phantom agent detection for missing output files
4. ✅ Added detection for old empty output files
5. ✅ Improved logging with phantom vs stale distinction
6. ✅ Added comprehensive documentation

## References

- **Agent Registry**: `.claude/agent-registry.json`
- **Config**: `.claude/config.yaml` (lines 10-14, 200-204)
- **Fix Tool**: `.claude/hooks/fix_phantom_agents.py`
- **Cleanup Hook**: `.claude/hooks/cleanup_stale_agents.py`
- **Task Outputs**: `/tmp/claude/-<project>/tasks/*.output`
- **Full Analysis**: `.claude/docs/phantom-agents-solution.md`

## Support

If you encounter issues:

1. Run `fix_phantom_agents.py --dry-run` to diagnose
2. Check `.claude/hooks/errors.log` for hook failures
3. Review agent registry structure with `jq` commands above
4. As last resort, backup and reset registry
