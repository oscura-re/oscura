---
name: cleanup
description: Run maintenance tasks to clean up orchestration artifacts
arguments: [--dry-run, --force]
---

# Cleanup Command

Execute maintenance tasks to archive old files, clean stale agents, and optimize disk usage.

## Usage

```bash
/cleanup              # Run all cleanup tasks
/cleanup --dry-run    # Show what would be cleaned without doing it
/cleanup --force      # Skip confirmations and force cleanup
```

## Examples

```bash
/cleanup              # Interactive cleanup with confirmations
/cleanup --dry-run    # Preview cleanup actions
/cleanup --force      # Force cleanup without prompts
```

## Cleanup Tasks

### 1. Archive Old Completion Reports

**Target**: `.claude/agent-outputs/`

- Archives reports older than 7 days
- Moves to `.claude/agent-outputs/archive/`
- Compresses archives older than 30 days
- Removes archives older than 90 days

**Command**: `.claude/hooks/cleanup_completed_workflows.sh`

### 2. Clean Coordination Files

**Target**: `.coordination/`

- Archives coordination JSON files older than 30 days
- Archives markdown files older than 3 days
- Cleans completed workflow progress files
- Removes expired locks (>60 minutes)

**Command**: `.claude/hooks/cleanup_completed_workflows.sh`

### 3. Mark Stale Agents

**Target**: `.claude/agent-registry.json`

- Marks agents running >60 minutes as failed
- Updates agent registry metadata
- Logs stale agent warnings

**Command**: `.claude/hooks/cleanup_stale_agents.py`

### 4. Archive Old Checkpoints

**Target**: `.coordination/checkpoints/`

- Archives checkpoints older than 7 days
- Moves to `.coordination/checkpoints/.archive/`
- Removes archived checkpoints older than 30 days

**Command**: `.claude/hooks/archive_old_checkpoints.py`

### 5. Clean Session Artifacts

**Target**: Various temporary files

- Removes temporary coordination files
- Cleans up orphaned lock files
- Removes empty directories

**Command**: `.claude/hooks/session_cleanup.sh`

## Retention Policies

From `.claude/paths.yaml`:

|Artifact|Retention|Archive|Delete|
|---|---|---|---|
|Coordination files|Active|30 days|90 days|
|Agent outputs|Active|7 days|30 days|
|Checkpoints|Active|7 days|30 days|
|Markdown files|Active|3 days|N/A|
|Locks|Active|60 min|Immediate|

## Sample Output

```
Orchestration Cleanup
=====================

1. Archiving old completion reports...
   ✓ Archived 12 reports (older than 7 days)
   ✓ Compressed 5 archives (older than 30 days)
   ✓ Removed 2 archives (older than 90 days)
   Freed: 2.3 MB

2. Cleaning coordination files...
   ✓ Archived 8 JSON files (older than 30 days)
   ✓ Archived 15 markdown files (older than 3 days)
   ✓ Removed 3 expired locks
   Freed: 1.1 MB

3. Marking stale agents...
   ✓ Marked 1 stale agent as failed (running >60 min)
   ✓ Updated agent registry

4. Archiving old checkpoints...
   ✓ Archived 3 checkpoints (older than 7 days)
   ✓ Removed 1 old archive (older than 30 days)
   Freed: 5.8 MB

5. Cleaning session artifacts...
   ✓ Removed 7 temporary files
   ✓ Cleaned 2 empty directories
   Freed: 0.3 MB

Total Freed: 9.5 MB
```

## Dry-Run Output

When using `--dry-run`, shows what would be cleaned:

```
Cleanup Preview (Dry Run)
==========================

Would archive:
  • 12 completion reports (.claude/agent-outputs/)
  • 8 coordination JSON files (.coordination/)
  • 15 markdown files (.coordination/)
  • 3 checkpoints (.coordination/checkpoints/)

Would remove:
  • 2 old archives (>90 days)
  • 1 old checkpoint archive (>30 days)
  • 3 expired locks
  • 7 temporary files

Estimated space freed: 9.5 MB

Run without --dry-run to perform cleanup.
```

## Safety Features

### Confirmation Prompts

Without `--force`, prompts before:

- Archiving active coordination files
- Removing old archives
- Marking agents as failed
- Deleting checkpoints

### Protected Files

Never touches:

- `.coordination/README.md`
- `.claude/agent-outputs/.gitkeep`
- Any file modified in last 24 hours (unless `--force`)

### Backup Before Delete

- Creates backup in `.coordination/archive/cleanup-backup-TIMESTAMP/`
- Keeps backup for 7 days
- Can restore with `/cleanup --restore TIMESTAMP`

## Manual Cleanup

Individual cleanup scripts can be run manually:

```bash
# Clean coordination directory
.claude/hooks/session_cleanup.sh

# Mark stale agents
.claude/hooks/cleanup_stale_agents.py

# Clean completed workflows
.claude/hooks/cleanup_completed_workflows.sh
```

## Health Check Integration

Cleanup recommendations also shown in:

- `/status` - Displays cleanup suggestions
- `/context` - Shows context optimization via cleanup

## Automation

### Automatic Cleanup

Automatic cleanup runs:

- **SessionEnd**: Light cleanup (temp files, expired locks)
- **Weekly**: Full cleanup (archives, old checkpoints)

### Disable Automatic Cleanup

Edit `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionEnd": [] // Remove session_cleanup.sh
  }
}
```

## See Also

- `/status` - View cleanup recommendations
- `.claude/hooks/README.md` - Hook documentation
- `.claude/paths.yaml` - Retention policy configuration
