# Lifecycle Hooks Documentation

Comprehensive guide to Claude Code lifecycle hooks in workspace template.

## Overview

Lifecycle hooks allow automated actions at specific points in Claude's workflow:

- **PreToolUse**: Validate operations before they execute (blocking)
- **PostToolUse**: Process results after tool execution (non-blocking)
- **PreCompact**: Clean up before context compaction
- **SessionStart**: Initialize after session starts or recovery
- **SessionEnd**: Clean up when session ends
- **Stop**: Handle graceful shutdown
- **SubagentStop**: Handle subagent completion

## Hook Inventory

### Security & Validation Hooks

#### PreToolUse: validate_path.py

**Purpose**: Prevent writes to sensitive files and validate path safety

**Triggers**: Before Write, Edit, NotebookEdit tool calls

**Behavior**: Blocking (non-zero exit = operation canceled)

**Features**:

- Blocks credentials (.env\*, \*.key, \*.pem, secrets.\*)
- Blocks .git directory internals
- Warns on critical configs (pyproject.toml, package.json, .claude/settings.json)
- Prevents path traversal attacks (../)
- Validates paths within project root

### Auto-Formatting Hooks

#### PostToolUse: auto_format.py

**Purpose**: Automatically format files after Claude writes/edits them

**Triggers**: After Write, Edit, NotebookEdit tool calls

**Behavior**: Non-blocking (always exits 0)

**Formatters**:

- **Python** -> `uv run ruff format` (via pyproject.toml)
- **Markdown/YAML/JSON** -> `npx prettier --write` (via .prettierrc.yaml)
- **Shell** -> `shfmt -i 2 -w` (2-space indent)
- **LaTeX** -> `latexindent -w` (via .latexindent.yaml)

### Context Management Hooks

#### PreCompact: pre_compact_cleanup.sh

**Purpose**: Clean up temporary files before context compaction

**Triggers**: Before automatic context compaction

**Features**:

- Archives old agent outputs (>7 days -> `.claude/agent-outputs/archive/`)
- Archives old coordination files (>30 days -> `.coordination/archive/`)
- Compresses large JSON files (>100KB)
- Cleans up test data temporary files
- Archives old validation reports (>7 days)
- Deduplicates redundant audit files
- Logs cleanup metrics

**Timeout**: 30s

#### SessionStart: post_compact_recovery.sh

**Purpose**: Verify session state after compaction or recovery

**Triggers**: After session starts (normal or post-compact)

**Features**:

- Validates .claude/ structure (warns but doesn't fail on CLAUDE.md)
- Checks .coordination/spec/ directory
- Verifies hook executability (auto-fixes if needed)
- Provides recovery suggestions on errors
- Logs recovery status

**Timeout**: 10s

#### SessionEnd: session_cleanup.sh

**Purpose**: Clean up session-specific temporary files

**Triggers**: When session ends normally

**Features**:

- Archives session logs
- Cleans up temporary files
- Removes expired lock files
- Validates final state

**Timeout**: 30s

#### check_context_usage.sh

**Purpose**: Monitor context usage and trigger warnings at thresholds

**Usage**: Can be called manually or periodically

**Features**:

- Estimates current token usage
- Triggers warnings at configured thresholds
- Tracks context growth rate
- Suggests cleanup actions
- Provides JSON or human-readable output

**Thresholds** (see `.claude/config.yaml` for authoritative values):

- 60%: Warning - consider summarizing completed work
- 65%: Checkpoint - create checkpoint now
- 75%: Critical - complete current task only, then checkpoint

**Example**:

```bash
# Get JSON status
.claude/hooks/check_context_usage.sh

# Get detailed report
.claude/hooks/check_context_usage.sh --report
```

### Checkpoint System

#### checkpoint_state.sh

**Purpose**: Create and manage checkpoints for long-running tasks

**Commands**:

- `create <task-id> [description]` - Create new checkpoint
- `list` - List all checkpoints
- `show <task-id>` - Show checkpoint details
- `delete <task-id>` - Delete checkpoint (archives first)
- `update <task-id> <key> <value>` - Update state

**Checkpoint Structure**:

```
.coordination/checkpoints/<task-id>/
  manifest.json   - Metadata
  state.json      - Task state
  context.md      - Human-readable summary
  artifacts/      - Important files
```

**Example**:

```bash
# Create checkpoint before compaction
.claude/hooks/checkpoint_state.sh create validation-task "Validating WFM files"

# Update progress
.claude/hooks/checkpoint_state.sh update validation-task phase "testing"
```

#### restore_checkpoint.sh

**Purpose**: Restore checkpoint state after compaction

**Usage**:

```bash
# Restore specific checkpoint
.claude/hooks/restore_checkpoint.sh <task-id>

# Restore most recent checkpoint
.claude/hooks/restore_checkpoint.sh --latest

# Get JSON output
.claude/hooks/restore_checkpoint.sh <task-id> --json
```

### Stop Handlers

#### Stop: check_stop.py

**Purpose**: Handle graceful shutdown of main session

**Triggers**: When session stops (user command or system)

**Timeout**: 10s

#### SubagentStop: check_subagent_stop.py

**Purpose**: Validate subagent completion reports

**Triggers**: When subagent completes (Task tool)

**Features**:

- Validates completion reports exist
- Checks for blocked status
- Warns on needs-review status

**Timeout**: 10s

## Utility Scripts

Located in `scripts/maintenance/`:

### batch_file_processor.py

Process files in batches to reduce context overhead:

```bash
# Process WFM files
python scripts/maintenance/batch_file_processor.py test_data "*.wfm" --batch-size 20

# Get size summary
python scripts/maintenance/batch_file_processor.py . "*.json" --output sizes
```

### json_summarizer.py

Extract key metrics from large JSON files:

```bash
# Summarize validation report
python scripts/maintenance/json_summarizer.py validation_report.json

# Extract specific keys
python scripts/maintenance/json_summarizer.py report.json --keys status,error_count
```

### context_optimizer.py

Analyze project for context optimization:

```bash
# Quick analysis
python scripts/maintenance/context_optimizer.py

# Detailed report
python scripts/maintenance/context_optimizer.py --report

# Suggestions only
python scripts/maintenance/context_optimizer.py --suggestions
```

### archive_coordination.sh

Archive old coordination files:

```bash
# Archive files older than 7 days
scripts/maintenance/archive_coordination.sh

# Dry run (show what would be archived)
scripts/maintenance/archive_coordination.sh --days 30 --dry-run
```

## Hook Execution Flow

### File Write Operation

```
User request: "Create example.py with..."
    |
PreToolUse: validate_path.py
    |-- Check: Is .env or .key? -> BLOCK
    |-- Check: Is pyproject.toml? -> WARN but allow
    +-- Check: Path traversal? -> BLOCK if detected
    |
Write tool executes (if allowed)
    |
PostToolUse: auto_format.py
    |-- Detect: example.py -> Python file
    |-- Check: uv available? -> Yes
    |-- Run: uv run ruff format example.py
    +-- Output: Auto-formatted: example.py (ruff formatted)
```

### Session Lifecycle

```
Session Start
    |
SessionStart: post_compact_recovery.sh
    +-- Validate .claude/ structure
    |
[Work happens]
    |
Context approaching limit (monitor with check_context_usage.sh)
    |
[Optional: checkpoint_state.sh create]
    |
PreCompact: pre_compact_cleanup.sh
    +-- Archive old files, compress large JSON
    |
[Compaction occurs]
    |
SessionStart: post_compact_recovery.sh (again)
    +-- Verify post-compact state
    |
[Optional: restore_checkpoint.sh --latest]
    |
Session End
    |
SessionEnd: session_cleanup.sh
    +-- Clean up and log
```

## Configuration

Hooks are configured in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR/.claude/hooks/pre_compact_cleanup.sh\"",
            "timeout": 30
          }
        ],
        "matcher": "auto"
      }
    ]
  }
}
```

## Environment Variables

Hooks have access to:

- `CLAUDE_PROJECT_DIR` = Project root directory
- `CLAUDE_SESSION_ID` = Current session ID
- `PATH` = System PATH (includes tool directories)

## Testing Hooks

### Manual Testing

```bash
# Test pre_compact_cleanup.sh
CLAUDE_PROJECT_DIR=/path/to/project .claude/hooks/pre_compact_cleanup.sh

# Test post_compact_recovery.sh
CLAUDE_PROJECT_DIR=/path/to/project .claude/hooks/post_compact_recovery.sh

# Test context monitoring
.claude/hooks/check_context_usage.sh --report

# Test checkpoint system
.claude/hooks/checkpoint_state.sh create test-task "Test checkpoint"
.claude/hooks/checkpoint_state.sh list
.claude/hooks/restore_checkpoint.sh test-task
.claude/hooks/checkpoint_state.sh delete test-task
```

### Integration Testing

Use Claude Code to trigger hooks naturally through normal operations.

## Log Files

- `compaction.log` - Pre/post compact operations
- `sessions.log` - Session start/end events
- `context_metrics.log` - Context usage monitoring
- `checkpoints.log` - Checkpoint operations
- `errors.log` - Hook errors

## Troubleshooting

### Hook Not Running

**Check**:

1. Script exists in `.claude/hooks/`
2. Script is executable (`chmod +x`)
3. Matcher pattern matches tool name
4. No syntax errors

**Debug**:

```bash
# Check hook logs
cat .claude/hooks/*.log

# Test hook directly
CLAUDE_PROJECT_DIR=$(pwd) .claude/hooks/<hook_name>
```

### Context Issues

**Use context monitoring**:

```bash
# Check current usage
.claude/hooks/check_context_usage.sh

# Get optimization suggestions
python scripts/maintenance/context_optimizer.py --suggestions
```

### Checkpoint Recovery

**If checkpoint restore fails**:

1. Check checkpoint exists: `.claude/hooks/checkpoint_state.sh list`
2. Verify checkpoint files: `.claude/hooks/checkpoint_state.sh show <task-id>`
3. Check state.json is valid JSON
4. Restore manually from context.md if needed

## Best Practices

### For Long-Running Tasks

1. Create checkpoint before 65% context usage
2. Use batch operations for file processing
3. Save large results to disk, reference by path
4. Monitor context with `check_context_usage.sh`
5. Prepare for compaction proactively

### For Compaction Readiness

1. Archive old coordination files regularly
2. Compress large JSON files
3. Remove test data temporary files
4. Create checkpoint with essential state
5. Document resumption steps in context.md

### Hook Design

1. **Single Responsibility**: One hook = one purpose
2. **Fast Execution**: < 1s for common operations
3. **Graceful Degradation**: Work without optional tools
4. **Clear Output**: Informative success/error messages
5. **Non-Blocking**: PostToolUse always exits 0

---

**Last Updated**: 2026-01-06
**Module**: base (workspace template)
**Hook Count**: 33 total hooks (6 security/validation, 2 formatting, 6 lifecycle/context, 2 stop handlers, 3 agent management, 3 checkpoint system, 2 monitoring, 4 testing, 4 pre-commit, 1 utility)
