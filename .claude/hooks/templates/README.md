# Hook Development Guide

Templates and guidelines for creating new Claude Code hooks.

## Templates

### Python Validation Hook

Use `python_validation_hook.py.template` for hooks that:

- Validate configuration files
- Check code patterns
- Enforce policies

```bash
cp .claude/hooks/templates/python_validation_hook.py.template \
   .claude/hooks/my_validation_hook.py

# Edit the template placeholders:
# {{HOOK_NAME}} - e.g., "check_imports"
# {{HOOK_DESCRIPTION}} - One-line description
# {{HOOK_PURPOSE}} - Detailed explanation
# {{DATE}} - Creation date
```

### Bash Cleanup Hook

Use `bash_cleanup_hook.sh.template` for hooks that:

- Archive old files
- Clean up temporary data
- Manage file lifecycle

```bash
cp .claude/hooks/templates/bash_cleanup_hook.sh.template \
   .claude/hooks/my_cleanup_hook.sh

chmod +x .claude/hooks/my_cleanup_hook.sh
```

## Hook Requirements

### Mandatory

1. **Absolute path resolution** - Always resolve paths from script location:

   ```python
   SCRIPT_DIR = Path(__file__).parent.resolve()
   REPO_ROOT = SCRIPT_DIR.parent.parent
   ```

   ```bash
   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
   ```

2. **Error handling** - Wrap main logic in try-except/set -e:

   ```python
   try:
       result = validate()
   except Exception as e:
       logger.exception("Hook failed")
       sys.exit(0)  # Fail open
   ```

3. **Logging** - Log to `.claude/hooks/hook.log`:

   ```python
   logging.basicConfig(
       handlers=[logging.FileHandler(LOG_FILE, mode="a")]
   )
   ```

4. **JSON output** - Return structured JSON for programmatic use:

   ```json
   { "ok": true, "message": "Validation passed" }
   ```

### Recommended

1. **Emergency bypass** - Check `CLAUDE_BYPASS_HOOKS=1`
2. **Dry-run mode** - Support `--dry-run` flag
3. **Verbose mode** - Support `--verbose` flag
4. **Dependency checking** - Verify required tools exist

## Testing Hooks

### Unit Testing

```bash
# Test hook directly
.claude/hooks/my_hook.sh
echo "Exit code: $?"

# Test with dry-run
.claude/hooks/my_hook.sh --dry-run

# Test Python hooks
uv run python .claude/hooks/my_hook.py --verbose
```

### Edge Case Testing

```bash
# Test with missing dependencies
PATH="" .claude/hooks/my_hook.sh

# Test with corrupted input
echo "invalid" > /tmp/test.json
uv run python .claude/hooks/my_hook.py /tmp/test.json

# Test with empty files
touch /tmp/empty.yaml
uv run python .claude/hooks/my_hook.py /tmp/empty.yaml
```

### Integration Testing

```bash
# Run stress test suite
uv run python .claude/hooks/stress_test_suite.py

# Check results
cat .claude/stress_test_report.md
```

## Hook Registration

### In settings.json

Register hooks in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/my_hook.py\"",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

### Hook Types

| Hook Type      | When It Runs                    |
| -------------- | ------------------------------- |
| `PreToolUse`   | Before tool execution           |
| `PostToolUse`  | After tool execution            |
| `Stop`         | When agent stops                |
| `SubagentStop` | When subagent stops             |
| `PreCompact`   | Before context compaction       |
| `SessionStart` | On session start (with matcher) |
| `SessionEnd`   | On session end                  |

## Best Practices

### DO

- Use absolute paths
- Handle missing files gracefully
- Log all actions
- Provide structured output
- Support dry-run mode
- Document dependencies

### DON'T

- Use relative paths
- Crash on invalid input
- Print to stdout without structure
- Block on non-critical failures
- Skip logging
- Assume dependencies exist

## Troubleshooting

### Hook Not Running

1. Check permissions: `chmod +x .claude/hooks/my_hook.sh`
2. Verify registration in `settings.json`
3. Check error log: `tail .claude/hooks/errors.log`

### Hook Failing

1. Check hook log: `tail .claude/hooks/hook.log`
2. Test manually: `.claude/hooks/my_hook.sh`
3. Check dependencies: `.claude/hooks/check_dependencies.sh`

### Hook Blocking Session

Use emergency bypass:

```bash
export CLAUDE_BYPASS_HOOKS=1
# Do work
unset CLAUDE_BYPASS_HOOKS
```

Then fix the hook immediately.
