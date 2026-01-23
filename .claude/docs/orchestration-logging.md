# Orchestration Activity Logging

## Overview

Lightweight debugging and tracking of orchestration decisions without consuming context.

**Log file**: `.claude/hooks/orchestration.log (file created at runtime)` (git-ignored, auto-rotated)

## Purpose

- Debug routing issues without verbose context
- Track system behavior over time
- Identify patterns in complexity detection
- Quick post-mortem analysis

## What to Log

### Essential (always log)

- Routing decisions with complexity scores
- Agent selections and workflow paths
- Errors and failures
- Agent completions with duration

### Optional (only if useful)

- Keyword matches
- Disambiguation reasoning
- Context usage warnings

## Log Format

Simple text format (not verbose JSON):

````python
[2026-01-09 14:30:45] ROUTE | Complexity: 25 | Path: AD_HOC | Agent: code_assistant
[2026-01-09 14:31:12] ROUTE | Complexity: 65 | Path: AUTO_SPEC | Agent: orchestrator→spec_implementer
[2026-01-09 14:32:00] ERROR | Agent: code_assistant | Message: File not found
[2026-01-09 14:33:15] COMPLETE | Agent: code_assistant | Duration: 45s | Status: success
```python

## Implementation Example

```python
from datetime import datetime
from pathlib import Path

def log_orchestration(event_type: str, **data) -> None:
    """Log orchestration event to .claude/hooks/orchestration.log (file created at runtime)"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {event_type} | {' | '.join(f'{k}: {v}' for k, v in data.items())}\n"

    log_file = Path('.claude/hooks/orchestration.log (file created at runtime)')
    with open(log_file, 'a') as f:
        f.write(log_line)

# Usage examples
log_orchestration('ROUTE', Complexity=score, Path=path_name, Agent=agent_name)
log_orchestration('ERROR', Agent=agent_name, Message=error_msg)
log_orchestration('COMPLETE', Agent=agent_name, Duration=duration, Status='success')
```markdown

## Retention

- Retention: See `config.yaml:retention` section (retention policies for logs)
- Auto-cleanup: Weekly via retention policies
- Log rotation: Automatic when size exceeds 10MB

## Benefits

- Debug routing issues without verbose context
- Track system behavior over time
- Identify patterns in complexity detection
- Quick post-mortem analysis

## What NOT to Log

- Full tool outputs (too verbose)
- File contents (use agent outputs for that)
- User prompts (already in conversation)
- Internal Claude reasoning

## Configuration

See `.claude/config.yaml`:

```yaml
retention:
  orchestration_log_days: 14  # Keep orchestration logs for 14 days

logging:
  enabled: true
  files:
    orchestration: ".claude/hooks/orchestration.log (file created at runtime)"
```markdown

## See Also

- `.claude/hooks/shared/logging_utils.py` - Shared logging utilities
- `.claude/config.yaml` - Configuration settings
- `.claude/agents/orchestrator.md` - Orchestrator agent documentation
````
