# Agent Completion Report Template

Standard format for agent completion reports written to `.claude/agent-outputs/`.

## JSON Format

All agent completion reports should follow this structure:

```json
{
  "task_id": "YYYY-MM-DD-HHMMSS-agent-name",
  "agent": "agent_name",
  "status": "complete|in-progress|blocked|needs-review|failed",
  "started_at": "ISO-8601 timestamp",
  "completed_at": "ISO-8601 timestamp",
  "request": "Original user request",
  "deliverables": {
    "files_created": ["path/to/file1.py", "path/to/file2.py"],
    "files_modified": ["path/to/modified.py:42-68"],
    "files_deleted": ["path/to/removed.py"]
  },
  "metrics": {
    "lines_of_code": 150,
    "functions_created": 3,
    "classes_created": 1,
    "tests_written": 5,
    "documentation_added": true
  },
  "validation": {
    "validation_performed": true,
    "validation_passed": true,
    "checks": [
      {"name": "syntax", "passed": true},
      {"name": "type_check", "passed": true},
      {"name": "tests", "passed": true, "details": "5/5 tests passing"}
    ]
  },
  "artifacts": [
    "path/to/generated/file.py",
    "path/to/documentation.md"
  ],
  "notes": "Additional context or observations",
  "potential_gaps": [
    "Error handling for edge case X could be improved",
    "Performance testing needed for large datasets"
  ],
  "open_questions": [
    "Should we add caching for repeated calls?",
    "Does this need integration tests?"
  ],
  "next_agent": "agent-name|none",
  "handoff_context": {
    "for_next_agent": "Context needed by next agent in workflow"
  }
}
```

## Field Definitions

### Required Fields

- **task_id**: Unique identifier (format: `YYYY-MM-DD-HHMMSS-agent-name`)
- **agent**: Agent name that created this report
- **status**: Current task status (see Status Values below)
- **started_at**: ISO-8601 timestamp when task began
- **request**: Original user request or task description

### Status Values

| Status | Meaning | Next Action |
|---|---|---|
| `complete` | Task finished successfully | Continue workflow or return to user |
| `in-progress` | Task still running | Wait for completion |
| `blocked` | Cannot proceed without user input | Report to user, wait for input |
| `needs-review` | Work complete but needs approval | Report to user for review |
| `failed` | Task failed with error | Report error, potentially retry |

### Optional Fields

- **completed_at**: ISO-8601 timestamp when task finished
- **deliverables**: Dictionary of files created/modified/deleted
- **metrics**: Quantitative measurements of work done
- **validation**: Validation checks performed and results
- **artifacts**: List of key files produced (subset of deliverables)
- **notes**: Free-form observations or context
- **potential_gaps**: Known limitations or areas for improvement
- **open_questions**: Questions for user or next agent
- **next_agent**: Which agent should handle next phase (if any)
- **handoff_context**: Context needed by next agent in workflow

## Agent-Specific Conventions

### code_assistant

```json
{
  "agent": "code_assistant",
  "deliverables": {
    "files_created": ["src/module.py"],
    "files_modified": []
  },
  "metrics": {
    "lines_of_code": 120,
    "functions_created": 3,
    "classes_created": 1,
    "documentation_added": true
  }
}
```

### knowledge_researcher

```json
{
  "agent": "knowledge_researcher",
  "deliverables": {
    "files_created": [".coordination/research-report-TIMESTAMP.md"]
  },
  "metrics": {
    "sources_consulted": 15,
    "documentation_pages": 8
  },
  "artifacts": [".coordination/research-report-TIMESTAMP.md"],
  "next_agent": "code_assistant",
  "handoff_context": {
    "for_next_agent": "Research findings suggest using library X for implementation"
  }
}
```

### technical_writer

```json
{
  "agent": "technical_writer",
  "deliverables": {
    "files_created": ["docs/api/module.md", "docs/tutorials/getting-started.md"],
    "files_modified": ["docs/index.md"]
  },
  "metrics": {
    "documentation_pages": 3,
    "word_count": 2500,
    "examples_included": 8
  }
}
```

### code_reviewer

```json
{
  "agent": "code_reviewer",
  "deliverables": {
    "files_created": [".coordination/review-report-TIMESTAMP.md"]
  },
  "metrics": {
    "files_reviewed": 5,
    "issues_found": 12,
    "critical_issues": 2,
    "overall_grade": "B+"
  },
  "artifacts": [".coordination/review-report-TIMESTAMP.md"],
  "potential_gaps": [
    "Security vulnerability in authentication module (CRITICAL)",
    "Performance issue in data processing loop (MAJOR)"
  ],
  "next_agent": "code_assistant",
  "handoff_context": {
    "for_next_agent": "Fix critical security issue in src/auth.py:45"
  }
}
```

### git_commit_manager

```json
{
  "agent": "git_commit_manager",
  "deliverables": {
    "commits_created": ["abc123f", "def456a"],
    "pushed_to": "origin/main"
  },
  "metrics": {
    "commits": 2,
    "files_committed": 8,
    "push_successful": true
  }
}
```

### orchestrator

```json
{
  "agent": "orchestrator",
  "status": "complete",
  "routing_decision": {
    "user_intent": "Write authentication module",
    "complexity": "multi-agent",
    "agents_discovered": ["code_assistant", "code_reviewer", "technical_writer"],
    "agents_selected": ["code_assistant"],
    "keyword_matches": {
      "code_assistant": ["write", "module", "implement"]
    }
  },
  "workflow": {
    "phases": ["implement", "review", "document"],
    "current_phase": "complete",
    "execution_mode": "serial"
  },
  "progress": {
    "phases_completed": 3,
    "phases_total": 3,
    "context_used_percent": 45,
    "checkpoint_created": true
  }
}
```

## File Naming Convention

Completion reports should be named:

```
YYYY-MM-DD-HHMMSS-agent-name-status.json
```

**Examples**:

- `2026-01-16-143045-code-assistant-complete.json`
- `2026-01-16-143120-orchestrator-in-progress.json`
- `2026-01-16-143200-knowledge-researcher-complete.json`

## Storage Location

- **Active reports**: `.claude/agent-outputs/`
- **Archived reports**: `.claude/agent-outputs/archive/` (after retention period)
- **Summaries**: `.claude/summaries/` (lightweight summaries for context management)

## Retention Policy

See `.claude/config.yaml:retention.agent_outputs` (default: 7 days before archiving)

## Usage in Workflows

### Reading Completion Reports

```python
import json
from pathlib import Path

def read_completion_report(agent_id: str) -> dict:
    """Read agent completion report."""
    report_path = Path(f".claude/agent-outputs/{agent_id}-complete.json")
    with open(report_path) as f:
        return json.load(f)

# Check status before proceeding
report = read_completion_report("2026-01-16-143045-code-assistant")
if report["status"] == "complete" and report["validation"]["validation_passed"]:
    # Safe to proceed to next phase
    next_agent = report.get("next_agent")
```

### Writing Completion Reports

```python
import json
from datetime import datetime
from pathlib import Path

def write_completion_report(agent_name: str, data: dict) -> None:
    """Write agent completion report."""
    task_id = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-{agent_name}"
    data["task_id"] = task_id
    data["completed_at"] = datetime.now().isoformat()

    report_path = Path(f".claude/agent-outputs/{task_id}-{data['status']}.json")
    with open(report_path, "w") as f:
        json.dump(data, f, indent=2)
```

## Best Practices

1. **Always include task_id and agent** - Required for tracking and recovery
2. **Use descriptive status values** - Helps orchestrator make decisions
3. **List all artifacts** - Enables recovery after context compaction
4. **Include validation results** - Prevents propagating errors through workflow
5. **Note potential gaps** - Helps next agent or user address issues
6. **Provide handoff context** - Enables seamless workflow transitions
7. **Keep notes concise** - Detailed output goes in artifact files, not report
8. **Update status as task progresses** - Allows monitoring long-running tasks

## Related Documentation

- **Orchestrator Logic**: `.claude/agents/orchestrator.md` - How reports are used
- **Agent Registry**: See agent registry format in orchestrator documentation
- **Checkpointing**: `.claude/docs/checkpointing.md` - State management
- **Context Management**: `.claude/commands/context.md` - Optimization strategies

## Version

v1.0.0 (2026-01-16) - Initial completion report template
