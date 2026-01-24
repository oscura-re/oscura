# Agent Completion Report Template

Standard format for agent completion reports written to `.claude/agent-outputs/`.

**Version**: 2.0.0
**Last Updated**: 2026-01-22
**Breaking Changes**: Status values standardized to 5 values only

## Standard Status Values

ALL agents MUST use ONLY these 5 status values:

| Status | Meaning | Next Action |
|--------|---------|-------------|
| `complete` | Task finished successfully | Continue workflow or return to user |
| `in_progress` | Task currently executing | Wait for completion |
| `blocked` | Cannot proceed without user input | Report to user, wait for input |
| `needs_review` | Work complete but needs approval | Report to user for review |
| `failed` | Task failed with unrecoverable error | Report error, potentially retry |

**DEPRECATED** (do NOT use): `in-progress`, `needs-review`, or any other status values

## JSON Format

All agent completion reports MUST follow this structure:

```json
{
  "task_id": "YYYY-MM-DD-HHMMSS-agent-name",
  "agent": "agent_name",
  "status": "complete|in_progress|blocked|needs_review|failed",
  "started_at": "ISO-8601 timestamp",
  "completed_at": "ISO-8601 timestamp",
  "request": "Original user request",
  "artifacts": ["path/to/file1", "path/to/file2"],
  "metrics": {
    "agent_specific_metric": 0
  },
  "validation": {
    "validation_performed": true,
    "validation_passed": true,
    "checks": []
  },
  "notes": "Brief summary of what was accomplished",
  "next_agent": "agent_name|none",
  "handoff_context": {
    "for_next_agent": "Context needed by next agent"
  }
}
```

## Required Fields

Every completion report MUST include:

- **task_id**: Unique identifier (format: `YYYY-MM-DD-HHMMSS-agent-name`)
- **agent**: Agent name that created this report
- **status**: Current task status (ONLY use 5 standard values)
- **started_at**: ISO-8601 timestamp when task began
- **request**: Original user request or task description

## Optional Fields

Include as appropriate for the agent:

- **completed_at**: ISO-8601 timestamp when task finished
- **artifacts**: List of key files produced
- **metrics**: Quantitative measurements of work done
- **validation**: Validation checks performed and results
- **notes**: Free-form observations or context
- **next_agent**: Which agent should handle next phase (if any)
- **handoff_context**: Context needed by next agent in workflow

## Agent-Specific Examples

### code_assistant

```json
{
  "task_id": "2026-01-22-143045-code-assistant",
  "agent": "code_assistant",
  "status": "complete",
  "started_at": "2026-01-22T14:30:45Z",
  "completed_at": "2026-01-22T14:35:12Z",
  "request": "Write a function to parse CSV files",
  "artifacts": ["src/utils/csv_parser.py", "tests/test_csv_parser.py"],
  "metrics": {
    "lines_of_code": 120,
    "functions_created": 3,
    "tests_written": 5
  },
  "validation": {
    "validation_performed": true,
    "validation_passed": true,
    "checks": [
      {"name": "syntax", "passed": true},
      {"name": "tests", "passed": true, "details": "5/5 passing"}
    ]
  },
  "notes": "Created CSV parser with error handling and tests",
  "next_agent": "none"
}
```

### knowledge_researcher

```json
{
  "task_id": "2026-01-22-150000-research",
  "agent": "knowledge_researcher",
  "status": "complete",
  "started_at": "2026-01-22T15:00:00Z",
  "completed_at": "2026-01-22T15:45:30Z",
  "request": "Research Docker networking best practices",
  "topic": "Docker networking",
  "artifacts": ["docs/research/docker-networking.md"],
  "metrics": {
    "sources_consulted": 7,
    "sources_verified": 7,
    "citations_formatted": 7
  },
  "validation": {
    "validation_performed": true,
    "validation_passed": true,
    "checks": [
      {"name": "source_quality", "passed": true},
      {"name": "citations", "passed": true}
    ]
  },
  "notes": "Researched Docker networking, 7 authoritative sources",
  "next_agent": "technical_writer",
  "handoff_context": {
    "for_next_agent": "Research complete, ready for tutorial creation"
  }
}
```

### technical_writer

```json
{
  "task_id": "2026-01-22-160000-writing",
  "agent": "technical_writer",
  "status": "complete",
  "started_at": "2026-01-22T16:00:00Z",
  "completed_at": "2026-01-22T16:30:00Z",
  "request": "Create Docker networking tutorial",
  "artifacts": ["docs/tutorials/docker-networking.md"],
  "metrics": {
    "word_count": 2500,
    "sections": 8,
    "examples": 10
  },
  "validation": {
    "validation_performed": true,
    "validation_passed": true,
    "checks": [
      {"name": "formatting", "passed": true},
      {"name": "links", "passed": true}
    ]
  },
  "notes": "Created tutorial with 10 tested examples",
  "next_agent": "none"
}
```

### code_reviewer

```json
{
  "task_id": "2026-01-22-170000-review",
  "agent": "code_reviewer",
  "status": "complete",
  "started_at": "2026-01-22T17:00:00Z",
  "completed_at": "2026-01-22T17:20:00Z",
  "request": "Review authentication module",
  "artifacts": [".coordination/review-report-TIMESTAMP.md"],
  "metrics": {
    "files_reviewed": 5,
    "issues_found": 12,
    "critical_issues": 2,
    "overall_score": 7.5
  },
  "validation": {
    "validation_performed": true,
    "validation_passed": false,
    "checks": [
      {"name": "security", "passed": false, "details": "SQL injection found"}
    ]
  },
  "notes": "Found 2 critical security issues requiring immediate fix",
  "next_agent": "code_assistant",
  "handoff_context": {
    "for_next_agent": "Fix critical SQL injection in auth.py:45"
  }
}
```

### git_commit_manager

```json
{
  "task_id": "2026-01-22-180000-git-commit",
  "agent": "git_commit_manager",
  "status": "complete",
  "started_at": "2026-01-22T18:00:00Z",
  "completed_at": "2026-01-22T18:05:00Z",
  "request": "Commit changes to authentication module",
  "artifacts": [],
  "metrics": {
    "commits_created": 2,
    "files_committed": 8,
    "push_successful": true
  },
  "commit_details": [
    {
      "hash": "abc1234",
      "message": "feat(auth): add JWT authentication",
      "files": ["src/auth.py", "tests/test_auth.py"]
    },
    {
      "hash": "def5678",
      "message": "docs(auth): add authentication guide",
      "files": ["docs/guides/auth.md"]
    }
  ],
  "notes": "Created 2 atomic commits, pushed to origin/main",
  "next_agent": "none"
}
```

### orchestrator

```json
{
  "task_id": "2026-01-22-190000-orchestration",
  "agent": "orchestrator",
  "status": "complete",
  "started_at": "2026-01-22T19:00:00Z",
  "completed_at": "2026-01-22T19:45:00Z",
  "request": "Implement and document authentication feature",
  "routing_decision": {
    "user_intent": "Implement authentication with documentation",
    "complexity": "multi-agent",
    "agents_discovered": ["code_assistant", "code_reviewer", "technical_writer", "git_commit_manager"],
    "agents_selected": ["code_assistant", "code_reviewer", "technical_writer", "git_commit_manager"]
  },
  "workflow": {
    "phases": ["implement", "review", "document", "commit"],
    "execution_mode": "serial"
  },
  "progress": {
    "phases_completed": 4,
    "phases_total": 4
  },
  "artifacts": [
    "src/auth.py",
    "tests/test_auth.py",
    "docs/guides/authentication.md"
  ],
  "notes": "Completed full workflow: implementation → review → documentation → commit",
  "next_agent": "none"
}
```

## File Naming Convention

Completion reports should be named:

```
YYYY-MM-DD-HHMMSS-agent-name-status.json
```

**Examples**:

- `2026-01-22-143045-code-assistant-complete.json`
- `2026-01-22-143120-orchestrator-in_progress.json`
- `2026-01-22-143200-knowledge-researcher-complete.json`

## Storage Location

- **Active reports**: `.claude/agent-outputs/`
- **Archived reports**: `.claude/agent-outputs/archive/` (after retention period)
- **Summaries**: `.claude/summaries/` (lightweight summaries for context management)

## Retention Policy

See `.claude/config.yaml:retention.agent_outputs` (default: 7 days before archiving)

## Validation

Before writing a completion report, verify:

1. **Status value is valid**: Only use the 5 standard values
2. **Required fields present**: task_id, agent, status, started_at, request
3. **Timestamps are ISO-8601**: `YYYY-MM-DDTHH:MM:SSZ` format
4. **Artifacts list actual files**: Verify files exist on filesystem
5. **Metrics are accurate**: Count files, lines, etc. correctly

## Best Practices

1. **Use standard status values** - Only the 5 defined values, no variations
2. **Include task_id and agent** - Required for tracking and recovery
3. **List all artifacts** - Enables recovery after context compaction
4. **Include validation results** - Prevents propagating errors through workflow
5. **Provide handoff context** - Enables seamless workflow transitions
6. **Keep notes concise** - Detailed output goes in artifact files, not report
7. **Update status as task progresses** - Allows monitoring long-running tasks

## Related Documentation

- **Agent Template**: `.claude/templates/agent-definition.md` - Standard agent structure
- **Orchestrator**: `.claude/agents/orchestrator.md` - How reports are used
- **Context Management**: `.claude/docs/agent-context-best-practices.md` - Optimization
- **Configuration**: `.claude/config.yaml` - Retention and enforcement settings
