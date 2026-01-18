---
name: status
description: Show orchestration health, running agents, and context usage
arguments: [--json]
---

# Status Command

Display comprehensive orchestration system health, running agents, and resource usage.

## Usage

```bash
/status            # Human-readable dashboard
/status --json     # JSON output for scripting
```

## Examples

```bash
/status            # Show system health dashboard
/status --json     # Get JSON status for automation
```

## Information Displayed

### Agent Status

- Currently running agents (from `.claude/agent-registry.json`)
- Recently completed agents (last 10)
- Failed agents requiring attention
- Agent resource usage and runtime

### Context Usage

- Current context usage percentage
- Tokens used vs. available
- Warning threshold status (see `config.yaml:orchestration.context.*` for thresholds)
- Recommendations for optimization

### System Health

- Coordination directory health (`.coordination/`)
- Agent output directory status (`.claude/agent-outputs/`)
- Checkpoint availability
- Stale agent detection
- Lock status (active locks in `.coordination/locks/`)

### Resource Metrics

- Number of completion reports in last 7 days
- Agent output file sizes
- Checkpoint disk usage
- Archive status and cleanup recommendations

## Health Indicators

**Healthy** 🟢

- Context usage below warning threshold (see `config.yaml:orchestration.context.warning_threshold`)
- No stale agents
- No failed agents
- Checkpoints available

**Warning** 🟡

- Context usage between warning and critical thresholds (see `config.yaml:orchestration.context.*`)
- Agents running for >30 minutes
- Old completion reports (see `config.yaml:retention.agent_outputs`)
- Archive cleanup recommended

**Critical** 🔴

- Context usage above critical threshold (see `config.yaml:orchestration.context.critical_threshold`)
- Exceeding max concurrent agents (see `config.yaml:orchestration.agents.max_concurrent`)
- Multiple failed agents
- Disk space issues

## Sample Output

```
Orchestration System Status
============================

Agent Status:
  Running:    2 agents (orchestrator, code_assistant)
  Completed:  15 agents (last 24h)
  Failed:     0 agents

Context Usage:
  Current:    45% (90,000 / 200,000 tokens)
  Status:     🟢 Healthy
  Threshold:  Next warning at 60%

System Health:
  Agent Registry:     ✓ Healthy
  Coordination:       ✓ Healthy
  Checkpoints:        3 available
  Stale Agents:       0 detected
  Active Locks:       0

Recommendations:
  • No action required - system operating normally
```

## JSON Output Format

```json
{
  "timestamp": "2025-01-09T13:00:00Z",
  "health": "healthy|warning|critical",
  "agents": {
    "running": 2,
    "completed_24h": 15,
    "failed": 0,
    "running_list": [
      { "id": "abc123", "name": "orchestrator", "runtime": "5m30s" },
      { "id": "def456", "name": "code_assistant", "runtime": "2m15s" }
    ]
  },
  "context": {
    "usage_percent": 45,
    "tokens_used": 90000,
    "tokens_available": 200000,
    "status": "healthy",
    "next_threshold": "60%"
  },
  "system": {
    "agent_registry": "healthy",
    "coordination": "healthy",
    "checkpoints": 3,
    "stale_agents": 0,
    "active_locks": 0
  },
  "recommendations": []
}
```

## Implementation

This command reads:

- `.claude/agent-registry.json` - Agent state
- `.coordination/active_work.json` - Current work
- `.coordination/completed.jsonl` - Completion history
- `.coordination/locks/` - Active coordination locks
- `.claude/agent-outputs/` - Recent outputs

## See Also

- `/context` - Detailed context optimization advice
- `/cleanup` - Run maintenance tasks
- `.claude/hooks/check_context_usage.sh` - Context monitoring hook
- `.claude/config.yaml` - Threshold configuration
