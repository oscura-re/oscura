# Cross-Reference Map

## Overview

This document maps relationships between commands, agents, and documentation for easy navigation.

## Command → Agent Mapping

| Command | Target Agent | Documentation |
|---------|--------------|---------------|
| `/help` | none (info) | `.claude/commands/help.md` |
| `/status` | none (info) | `.claude/commands/status.md` |
| `/context` | none (info) | `.claude/commands/context.md` |
| `/cleanup` | none (maintenance) | `.claude/commands/cleanup.md` |
| `/agents` | none (info) | `.claude/commands/agents.md` |
| `/git` | git_commit_manager | `.claude/commands/git.md` |
| `/research` | knowledge_researcher | `.claude/commands/research.md` |
| `/review` | code_reviewer | `.claude/commands/review.md` |
| `/route` | (user-specified) | `.claude/commands/route.md` |
| `/swarm` | orchestrator (parallel) | `.claude/commands/swarm.md` |

## Agent → Related Commands

| Agent | Primary Commands | Documentation |
|-------|-----------------|---------------|
| orchestrator | `/swarm` | `.claude/agents/orchestrator.md` |
| code_assistant | (via routing) | `.claude/agents/code_assistant.md` |
| technical_writer | (via routing) | `.claude/agents/technical_writer.md` |
| knowledge_researcher | `/research` | `.claude/agents/knowledge_researcher.md` |
| code_reviewer | `/review` | `.claude/agents/code_reviewer.md` |
| git_commit_manager | `/git` | `.claude/agents/git_commit_manager.md` |

## Documentation → Related Resources

### Core Concepts

| Document | Related Docs | Related Commands |
|----------|--------------|------------------|
| `routing-concepts.md` | `orchestrator.md`, `agent-context-best-practices.md` | `/route`, `/agents` |
| `orchestration-logging.md` | `orchestrator.md`, `config.yaml` | `/status` |
| `agent-context-best-practices.md` | `technical_writer.md`, `knowledge_researcher.md` | `/context` |
| `fuzzy-routing.md` | `routing-concepts.md`, `routing.py` | `/route`, `/agents` |

### Implementation Files

| File | Purpose | Related Docs |
|------|---------|--------------|
| `.claude/config.yaml` | System configuration | `orchestration-logging.md`, `routing-concepts.md` |
| `.claude/hooks/shared/routing.py` | Fuzzy routing implementation | `fuzzy-routing.md` |
| `.claude/hooks/shared/logging_utils.py` | Logging utilities | `orchestration-logging.md` |
| `.claude/agent-outputs/*.json` | Active agent tracking | `orchestrator.md` |

## Workflow Patterns

### Research Workflow

```markdown
User Request → knowledge_researcher → technical_writer
                     ↓                       ↓
            research-complete.json   writing-complete.json
```markdown

**Related**:
- Commands: `/research`
- Agents: `knowledge_researcher.md`, `technical_writer.md`
- Docs: `routing-concepts.md`

### Code Quality Workflow

```markdown
User Request → code_assistant → code_reviewer → git_commit_manager
                    ↓                ↓                 ↓
            code-complete.json  review-complete.json  commit
```markdown

**Related**:
- Commands: `/review`, `/git`
- Agents: `code_assistant.md`, `code_reviewer.md`, `git_commit_manager.md`
- Docs: `routing-concepts.md`

### Parallel Workflow

```markdown
User Request → orchestrator (swarm) → [agent1, agent2, agent3]
                      ↓                       ↓
            orchestration-complete.json   summary files
```markdown

**Related**:
- Commands: `/swarm`
- Agents: `orchestrator.md`
- Docs: `orchestration-logging.md`, `config.yaml`

## Configuration References

### Context Management

- **Primary Config**: `.claude/config.yaml:orchestration.context`
- **Related Docs**: `agent-context-best-practices.md`, `orchestration-logging.md`
- **Related Commands**: `/context`, `/status`
- **Thresholds**: warning (60%), checkpoint (65%), critical (75%)

### Agent Limits

- **Primary Config**: `.claude/config.yaml:orchestration.agents`
- **Related Docs**: `orchestrator.md`
- **Related Commands**: `/status`, `/swarm`
- **Enforcement**: `enforce_agent_limit.py` hook

### Logging

- **Primary Config**: `.claude/config.yaml:logging`
- **Related Docs**: `orchestration-logging.md`
- **Related Files**: `logging_utils.py`
- **Log Files**: `.claude/hooks/orchestration.log (file created at runtime)`, `.claude/hooks/*.log`

### Retention Policies

- **Primary Config**: `.claude/config.yaml:retention`
- **Related Docs**: N/A
- **Related Commands**: `/cleanup`
- **Cleanup Scripts**: `session_cleanup.sh`, `cleanup_stale_agents.py`

## Quick Navigation

### By Task Type

**Writing Documentation**:
- Agent: `technical_writer.md`
- Context: `agent-context-best-practices.md`

**Research & Validation**:
- Agent: `knowledge_researcher.md`
- Command: `/research`
- Context: `agent-context-best-practices.md`

**Code Review**:
- Agent: `code_reviewer.md`
- Command: `/review`

**Git Operations**:
- Agent: `git_commit_manager.md`
- Command: `/git`

**Routing & Orchestration**:
- Agent: `orchestrator.md`
- Docs: `routing-concepts.md`, `fuzzy-routing.md`
- Commands: `/route`, `/swarm`, `/agents`

**System Monitoring**:
- Commands: `/status`, `/context`, `/cleanup`
- Docs: `orchestration-logging.md`
- Config: `config.yaml:logging`, `config.yaml:orchestration`

### By User Intent

**"I want to understand how routing works"**:
- Start: `routing-concepts.md`
- See also: `orchestrator.md`, `fuzzy-routing.md`
- Try: `/agents`, `/route`

**"I want to write documentation"**:
- Start: `technical_writer.md`
- See also: `agent-context-best-practices.md`
- Try: Natural language request or `/route technical_writer "write docs"`

**"I want to check system health"**:
- Start: `/status`
- See also: `/context`, `/cleanup`
- Config: `config.yaml:orchestration`

**"I want to run multiple agents in parallel"**:
- Start: `/swarm`
- See also: `orchestrator.md`, `orchestration-logging.md`
- Config: `config.yaml:orchestration.agents`

## Index of All Documentation

### Commands (`.claude/commands/`)

- `agents.md` - List available agents
- `cleanup.md` - System maintenance
- `context.md` - Context monitoring
- `git.md` - Git operations
- `help.md` - Command help
- `research.md` - Web research
- `review.md` - Code review
- `route.md` - Manual routing
- `status.md` - System status
- `swarm.md` - Parallel coordination

### Agents (`.claude/agents/`)

- `orchestrator.md` - Central coordinator
- `code_assistant.md` - Code implementation
- `technical_writer.md` - Documentation
- `knowledge_researcher.md` - Research & validation
- `code_reviewer.md` - Quality review
- `git_commit_manager.md` - Git operations

### Documentation (`.claude/docs/`)

- `routing-concepts.md` - Routing explanation
- `orchestration-logging.md` - Logging details
- `agent-context-best-practices.md` - Context management
- `fuzzy-routing.md` - Fuzzy matching
- `cross-reference-map.md` - This file

### Configuration

- `.claude/config.yaml` - System configuration (SSOT)
- `.claude/paths.yaml` - Path definitions
- `.claude/project-metadata.yaml` - Project identity
- `.claude/coding-standards.yaml` - Code quality rules

### Templates

- `.claude/templates/command-definition.md` - Command template
