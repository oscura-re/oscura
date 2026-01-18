# Routing Concepts

Comprehensive explanation of how task routing works in the orchestration system.

## Overview

The orchestration system uses **keyword-based routing** to automatically select the best agent for each task. This document explains the routing mechanism, how to influence routing decisions, and how to troubleshoot routing issues.

## How Routing Works

### 1. Agent Discovery

The system dynamically discovers available agents at runtime:

```
1. Scan `.claude/agents/*.md` for agent files
2. Parse YAML frontmatter from each file
3. Extract `routing_keywords` list
4. Build routing table in memory
```

**Key principle**: No hardcoded routing tables. Adding a new agent file automatically makes it available for routing.

### 2. Keyword Matching Algorithm

When a user makes a request:

```
1. Parse user request → Extract keywords
2. For each discovered agent:
   - Calculate keyword overlap score
   - Weight by keyword specificity
3. Sort agents by score (highest first)
4. Select top-scoring agent
5. Route task to selected agent
```

**Scoring example**:

```
User request: "write a function to parse JSON"

Agent keyword matches:
- code_assistant: ["write", "function"] → Score: 8/10
- knowledge_researcher: ["parse"] → Score: 2/10
- technical_writer: [] → Score: 0/10

Selected: code_assistant (highest score)
```

### 3. Complexity Detection

For certain keywords, the orchestrator may detect that a task requires multiple agents:

**Single-agent indicators**:

- Specific, focused task
- Single domain (code, docs, research)
- Clear deliverable

**Multi-agent indicators**:

- "comprehensive", "full", "complete"
- "multiple perspectives"
- "from different angles"
- Cross-domain requirements

**Example**:

```bash
# Single-agent (routes to code_assistant)
/ai write a function to validate emails

# Multi-agent (creates workflow with code_assistant + code_reviewer + technical_writer)
/ai comprehensive email validation with tests and documentation
```

## Agent Frontmatter Format

Each agent file defines routing keywords in YAML frontmatter:

```yaml
---
name: agent_name
description: Brief description of agent purpose
routing_keywords:
  - keyword1
  - keyword2
  - keyword3
---
```

**Example** (code_assistant):

```yaml
---
name: code_assistant
description: Writes code for all implementation tasks
routing_keywords:
  - write
  - create
  - add
  - function
  - script
  - utility
  - helper
  - implement
  - build
  - develop
---
```

## Current Agents and Keywords

See `/agents --verbose` for complete list with keywords, or check individual agent files:

| Agent | Primary Keywords | Purpose |
|---|---|---|
| code_assistant | write, create, function, implement | Code implementation |
| knowledge_researcher | research, learn, investigate, compare | Web research |
| technical_writer | document, docs, guide, tutorial | Documentation creation |
| code_reviewer | review, audit, check, analyze | Code quality review |
| git_commit_manager | commit, git, push | Git operations |
| orchestrator | coordinate, workflow, multi-step | Multi-agent coordination |

## Influencing Routing

### Method 1: Phrase Your Request Clearly

Include agent-specific keywords:

```bash
# Routes to code_assistant
/ai write a helper function for authentication

# Routes to knowledge_researcher
/ai research authentication best practices

# Routes to technical_writer
/ai document the authentication module
```

### Method 2: Use Explicit Commands

Bypass routing intelligence with direct commands:

```bash
/research <topic>     # Always → knowledge_researcher
/review <path>        # Always → code_reviewer
/git <message>        # Always → git_commit_manager
```

### Method 3: Force Specific Agent with /route

Override orchestrator completely:

```bash
/route code_assistant "your task here"
/route knowledge_researcher "your task here"
```

**Warning**: `/route` bypasses safety checks and complexity detection. Use only when you're certain of the right agent.

## Common Routing Scenarios

### Scenario 1: Wrong Agent Selected

**Problem**: Orchestrator routes to wrong agent

**Solution**:

1. **Rephrase with explicit keywords**:

   ```bash
   # Instead of: "make authentication"
   /ai write authentication function  # → code_assistant
   ```

2. **Use explicit command**:

   ```bash
   /code write authentication function  # Force ad-hoc code
   ```

3. **Force route**:

   ```bash
   /route code_assistant "write authentication function"
   ```

### Scenario 2: Ambiguous Request

**Problem**: Request could match multiple agents

**Example**:

```bash
/ai authentication
```

Could mean:

- Write auth code → code_assistant
- Research auth patterns → knowledge_researcher
- Document auth → technical_writer

**Solution**: Be specific with keywords:

```bash
/ai write authentication code        # → code_assistant
/ai research authentication methods  # → knowledge_researcher
/ai document authentication flow     # → technical_writer
```

### Scenario 3: Complex Multi-Agent Task

**Problem**: Task needs multiple agents but routes to single agent

**Example**:

```bash
/ai implement authentication
# Routes to code_assistant (single agent)
# But you want code + tests + docs
```

**Solution**: Use multi-agent keywords:

```bash
/ai comprehensive authentication implementation
# Routes to orchestrator → creates workflow:
#   1. code_assistant (implementation)
#   2. code_reviewer (review)
#   3. technical_writer (documentation)
```

Or use `/swarm` for parallel execution:

```bash
/swarm complete authentication with code, tests, and docs
```

## Routing Decision Visibility

### Method 1: Check Orchestrator Output

When orchestrator routes a task, it shows:

```
Routing Decision:
  Task: "write a function"
  Complexity: Single-agent (low)
  Matched Agent: code_assistant
  Keywords: ["write", "function"]
  Score: 8/10
```

### Method 2: Enable Logging

Set in `.claude/config.yaml`:

```yaml
orchestration:
  workflow:
    show_routing_decisions: true
```

The orchestrator will log routing decisions to `.claude/orchestration.log`:

```
[2026-01-16 14:30:45] ROUTE | Complexity: 25 | Path: SINGLE | Agent: code_assistant | Keywords: write,function
```

### Method 3: Use /agents Command

See which keywords would match:

```bash
/agents --verbose     # Show all agents with keywords
/agents write         # Search for agents matching "write"
```

## Troubleshooting Routing Issues

### Issue 1: Agent Not Found

**Symptom**: "No agent found for task"

**Cause**: No agent keywords match request

**Solution**:

1. Check available agents: `/agents`
2. Rephrase with explicit keywords
3. Use direct command (e.g., `/code`, `/research`)

### Issue 2: Wrong Agent Selected Repeatedly

**Symptom**: Orchestrator consistently routes to wrong agent

**Possible causes**:

1. Agent keywords too broad
2. Request phrasing ambiguous
3. Keyword overlap between agents

**Solution**:

1. Use explicit commands (`/code`, `/research`, etc.)
2. Use `/route` to force correct agent
3. Improve request phrasing with specific keywords
4. Check agent keywords: `/agents --verbose`

### Issue 3: Task Incorrectly Detected as Complex

**Symptom**: Simple task routes to orchestrator workflow

**Cause**: Request contains multi-agent keywords ("comprehensive", "complete", "all aspects")

**Solution**:

1. Remove multi-agent keywords from request
2. Be specific about single deliverable
3. Use direct command for simple task

**Example**:

```bash
# Detected as complex (avoid)
/ai complete authentication solution

# Detected as simple (better)
/ai write authentication function
```

## Best Practices

### 1. Start with Clear Verbs

Use agent-specific action verbs:

- **Code**: write, create, implement, build, add
- **Research**: research, learn, investigate, compare, study
- **Documentation**: document, explain, describe, guide
- **Review**: review, audit, check, analyze

### 2. Include Domain Keywords

Specify what you're working with:

```bash
/ai write Python function      # "Python" + "function" → code_assistant
/ai research React patterns    # "React" → knowledge_researcher
/ai document API endpoints     # "API" + "document" → technical_writer
```

### 3. Use Explicit Commands When Certain

If you know which agent you want, use direct commands:

```bash
/code <task>         # Direct to code_assistant
/research <topic>    # Direct to knowledge_researcher
/review <path>       # Direct to code_reviewer
/git <message>       # Direct to git_commit_manager
```

### 4. Let /ai Decide for Unknown Tasks

If unsure which agent to use, let orchestrator decide:

```bash
/ai <description of what you want>
```

The keyword matching will find the best agent.

### 5. Check Routing Before Large Tasks

For important work, verify routing will be correct:

```bash
/agents <keyword>    # Check which agents match
```

## Configuration

Routing behavior is controlled in `.claude/config.yaml`:

```yaml
orchestration:
  workflow:
    allow_command_overrides: true    # Allow /route to override
    show_routing_decisions: true     # Show routing in output

  agent_registry:
    enabled: true                    # Track agent execution
    auto_persist: true               # Save registry on launch
```

## Related Documentation

- **Agent Listing**: See `/agents` or `.claude/commands/agents.md`
- **Force Routing**: See `/route` or `.claude/commands/route.md`
- **Orchestrator Logic**: See `.claude/agents/orchestrator.md`
- **Command Reference**: See `/help` or `.claude/commands/help.md`

## Version

v1.0.0 (2026-01-16) - Initial routing concepts documentation
