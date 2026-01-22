# Getting Started with Claude Code Orchestration

Welcome to the Claude Code orchestration system! This guide explains how to use the intelligent routing system that automatically selects the right agent for your task.

**Version**: 4.0.0 (2026-01-22) - Accurate system description

## Quick Start

### Your First Task

**Not sure what to do? Just describe your task** - the system will figure out the rest:

```bash
# Write code (routes to code_assistant)
"Write a function to calculate factorial"

# Research something (routes to knowledge_researcher)
"Research Docker networking best practices"

# Create documentation (routes to technical_writer)
"Document the API endpoints"

# Review code (routes to code_reviewer)
"Review src/auth/jwt.py for security issues"
```

### Check System Status

```bash
/status              # System health
/agents              # List available agents
/help                # Show all commands
```

---

## How It Works

The orchestration system uses **intelligent keyword-based routing** to automatically select the best agent for your task:

```
User describes task in natural language
  ↓
Orchestrator analyzes keywords and context
  ↓
Fuzzy matching selects best agent (RapidFuzz)
  ↓
Agent executes task
  ↓
Returns results to user
```

### Routing Intelligence

The orchestrator:

1. **Discovers available agents** dynamically from `.claude/agents/*.md`
2. **Extracts routing keywords** from agent frontmatter
3. **Scores each agent** based on keyword overlap with user request
4. **Routes to highest-scoring agent** automatically

For complete routing details, see `.claude/docs/routing-concepts.md` and `.claude/docs/fuzzy-routing.md`.

---

## Available Agents

The system includes **6 specialized agents**:

| Agent | Purpose | Triggers |
|-------|---------|----------|
| **orchestrator** | Task routing and workflow coordination | Automatic (routes all tasks) |
| **code_assistant** | Write code for all implementation tasks | "write", "create", "implement", "function" |
| **knowledge_researcher** | Web research with citations | "research", "investigate", "validate" |
| **technical_writer** | Create documentation and guides | "document", "write guide", "tutorial" |
| **code_reviewer** | Code quality and security reviews | "review", "audit", "quality" |
| **git_commit_manager** | Git operations and commits | "commit", "/git" |

For detailed agent documentation, see `.claude/agents/*.md`

To list all agents with their keywords: `/agents --verbose`

---

## Essential Commands

### Core Commands

| Command | Purpose | Example |
|---------|---------|---------|
| Natural language | Describe task, system routes automatically | "Implement authentication" |
| `/help` | Show all commands | `/help` |
| `/status [--json]` | System health check | `/status` |
| `/agents [keyword]` | List available agents | `/agents` |
| `/context` | Show context usage | `/context` |
| `/cleanup [--dry-run]` | Run maintenance tasks | `/cleanup` |

### Specialized Commands

| Command | Purpose | Agent |
|---------|---------|-------|
| `/git [message]` | Create atomic commits | git_commit_manager |
| `/swarm <task>` | Parallel multi-agent workflows | orchestrator |
| `/route <agent> <task>` | Force route to specific agent | (bypass routing) |
| `/research <topic>` | Conduct comprehensive research | knowledge_researcher |
| `/review [path]` | Code quality review | code_reviewer |

For complete command reference, see `.claude/commands/*.md`

---

## Common Workflows

### 1. Research → Document → Implement

```bash
/research "GraphQL best practices 2026"
# Review research findings

/ai document "Create GraphQL implementation guide"
# Documentation created

/ai implement "Add GraphQL endpoint for users"
# Implementation complete
```

### 2. Implement → Review → Commit

```bash
/ai "Implement user authentication with JWT"
# Code implementation complete

/review src/auth/
# Review findings: 2 critical, 3 high priority

# Fix critical issues

/git "feat: add JWT authentication"
# Committed with conventional format
```

### 3. Parallel Research

```bash
/swarm research authentication approaches: OAuth, JWT, session-based
# Launches 3 parallel research tasks
# Synthesizes results
```

### 4. Force Specific Agent

```bash
# Let orchestrator decide
"Write a caching utility"

# OR force specific agent
/route code_assistant "Write a caching utility"
```

---

## Routing Examples

The orchestrator intelligently routes based on keywords:

### Code Implementation

```bash
"Write a function to validate emails"           → code_assistant
"Implement user registration endpoint"          → code_assistant
"Create a helper script for migrations"         → code_assistant
```

### Research & Learning

```bash
"Research async patterns in Python"             → knowledge_researcher
"Investigate Docker networking best practices"  → knowledge_researcher
"Validate JWT security approaches"              → knowledge_researcher
```

### Documentation

```bash
"Document the REST API endpoints"               → technical_writer
"Write a tutorial for new contributors"         → technical_writer
"Create architecture documentation"             → technical_writer
```

### Code Review

```bash
"Review src/auth/ for security issues"          → code_reviewer
"Audit code quality in src/analyzers/"          → code_reviewer
"Check for performance bottlenecks"             → code_reviewer
```

### Git Operations

```bash
/git "feat: add user authentication"            → git_commit_manager
/git "fix: resolve validation bug"              → git_commit_manager
```

---

## Fuzzy Routing

The system uses **fuzzy keyword matching** (RapidFuzz) to handle variations:

```bash
"write"     → matches "write", "writer", "writing"
"research"  → matches "research", "researcher", "investigate"
"document"  → matches "document", "documentation", "docs"
```

**Similarity threshold**: 80% by default (configurable in `.claude/config.yaml`)

For complete fuzzy routing details, see `.claude/docs/fuzzy-routing.md`.

---

## Configuration

### Main Config: `.claude/config.yaml`

Controls routing behavior:

```yaml
orchestration:
  default_agent: orchestrator         # Routes all tasks

  routing:
    fuzzy_matching:
      enabled: true                   # Enable fuzzy keyword matching
      similarity_threshold: 80        # Match threshold (0-100)
      algorithm: token_set_ratio      # RapidFuzz algorithm
    confidence_threshold: 0.7         # Min confidence to auto-route
    show_routing_decisions: false     # Log routing decisions

  agents:
    max_concurrent: 2                 # Max simultaneous agents
    max_batch_size: 2                 # Max agents per batch
```

### Agent-Specific Settings

See `.claude/config.yaml:orchestration.agents.*` for agent-specific configuration.

---

## Best Practices

### ✅ DO

- **Start with natural language** - Describe your task and let the system route intelligently
- **Be specific with keywords** - Use clear, descriptive language
- **Use specialized commands** - `/research`, `/review`, `/git` for common tasks
- **Check `/status`** before big tasks
- **Review routing decisions** - Use `/agents` to understand routing

### ❌ DON'T

- **Override routing unnecessarily** - Trust the fuzzy matching intelligence
- **Use vague descriptions** - "do something" won't route well
- **Force routing with `/route` unless needed** - Automatic routing is usually best
- **Ignore agent suggestions** - If system recommends different agent, consider why

---

## Troubleshooting

### "No matching agent found"

**Issue**: Task description didn't match any agent keywords
**Solution**: Be more specific, use recognized keywords, or try `/agents` to see available agents

### "Multiple agents matched"

**Issue**: Task description matches multiple agents equally
**Solution**: Be more specific about intent (e.g., "research Docker" vs "implement Docker client")

### "Agent not responding"

**Issue**: Agent seems stuck
**Solution**: Check `/status`, review `.claude/agent-outputs/` for errors

### Wrong Agent Selected

**Issue**: Orchestrator routed to wrong agent
**Solution**: Use `/route <agent> <task>` to force correct agent, or rephrase with better keywords

---

## Understanding Routing Decisions

To see why the orchestrator chose a specific agent:

1. Enable routing decision logging in `.claude/config.yaml`:

   ```yaml
   orchestration:
     routing:
       show_routing_decisions: true
   ```

2. Review structured logs (if configured):

   ```bash
   cat .claude/logs/routing-decisions.jsonl
   ```

3. Use `/agents` to see keyword mappings:

   ```bash
   /agents --verbose
   ```

For complete routing explanation, see `.claude/docs/routing-concepts.md`.

---

## Advanced Features

### Structured Logging

Enable JSON logging for machine-readable output:

```yaml
orchestration:
  logging:
    structured: true
    output_path: .claude/logs/
    retention_days: 30
```

See `.claude/docs/orchestration-logging.md` for complete logging documentation.

### Swarm Workflows

Execute complex tasks with parallel agent coordination:

```bash
/swarm research authentication: OAuth2, JWT, session-based
/swarm implement feature: backend, frontend, tests
```

See `.claude/commands/swarm.md` for swarm workflow details.

### Manual Routing

Force routing to specific agent bypassing orchestrator:

```bash
/route code_assistant "write a function"
/route knowledge_researcher "research Docker"
```

See `.claude/commands/route.md` for manual routing guide.

---

## Next Steps

1. **Try the examples** above with natural language
2. **Explore available agents** with `/agents --verbose`
3. **Read agent documentation** in `.claude/agents/`
4. **Review command reference** in `.claude/commands/`
5. **Check project workflow** in `CLAUDE.md`
6. **Understand routing** in `.claude/docs/routing-concepts.md`

---

## File Locations

| Documentation Type | Location |
|-------------------|----------|
| Agent definitions | `.claude/agents/*.md` |
| Command reference | `.claude/commands/*.md` |
| Routing concepts | `.claude/docs/routing-concepts.md` |
| Fuzzy routing | `.claude/docs/fuzzy-routing.md` |
| Templates | `.claude/templates/*.md` |
| Configuration | `.claude/config.yaml` |
| Project workflow | `CLAUDE.md` |
| Contributing | `CONTRIBUTING.md` |

---

## Version History

- **v4.0.0** (2026-01-22): Complete rewrite to reflect actual system (6 agents, fuzzy routing, no spec system)
- **v3.1.0** (2026-01-16): Optimized and streamlined (OBSOLETE - described non-existent system)
- **v3.0.0** (2026-01-09): Complete three-path workflow system (REMOVED)
- **v2.0.0**: Added auto-spec workflow (REMOVED)
- **v1.0.0**: Initial orchestration system

---

For detailed project development instructions, see `CLAUDE.md` in the repository root.
