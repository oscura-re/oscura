---
name: agents
description: List available agents with their capabilities and routing keywords
arguments: [keyword, --verbose]
---

# /agents - List Available Agents

Show all available agents with their capabilities, keywords, and use cases.

## Usage

```bash
/agents               # List all agents
/agents code          # Search agents by keyword
/agents --verbose     # Show routing keywords
```markdown

## Purpose

Help users understand:

- What agents are available
- When to use each agent
- What keywords trigger which agent
- How routing works

## Output Format

### Default View

```bash
Available Agents (6):

1. orchestrator (opus)
   → Routes tasks to specialists and coordinates multi-agent workflows
   Use: Complex multi-step tasks, workflow coordination

2. code_assistant (sonnet)
   → Writes code for all implementation tasks
   Use: Functions, utilities, features, prototypes, bug fixes

3. knowledge_researcher (opus)
   → Web research with citations and comprehensive analysis
   Use: Learning new technologies, best practices, comparisons

4. technical_writer (sonnet)
   → Creates documentation, tutorials, and guides
   Use: API docs, user guides, architecture documentation

5. code_reviewer (sonnet)
   → Code quality audits, security reviews, best practices
   Use: Pre-commit reviews, security audits, quality checks

6. git_commit_manager (sonnet)
   → Git operations with conventional commit format
   Use: Creating commits, managing version control
```markdown

### Verbose View (`--verbose`)

```markdown
Available Agents (6):

1. orchestrator (opus) - .claude/agents/orchestrator.md
   → Routes tasks to specialists and coordinates multi-agent workflows
   Keywords: route, coordinate, delegate, workflow, multi-step
   Use: Complex multi-step tasks, workflow coordination

2. code_assistant (sonnet) - .claude/agents/code_assistant.md
   → Writes code for all implementation tasks
   Keywords: write, create, add, function, script, utility, helper, implement, build, develop
   Use: Functions, utilities, features, prototypes, bug fixes

[... etc for all agents]
```markdown

### Search View (`/agents code`)

```bash
Agents matching "code" (2):

1. code_assistant (sonnet)
   → Writes code for all implementation tasks
   Match: Keywords include "code, write, implement"

2. code_reviewer (sonnet)
   → Code quality audits and reviews
   Match: Keywords include "code review, quality"
```markdown

## When to Use /agents

✅ **Use /agents when**:

- Learning the system for the first time
- Unsure which agent handles a task
- Want to understand routing keywords
- Debugging routing issues (wrong agent selected)
- Discovering available capabilities

## Examples

### Example 1: List All

```bash
/agents
```markdown

Shows complete list with descriptions.

### Example 2: Search

```bash
/agents document
```markdown

Returns: technical_writer agent details.

### Example 3: Verbose

```bash
/agents --verbose
```markdown

Shows all agents with full keyword lists and file paths.

## Understanding Agent Roles

### Code Writing

- **code_assistant**: All code implementation tasks

### Quality & Review

- **code_reviewer**: Reviews for quality/security

### Knowledge & Documentation

- **knowledge_researcher**: Web research
- **technical_writer**: Create docs

### Operations

- **git_commit_manager**: Git operations
- **orchestrator**: Coordinates all agents

## Routing Priority

The orchestrator uses keyword matching to select agents. For complete details, see `.claude/docs/routing-concepts.md`.

**Quick overview**:

1. **Discover Available Agents**: Scan `.claude/agents/*.md`
2. **Parse Keywords**: Extract `routing_keywords` from frontmatter
3. **Match Intent**: Score agents by keyword overlap with request
4. **Select Best Match**: Route to highest-scoring agent

## How to Force Specific Agent

If orchestrator routes to wrong agent:

1. **Use /route command**:

   ```bash
   /route code_assistant "write a function"
   /route knowledge_researcher "research Docker networking"
```bash

2. **Improve request phrasing**:

   ```bash
   # Instead of: "make auth"
   # Use: "write an auth function" → code_assistant
   # Or: "research auth patterns" → knowledge_researcher
```markdown

## Agent Discovery

Agents are **dynamically discovered** from `.claude/agents/*.md`. See `.claude/docs/routing-concepts.md` for complete routing explanation.

**Key benefits**:

- ✅ Adding new agents is automatic
- ✅ No hardcoded routing tables
- ✅ System is extensible
- ✅ Custom agents work immediately

## Configuration

Agent behavior controlled in `.claude/config.yaml:orchestration.agents`:

- `max_concurrent`: Max simultaneous agents (default: 2)
- `max_batch_size`: Max per batch (default: 2)
- `polling_interval_seconds`: Poll interval for agent monitoring (default: 10)

## Related Commands

|Command|Purpose|
|---|---|
|`/agents`|List available agents|
|`/route <agent> <task>`|Force route to specific agent|
|`/help`|Show all commands|
|`/status`|System health|

## See Also

- `.claude/docs/routing-concepts.md` - Complete routing explanation
- `.claude/commands/route.md` - Force routing to specific agent
- `.claude/agents/orchestrator.md` - Orchestration logic

## Version

v2.0.0 (2026-01-16) - Spec system removed, simplified routing
