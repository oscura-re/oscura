# Getting Started with Claude Code

Welcome to the Claude Code orchestration system! This guide explains the **intelligent three-path workflow system** that automatically routes your tasks to the right agent.

**Version**: 3.1.0 (2026-01-16) - Optimized and streamlined

## Quick Start

### Your First Command

**Not sure what to do? Just use `/ai`** - the system will figure out the rest:

```bash
# Write a simple function (routes to ad-hoc workflow)
/ai write a function to calculate factorial

# Build a feature (routes to auto-spec workflow)
/ai implement user authentication with JWT

# Research something (routes to research agent)
/ai research Docker networking best practices
```

### Check System Status

```bash
/status              # System health
/agents              # List available agents
/help                # Show all commands
```

---

## Understanding the Three Workflows

The system intelligently routes your tasks through **three different paths** based on complexity:

### Path 1: Ad-Hoc (No Requirements) 🏃 FAST

**For**: Quick utilities, simple functions, prototypes, bug fixes
**Speed**: ⚡ Minutes
**Output**: Code only (no spec, no validation)

**Example**:

```bash
/ai write a function to validate email addresses
```

**What happens**: Direct code generation with docstring, ready to use immediately.

---

### Path 2: Auto-Spec (System Generates) 🤖 SMART

**For**: Medium complexity features (not trivial, not massive)
**Speed**: ⏱️ 15-30 minutes
**Output**: Auto-generated requirements + code + validation report

**Example**:

```bash
/ai implement API endpoint for user registration
```

**What happens**:

1. System analyzes complexity (score: 55)
2. Generates AUTO-XXX requirements with acceptance criteria
3. Asks: "Review requirements? [Y/n]"
4. Implements with validation
5. Returns: Code + tests + validation report

---

### Path 3: Manual Spec (User Writes) 📋 FORMAL

**For**: Complex features with detailed requirements
**Speed**: ⏱️ Hours (thorough)
**Output**: Full requirements + task graph + code + validation

**Example**:

```bash
# 1. Write requirements.md manually
vim requirements.md

# 2. Extract to formal spec
/spec extract requirements.md

# 3. Implement
/implement TASK-001

# 4. Validate
/validate TASK-001
```

---

## How the System Decides

The orchestrator calculates a **complexity score (0-100)** to choose the path:

| Score      | Path        | Example                                |
| ---------- | ----------- | -------------------------------------- |
| **0-30**   | Ad-Hoc      | "write a function to reverse string"   |
| **31-70**  | Auto-Spec   | "implement JWT authentication"         |
| **71-100** | Manual Spec | "build complete OAuth system"          |

### Complexity Factors

**Increases Score (+)**:

- Security/auth keywords: +30
- Database/state: +20
- API integration: +15
- Multiple components: +15 each
- Validation needed: +10

**Decreases Score (-)**:

- "function", "helper", "utility": -30
- "quick", "simple", "small": -20
- "prototype", "example": -15

---

## Common Workflows

### 1. Research → Document → Implement

```bash
/ai research GraphQL best practices
/ai document GraphQL implementation guide
/ai implement GraphQL endpoint
```

### 2. Implement → Test → Review → Commit

```bash
/ai implement user authentication
/ai write tests for authentication
/review src/auth/
/git "implement user authentication"
```

### 3. Prototype → Formalize → Validate

```bash
/ai quick prototype of caching system
# Test prototype, decide to formalize
/spec extract from prototype
/implement TASK-001
/validate TASK-001
```

### 4. Parallel Research (Multi-Agent)

```bash
/swarm research authentication: OAuth, JWT, session-based
# Launches 3 agents in parallel
```

---

## Essential Commands

### Core Orchestration

| Command                     | Purpose                         | Example                                  |
| --------------------------- | ------------------------------- | ---------------------------------------- |
| `/ai <task>`                | Universal AI routing            | `/ai implement authentication`           |
| `/status`                   | System health check             | `/status`                                |
| `/help`                     | Show all commands               | `/help`                                  |

### Specialized Tasks

| Command                     | Purpose                         | Agent                |
| --------------------------- | ------------------------------- | -------------------- |
| `/swarm <task>`             | Parallel multi-agent workflows  | orchestrator (swarm) |
| `/research <topic>`         | Web research                    | knowledge_researcher |
| `/review <path>`            | Code review                     | code_reviewer        |
| `/implement TASK-XXX`       | Implement from spec             | spec_implementer     |
| `/validate TASK-XXX`        | Validate implementation         | spec_validator       |

### Utilities

| Command                     | Purpose                         |
| --------------------------- | ------------------------------- |
| `/agents`                   | List available agents           |
| `/context`                  | Monitor system state            |
| `/cleanup [--aggressive]`   | Clean up old files              |

For complete command reference, see `.claude/commands/*.md`

---

## Available Agents

The system includes 8 specialized agents:

| Agent                     | Purpose                                | Triggers                              |
| ------------------------- | -------------------------------------- | ------------------------------------- |
| **orchestrator**          | Task routing and workflow coordination | All `/ai` commands                    |
| **code_assistant**        | Quick ad-hoc code writing              | "write", "create", "function"         |
| **knowledge_researcher**  | Web research and synthesis             | "research", "find", "investigate"     |
| **spec_implementer**      | Implement from specifications          | "implement TASK-XXX"                  |
| **spec_validator**        | Validate against requirements          | "validate TASK-XXX"                   |
| **technical_writer**      | Documentation creation                 | "document", "write guide"             |
| **code_reviewer**         | Code review and security audit         | "review", "audit"                     |
| **git_commit_manager**    | Git operations and commits             | "commit", "/git"                      |

For detailed agent documentation, see `.claude/agents/*.md`

---

## Configuration

### Main Config: `.claude/orchestration-config.yaml`

Controls routing behavior:

```yaml
orchestration:
  default_agent: orchestrator

  workflow:
    ad_hoc_max: 30           # Complexity threshold for ad-hoc
    auto_spec_max: 70        # Max for auto-spec
    auto_spec_prompt: true   # Ask before generating

  routing:
    allow_auto_fallback: true
    confidence_threshold: 0.7
```

### Agent Config: `.claude/settings.json`

Agent-specific settings (see file for details).

---

## Best Practices

### ✅ DO

- **Start with `/ai`** - Let the system choose the right workflow
- **Be specific** - Clear task descriptions get better results
- **Review auto-generated specs** - Edit before implementation
- **Use `/status`** - Check system health before big tasks
- **Commit frequently** - Use `/git` after completing tasks

### ❌ DON'T

- **Override routing unnecessarily** - Trust the intelligence
- **Skip validation** - Always validate specs before committing
- **Use ad-hoc for complex tasks** - System will warn you
- **Ignore warnings** - If system suggests different workflow, listen
- **Mix workflows** - Complete one workflow before starting another

---

## Troubleshooting

### "Task too complex for ad-hoc"

**Issue**: Tried to use ad-hoc workflow for complex task
**Solution**: Use `/ai` and let system choose, or explicitly use `/feature`

### "No TASK-XXX spec found"

**Issue**: Tried to implement without spec
**Solution**: Create spec first with `/spec extract` or use `/ai` for auto-spec

### "Agent not responding"

**Issue**: Agent seems stuck
**Solution**: Check `/status`, restart if needed, check `.claude/reports/` for errors

### "Multiple agents suggested"

**Issue**: Task is ambiguous
**Solution**: Be more specific, or use `/swarm` for parallel approaches

---

## Next Steps

1. **Try the quick start examples** above
2. **Read agent documentation** in `.claude/agents/`
3. **Review command reference** in `.claude/commands/`
4. **Check project docs** in `CLAUDE.md` for development workflow
5. **Explore configuration** in `.claude/orchestration-config.yaml`

---

## Version History

- **v3.1.0** (2026-01-16) - Optimized and streamlined
- **v3.0.0** (2026-01-09) - Complete three-path workflow system
- **v2.0.0** - Added auto-spec workflow
- **v1.0.0** - Initial orchestration system

---

For detailed project development instructions, see `CLAUDE.md` in the repository root.
