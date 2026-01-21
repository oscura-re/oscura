---
name: route
description: Force route to specific agent bypassing orchestrator
arguments: <agent> <task>
---

# /route - Force Route to Specific Agent

Bypass orchestrator intelligence and route directly to a specific agent.

## Usage

```bash
/route <agent> <task>                           # Force routing
/route code_assistant "write a function"        # Bypass orchestrator
/route knowledge_researcher "research Docker"   # Direct to researcher
/route technical_writer "document API"          # Direct to docs writer
```

## Purpose

Give power users explicit control over agent routing when:

- Orchestrator routes to wrong agent
- Testing specific agent behavior
- Know exactly which agent you want
- Debugging routing issues
- Want deterministic routing (scripts, automation)

## When to Use

✅ **Use /route when**:

- Orchestrator selected wrong agent
- Want to test specific agent
- Need deterministic routing for automation
- Debugging agent behavior
- Power user who knows the system

❌ **Don't use /route when**:

- Unsure which agent to use (use `/ai` or explicit commands instead)
- Learning the system (let orchestrator decide)
- Normal workflow (use `/code`, `/feature`, etc. instead)

## How It Works

```
/route <agent> <task>
  ↓
Skip orchestrator intelligence
  ↓
Validate agent exists
  ↓
Route directly to agent
  ↓
Agent executes task
  ↓
Return results
```

No complexity detection, no keyword matching, no workflow logic.

**For understanding normal routing**: See `.claude/docs/routing-concepts.md`

## Examples

### Example 1: Force Ad-Hoc Code

```bash
/route code_assistant "implement user authentication"
```

Even though "implement" + "authentication" would normally trigger auto-spec workflow (complexity: 75), this forces ad-hoc code writing.

Warning shown:

```
⚠️ Bypassing orchestrator intelligence
Task complexity appears high (authentication, security keywords)
Forcing: code_assistant (ad-hoc, no spec)
Recommended: /feature or /ai instead

Proceed? [y/N]:
```

### Example 2: Force Research

```bash
/route knowledge_researcher "write a function to parse JSON"
```

Forces research even though this would normally go to code_assistant.

### Example 3: Force Implementation

```bash
/route technical_writer "document API"
```

Directly routes to technical_writer for documentation creation.

## Available Agents

|Agent|Purpose|Alternative|
|---|---|---|
|`code_assistant`|Code implementation|`/ai write...`|
|`knowledge_researcher`|Web research|`/research <topic>`|
|`technical_writer`|Documentation|Use `/ai document...`|
|`code_reviewer`|Code review|`/review [path]`|
|`git_commit_manager`|Git operations|`/git "message"`|
|`orchestrator`|Workflow coordination|Automatic|
|`git_commit_manager`|Git operations|`/git [msg]`|
|`orchestrator`|Coordination|`/ai <task>`|

To see full list: `/agents`

## Validation and Safety

When you use `/route`, the system:

1. **Validates agent exists**
   - Checks `.claude/agents/<agent>.md` exists
   - If not found: Lists available agents

2. **Checks task appropriateness**
   - Analyzes if agent can handle task
   - Warns if mismatch detected

3. **Requires confirmation** (for mismatches)
   - Shows why mismatch detected
   - Asks user to confirm

### Example Validation

```bash
/route technical_writer "fix bug in login"
```

System response:

```
⚠️ Task/Agent Mismatch Detected

Agent: technical_writer
Expected: Documentation creation, tutorials, guides
Task: "fix bug in login"
Analysis: Bug fixing typically requires code_assistant

Did you mean:
1. /code fix bug in login
2. /route code_assistant "fix bug in login"
3. Proceed anyway with technical_writer

Choice [1/2/3]:
```

## Comparison with Other Approaches

|Approach|Intelligence|Safety|Speed|Use Case|
|---|---|---|---|---|
|`/ai <task>`|✅ Full|✅ High|⏱️ Medium|Let system decide|
|`/code <task>`|⚠️ Partial|✅ High|⚡ Fast|Force ad-hoc|
|`/feature <task>`|⚠️ Partial|✅ High|⏱️ Medium|Force auto-spec|
|`/route <agent> <task>`|❌ None|⚠️ Medium|⚡ Fastest|Manual control|

## Use Cases

### 1. Testing Agent Behavior

```bash
# Test how code_assistant handles complex task
/route code_assistant "implement complete REST API"

# See if it warns or proceeds
```

### 2. Overriding Wrong Routing

```bash
# User: /ai write a quick helper function
# System routes to knowledge_researcher (wrong!)
# User corrects:
/route code_assistant "write a quick helper function"
```

### 3. Automation/Scripting

```bash
# In script: deterministic routing
/route code_assistant "generate migration script"
/route code_reviewer src/module.py
/route git_commit_manager "feat: add migrations"
```

### 4. Agent Development/Debugging

```bash
# Testing new agent
/route my_custom_agent "test task"
```

## Error Handling

### Agent Not Found

```bash
/route nonexistent_agent "do something"
```

Response:

```
❌ Agent Not Found: nonexistent_agent

Available agents:
- code_assistant
- knowledge_researcher
- technical_writer
- code_reviewer
- git_commit_manager
- orchestrator

Use: /agents for details
```

### Invalid Task

```bash
/route code_assistant ""
```

Response:

```
❌ Invalid Task

Task description cannot be empty.
Usage: /route <agent> <task>
```

## Configuration

Routing behavior in `.claude/config.yaml`:

```yaml
orchestration:
  workflow:
    allow_command_overrides: true # Must be true for /route to work
    show_routing_decisions: true # Show why route succeeded
```

## Workflow

```
/route → Validate agent exists
       → Validate task not empty
       → Check task/agent appropriateness
       → Warn if mismatch (optional confirmation)
       → Route directly to agent
       → Skip all orchestrator logic
```

## Comparison with /ai

|Aspect|/ai <task>|/route <agent> <task>|
|---|---|---|
|**Intelligence**|Full routing logic|None|
|**Complexity detection**|✅ Yes|❌ No|
|**Keyword matching**|✅ Yes|❌ No|
|**Safety checks**|✅ Yes|⚠️ Basic|
|**Best for**|Normal use|Power users, testing|
|**Speed**|Medium|Fastest|
|**Flexibility**|High|Maximum|

## When NOT to Use /route

**Don't use /route when**:

1. **Unsure which agent**: Use `/ai` or ask `/agents`
2. **Normal workflow**: Use explicit commands (`/code`, `/feature`, etc.)
3. **Learning system**: Let orchestrator teach you through routing
4. **Team projects**: Explicit commands are clearer than `/route`

## Pro Tips

### 1. Combine with /agents

```bash
# First, find the right agent
/agents research

# Then route directly
/route knowledge_researcher "Docker networking 2026"
```

### 2. Use for A/B Testing

```bash
# Try ad-hoc
/route code_assistant "implement caching"

# Compare with formal
/feature implement caching

# See which approach works better
```

### 3. Debug Routing Issues

```bash
# See what orchestrator would do
/ai --dry-run implement auth

# Override if needed
/route code_assistant "implement auth"
```

## See Also

- `.claude/docs/routing-concepts.md` - Complete explanation of normal routing
- `.claude/commands/agents.md` - List available agents
- `.claude/agents/orchestrator.md` - Orchestration logic

## Version

v1.0.0 (2026-01-09) - Initial creation as part of utility command system
v1.1.0 (2026-01-16) - Added reference to routing-concepts.md
