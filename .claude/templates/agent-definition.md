# Agent Definition Template

Standard template for all agent definitions. All agents MUST follow this structure.

---

name: agent_name
description: 'Brief one-line description of what this agent does (max 80 chars)'
tools: [Tool1, Tool2, Tool3] # Available tools: Read, Write, Edit, Bash, Grep, Glob, Task, WebFetch, WebSearch
model: sonnet # sonnet (fast, efficient) or opus (complex reasoning)
routing_keywords:

- keyword1
- keyword2
- keyword3
- keyword4

---

# {Agent Name}

One-sentence description of what this agent does and when to use it.

## Core Capabilities

List 4-6 specific capabilities this agent provides:

- **Capability 1** - Brief description with concrete example
- **Capability 2** - Brief description with concrete example
- **Capability 3** - Brief description with concrete example
- **Capability 4** - Brief description with concrete example

## Routing Keywords

Explain when these keywords trigger this agent and how they're used:

- **keyword1/keyword2**: Pattern description (e.g., "Direct X requests")
- **keyword3**: Pattern description (e.g., "Y-specific operations")
- **keyword4**: Pattern description (e.g., "Overlaps with AgentZ - see disambiguation")

**Note**: If keywords overlap with other agents, reference `.claude/docs/keyword-disambiguation.md`.

## Triggers

When to invoke this agent:

- User requests {specific type of work}
- After {workflow phase} completes
- When {condition} is detected
- Keywords: {list keywords}
- `/{command}` invocation (if applicable)

When NOT to invoke (anti-triggers):

- {Anti-trigger 1} → Route to `{other_agent}`
- {Anti-trigger 2} → Route to `{other_agent}`
- {Anti-trigger 3} → Route to `{other_agent}`

## Workflow

Document the step-by-step process this agent follows.

### Step 1: {Action Name}

**Purpose**: What this step accomplishes

**Actions**:

- Specific task 1
- Specific task 2
- Specific task 3

**Inputs**: What's needed to start
**Outputs**: What this step produces

### Step 2: {Action Name}

**Purpose**: What this step accomplishes

**Actions**:

- Specific task 1
- Specific task 2
- Specific task 3

**Dependencies**: What must complete first
**Outputs**: What this step produces

### Step 3: {Action Name}

**Purpose**: What this step accomplishes

**Actions**:

- Specific task 1
- Specific task 2
- Quality checks
- Validation steps

**Outputs**: What this step produces

### Step 4: Report & Handoff

**Actions**:

- Verify completion criteria met
- Generate artifacts
- Write completion report
- Prepare handoff context (if applicable)

## Definition of Done

Task is complete when ALL criteria are met:

- [ ] {Specific completion criterion 1}
- [ ] {Specific completion criterion 2}
- [ ] {Specific completion criterion 3}
- [ ] {Specific completion criterion 4}
- [ ] {Specific completion criterion 5}
- [ ] Completion report written to `.claude/agent-outputs/[task-id]-complete.json`

## Anti-Patterns

Common mistakes to avoid:

- **Pattern Name**: Description of what NOT to do, why it's wrong, and what to do instead
- **Pattern Name**: Description of what NOT to do, why it's wrong, and what to do instead
- **Pattern Name**: Description of what NOT to do, why it's wrong, and what to do instead

## Completion Report Format

Write to `.claude/agent-outputs/[task-id]-complete.json`:

```json
{
  "task_id": "YYYY-MM-DD-HHMMSS-{agent-name}",
  "agent": "{agent_name}",
  "status": "complete|in_progress|blocked|needs_review|failed",
  "started_at": "ISO-8601 timestamp",
  "completed_at": "ISO-8601 timestamp",
  "request": "Original user request or task description",
  "artifacts": ["path/to/file1", "path/to/file2"],
  "metrics": {
    "agent_specific_metric_1": 0,
    "agent_specific_metric_2": 0
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

**Status Values** (ONLY use these 5 values):

- `complete` - Task finished successfully
- `in_progress` - Currently executing (for long-running tasks)
- `blocked` - Cannot proceed without user input
- `needs_review` - Work complete but requires human review before proceeding
- `failed` - Encountered unrecoverable error

**Required Fields**: `task_id`, `agent`, `status`, `started_at`, `request`

**Optional Fields**: `completed_at`, `artifacts`, `metrics`, `validation`, `notes`, `next_agent`, `handoff_context`

## Examples

Provide 2-3 concrete examples showing common use cases.

### Example 1: {Primary Use Case}

**User Request**: "{exact user request}"

**Agent Actions**:

1. {Step 1 description with specific details}
2. {Step 2 description with specific details}
3. {Step 3 description with specific details}

**Output**: {What user receives - be specific}

**Artifacts**: `path/to/file.ext`

### Example 2: {Secondary Use Case}

**User Request**: "{exact user request}"

**Agent Actions**:

1. {Step 1 description}
2. {Step 2 description}
3. {Step 3 description}

**Output**: {What user receives}

**Artifacts**: `path/to/file.ext`

### Example 3: {Edge Case or Handoff}

**User Request**: "{exact user request}"

**Agent Actions**:

1. {Step 1 description}
2. {Detection of handoff condition}
3. {Preparation for handoff}

**Output**: {What user receives}

**Handoff**: Routes to `other_agent` with context: "{handoff reason}"

## See Also

Related documentation and agents:

- **Agent**: `{agent_name}` - Use when {condition}
- **Command**: `/{command}` - Alternative approach for {use case}
- **Documentation**: `.claude/docs/{doc}.md` - Background information
- **Configuration**: See `.claude/config.yaml:{section}` for settings

---

## Template Metadata

**Version**: 1.0.0
**Last Updated**: 2026-01-22
**Schema**: Agent Definition v1

## Validation Checklist

Before committing new/updated agent definition:

- [ ] Frontmatter complete (name, description, tools, model, keywords)
- [ ] All required sections present
- [ ] Status values match standard (5 values only)
- [ ] Completion report format matches template
- [ ] Examples are concrete and tested
- [ ] Cross-references exist for related agents
- [ ] Keywords don't duplicate others without disambiguation
- [ ] No duplicate content from other agents
- [ ] Run validation: `.claude/hooks/validate_agents.py`
