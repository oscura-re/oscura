# Agent Context Management Best Practices

## Overview

Effective context management ensures agents receive high-signal inputs and produce quality outputs within Claude's context window constraints.

## High-Signal vs Low-Signal Inputs

### High-Signal Inputs

Provide:

- Specific file paths and line numbers
- Clear, focused questions
- Relevant error messages with stack traces
- Concrete examples of desired output
- Well-defined success criteria

### Low-Signal Inputs

Avoid:

- Vague requests ("make it better")
- Entire directory dumps without focus
- Irrelevant background information
- Redundant examples
- Ambiguous requirements

## Context Optimization Strategies

### 1. Input Filtering

Before invoking an agent:

- Identify the minimum required context
- Exclude boilerplate and generated code
- Focus on changed lines, not entire files
- Provide targeted examples, not comprehensive lists

### 2. Progressive Disclosure

When uncertain about requirements:

1. Start with minimal context
2. Agent requests additional context if needed
3. User provides focused additions
4. Iterate until sufficient

### 3. Checkpointing

For long tasks:

- Break into phases
- Checkpoint after each phase
- Each checkpoint creates resumable state
- Reduces re-work if context compacts

## Quality Indicators

Inputs are high-quality when:

- ✓ Task can be completed without clarification
- ✓ Success criteria are measurable
- ✓ Examples match desired output format
- ✓ Scope is clear and bounded

Inputs need improvement when:

- ✗ Agent asks many clarifying questions
- ✗ Multiple iterations needed to understand task
- ✗ Output doesn't match expectations
- ✗ Task scope keeps expanding

## Context Budget Management

Claude has a context window limit. Manage it by:

1. **Monitor usage**: Use `/context` command regularly
2. **Early cleanup**: Remove completed agent outputs
3. **Summarize**: Create summaries of long discussions
4. **Checkpoint**: Save state between phases

## Agent-Specific Context Needs

Different agents have different context requirements:

| Agent                | Optimal Context Size | Key Information Needed                         |
| -------------------- | -------------------- | ---------------------------------------------- |
| code_assistant       | Small-Medium         | File to edit, specific function, requirements  |
| technical_writer     | Medium-Large         | Existing docs, codebase structure, audience    |
| knowledge_researcher | Small                | Research question, quality criteria, sources   |
| code_reviewer        | Large                | Full file(s), test coverage, security concerns |
| git_commit_manager   | Small                | Changed files, commit history                  |

## Best Practices Summary

1. **Be specific** - Provide exact paths, line numbers, error messages
2. **Be focused** - Include only relevant context
3. **Be clear** - Define success criteria upfront
4. **Be iterative** - Checkpoint complex tasks
5. **Be mindful** - Monitor context usage

## See Also

- `/context` command for usage monitoring
- `.claude/docs/routing-concepts.md` for agent selection
- `.claude/config.yaml` for context thresholds
