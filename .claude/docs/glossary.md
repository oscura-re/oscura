# Orchestration System Glossary

Consistent terminology for the Claude Code orchestration system.

## Core Concepts

### Agent

A specialized AI worker that handles a specific domain of tasks (code, research, documentation, review, git operations). Each agent is defined by a markdown file in `.claude/agents/` with YAML frontmatter specifying routing keywords and capabilities.

**Examples**: code_assistant, knowledge_researcher, technical_writer

### Orchestrator

The central coordination agent that routes tasks to specialists and manages multi-agent workflows. Responsible for agent discovery, keyword matching, workflow execution, and context management.

**File**: `.claude/agents/orchestrator.md`

### Task

A unit of work assigned to an agent. Can be simple (single action) or complex (multi-step workflow).

**Examples**: "write a function", "research Docker networking", "comprehensive analysis"

### Workflow

A sequence of phases executed by one or more agents to complete a complex task. Can be serial (sequential) or parallel (concurrent).

**Types**: Single-agent, multi-agent serial, multi-agent parallel (swarm)

## Routing Concepts

### Routing Keywords

A list of keywords in an agent's frontmatter that trigger routing to that agent. Used by the orchestrator for keyword-based matching.

**Example**:

```yaml
routing_keywords:
  - write
  - code
  - implement
```markdown

### Keyword Matching

The algorithm used by the orchestrator to score agents based on overlap between user request keywords and agent routing keywords. Higher score = better match.

### Dynamic Discovery

The process of scanning `.claude/agents/*.md` at runtime to build the routing table. Ensures no hardcoded agent lists.

### Complexity Detection

Analysis of user request to determine if task requires single agent or multi-agent workflow. Looks for keywords like "comprehensive", "complete", "multiple perspectives".

### Forced Routing

Using `/route <agent> <task>` to bypass orchestrator intelligence and route directly to a specific agent.

## Execution Patterns

### Serial Execution

Agents execute one after another in sequence. Used when tasks have dependencies.

**Example**: implement → review → document

### Parallel Execution (Swarm)

Multiple agents execute simultaneously on independent subtasks. Used for comprehensive analysis or multi-perspective work.

**Example**: research from academic + industry + community sources concurrently

### Batch Execution

Grouping parallel agents into batches to stay within max concurrent limit. Each batch completes before next batch starts.

**Example**: 6 agents → 3 batches of 2 agents each

## State Management

### Completion Report

A JSON file written by each agent upon task completion. Contains status, deliverables, metrics, validation results, and handoff context for next agent.

**Location**: `.claude/agent-outputs/`
**Format**: See `.claude/templates/completion-report.md`

### Agent Registry

JSON file tracking all active and completed agents. Includes agent ID, status, start time, completion time, and output location. Critical for recovery after context compaction.

**File**: `.claude/agent-outputs/*.json`

### Checkpoint

A snapshot of workflow state saved at key points (between batches, before context compaction). Enables recovery if context is lost.

**Location**: `.coordination/checkpoints/`

### Active Work

JSON file tracking current workflow state, including current phase, completed phases, and remaining work.

**File**: `.coordination/active_work.json`

### Summary

A condensed version of an agent's output written to file to reduce context usage. Contains key findings without verbose details.

**Location**: `.claude/summaries/`

## Context Management

### Context

The conversational memory available to Claude, measured in tokens. Limited to ~200K tokens total.

### Context Usage

Percentage of available context currently in use. Monitored continuously to prevent context overflow.

### Context Thresholds

Configured limits that trigger optimization actions:

- **Warning threshold** (default: 60%): Start optimizing
- **Checkpoint threshold** (default: 65%): Create checkpoint now
- **Critical threshold** (default: 75%): Complete current task, then compact

**Configuration**: `.claude/config.yaml:orchestration.context`

### Context Compaction

Automatic reduction of context by Claude when usage becomes too high. Can lose agent task IDs and make outputs unretrievable if not properly managed.

### Context Optimization

Strategies to reduce context usage: summarizing outputs, archiving old reports, checkpointing state, deferring large file reads.

### Token

Basic unit of text for Claude (roughly 4 characters). Context capacity is measured in tokens.

## Coordination Files

### Coordination Directory

Directory containing workflow state, checkpoints, handoffs, and temporary coordination files.

**Location**: `.coordination/`

### Lock File

Temporary file preventing concurrent access to shared resources. Automatically cleaned up after use or when stale.

**Location**: `.coordination/locks/`

### Handoff

Context passed from one agent to another in a workflow. Includes artifacts, notes, and guidance for next agent.

**File pattern**: `.coordination/handoff-{source}-to-{target}-{timestamp}.md`

### Deliverable

A file or artifact produced by an agent. Listed in completion report for tracking and recovery.

## Agent Limits

### Max Concurrent Agents

Maximum number of agents allowed to run simultaneously. Enforced by runtime hooks to prevent context overflow.

**Configuration**: `.claude/config.yaml:orchestration.agents.max_concurrent` (default: 2)

### Max Batch Size

Maximum number of agents in a single batch for parallel execution.

**Configuration**: `.claude/config.yaml:orchestration.agents.max_batch_size` (default: 2)

### Polling Interval

Time in seconds between checks for agent completion when monitoring parallel agents.

**Configuration**: `.claude/config.yaml:orchestration.agents.polling_interval_seconds` (default: 10)

## Status Values

### Task Status

State of a task execution:

- **complete**: Successfully finished
- **in-progress**: Currently running
- **blocked**: Waiting for user input
- **needs-review**: Awaiting user approval
- **failed**: Execution failed with error

### Agent Status

State of an agent in the registry:

- **running**: Currently executing
- **completed**: Finished successfully
- **failed**: Execution failed
- **stale**: Running too long (>60 minutes), marked for cleanup

## Commands

### Slash Command

User-facing command that routes tasks to agents. Format: `/command <args>`

**Examples**: natural language requests, code implementation requests, `/research`, `/review`, `/git`

### Explicit Command

Command that forces routing to a specific agent, bypassing orchestrator intelligence.

**Examples**: `/research`, `/review`, `/git`, `/route <agent> <task>`

### Universal Command

The natural language requests command that uses full orchestrator intelligence to route tasks automatically.

## Hooks

### Runtime Hook

Python script executed automatically at specific points in Claude Code lifecycle. Used for enforcement, monitoring, and automation.

**Types**: PreToolUse, PostToolUse, SubagentStop, SessionEnd

### Enforcement Hook

Hook that blocks invalid operations (e.g., `enforce_agent_limit.py` blocks spawning >max_concurrent agents).

### Monitoring Hook

Hook that observes behavior and logs metrics (e.g., context usage monitoring).

## Retention and Cleanup

### Retention Policy

Rules defining how long coordination artifacts are kept before archiving or deletion.

**Configuration**: `.claude/config.yaml:retention`

### Archive

Moving old files to archive directory for later deletion. Keeps active directories clean while preserving history.

### Stale Agent

Agent that has been running for an unusually long time (>60 minutes by default). Automatically marked as failed during cleanup.

### Orphaned File

File without a corresponding counterpart, indicating incomplete work or failed cleanup.

**Example**: `chunk-001.md` without `chunk-001-translated.md`

## Workflow Patterns

### Ad-Hoc Workflow

Quick, informal task execution without formal specification or validation. Good for prototypes and simple utilities.

### Formal Workflow

Structured execution with specification, validation, and documentation. Used for production features.

### Swarm Workflow

Parallel execution pattern where multiple agents work on independent subtasks simultaneously.

### Pipeline Workflow

Serial execution pattern where output of one agent feeds into input of next agent.

## File Patterns

### Frontmatter

YAML metadata block at the beginning of markdown files, enclosed by `---`. Used in agent and command definitions.

**Example**:

```yaml
---
name: agent_name
description: Brief description
routing_keywords:
  - keyword1
  - keyword2
---
```markdown

### ISO-8601 Timestamp

Standardized date/time format: `YYYY-MM-DDTHH:MM:SSZ`

**Example**: `2026-01-16T14:30:45Z`

### Task ID

Unique identifier for a task: `YYYY-MM-DD-HHMMSS-agent-name`

**Example**: `2026-01-16-143045-code-assistant`

## Configuration

### Config File

Single source of truth for all behavioral configuration.

**File**: `.claude/config.yaml`

### Settings File

Generated file containing Claude Code settings. Do not edit manually.

**File**: `.claude/settings.json` (generated from config.yaml)

### Paths File

Definitions of file system paths used by the orchestration system.

**File**: `.claude/paths.yaml`

### Metadata File

Project identity and metadata (name, version, description).

**File**: `.claude/project-metadata.yaml`

## Error Handling

### Blocked Status

Task cannot proceed without user input. Orchestrator reports to user and waits.

### Failed Status

Task execution failed with error. Orchestrator logs error and may retry based on configuration.

### Needs-Review Status

Task completed but requires user approval before proceeding to next phase.

### Recovery

Process of restoring workflow state after interruption or context compaction. Uses checkpoints, registry, and deliverables on filesystem.

## Best Practices Terms

### Immediate Retrieval

Pattern of retrieving agent outputs as soon as they complete, rather than waiting for all agents to finish. Critical for preventing context compaction.

### Checkpoint Between Batches

Pattern of saving workflow state after each batch completes, before starting next batch. Enables recovery if compaction occurs.

### Summarize to File

Pattern of writing condensed agent output to file immediately, reducing context usage while preserving key information.

### Persist to Registry

Pattern of saving agent metadata to registry file immediately upon launch, ensuring recoverability.

## Anti-Patterns

### Spawn and Wait

❌ Launching all agents then waiting for all to complete. Can cause context overflow.
✅ Use batching with immediate retrieval instead.

### Hardcoded Routing

❌ Using fixed routing tables instead of dynamic discovery.
✅ Always scan agent files and use keyword matching.

### Batched Retrieval

❌ Waiting to retrieve outputs until all agents complete.
✅ Retrieve each output immediately upon completion.

### No Checkpointing

❌ Running long workflow without checkpoints.
✅ Checkpoint between batches and at key milestones.

## Version

v1.0.0 (2026-01-16) - Initial glossary
