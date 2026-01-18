---
name: swarm
description: Execute complex tasks with parallel agent coordination
arguments: <task>
---

# Parallel Agent Swarm

Execute tasks requiring multiple independent agents working simultaneously.

## When to Use

|Command|Pattern|Use When|
|---|---|---|
|`/ai`|Serial|Tasks with dependencies, step-by-step workflows|
|`/swarm`|Parallel|Independent subtasks, multiple perspectives|

## Parallel Detection Keywords

- **comprehensive** / **full** / **complete** - Needs broad coverage
- **multiple** / **various** / **diverse** - Different perspectives
- **all aspects** / **from different angles** - Multi-faceted analysis

## Process

**CRITICAL**: Use batch execution to prevent context compaction

1. Decompose task into independent subtasks
2. Group subtasks into batches (see `config.yaml:orchestration.agents.max_batch_size`)
3. For each batch:
   a. Spawn agents in parallel (max set by `config.yaml:orchestration.agents.max_concurrent`, enforced by hooks)
   b. **Retrieve outputs IMMEDIATELY** upon each completion
   c. Summarize each output to `.claude/summaries/{agent-id}.md`
   d. Update `.claude/agent-registry.json` with completion
   e. Checkpoint batch progress to `.coordination/checkpoints/`
4. Synthesize all results into unified response

## Examples

```bash
# Comprehensive research
/swarm comprehensive research on Docker networking from academic, industry, and community sources

# Multi-perspective analysis
/swarm security review from multiple perspectives: threat modeling, code audit, compliance

# Full documentation suite
/swarm complete documentation: API reference, tutorial, and architecture diagram
```

## Synthesis Process

After all agents complete:

1. Read all completion reports
2. Identify overlapping findings
3. Note conflicting information
4. Synthesize into unified document
5. Cross-reference related sections

## Anti-patterns

**CRITICAL - Context Compaction Failures**:

- Spawning >max_concurrent agents simultaneously (see `config.yaml:orchestration.agents.max_concurrent`, blocked by runtime hooks)
- Waiting for all agents before retrieving outputs (IDs lost to compaction)
- Not persisting agent registry (cannot recover after compaction)
- Batching output retrieval (retrieve immediately)
- No checkpointing between batches (lose progress on compaction)

**Other Anti-patterns**:

- Using /swarm for tasks with dependencies (use /ai for serial)
- Not synthesizing results
- Parallel agents that need each other's output

## Context Compaction Mitigation

**Why This Matters**: Multiple parallel agents can generate excessive tokens, triggering context compaction that loses agent task IDs and makes outputs unretrievable.

**Mitigation Strategy** (See `.claude/config.yaml`):

1. **Agent Registry**: Persist agent IDs to `.claude/agent-registry.json` on launch
2. **Immediate Retrieval**: Call `TaskOutput(agent_id, block=False)` in polling loop
3. **Batch Execution**: Launch max 2 agents, wait for completion, then next batch (enforced)
4. **Checkpoint Progress**: Save state to `.coordination/checkpoints/` after each batch
5. **Summarize Immediately**: Write outputs to files before moving to next batch

**Recovery Path** (if context compaction occurs):

1. Read `.claude/agent-registry.json` for agent IDs and status
2. Load latest checkpoint from `.coordination/checkpoints/`
3. Check filesystem for deliverables (agents write files successfully)
4. Resume from last completed batch

**Optimal Configuration** (see `.claude/config.yaml:orchestration.agents.*`):

- **Batch size**: See `max_batch_size` (default: 2)
- **Poll interval**: See `polling_interval_seconds` (default: 10)
- **Retrieval**: Immediate (non-blocking check, then blocking retrieve)
- **Model**: Haiku for simple agents (reduces token usage)

## See Also

- `.claude/config.yaml` - Complete orchestration configuration
- `.claude/agents/orchestrator.md` - Handles parallel dispatch with registry
- `/ai` - For serial workflows
