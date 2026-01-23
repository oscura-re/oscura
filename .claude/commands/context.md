---
name: context
description: Display context usage and optimization recommendations
arguments: []
---

# Context Command

Analyze current context usage and provide actionable optimization recommendations.

## Usage

````bash
/context           # Show context analysis and recommendations
```python

## Examples

```bash
/context           # Display current context status with optimization advice
```markdown

## Information Displayed

### Current Usage

- Total tokens used and available
- Usage percentage
- Current threshold level (healthy/warning/critical)
- Historical usage trend (if available)

### Context Breakdown

- Conversation messages
- File contents read
- Tool outputs
- Agent completion reports
- Coordination state files

### Optimization Recommendations

Based on current usage, provides specific recommendations:

#### Below Warning Threshold (Healthy)

See `config.yaml:orchestration.context.warning_threshold` (default: 60%)

- Continue normal operation
- No optimization needed
- Consider checkpointing for long-running tasks

#### Warning to Checkpoint (Warning)

Between `warning_threshold` and `checkpoint_threshold` (defaults: 60-65%)

- Archive old completion reports
- Summarize long conversation threads
- Move analysis files to `.coordination/`
- Consider creating checkpoint before next major task

#### Checkpoint to Critical (Urgent)

Between `checkpoint_threshold` and `critical_threshold` (defaults: 65-75%)

- **Immediate**: Create checkpoint now
- Archive all completed agent outputs
- Summarize conversation history
- Defer non-critical file reads
- Complete current task before starting new work

#### Above Critical Threshold (Critical)

Above `critical_threshold` (default: 75%)

- **Stop**: Complete current task immediately
- **Required**: Create checkpoint
- **Required**: Trigger context compaction
- Restore from checkpoint after compaction
- Do not start new multi-agent workflows

## Sample Output

```python
Context Usage Analysis
======================

Current Usage:
  Tokens:         90,000 / 200,000 (45%)
  Status:         🟢 Healthy
  Trend:          ↑ Increasing (15k in last 10 messages)

Context Breakdown:
  Conversation:   25,000 tokens (28%)
  File Contents:  35,000 tokens (39%)
  Tool Outputs:   20,000 tokens (22%)
  Agent Reports:  10,000 tokens (11%)

Optimization Recommendations:
  ✓ Operating normally - no action required
  • Consider checkpoint if planning large multi-agent workflow
  • Next warning threshold at 120,000 tokens (60%)

Context Efficiency:
  Signal-to-Noise: High
  Large Files Read: 2 files >5000 tokens
  Suggestions:
    - Consider using offset/limit when reading large files
    - Archive old agent outputs (5 reports >7 days old)
```markdown

## Context Management Thresholds

From `.claude/config.yaml:orchestration.context` (defaults shown, see config for current values):

- **Below warning_threshold** (default: <60%): Healthy - Normal operation
- **At warning_threshold** (default: 60%): Warning - Start optimizing
- **At checkpoint_threshold** (default: 65%): Checkpoint - Create checkpoint now
- **At critical_threshold** (default: 75%): Critical - Complete task, then compact
- **85%**: Emergency - Automatic compaction triggered

## Optimization Strategies

### Immediate Actions (any threshold)

```bash
# Clean coordination files
.claude/hooks/session_cleanup.sh

# Create checkpoint
checkpoint management create task-name "Description"
```markdown

### File Reading Optimization

```python
# Instead of reading entire large file
Read(file_path="/large/file.py")

# Use offset/limit for large files
Read(file_path="/large/file.py", offset=100, limit=50)
```markdown

### Agent Coordination

- Limit to max_concurrent agents (see `config.yaml:orchestration.agents.max_concurrent`, enforced by `enforce_agent_limit.py`)
- Use batching for multi-agent workflows
- Create checkpoints between batches
- Archive completion reports after synthesis

## See Also

- `/status` - System health and agent status
- `/cleanup` - Run maintenance tasks
- `context monitoring` - Context monitoring
- `.claude/config.yaml` - Threshold configuration
- `checkpoint management` - Checkpoint management

## Version History

- v1.0.0 (2026-01-16): Initial creation with context monitoring and optimization
````
