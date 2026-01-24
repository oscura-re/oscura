# EXHAUSTIVE CLAUDE CODE FEATURES & REFERENCE GUIDE

**Terminal-Formatted Comprehensive Reference | 2026 Edition**

_Last Updated: 2026-01-22_

---

## Table of Contents

1. [Core Features & Execution Modes](#1-core-features--execution-modes)
2. [Built-in CLI Commands & Flags](#2-built-in-cli-commands--flags)
3. [Interactive Slash Commands](#3-interactive-slash-commands)
4. [Tools Available to Claude](#4-tools-available-to-claude)
5. [Memory & Context Management](#5-memory--context-management)
6. [Configuration (.claude/config.yaml & settings.json)](#6-configuration-claudeconfigyaml--settingsjson)
7. [Hooks: Complete Reference](#7-hooks-complete-reference)
8. [Custom Extensions](#8-custom-extensions)
9. [Agent System](#9-agent-system)
10. [Session Management](#10-session-management)
11. [Git Integration](#11-git-integration)
12. [Developer Tools & Capabilities](#12-developer-tools--capabilities)
13. [File Operations & Patterns](#13-file-operations--patterns)
14. [Execution & Scripting](#14-execution--scripting)
15. [Monitoring & Observability](#15-monitoring--observability)
16. [Security & Safety](#16-security--safety)
17. [Output & Formatting](#17-output--formatting)
18. [Environment & Platform](#18-environment--platform)
19. [Documentation & Help](#19-documentation--help)
20. [Advanced & Experimental Features](#20-advanced--experimental-features)
21. [Keyboard Shortcuts & Hotkeys](#21-keyboard-shortcuts--hotkeys)
22. [Best Practices & Guidance](#22-best-practices--guidance)
23. [Configuration Examples](#23-configuration-examples)
24. [Troubleshooting Guide](#24-troubleshooting-guide)
25. [Resources & References](#25-resources--references)

- [Appendix A: Quick Reference Tables](#appendix-a-quick-reference-tables)
- [Appendix B: Glossary](#appendix-b-glossary)
- [Appendix C: File Structure](#appendix-c-file-structure)

---

## 1. CORE FEATURES & EXECUTION MODES

### 1.1 Execution Modes

**Interactive REPL Mode**

- `claude` - Start interactive REPL session in terminal
- Persistent bash session across commands
- Access to all tools and slash commands
- Conversation history maintained within session

**Print Query Mode (Non-Interactive)**

- `claude -p "<query>"` - Execute query and exit
- Returns response, no REPL interaction
- Useful for scripting and automation
- Supports piped input: `cat file | claude -p`

**Continue/Resume Mode**

- `claude -c` - Resume most recent session in current directory
- `claude -r "<session-id>"` - Resume named session
- Restores full session context and state
- Can pass additional prompt: `claude -c "Continue with..."`

**Plan Mode**

- Focus on architecture and strategy
- Creates detailed implementation plans
- Best for complex, multi-step projects
- Generates task breakdowns before coding

**Build Mode**

- Focuses on implementation details
- Executes code modifications directly
- Runs tests and validates implementations
- Optimizes for working code delivery

**Deploy Mode**

- Handles infrastructure and deployment tasks
- Manages CI/CD configuration
- Orchestrates multi-step deployments
- Integrates with cloud platforms

---

## 2. BUILT-IN CLI COMMANDS & FLAGS

### 2.1 Primary Commands

**Session Management**

```bash
claude                          # Start interactive REPL
claude "query"                  # Initialize REPL with prompt
claude -p "query"               # Query via SDK, exit (non-interactive)
cat file | claude -p            # Process piped content
claude -c                       # Continue most recent session
claude -r "<session-id>"        # Resume named/numbered session
claude update                   # Update to latest version
claude mcp                      # Configure Model Context Protocol servers
```bash

**CLI Flags - Model & Behavior**

```bash
--model <model>                 # Set model (sonnet|opus|haiku)
--max-turns <n>                 # Limit agentic turns (print mode only)
--max-budget-usd <amount>       # Spending cap before stopping
--reasoning <mode>              # Enable reasoning (basic|extended)
--temperature <0-2>             # Control creativity/precision
```python

**CLI Flags - System Prompts**

```bash
--system-prompt "<text>"        # Replace entire default system prompt
--append-system-prompt "<text>" # Add instructions keeping defaults
--system-prompt-file <path>     # Load system prompt from file
--append-system-prompt-file <path> # Load additional instructions
```bash

**CLI Flags - Permissions & Tools**

```bash
--permission-mode <mode>        # Set initial permission mode
--dangerously-skip-permissions  # Skip all permission prompts (⚠️ use cautiously)
--tools <list>                  # Restrict tools (Bash,Read,Edit,Write,Glob,Grep)
--allowedTools <list>           # Only allow specified tools
--disallowedTools <list>        # Deny specific tools
--add-dir <path>                # Add working directory
```bash

**CLI Flags - Output & Integration**

```bash
--output-format <format>        # text | json | stream-json
--json                          # Shorthand for --output-format json
--chrome                        # Enable browser automation
--ide                           # Auto-connect to available IDE
--headless                      # Run in headless mode
--verbose                       # Enable detailed turn-by-turn output
--debug [filter]                # Enable debug mode with optional filtering
```bash

**CLI Flags - Advanced**

```bash
--agents <json>                 # Define custom subagents via JSON
--version                       # Show version
--help                          # Show help
--init                          # Initialize project setup
--init-only                     # Run initialization without starting REPL
--maintenance                   # Run maintenance operations
--profile                       # Enable performance profiling
```markdown

---

## 3. INTERACTIVE SLASH COMMANDS

### 3.1 Built-in Slash Commands (Universal)

**Core Navigation & Help**

```bash
/help [command]                 # Show available commands or detailed help
/status [--json]                # System health, agents, context usage
/context                        # Context optimization recommendations
/usage                          # Token usage and cost information
/model                          # Show/change current model
/models                         # List available models
```bash

**Session & State Management**

```bash
/memory [file]                  # Open memory file in editor
/rename <name>                  # Rename current session
/resume <n>                     # Resume numbered or named session (REPL only)
/stats                          # Usage statistics and trends
/todos                          # List tracked TODO items
/todos add <item>               # Add TODO item
/todos done <id>                # Mark TODO complete
```bash

**Code & File Operations**

```bash
/rewind                         # Access checkpoint/undo menu (ESC ESC)
/review [path]                  # Code quality review
/search <pattern>               # Search codebase
/grep <pattern>                 # Advanced grep search
```bash

**Permissions & Security**

```bash
/permissions                    # View and update tool permissions
/deny <tool>                    # Deny specific tool access
/allow <tool>                   # Allow specific tool access
```markdown

**System & Debug**

```bash
/clear                          # Clear conversation history (session only)
/compact                        # Manual context compaction (rarely needed)
/checkpoint                     # Create manual checkpoint
/version                        # Show Claude Code version
/settings                       # View current settings
/feedback                       # Report bug or feature request
```markdown

### 3.2 Custom Project Slash Commands (Oscura Example)

**Orchestration & Routing**

```bash
/ai <task>                      # Universal routing to specialized agents
/agents                         # List available agents with capabilities
/route <agent> <task>           # Force route to specific agent
/swarm <task>                   # Parallel agent coordination
/status [--json]                # Show orchestration health
```bash

**Development Workflows**

```bash
/research <topic>               # Web research with citations
/review [path]                  # Code quality review
/git [message]                  # Smart atomic commits
/security-review                # Security audit of pending changes
```markdown

**System Management**

```bash
/context                        # Context usage analysis
/cleanup [--dry-run]            # Clean stale files and agents
/help [command]                 # Command reference
```python

---

## 4. TOOLS AVAILABLE TO CLAUDE

### 4.1 File Operation Tools

**Read(file_path)**

- Read files from filesystem
- Parameters: `file_path` (absolute), `offset` (optional), `limit` (optional)
- Supports: Text files, images (PNG, JPG), PDFs, Jupyter notebooks
- Returns: File contents with line numbers
- Gotcha: Long lines (>2000 chars) are truncated

**Write(file_path, file_contents)**

- Create or overwrite entire files
- Parameters: `file_path` (absolute), `file_contents` (string)
- Returns: Confirmation of write
- Gotcha: Overwrites completely - use Edit for modifications

**Edit(file_path, original_text, modified_text)**

- Perform exact string replacements
- Parameters: `file_path`, `original_text` (exact match), `modified_text`
- Returns: Confirmation with context
- Gotcha: String must match exactly (whitespace matters)

**MultiEdit(file_path, edits[])**

- Batch multiple edits to single file
- Parameters: `file_path`, array of `{original, modified}` pairs
- Returns: Confirmation of all edits
- Gotcha: Edits applied in order; earlier edits affect line numbers

**NotebookRead(file_path)**

- Read Jupyter notebooks (.ipynb)
- Parameters: `file_path` (absolute)
- Returns: All cells with outputs, combining code/text/visualizations
- Gotcha: Large notebooks may truncate output

**NotebookEdit(file_path, edits[])**

- Edit Jupyter notebooks
- Parameters: `file_path`, array of cell edits
- Returns: Confirmation with cell references
- Gotcha: Cell IDs must exist

### 4.2 Search & Pattern Tools

**Glob(pattern, path?)**

- Fast file pattern matching (any codebase size)
- Parameters: `pattern` (glob), `path` (optional directory)
- Supports: `**/*.js`, `src/**/*.ts`, brace expansion `**/*.{ts,tsx}`
- Returns: Matching file paths sorted by modification time
- Gotcha: `**` requires explicit use for recursive search

**Grep(pattern, path?, options)**

- Ripgrep-based content search (powerful regex)
- Parameters: `pattern` (regex), `path` (optional), `glob` (optional), `type` (optional)
- Options: `-n` (line numbers), `-A/-B/-C` (context lines), `-i` (case-insensitive)
- Output modes: `content` (matching lines), `files_with_matches` (paths), `count`
- Supports: Full regex syntax, multiline matching with `multiline: true`
- Gotcha: Literal braces in Go code need escaping (`interface\\{\\}`)

### 4.3 Execution Tools

**Bash(command, cwd?)**

- Execute shell commands in persistent bash session
- Parameters: `command` (string), `cwd` (working directory)
- Returns: Stdout and stderr
- Environment: Inherits from Claude Code shell
- Special: Timeout handling via `BASH_DEFAULT_TIMEOUT_MS` env var
- Gotcha: Working directory resets between calls in agent threads

### 4.4 Web & Network Tools

**WebFetch(url, prompt)**

- Fetch URL content and process with AI
- Parameters: `url` (valid URL), `prompt` (extraction instructions)
- Returns: Processed markdown content
- Auto-upgrades HTTP to HTTPS
- Features: 15-minute cache, HTML to markdown conversion
- Gotcha: May summarize very large content

**WebSearch(query, allowed_domains?, blocked_domains?)**

- Web search with up-to-date results
- Parameters: `query` (search term), optional domain filtering
- Returns: Search results with markdown links
- US-only availability
- Recent data access (beyond knowledge cutoff)
- Requires Sources section in response

### 4.5 Task & List Management Tools

**Task(config, instructions?)**

- Launch subagent/agent tasks
- Parameters: `config` (agent spec), optional `instructions`
- Returns: Task ID for monitoring
- Advanced: Non-blocking retrieval via `TaskOutput(id, block=False)`

**TaskOutput(task_id, block?, timeout?)**

- Retrieve subagent results
- Parameters: `task_id`, `block` (wait?), `timeout` (milliseconds)
- Returns: Agent output and status
- Pattern: For swarm mode, retrieve immediately (non-blocking)

**TodoRead()**

- Read current TODO list
- Returns: Formatted list of tracked items

**TodoWrite(todos[])**

- Update TODO list
- Parameters: Array of TODO item objects
- Returns: Confirmation

### 4.6 System Tools

**LS(path)**

- List directory contents
- Parameters: `path` (directory)
- Returns: Formatted directory listing

---

## 5. MEMORY & CONTEXT MANAGEMENT

### 5.1 CLAUDE.md (Project Memory)

**Locations (Hierarchical Loading)**

```bash
1. Managed Policy (system-wide)
   macOS: /Library/Application Support/ClaudeCode/CLAUDE.md
   Linux: /etc/claude-code/CLAUDE.md
   Windows: C:\Program Files\ClaudeCode\CLAUDE.md

2. User Memory (personal, all projects)
   ~/.claude/CLAUDE.md

3. Project Memory (shared with team)
   ./CLAUDE.md or ./.claude/CLAUDE.md

4. Local Project Memory (personal, auto-gitignored)
   ./CLAUDE.local.md
```python

**CLAUDE.md Features**

- Plain markdown with arbitrary structure
- Supports file imports via `@path/to/file` syntax
- Recursive upward discovery from cwd to root
- Recursive downward discovery in active subdirectories
- Imports support 5-hop deep recursive loading
- Automatically loaded into context on session start

**CRITICAL FINDING: CLAUDE.md Context Behavior**

- ✅ **Loaded at session start ONLY** - not during conversation
- ✅ **PERSISTS through compaction** - stored separately from conversation history
- ✅ **No explicit reload command** - by design (would bloat context)
- ✅ **To force reload**: Start new session (`claude --new`) or restart

**Best Practices**

- Document coding standards and conventions
- Include frequently-used build/test commands
- Explain architectural patterns and abstractions
- Define project-specific terminology
- Keep focused and organized (use .claude/rules/ for modular approach)

### 5.2 Path-Specific Rules (.claude/rules/)

**Structure**

```yaml
.claude/rules/
  testing.md          # Testing-specific instructions
  security.md         # Security guidelines
  performance.md      # Performance best practices
  api-design.md       # API design patterns
```yaml

**YAML Frontmatter**

```yaml
---
paths:
  - "src/**/*.ts"
  - "lib/**/*.ts"
  - "tests/**/*.test.ts"
---

# Rule content here
```markdown

**Matching**

- Rules without `paths` apply globally
- Rules with `paths` only load for matching files
- Supports glob patterns and brace expansion: `src/**/*.{ts,tsx}`
- Evaluated when Claude works with matching files

### 5.3 Context Management

**Context Window**

- Fixed 200,000 token window (Sonnet/Opus)
- Contains: Conversation history, file contents, tool outputs, CLAUDE.md
- Working memory for current session

**Context Thresholds**

- Warning: 60% usage - Consider summarizing
- Checkpoint: 65% usage - Create checkpoint now
- Critical: 75% usage - Complete task only, then checkpoint

**Context Monitoring Commands**

```bash
/context                        # Detailed analysis with recommendations
/status                         # Current usage percentage
```markdown

**Context Optimization Techniques**

1. Checkpoint and compact before critical tasks
2. Summarize completed work to files
3. Archive old outputs to .claude/agent-outputs/
4. Use .claudeignore to exclude large files
5. Explicit cleanup of stale coordination files

### 5.4 Auto-Compaction

**How It Works**

- Triggered when context approaches threshold
- System analyzes conversation to identify key info
- Creates concise summary of previous interactions
- Compacts by replacing old messages with summary
- Continues seamlessly with preserved context

**Characteristics**

- Instant (as of v2.0.64+)
- Fires around 75-92% context usage (not officially documented)
- Preserves decisions, progress, dependencies
- Drops verbose outputs, exploration paths, resolved errors

**Triggering**

- Automatic at context thresholds
- Manual via `/cleanup` (rarely needed)
- Can be customized via PreCompact hooks

**Recovery Post-Compaction**

- System automatically loads checkpoints
- Agent registry persists status
- Coordination files preserve state
- FileSystem recovery fallback available

---

## 6. CONFIGURATION (.claude/config.yaml & settings.json)

### 6.1 Orchestration Configuration

**Agent Limits**

```yaml
orchestration:
  agents:
    max_concurrent: 3           # Simultaneous agents (enforced by hooks)
    max_batch_size: 3           # Agents per batch
    recommended_batch_size: 1   # Safest setting
    polling_interval_seconds: 10 # Swarm mode polling
```yaml

**Workflow Tracking**

```yaml
orchestration:
  workflow:
    track_progress: true
    max_phases_without_checkpoint: 3
    require_completion_reports: true
```yaml

**Context Management**

```yaml
orchestration:
  context:
    enable_monitoring: true
    warning_threshold: 60       # Percent
    checkpoint_threshold: 65    # Percent
    critical_threshold: 75      # Percent
    max_tool_output_size: 10000 # Tokens
    truncate_large_outputs: true
```yaml

**Checkpointing**

```yaml
orchestration:
  checkpointing:
    enabled: true
    auto_checkpoint: true
    checkpoint_between_batches: true
    checkpoint_on_context_threshold: true
    checkpoint_interval_phases: 1
```yaml

**Output Summarization**

```yaml
orchestration:
  summaries:
    enabled: true
    auto_summarize: true
    immediate_capture: true
    immediate_summarization: true
    auto_summarize_threshold_tokens: 5000
    max_summary_length_tokens: 2000
    write_summaries_to_file: true
    max_output_tokens_in_context: 5000
    format: markdown
```yaml

### 6.2 Retention Policies (days)

```yaml
retention:
  agent_registry: 30            # Analysis/metrics
  agent_outputs: 7              # Archive after 7 days
  checkpoints: 7                # Delete temporary checkpoints
  coordination_files: 30        # Archive after 30 days
  handoffs: 7                   # Delete handoff data
  summaries: 30                 # Keep with registry
  reports: 7                    # Archive temporary reports
  reports_archive: 30           # Delete archived reports
  locks_stale_minutes: 60       # Delete stale locks (1 hour)
  archives_max_days: 180        # Delete old archives (6 months)
  orchestration_log_days: 14    # Keep logs (14 days)
```markdown

### 6.3 Hook Configuration

**Hook Event Locations in settings.json**

```json
{
  "hooks": {
    "PreToolUse": [...],
    "PostToolUse": [...],
    "SubagentStop": [...],
    "PreCompact": [...],
    "SessionStart": [...],
    "SessionEnd": [...],
    "Stop": [...],
    "PermissionRequest": [...],
    "UserPromptSubmit": [...],
    "Notification": [...],
    "Setup": [...]
  }
}
```bash

**Hook Configuration Structure**

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "auto|compact|Bash|Edit|Write|*",
        "hooks": [
          {
            "type": "command",
            "command": "script content",
            "timeout": 30
          },
          {
            "type": "action",
            "action": "allow|deny|prompt"
          }
        ]
      }
    ]
  }
}
```yaml

### 6.4 Enforcement Configuration

```yaml
enforcement:
  enabled: true
  agent_limit: true             # Enforce max_concurrent
  auto_summarize: true          # Force summarization on large outputs
  path_validation: true         # Validate write paths
  report_limits: true           # Prevent report proliferation
  context_monitoring: true      # Monitor and warn about context

  hooks:
    enforce_agent_limit: ".claude/hooks/enforce_agent_limit.py"
    check_subagent_stop: ".claude/hooks/check_subagent_stop.py"
    validate_path: ".claude/hooks/validate_path.py"
    check_report_proliferation: ".claude/hooks/check_report_proliferation.py"
    health_check: ".claude/hooks/health_check.py"
```yaml

### 6.5 Logging & Metrics

```yaml
logging:
  enabled: true
  files:
    context_metrics: ".claude/hooks/context_metrics.log"
    compaction_events: ".claude/hooks/compaction.log"
    checkpoint_history: ".claude/hooks/checkpoints.log"
    session_history: ".claude/hooks/sessions.log"
    errors: ".claude/hooks/errors.log"
    enforcement: ".claude/hooks/enforcement.log"

  log_context_metrics: true
  log_agent_launches: true
  log_completions: true
  log_failures: true
  detect_compaction: true
```yaml

### 6.6 Security Configuration

**Denied Read Paths** (prevent context pollution)

```yaml
security:
  denied_reads:
    - ".claude/agent-outputs/archive/**"
    - ".coordination/archive/**"
    - ".git/**"
    - "__pycache__/**"
    - ".venv/**"
    - ".ruff_cache/**"
    - ".mypy_cache/**"
    - ".pytest_cache/**"
```yaml

**Denied Write Paths** (protect critical files)

```yaml
security:
  denied_writes:
    - ".claude/settings.json"
    - ".claude/config.yaml"
    - ".claude/template.lock"
    - "**/.env*"
    - "**/*.key"
    - "**/*.pem"
    - "**/*credentials*"
    - ".git/**"
```markdown

---

## 7. HOOKS: COMPLETE REFERENCE

### 7.1 Hook Events & When They Fire

**PreToolUse**

- Fires: Before any tool call
- Can: Block tool, provide feedback, log usage
- Matchers: Tool-specific, multi-tool (|), wildcard (*)
- Example: Enforce tool restrictions, validate paths, rate limit

**PostToolUse**

- Fires: After tool completes successfully
- Can: Format output, trigger side effects, log results
- Example: Auto-format code, notify, cleanup

**SubagentStop**

- Fires: When subagent task completes
- Can: Capture output, summarize, validate results
- Example: Track agent completion, enforce output size limits

**PreCompact**

- Fires: Before context compaction/summarization
- Can: Create pre-compaction checkpoint, log state
- Example: Preserve critical context, archive outputs

**SessionStart**

- Fires: New session or resumed session
- Output: Added to conversation context
- Example: Load project state, health checks, recovery

**SessionEnd**

- Fires: When session terminates
- Example: Cleanup, metrics, archive work

**Stop**

- Fires: When Claude Code finishes responding
- Example: Final cleanup, notification

**PermissionRequest**

- Fires: When permission dialog appears
- Can: Allow/deny/prompt
- Example: Auto-approve trusted tools

**UserPromptSubmit**

- Fires: User submits prompt (before processing)
- Can: Validate, modify, block prompt
- Example: Rate limiting, prompt validation

**Notification**

- Fires: Claude Code sends notification
- Example: Custom notification routing

**Setup**

- Fires: With `--init`, `--init-only`, `--maintenance` flags
- Example: Initialize project, run setup tasks

### 7.2 Hook Matchers

**Syntax Options**

```bash
"matcher": "auto"               # Auto-detect session type
"matcher": "compact"            # Only compact sessions
"matcher": "Bash"               # Only Bash tool
"matcher": "Edit|Write"         # Multiple tools (pipe-separated)
"matcher": "Read|Glob|Grep"     # Search tools only
"matcher": "*"                  # All tools (wildcard)
```markdown

**Common Tool Matcher Combinations**

```markdown
"Bash"                          # Shell commands
"Write|Edit|NotebookEdit"       # File modifications
"Read|Glob|Grep"                # File searches
"Task"                          # Subagent launches
```markdown

### 7.3 Hook Actions

**Command Hooks**

```json
{
  "type": "command",
  "command": "bash script here",
  "timeout": 30
}
```markdown

**Action Hooks**

```json
{
  "type": "action",
  "action": "allow|deny|prompt"
}
```bash

### 7.4 Hook Examples

**Pre-Commit Validation (PreToolUse)**

```json
{
  "matcher": "Bash",
  "hooks": [{
    "type": "command",
    "command": "if echo \"$COMMAND\" | grep -qE '^pip'; then echo 'Use uv instead'; exit 1; fi",
    "timeout": 5
  }]
}
```bash

**Auto-Formatting (PostToolUse)**

```json
{
  "matcher": "Write|Edit",
  "hooks": [{
    "type": "command",
    "command": "python3 .claude/hooks/auto_format.py",
    "timeout": 60
  }]
}
```bash

**Subagent Output Validation (SubagentStop)**

```json
{
  "matcher": "*",
  "hooks": [{
    "type": "command",
    "command": "python3 .claude/hooks/check_subagent_stop.py",
    "timeout": 30
  }]
}
```bash

**Context Monitoring (SessionStart)**

```json
{
  "matcher": "auto",
  "hooks": [{
    "type": "command",
    "command": "python3 .claude/hooks/health_check.py",
    "timeout": 5
  }]
}
```markdown

---

## 8. CUSTOM EXTENSIONS

### 8.1 Creating Slash Commands

**Command Definition Structure** (`.claude/commands/[your-command].md`)

```yaml
---
name: custom-cmd
description: What this command does
arguments: [arg1, arg2]
---

# Command Documentation

## Usage
```markdown

/custom-cmd <arg1> [arg2]

```markdown

## Examples
```bash
/custom-cmd value1 value2
```markdown

```markdown

**Command Implementation**
- Name: `/custom-cmd` (file prefix becomes command name)
- Args: Space-separated after command
- Accessible via `/help custom-cmd`
- Can call other commands, route to agents

### 8.2 Creating Custom Agents

**Agent Definition Structure** (`.claude/agents/[your-agent].md`)
```yaml
---
name: custom-agent
description: Clear description of purpose
model: sonnet|opus|haiku
tools: [Tool1, Tool2, Tool3]
routing_keywords: [keyword1, keyword2, keyword3]
---

# Agent Documentation

## Triggers
- List of conditions that activate this agent

## Responsibilities
- Core duties and capabilities

## Workflow
### Step 1: ...
### Step 2: ...

## Completion Report Format
```json
{
  "status": "complete|blocked|needs-review",
  "artifacts": [...],
  "next_steps": [...]
}
```markdown

```python

**Agent Registration**
- File: `.claude/agents/<name>.md`
- Frontmatter required for routing
- Scanned dynamically by orchestrator
- Must include `routing_keywords` for discovery

### 8.3 Hook Development

**Hook Script Template (Python)**
```python
#!/usr/bin/env python3
import sys
import json
import os
from pathlib import Path

# Hook receives context via environment
tool_name = os.getenv("TOOL_NAME")
command = os.getenv("COMMAND")  # For Bash
file_path = os.getenv("FILE_PATH")  # For Write/Edit

# Process
result = validate_something(command)

# Exit codes:
# 0 = success (allow)
# 1 = failure (deny/error)
sys.exit(0 if result else 1)
```bash

**Hook Script Template (Bash)**

```bash
#!/bin/bash
# Hook receives tool info via environment
TOOL_NAME="${TOOL_NAME}"
COMMAND="${COMMAND}"
FILE_PATH="${FILE_PATH}"

# Validation logic
if [[ "$FILE_PATH" == *.env* ]]; then
  echo "Cannot modify .env files"
  exit 1
fi

exit 0
```yaml

---

## 9. AGENT SYSTEM

### 9.1 Agent Types & Availability

**Built-in Available Agents**

```yaml
orchestrator         - Task routing and multi-agent coordination
code_assistant       - Quick ad-hoc code writing
knowledge_researcher - Web research and synthesis
technical_writer     - Documentation creation
code_reviewer        - Code quality and security review
git_commit_manager   - Git operations and commits
```markdown

**Agent Discovery**

- Location: `.claude/agents/*.md` (project root)
- Routing: Via frontmatter `routing_keywords`
- Dynamic: Scanned on each routing decision
- Never hardcode routing tables

### 9.2 Agent Orchestration

**Routing Process (Orchestrator)**

1. Parse user intent and extract task type
2. Assess complexity (single vs multi-agent)
3. Scan `.claude/agents/` for available agents
4. Match keywords from agent frontmatter
5. Score agents by keyword overlap
6. Route to best-matching agent or coordinate workflow

**Execution Loop**

```python
1. Update active_work.json: status = "in_progress"
2. Spawn agent(s) for current phase
3. WAIT for completion report(s)
4. Check report status:
   - "blocked": Report to user, wait for input
   - "needs-review": Report to user, wait for input
   - "complete": Continue
5. More phases remaining?
   - YES: Return to step 2
   - NO: Continue
6. Synthesize results from all completion reports
7. Update active_work.json: status = "complete"
8. Write final orchestration completion report
```python

### 9.3 Swarm Mode (Parallel Execution)

**Pattern: Parallel Agent Dispatch**

```python
1. Load configuration from .claude/config.yaml
2. Initialize registry: .claude/agent-registry.json
3. Decompose task into independent subtasks
4. Batch planning (max per config, enforced by hooks)
5. Execute each batch:
   - Launch agents
   - Persist to registry immediately
   - Non-blocking monitoring (polling loop)
   - Immediate output capture & summarization
   - Move agent to registry as "completed"
6. Checkpoint after each batch
7. Synthesize all summaries
8. Archive old outputs
9. Return unified response
```yaml

**Configuration**

```yaml
orchestration:
  agents:
    max_concurrent: 3           # Simultaneous agents
    max_batch_size: 3
    polling_interval_seconds: 10 # Check for completion
  batch:
    strategy: sequential        # or parallel
    wait_for_completion: true
    checkpoint_between_batches: true
    summarize_between_batches: true
    immediate_retrieval: true
```markdown

**Usage**

```bash
/swarm research authentication: OAuth, JWT, session-based
# Launches 3 agents in parallel for each approach
```json

### 9.4 Multi-Agent Coordination

**Completion Reports** (Required for handoff)

```json
{
  "task_id": "timestamp-task-name",
  "agent": "agent-name",
  "status": "complete|blocked|needs-review",
  "artifacts": [
    {
      "type": "code|documentation|analysis",
      "path": "file/path.ext",
      "description": "What this artifact does"
    }
  ],
  "summary": "2-3 sentence summary of work done",
  "next_agent": "agent-name|none",
  "potential_gaps": ["list of", "incomplete items"],
  "validation_passed": true,
  "completed_at": "ISO-8601 timestamp"
}
```markdown

**Handoff Protocol**

- Before routing to next agent: Read completion report
- Verify `status: complete`
- Check `validation_passed: true` if applicable
- Extract `artifacts` for context
- Note `potential_gaps` for next phase

---

## 10. SESSION MANAGEMENT

### 10.1 Session Persistence

**Automatic Session Tracking**

- Sessions stored in `~/.claude/sessions/` (SQLite database)
- Contains: Conversation history, file contexts, permissions, working dirs
- Persists across machine restarts
- Auto-saved after each interaction

**Named Sessions**

```bash
/rename <session-name>          # Name current session
/resume <n>                     # Resume numbered or named session
claude -r "session-name"        # Resume from terminal
```markdown

**Session Resume Behavior**

- Restores full context and state
- Reloads CLAUDE.md and rules
- Restores working directory
- Preserves tool permissions
- Can pass new prompt: `claude -c "Continue with..."`

### 10.2 Session Lifecycle

**Session Start**

- Loads CLAUDE.md and .claude/rules/
- Runs SessionStart hooks
- Initializes working directory
- Establishes tool permissions
- Available in REPL and print modes

**Session Active**

- Maintains conversation history
- Tracks file modifications
- Monitors context usage
- Creates automatic checkpoints
- Pre-compaction on threshold approach

**Session End**

- Runs SessionEnd hooks
- Archives outputs to .claude/summaries/
- Updates agent registry
- Triggers cleanup tasks
- Persists session state

### 10.3 Checkpoint System

**How Checkpoints Work**

- Automatic: Every user prompt creates new checkpoint
- Persist: Across sessions for 30 days
- Track: File modifications before each edit
- Restore: Via `/rewind` command (ESC ESC)

**Restoration Options**

```bash
1. Conversation Only - Revert messages, keep code
2. Code Only - Remove changes, keep history
3. Both - Full rewind to checkpoint
```bash

**Limitations**

- Does NOT track bash commands (rm, mv, cp, etc.)
- External changes may not be captured
- NOT a replacement for Git version control
- Local undo functionality only

**Best Practice**

- Use for session-level experimentation
- Maintain Git commits for permanent history
- Think checkpoints as complementary to version control

---

## 11. GIT INTEGRATION

### 11.1 Git Workflow

**Smart Commit Command**

```bash
/git [message]                  # Create smart atomic commits
/git "feat: implement auth"     # With explicit message
```bash

**Commit Analysis Process**

1. Sync with remote (REQUIRED - prevents merge conflicts)
2. Smart merge resolution if needed
3. Safe change review
4. Categorize changes for atomic commits
5. Execute commits with conventional format
6. Push all commits

**Conventional Commit Format**

```markdown
<type>(<scope>): <subject>

<body>

<footer>
```python

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `ci`

### 11.2 Multi-File Commit Strategies

**Splitting Complex Changes**

```bash
# Separate by domain
git add src/auth/login.py && git commit -m "feat(auth): implement login"
git add src/auth/jwt.py && git commit -m "feat(auth): add JWT support"

# Separate code from tests
git add src/module.py && git commit -m "feat: new feature"
git add tests/test_module.py && git commit -m "test: add feature tests"
```bash

### 11.3 Git Commands Available

**Pre-Commit Checks**

```bash
./scripts/check.sh              # All quality checks
./scripts/fix.sh                # Auto-fix issues
./scripts/pre-push.sh           # Full CI simulation
```markdown

---

## 12. DEVELOPER TOOLS & CAPABILITIES

### 12.1 Code Review

**Interactive Code Review**

```bash
/review [path]                  # Code quality review
/security-review                # Security audit of pending changes
```bash

**Review Categories**

1. Code Quality (style, design, complexity)
2. Security (vulnerabilities, injection, auth)
3. Performance (bottlenecks, optimization)
4. Testing (coverage, edge cases)
5. Maintainability (readability, documentation)

**Severity Levels**

- CRITICAL: Security/correctness issues
- HIGH: Must fix before merge
- MEDIUM: Tech debt, performance
- LOW: Style suggestions

### 12.2 Testing Support

**Test Execution**

```bash
./scripts/test.sh               # Run tests (auto-parallel)
./scripts/test.sh --fast        # Quick tests without coverage
uv run pytest tests/ -x         # Manual test run
```markdown

**Testing Patterns**

- Unit tests: Algorithm correctness, synthetic data
- Integration tests: Edge cases not in demos
- Demo validation: Living integration tests

### 12.3 Debugging Features

**Inspection Tools**

```bash
/status [--json]                # System state
/context                        # Context analysis
/rewind                         # Code changes browser
```markdown

**Bash Debugging**

- Error output captured and displayed
- Exit codes preserved
- Working directory context available

---

## 13. FILE OPERATIONS & PATTERNS

### 13.1 File Reading Patterns

**Basic Read**

```bash
Read(file_path="/absolute/path/file.txt")
```bash

**Reading with Offset/Limit**

```bash
Read(file_path="/path/file.txt", offset=100, limit=50)
# Reads lines 100-150
```markdown

**Special File Types**

```bash
Read(file_path="/path/image.png")     # Images display visually
Read(file_path="/path/document.pdf")  # PDFs extract text/images
Read(file_path="/path/notebook.ipynb") # Jupyter returns all cells
```markdown

### 13.2 File Writing Patterns

**Create/Overwrite File**

```bash
Write(file_path="/absolute/path/new.py", file_contents="code here")
```bash

**Safe Modification (Exact String Match)**

```bash
Edit(
  file_path="/absolute/path/file.py",
  original_text="old code\nmultiline match",
  modified_text="new code\nreplacement"
)
```markdown

**Batch Editing**

```bash
MultiEdit(
  file_path="/absolute/path/file.py",
  edits=[
    {"original": "old1", "modified": "new1"},
    {"original": "old2", "modified": "new2"}
  ]
)
```markdown

### 13.3 Search Patterns

**Glob Searches**

```bash
Glob(pattern="**/*.py", path="/project/src")
Glob(pattern="src/**/*.{ts,tsx}")           # Brace expansion
Glob(pattern="tests/**/test_*.py")          # Nested matching
```markdown

**Grep Searches**

```bash
Grep(pattern="function\\s+\\w+", type="js") # Regex patterns
Grep(pattern="TODO|FIXME", output_mode="files_with_matches")
Grep(pattern="class\\s+\\w+", path="/src", -n=True, -A=3) # Context
```markdown

### 13.4 Path Handling Rules

**Absolute Paths Required**

- All file operations use absolute paths
- Relative paths NOT supported
- Use environment variables: `$CLAUDE_PROJECT_DIR`
- In scripts: resolve with `pwd` or `realpath`

**Special Directories**

```markdown
$CLAUDE_PROJECT_DIR/.claude/     # Configuration and agents
$CLAUDE_PROJECT_DIR/.coordination/ # Coordination/checkpoints
$CLAUDE_PROJECT_DIR/src/         # Source code
$CLAUDE_PROJECT_DIR/tests/       # Test suites
```markdown

---

## 14. EXECUTION & SCRIPTING

### 14.1 Bash Execution

**Command Execution**

```bash
Bash(command="ls -la /path")
Bash(command="python script.py arg1", cwd="/project")
```bash

**Persistent Session**

- Single bash session across all commands
- Environment variables persist
- Working directory resets between agent calls (⚠️)
- Use absolute paths in scripts

**Timeout Handling**

```json
Default: 30 seconds
Configure: BASH_DEFAULT_TIMEOUT_MS environment variable
Example: export BASH_DEFAULT_TIMEOUT_MS=60000
```json

### 14.2 Background Processes

**Launching Subagents**

```bash
Task(config={
  "agent": "agent-name",
  "task": "description",
  "model": "sonnet"
})
```markdown

**Non-Blocking Retrieval**

```bash
result = TaskOutput(agent_id, block=False, timeout=5000)
# For swarm: immediate non-blocking check
```bash

### 14.3 Scripting Patterns

**Common Script Patterns**

```bash
# Find and process files
files=$(find . -name "*.py" -type f)
for file in $files; do
  echo "Processing: $file"
done

# Use absolute paths
full_path="$(cd "$(dirname "$file")" && pwd)/$(basename "$file")"
```markdown

---

## 15. MONITORING & OBSERVABILITY

### 15.1 Token Usage Tracking

**Usage Monitoring Commands**

```bash
/usage                          # Full token/cost breakdown
/status                         # Current usage percentage
/context                        # Detailed recommendations
```markdown

**Usage Information**

- Current tokens used vs available
- Cost in USD (if configured)
- Percentage toward limits
- Recommendations for optimization

### 15.2 Performance Metrics

**Agent Monitoring**

```bash
/status [--json]                # Running agents, runtimes
```bash

**Output in JSON**

```json
{
  "agents": {
    "running": 2,
    "completed_24h": 15,
    "failed": 0,
    "running_list": [
      {"id": "abc", "name": "agent", "runtime": "5m30s"}
    ]
  }
}
```markdown

### 15.3 Logging & Debug

**Debug Mode**

```bash
claude --debug [filter]         # Enable debug output
claude --verbose                # Detailed turn-by-turn output
```markdown

**Log Files** (when configured)

```markdown
.claude/hooks/context_metrics.log
.claude/hooks/compaction.log
.claude/hooks/checkpoints.log
.claude/hooks/sessions.log
.claude/hooks/errors.log
```markdown

---

## 16. SECURITY & SAFETY

### 16.1 Permission System

**Permission Modes**

```bash
--permission-mode always-allow   # No prompts (⚠️)
--permission-mode require-approval # Every action
--permission-mode reasonable     # Smart defaults
```markdown

**Viewing Permissions**

```bash
/permissions                    # View current settings
/allow <tool>                   # Enable tool
/deny <tool>                    # Disable tool
```markdown

### 16.2 Tool Restrictions

**Restricting Tools**

```bash
--tools "Read,Glob,Grep"        # Only these tools
--disallowedTools "Bash,Write"  # Deny these
--allowedTools "Edit,Read"      # Only allow these
```markdown

**Safe Tool Combinations**

- Read-only: Read, Glob, Grep, LS, WebFetch
- Modify: Read, Write, Edit, MultiEdit
- Execute: Bash (with caution)

### 16.3 Hook Validation

**Path Protection (PreToolUse hook)**

```python
# Validate write paths
if file_path in [".env", "secrets.json", ".git"]:
    exit(1)  # Block operation
```markdown

**Tool Blocking Examples**

```bash
# Block pip usage, require uv
if echo "$COMMAND" | grep -qE '^pip'; then
  exit 1
fi
```markdown

### 16.4 File Security

**Protected Paths** (cannot modify)

```bash
.claude/settings.json
.claude/config.yaml
.claude/template.lock
**/.env*
**/*.key
**/*.pem
**/*credentials*
.git/**
```markdown

**Denied Read Paths** (context protection)

```markdown
.claude/agent-outputs/archive/**
.coordination/archive/**
.git/**
__pycache__/**
.venv/**
```markdown

---

## 17. OUTPUT & FORMATTING

### 17.1 Markdown Support

**Code Blocks**

```python
# Syntax highlighting by language
```bash

**Tables**

| Column 1 | Column 2 |
|----------|----------|
| Value    | Value    |

**Formatting**

- **Bold** for emphasis
- `Inline code` for references
- Lists for organization
- Headings for structure

### 17.2 Terminal Rendering

**Status Indicators**

```markdown
🟢 Healthy - System operating normally
🟡 Warning - Action recommended
🔴 Critical - Immediate attention needed
✓ Success - Operation completed
✗ Failed - Operation failed
```markdown

**Progress Display**

- Spinner for running operations
- Percentage bars for progress
- Status messages for state changes

### 17.3 JSON Output

**JSON Mode**

```bash
claude --json                   # All output as JSON
/status --json                  # JSON for specific command
--output-format json            # Entire session
```json

**JSON Structure**

```json
{
  "status": "success|error",
  "message": "response text",
  "data": {...}
}
```markdown

---

## 18. ENVIRONMENT & PLATFORM

### 18.1 Platform Support

**Supported Platforms**

- macOS (Apple Silicon & Intel)
- Linux (Ubuntu, Debian, RHEL, etc.)
- Windows 10/11 (native & WSL)

**IDE Integration**

- VS Code (native extension)
- VS Code forks (Cursor, Windsurf)
- JetBrains IDEs (IntelliJ, PyCharm, etc.)
- GitHub Copilot (Claude model option)

### 18.2 Environment Detection

**Working Directory**

- Default: Current directory
- Override: `--add-dir <path>`
- In scripts: Absolute paths required (resets between calls)

**Git Repository Detection**

- Auto-detects git repos
- Enables git operations
- Loads .gitignore (respected)
- Access to git history

**Environment Variables**

```markdown
CLAUDE_PROJECT_DIR              # Project root
BASH_DEFAULT_TIMEOUT_MS         # Command timeout (default: 30000)
ANTHROPIC_API_KEY               # API authentication
```markdown

### 18.3 Configuration Detection

**Auto-Loaded on Startup**

1. CLAUDE.md (in memory hierarchy)
2. .claude/rules/ (path-specific rules)
3. ~/.claude/CLAUDE.md (user defaults)
4. ./CLAUDE.local.md (local overrides)
5. .claude/config.yaml (orchestration)
6. .claude/settings.json (Claude settings)

---

## 19. DOCUMENTATION & HELP

### 19.1 Built-in Help System

**Getting Help**

```bash
/help                           # All available commands
/help <command>                 # Specific command help
/help git                       # Git workflow help
claude --help                   # CLI help
```markdown

**Command Help Format**

- Usage syntax
- Available arguments/flags
- Examples
- Related commands

### 19.2 Documentation Sources

**Official Documentation**

- code.claude.com/docs - Main documentation
- Claude blog for feature announcements
- GitHub issues for known issues

**Project Documentation** (.claude/)

```markdown
.claude/README.md              # Configuration guide
.claude/GETTING_STARTED.md     # Quick start
.claude/commands/*.md          # Command reference
.claude/agents/*.md            # Agent documentation
.claude/docs/glossary.md       # Terminology
```markdown

---

## 20. ADVANCED & EXPERIMENTAL FEATURES

### 20.1 Model Context Protocol (MCP)

**MCP Servers**

```bash
claude mcp                      # Configure MCP servers
```bash

**Use Cases**

- Database integrations
- API connections
- File system operations
- Custom tools

**Configuration**

- Servers added via `claude mcp`
- Auto-discovered in projects
- Available to Claude in sessions

### 20.2 Browser Integration

**Chrome Automation**

```bash
--chrome                        # Enable browser control
```markdown

**Capabilities**

- Navigate websites
- Interact with web apps
- Screenshot pages
- Fill forms

### 20.3 IDE Integration Details

**Language Server Protocol (LSP)**

- Code intelligence features
- Go-to-definition
- Find references
- Hover documentation
- Real-time validation

**VS Code Integration**

- Visual diffs for changes
- Integrated terminal
- File tree navigation
- Git integration

### 20.4 Experimental/Unreleased Features

**Status Monitoring**

- Use `/feedback` to report issues
- Check GitHub issues for experimental features
- Follow Claude Code changelog for updates

**Recent 2026 Additions**

- Named session support (`/rename`)
- LSP tool integration
- Enhanced auto-compact thresholds
- Stats command (`/stats`)

---

## 21. KEYBOARD SHORTCUTS & HOTKEYS

### 21.1 REPL Shortcuts

**Navigation**

```markdown
Ctrl+C                         # Stop current operation
Ctrl+D                         # Exit REPL
ESC ESC                        # Open rewind/checkpoint menu
```markdown

**History**

```markdown
Up Arrow                       # Previous command
Down Arrow                     # Next command
Ctrl+R                        # Search history
```markdown

### 21.2 Editor Integration

**VS Code**

```markdown
Ctrl+Shift+L                  # Open Claude Code
Cmd+K Cmd+C                   # Show Claude chat
```markdown

**Cursor**

```markdown
Cmd+K (macOS)                 # Open Claude Code
Ctrl+K (Windows/Linux)
```markdown

---

## 22. BEST PRACTICES & GUIDANCE

### 22.1 Context Optimization

**DO**

- Monitor context usage regularly
- Create checkpoints before major tasks
- Archive completed outputs
- Use .claudeignore for large files
- Summarize long outputs to files
- Keep CLAUDE.md focused and updated

**DON'T**

- Exceed context thresholds before checkpointing
- Leave large tool outputs in conversation
- Load unnecessary files into context
- Create intermediate analysis files
- Ignore compaction warnings

### 22.2 Agent Coordination

**DO**

- Write clear completion reports
- Preserve artifacts on filesystem
- Use dynamic agent discovery
- Checkpoint between batches
- Monitor agent limits

**DON'T**

- Hardcode routing tables
- Spawn unlimited agents
- Ignore blocked/needs-review status
- Skip validation checks
- Wait for all agents before retrieving outputs

### 22.3 Hook Development

**DO**

- Keep hooks focused and fast
- Log important events
- Use absolute paths
- Test hooks before deployment
- Document hook behavior in comments

**DON'T**

- Create long-running hooks
- Write large outputs
- Assume relative paths
- Ignore timeout settings
- Add verbose logging in production

### 22.4 Session Management

**DO**

- Name important sessions
- Create checkpoints before risky operations
- Use `/status` to monitor health
- Review context before major tasks
- Archive completed work

**DON'T**

- Let sessions grow indefinitely without checkpoints
- Ignore context warnings
- Mix unrelated tasks in single session
- Rely on checkpoints instead of Git

---

## 23. CONFIGURATION EXAMPLES

### 23.1 Project-Specific Setup

**.claude/config.yaml Example**

```yaml
orchestration:
  agents:
    max_concurrent: 2
    recommended_batch_size: 1
  context:
    warning_threshold: 60
    checkpoint_threshold: 70
    critical_threshold: 85
  checkpointing:
    enabled: true
    auto_checkpoint: true

retention:
  agent_outputs: 14
  reports: 14

enforcement:
  agent_limit: true
  path_validation: true
  context_monitoring: true
```bash

**CLAUDE.md Example**

```markdown
# Project: Oscura

## Architecture
- Signal: Time-series container
- Loader: Format parser
- Analyzer: Measurement extractor
- Decoder: Protocol decoder

## Development Workflow
- Use ./scripts/test.sh for tests
- Use ./scripts/check.sh for linting
- Follow conventional commit format
- Update CHANGELOG.md on every PR

## Coding Standards
- Python 3.12+, follow PEP 8
- Use type hints everywhere
- Test coverage minimum 80%
- Docstrings for public APIs
```bash

**Hook Configuration Example**

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [{
          "type": "command",
          "command": "python3 .claude/hooks/validate_path.py",
          "timeout": 5
        }]
      }
    ],
    "SessionStart": [
      {
        "matcher": "auto",
        "hooks": [{
          "type": "command",
          "command": "python3 .claude/hooks/health_check.py",
          "timeout": 5
        }]
      }
    ]
  }
}
```markdown

---

## 24. TROUBLESHOOTING GUIDE

### 24.1 Common Issues

**Context Approaching Limits**

```bash
Problem: "Warning: Context usage at 65%"
Solution:
1. Run /context for recommendations
2. Create checkpoint: /checkpoint
3. Archive outputs: /cleanup
4. Manual compact if needed: /compact
```bash

**Agent Not Responding**

```bash
Problem: Agent seems stuck
Solution:
1. Check /status for agent status
2. Review .claude/reports/ for errors
3. Check logs: .claude/hooks/errors.log
4. Force-stop if needed (Ctrl+C)
```bash

**Hook Timeout**

```python
Problem: Hook hangs or times out
Solution:
1. Check hook script in .claude/hooks/
2. Verify script logic and loops
3. Increase timeout if legitimate
4. Add early exit conditions
```python

**Checkpoint Restore Failed**

```python
Problem: Cannot restore from checkpoint
Solution:
1. Check disk space (at least 5% free)
2. Verify checkpoint files exist
3. Use /rewind to see available checkpoints
4. Manual restore from .claude/summaries/ if needed
```markdown

**Permission Denied**

```markdown
Problem: "Permission denied for tool X"
Solution:
1. Check /permissions
2. Grant permission: /allow <tool>
3. Or adjust --permission-mode
4. Review .claude/settings.json
```markdown

### 24.2 Performance Issues

**Slow Command Execution**

```bash
Cause: Large file reads/writes
Solution:
1. Check file sizes with Glob
2. Use offset/limit for large files
3. Split edits into smaller chunks
4. Archive old outputs
```bash

**High Memory Usage**

```markdown
Cause: Too many agents or large outputs
Solution:
1. Reduce max_concurrent in config
2. Enable auto-summarization
3. Checkpoint more frequently
4. Clear summaries: /cleanup
```markdown

---

## 25. RESOURCES & REFERENCES

### 25.1 Official Documentation Links

**Core Documentation**

- Claude Code Docs: https://code.claude.com/docs
- CLI Reference: https://code.claude.com/docs/en/cli-reference
- Memory Management: https://code.claude.com/docs/en/memory
- Hooks Guide: https://code.claude.com/docs/en/hooks-guide
- Checkpointing: https://code.claude.com/docs/en/checkpointing

**Integration Docs**

- MCP Integration: https://code.claude.com/docs/en/mcp
- IDE Integration: https://code.claude.com/docs/en/ides
- GitHub Actions: https://code.claude.com/docs/en/github-actions

### 25.2 Community Resources

**External Guides**

- GitHub Awesome Claude Code: https://github.com/hesreallyhim/awesome-claude-code
- ClaudeLog Guides: https://claudelog.com/
- Claude Code Cheatsheets: Multiple community resources
- Medium Articles: Deep dives on specific features

### 25.3 Getting Help

**Support Channels**

```markdown
/feedback                      # Report bugs or request features
GitHub Issues                  # Known issues and discussions
Claude forum                   # Community discussions
```markdown

---

## APPENDIX A: QUICK REFERENCE TABLES

### Model Aliases

| Alias | Model |
|-------|-------|
| haiku | claude-haiku-4-5-20251001 |
| sonnet | claude-sonnet-4-20250514 |
| opus | claude-opus-4-5-20251101 |

### Hook Events Quick Reference

| Event | When | Can Block | Use Case |
|-------|------|-----------|----------|
| PreToolUse | Before tool call | Yes | Validation, logging |
| PostToolUse | After tool completes | No | Formatting, cleanup |
| PreCompact | Before compaction | Yes | Save state |
| SessionStart | Session begins | No | Load context |
| SessionEnd | Session ends | No | Cleanup |
| SubagentStop | Agent finishes | Yes | Output validation |

### Threshold Quick Reference

| Threshold | Percentage | Action |
|-----------|-----------|--------|
| Warning | 60% | Review context, consider summarizing |
| Checkpoint | 65% | Create checkpoint |
| Critical | 75% | Complete task only, then checkpoint |

### Retention Defaults

| Item | Days |
|------|------|
| Agent outputs | 7 |
| Checkpoints | 7 |
| Summaries | 30 |
| Logs | 14 |
| Archives | 180 |

---

## APPENDIX B: GLOSSARY

**Agent** - Specialized Claude Code persona with specific tools and responsibilities (code_reviewer, knowledge_researcher, etc.)

**Checkpoint** - Saved state of code and conversation, restorable via /rewind

**Completion Report** - JSON summary of agent work for orchestrator handoff

**Compaction** - Automatic context summarization when approaching token limits

**Hook** - Script that runs at lifecycle events (PreToolUse, SessionStart, etc.)

**MCP** - Model Context Protocol for extending Claude Code capabilities

**REPL** - Read-Eval-Print Loop; interactive terminal mode

**Swarm Mode** - Parallel execution of multiple agents for independent subtasks

**Token** - Smallest unit of text processed by Claude (~4 chars)

**Working Directory** - Current directory context for file operations

---

## APPENDIX C: FILE STRUCTURE

**Standard Claude Code Project Structure**

```bash
.claude/
├── config.yaml                 # Orchestration configuration
├── settings.json              # Claude Code settings (generated)
├── CLAUDE.md                  # Project memory/instructions
├── GETTING_STARTED.md         # Quick start guide
├── README.md                  # Configuration guide
├── paths.yaml                 # Path definitions
├── project-metadata.yaml      # Project identity
├── coding-standards.yaml      # Code quality rules
├── agents/                    # Custom agents
│   ├── orchestrator.md
│   ├── code_assistant.md
│   └── ...
├── commands/                  # Custom slash commands
│   ├── help.md
│   ├── status.md
│   └── ...
├── docs/                      # Project documentation
│   ├── glossary.md
│   └── routing-concepts.md
├── hooks/                     # Hook implementations
│   ├── validate_path.py
│   ├── health_check.py
│   └── ...
├── rules/                     # Path-specific rules
│   ├── testing.md
│   ├── security.md
│   └── ...
├── agent-outputs/             # Agent completion reports (gitignored)
│   ├── [timestamp]-complete.json
│   └── archive/
├── summaries/                 # Output summaries (gitignored)
├── reports/                   # Analysis reports (gitignored)
└── hooks/                     # Hook logs
    ├── errors.log
    ├── enforcement.log
    └── ...

.coordination/                 # Coordination data (gitignored)
├── active_work.json
├── checkpoints/               # Session checkpoints
├── locks/                     # Coordination locks
└── ...
```markdown

---

## KEY FINDINGS: CLAUDE.MD CONTEXT BEHAVIOR

### Critical Discovery

**When is CLAUDE.md read into context?**

- ✅ **At session start ONLY** - loaded when you launch Claude Code or start new conversation
- ✅ **NOT reloaded during conversation** - single load per session
- ✅ **NOT reloaded after compaction** - by design

**Does CLAUDE.md persist through compaction?**

- ✅ **YES** - CLAUDE.md is stored separately from conversation history
- ✅ **Compaction only affects conversation messages and tool outputs**
- ✅ **Project instructions remain available throughout entire session**

**Is there a reload command?**

- ❌ **NO explicit reload command exists**
- ✅ **By design** - since CLAUDE.md persists, reloading would bloat context unnecessarily
- ✅ **To force reload**: Start new session (`claude --new`) or restart Claude Code

**Official guidance from documentation:**
> "Put persistent rules in CLAUDE.md rather than relying on conversation history"

This is specifically because CLAUDE.md survives compaction while conversation history does not.

### Should You Create a Custom /refresh Command?

**Recommendation: Partially useful**

A custom `/refresh-context` command would be useful for:

- ✅ Triggering manual compaction via `/cleanup`
- ✅ Running validation hooks to enforce standards
- ✅ Displaying context metrics via `/context`
- ✅ Creating psychological checkpoint for standards review

But NOT necessary for:

- ❌ Re-reading CLAUDE.md (already persists automatically)
- ❌ Forcing reload (would waste tokens)

**Your existing setup already handles this well through:**

1. Hook system (PreCompact, SessionStart, PreToolUse hooks)
2. Context thresholds (Warning: 60%, Checkpoint: 65%, Critical: 75%)
3. Built-in commands (`/context`, `/cleanup`, `/status`)

**Bottom Line:**
CLAUDE.md is NOT forgotten after compaction - it's foundational context that persists throughout your session. You don't need to force re-reading. However, a command that compacts + validates standards could still be useful as a "reset point" during long sessions.

---

**END OF COMPREHENSIVE REFERENCE GUIDE**

---

## Document Metadata

- **Created**: 2026-01-22
- **Source**: Official Claude Code documentation research
- **Sections**: 25 main sections + 3 appendices
- **Coverage**: 50+ CLI commands, 15 tools, 11 hook types, 100+ config settings
- **Verified Against**: Claude Code docs, community resources, official guides

---

## Sources

This comprehensive guide was compiled from the following official Claude Code documentation and verified resources:

- [Claude Code CLI Reference](https://code.claude.com/docs/en/cli-reference)
- [Claude Code Memory Management](https://code.claude.com/docs/en/memory)
- [Claude Code Hooks Guide](https://code.claude.com/docs/en/hooks-guide)
- [Claude Code Checkpointing](https://code.claude.com/docs/en/checkpointing)
- [Claude Code Built-in Tools Reference](https://www.vtrivedy.com/posts/claudecode-tools-reference)
- [GitHub - awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code)
- [ClaudeLog - Comprehensive Claude Code Documentation](https://claudelog.com/claude-code-changelog/)
- [Manage Claude's Memory - Claude Code Docs](https://code.claude.com/docs/en/memory)
- [Claude Code Context Recovery - Medium](https://medium.com/@sonuyadav1/claude-code-context-recovery-stop-losing-progress-when-context-compacts-772830ee7863)
- [How I Use Every Claude Code Feature - Blog](https://blog.sshh.io/p/how-i-use-every-claude-code-feature)
- [The Ultimate Claude Code Cheat Sheet - Medium](https://medium.com/@tonimaxx/the-ultimate-claude-code-cheat-sheet-your-complete-command-reference-f9796013ea50)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
