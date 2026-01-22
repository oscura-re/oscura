---
name: help
description: List available commands and show usage information
arguments: [command]
---

# Help Command

Display available slash commands with descriptions and usage examples.

## Usage

```bash
/help              # List all available commands
/help <command>    # Show detailed help for specific command
```python

## Examples

```bash
/help              # Show all commands
/help git          # Detailed help for /git command
/help swarm        # Detailed help for /swarm command
```bash

## Available Commands

### Core Commands

- `/help [command]` - Show this help or detailed command usage
- `/status [--json]` - Show orchestration health, running agents, context usage
- `/context` - Display context usage and optimization recommendations
- `/cleanup [--dry-run]` - Run maintenance tasks (archive old files, clean stale agents)

### Domain Commands

- `/git [message]` - Smart atomic commits with conventional format
- `/swarm <task>` - Execute complex tasks with parallel agent coordination

## Command Categories

**Task Orchestration**: `/swarm`, `/route`
**Development Workflow**: `/git`
**System Management**: `/status`, `/context`, `/cleanup`, `/help`

## Getting Detailed Help

For detailed help on any command, use:

```bash
/help <command-name>
```markdown

This will show:

- Full command syntax
- Available arguments and options
- Usage examples
- Related commands

## See Also

- `.claude/commands/` - All command definitions
- `.claude/agents/` - Available agents
- `.claude/docs/routing-concepts.md` - How routing works
- `.claude/GETTING_STARTED.md` - Introduction to the orchestration system

## Version History

- v1.0.0 (2026-01-16): Initial creation with command listing and detailed help
