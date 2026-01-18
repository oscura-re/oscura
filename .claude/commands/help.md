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
```

## Examples

```bash
/help              # Show all commands
/help ai           # Detailed help for /ai command
/help git          # Detailed help for /git command
```

## Available Commands

### Core Commands

- `/ai <task>` - Universal routing to specialized agents for any task
- `/help [command]` - Show this help or detailed command usage
- `/status [--json]` - Show orchestration health, running agents, context usage
- `/context` - Display context usage and optimization recommendations
- `/cleanup [--dry-run]` - Run maintenance tasks (archive old files, clean stale agents)

### Domain Commands

- `/git [message]` - Smart atomic commits with conventional format
- `/swarm <task>` - Execute complex tasks with parallel agent coordination
- `/research <topic>` - Web research with citations
- `/review [path]` - Code quality review

## Command Categories

**Task Orchestration**: `/ai`, `/swarm`
**Development Workflow**: `/git`, `/research`, `/review`
**System Management**: `/status`, `/context`, `/cleanup`, `/help`

## Getting Detailed Help

For detailed help on any command, use:

```bash
/help <command-name>
```

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
