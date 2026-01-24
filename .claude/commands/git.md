---
name: git
description: Smart atomic commits with conventional format
arguments: [message]
---

# Git Workflow

Commit and push changes with intelligent grouping and conventional commit format.

## Usage

````bash
/git                    # Auto-generate commit message(s)
/git "custom message"   # Use provided message
/git --push             # Commit and push
```markdown

## Process

1. Invoke git_commit_manager agent
2. Fetch remote and check status
3. Analyze all changes (staged and unstaged)
4. Group changes by domain/type
5. Create atomic conventional commits
6. Push if requested

## Conventional Commit Types

|Type|Description|
|---|---|
|feat|New feature|
|fix|Bug fix|
|docs|Documentation|
|refactor|Code restructure|
|test|Tests|
|chore|Maintenance|

## Examples

```bash
/git                              # Auto-commit with smart grouping
/git "feat(auth): add OAuth support"  # Specific message
/git --push                       # Commit and push
```markdown

## Notes

- Changes automatically grouped by type and scope
- Multiple commits created if changes span different domains
- Never force pushes without explicit request
- Validates conventional commit format

## See Also

- `.claude/agents/git_commit_manager.md` - Primary agent handling this command
- `.claude/docs/routing-concepts.md` - How routing works
- `.claude/commands/agents.md` - All available agents

## Version History

- v1.0.0 (2026-01-16): Initial creation with atomic commits and conventional format
````
