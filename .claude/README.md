# Claude Code Configuration

This directory contains the Claude Code configuration for the Oscura project.

## Attribution Policy

**IMPORTANT**: This project uses `includeCoAuthoredBy: false` in `settings.json`.

### Why?

This is an **open-source project** where contributors should receive proper credit for their work. When using Claude Code as a development tool:

- **Contributors** (human developers) should be credited as sole authors
- **AI tools** (like Claude) are development assistants, not co-authors
- Commit authorship should reflect who is responsible for the contribution

### For All Contributors

If you're using Claude Code to contribute to this project:

1. The configuration is already set correctly in `.claude/settings.json`
2. Your commits will be attributed to you alone
3. Claude will never add `Co-authored-by` trailers to commit messages
4. This ensures proper credit and accountability in the project history

### Technical Details

The setting `"includeCoAuthoredBy": false` in `settings.json` disables AI attribution globally for this project. This applies to:

- Direct commits via git commands
- Pull request creation
- Any git operations performed through Claude Code

## For Maintainers

When reviewing PRs or commits:

- Verify that commits do NOT contain `Co-authored-by: Claude` trailers
- If you see such trailers, they indicate misconfiguration and should be removed
- Report any configuration issues in the repository

## Git Workflow

See `.claude/agents/git_commit_manager.md` and `.claude/commands/git.md` for the complete git workflow and commit conventions used in this project.
