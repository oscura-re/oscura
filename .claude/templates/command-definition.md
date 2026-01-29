# Command Definition Template

**Version**: 2.0.0
**Created**: 2026-01-16
**Updated**: 2026-01-22
**Purpose**: Standardized template for all slash commands in `.claude/commands/`

---

## Frontmatter (REQUIRED)

Every command file MUST start with YAML frontmatter:

```yaml
---
name: command_name # REQUIRED: Command name (no / prefix)
description: Brief one-line description # REQUIRED: Shown in /help listing
arguments: [arg1, arg2, --flag] # REQUIRED: Use [] for optional, <> for required
version: 1.0.0 # REQUIRED: Command version (semver)
created: 2026-01-22 # REQUIRED: ISO-8601 date
updated: 2026-01-22 # REQUIRED: ISO-8601 date (same as created initially)
status: stable # REQUIRED: stable|beta|experimental
target_agent: agent_name # OPTIONAL: Primary agent or "none" for utility commands
---
```

### Argument Notation

- `[arg]` - Optional argument
- `<arg>` - Required argument
- `--flag` - Optional flag
- `[arg1, arg2, ...]` - Multiple optional args
- `<arg1> [arg2]` - Required + optional

**Examples**:

- `arguments: [path]` → `/review [path]`
- `arguments: <agent> <task>` → `/route <agent> <task>`
- `arguments: [--json, --verbose]` → `/status [--json] [--verbose]`

---

## Standard Sections (REQUIRED)

### 1. Title and Brief Description

```markdown
# /command_name - One-Line Description

Brief paragraph (1-3 sentences) explaining what this command does.
```

### 2. Usage

Show all variations with clear syntax:

````markdown
## Usage

\```bash
/command # Basic usage
/command <required> # With required arg
/command [optional] # With optional arg
/command --flag # With flag
\```
````

### 3. Purpose

Explain **when** and **why** to use this command:

```markdown
## Purpose

This command is designed for:

- Use case 1
- Use case 2
- Use case 3

**When to use**: Specific scenarios
**When NOT to use**: Anti-patterns
```

### 4. Examples

Provide **concrete, working examples** with expected output:

````markdown
## Examples

### Example 1: Basic Usage

\```bash
/command basic
\```

**Output**:
\```
Expected output here
\```

**What happened**: Explain the result

### Example 2: Advanced Usage

\```bash
/command --advanced arg
\```

**Result**: Describe outcome
````

### 5. Arguments and Options

Document all arguments and flags:

```markdown
## Arguments

| Argument | Type   | Required | Default | Description     |
| -------- | ------ | -------- | ------- | --------------- |
| `arg1`   | string | Yes      | -       | Purpose of arg1 |
| `arg2`   | path   | No       | `.`     | Purpose of arg2 |

## Options

| Flag        | Description               |
| ----------- | ------------------------- |
| `--json`    | Output as JSON            |
| `--verbose` | Show detailed information |
```

### 6. How It Works (Optional but Recommended)

Explain the implementation/workflow:

````markdown
## How It Works

\```
User invokes /command
↓
System validates arguments
↓
Execute core logic
↓
Return results
\```

**Internal flow**:

1. Step 1: What happens
2. Step 2: What happens next
3. Step 3: Final result
````

### 7. Error Handling

Document common errors and solutions:

```markdown
## Error Handling

### Error 1: Description

**Symptom**: What user sees
**Cause**: Why it happens
**Solution**: How to fix

### Error 2: Description

**Symptom**: Error message
**Cause**: Root cause
**Solution**: Resolution steps
```

### 8. Configuration (If Applicable)

Link to relevant config settings:

````markdown
## Configuration

Behavior controlled by `.claude/config.yaml`:

\```yaml
section:
subsection:
setting: value # Description
\```

**Configurable options**:

- `setting1`: Purpose and default value
- `setting2`: Purpose and default value
````

### 9. Related Commands (REQUIRED)

List related commands in a table:

```markdown
## Related Commands

| Command   | Purpose              | When to Use Instead    |
| --------- | -------------------- | ---------------------- |
| `/other1` | Alternative approach | When X condition       |
| `/other2` | Complementary        | Use after this command |
```

**Ensure bidirectional links**: If command A links to B, B must link to A.

### 10. See Also (REQUIRED)

Link to related documentation:

```markdown
## See Also

- `.claude/docs/relevant-doc.md` - Detailed concepts
- `.claude/agents/agent.md` - Related agent
- `.claude/commands/related.md` - Related command
- `CLAUDE.md` - Project workflow
```

**Ensure all links are valid** and point to existing files.

### 11. Version History (REQUIRED)

Track changes over time:

```markdown
## Version History

- **v1.2.0** (2026-01-25): Added fuzzy matching support
- **v1.1.0** (2026-01-20): Added --verbose flag
- **v1.0.0** (2026-01-15): Initial release
```

---

## Optional Sections (Use When Appropriate)

### Pro Tips

```markdown
## Pro Tips

### 1. Tip Title

Explanation and example

### 2. Another Tip

Explanation and example
```

### Comparison with Alternatives

```markdown
## Comparison

| Aspect     | This Command | Alternative       |
| ---------- | ------------ | ----------------- |
| Speed      | Fast         | Slow              |
| Complexity | Simple       | Complex           |
| Use Case   | Quick tasks  | Complex workflows |
```

### Workflow Integration

```markdown
## Workflow Integration

This command fits into workflows:

**Common patterns**:

1. `/cmd1` → `/command` → `/cmd3`
2. `/command` (standalone)
3. Part of automated scripts
```

### Advanced Usage

````markdown
## Advanced Usage

### Scripting

\```bash

# Example script using this command

/command arg1 && /command arg2
\```

### Automation

Integration with CI/CD or other tools.
````

---

## Style Guidelines

### Writing Style

- **Be concise**: Users want answers fast
- **Use active voice**: "Show status" not "Status is shown"
- **Include examples**: Show, don't just tell
- **Explain WHY**: Don't just document WHAT

### Formatting

- **Use headings consistently**: Follow the template hierarchy
- **Code blocks**: Use \`\`\`bash for commands, specify language
- **Tables**: Use markdown tables for structured data
- **Lists**: Use `-` for unordered, `1.` for ordered
- **Emphasis**: Use **bold** for important terms, `code` for commands/paths

### Cross-References

- **Link generously**: Help users navigate documentation
- **Use relative paths**: `.claude/docs/file.md` not absolute paths
- **Verify links**: All links must resolve to existing files
- **Bidirectional**: If A links to B, ensure B links back to A

### Examples Quality

- **Real examples**: Use actual command syntax that works
- **Expected output**: Show what users will see
- **Context**: Explain what the example demonstrates
- **Variety**: Cover common, edge, and advanced cases

---

## Validation Checklist

Before committing a command file, verify:

- ✅ Frontmatter is complete and valid YAML
- ✅ All REQUIRED sections are present
- ✅ Arguments match frontmatter notation
- ✅ All code blocks have language specified
- ✅ All links resolve to existing files
- ✅ Related Commands section has bidirectional links
- ✅ Version history is up to date
- ✅ Examples are tested and accurate
- ✅ No typos or grammatical errors
- ✅ Formatting is consistent with template

---

## Template Usage

### Creating New Command

1. Copy this template
2. Replace `command_name` with actual command name
3. Fill in all REQUIRED sections
4. Add optional sections as needed
5. Validate using checklist above
6. Test command works as documented
7. Commit

### Updating Existing Command

1. Read current command file
2. Update content
3. Increment version in frontmatter
4. Add entry to Version History section
5. Update `updated` date in frontmatter
6. Validate using checklist
7. Commit

---

## Example Command File

See `.claude/commands/help.md` for a complete example following this template.

---

## Related Templates

- `.claude/templates/agent-definition.md` - Agent template
- `.claude/templates/completion-report.md` - Completion report template

---

## Version History

- **v2.0.0** (2026-01-22): Comprehensive update with all sections, style guidelines, and validation checklist
- **v1.0.0** (2026-01-16): Initial template creation
