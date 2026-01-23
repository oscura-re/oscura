---
name: review
description: Comprehensive code review for quality, security, and best practices
arguments: [path, --security, --verbose]
version: 1.0.0
created: 2026-01-22
updated: 2026-01-22
status: stable
target_agent: code_reviewer
---

# /review - Comprehensive Code Review

Perform thorough code review focusing on quality, security, maintainability, and adherence to project standards.

## Usage

````bash
/review [path]                  # Review changed files or specified path
/review src/module.py           # Review specific file
/review src/                    # Review directory
/review --security              # Focus on security issues
/review --verbose               # Detailed findings
```markdown

## Purpose

This command routes to the **code_reviewer** agent for:

- Pull request quality gates (pre-merge reviews)
- Security vulnerability detection
- Code quality assessment
- Best practices enforcement
- Performance bottleneck identification
- Test coverage analysis
- Maintainability scoring

**When to use**:
- Before merging pull requests
- Pre-commit quality checks
- Security audits for sensitive code
- Periodic code audits (monthly/quarterly)
- After major feature implementation

**When NOT to use**:
- Just want code written → Use natural language or code_assistant
- Just need documentation → Use technical_writer
- Git operations → Use `/git`

## Examples

### Example 1: Review Current Changes

```bash
/review
```bash

**Output**:
```bash
Code Review: src/auth/jwt.py (+127, -42 lines)

QUALITY (Score: 8.5/10)
✓ Type hints present and correct
✓ Docstrings follow Google style
✓ Function length appropriate (<50 lines)
⚠ Cyclomatic complexity: 12 (target: <10) in verify_token()

SECURITY (Score: 7/10)
✓ No hardcoded secrets
✓ Input validation on all user inputs
✗ CRITICAL: JWT signature not verified before decoding (jwt.py:45)
⚠ Missing rate limiting on token endpoint

MAINTAINABILITY (Score: 9/10)
✓ Clear separation of concerns
✓ Minimal coupling
✓ Good error messages

TEST COVERAGE
✓ Unit tests present: 12/12 functions
⚠ Missing edge case: expired token with valid signature

RECOMMENDATIONS
1. [CRITICAL] Add signature verification before decode
2. [HIGH] Reduce complexity in verify_token() - extract helper
3. [MEDIUM] Add rate limiting middleware
4. [LOW] Add edge case test for expired+valid tokens
```markdown

### Example 2: Security-Focused Review

```bash
/review src/auth/ --security
```bash

**Result**:
```bash
Security Review: src/auth/ (5 files, 847 lines)

CRITICAL ISSUES (2)
1. JWT signature not verified (jwt.py:45)
   Risk: Authentication bypass
   Fix: Use decode(verify=True, algorithms=['HS256'])

2. SQL query uses string interpolation (users.py:78)
   Risk: SQL injection
   Fix: Use parameterized queries

HIGH ISSUES (3)
[... detailed security findings ...]
```markdown

### Example 3: Specific File Review

```bash
/review src/oscura/analyzers/protocols/uart.py --verbose
```markdown

**Returns**:
- Detailed quality metrics
- Line-by-line analysis
- Specific improvement suggestions
- Code examples for fixes

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `path` | file/dir | No | git diff | File, directory, or use git changes |

## Options

| Flag | Description |
|------|-------------|
| `--security` | Focus on security vulnerabilities only |
| `--verbose` | Show detailed findings with code examples |

## How It Works

```bash
/review [path]
  ↓
Route to code_reviewer agent
  ↓
1. Scope Analysis: Identify files, load standards
  ↓
2. Quality Check: Naming, types, docs, complexity
  ↓
3. Security Scan: Vulnerabilities, input validation
  ↓
4. Performance Analysis: Algorithm efficiency, bottlenecks
  ↓
5. Test Coverage: Unit/integration, edge cases
  ↓
6. Maintainability: Coupling, cohesion, patterns
  ↓
Return comprehensive review with severity levels
```markdown

## Review Output Format

Reviews include:

1. **Executive Summary**: Overall scores and critical issues
2. **Quality Assessment**: Readability, complexity, documentation
3. **Security Findings**: Vulnerabilities categorized by severity
4. **Performance Analysis**: Bottlenecks and optimization opportunities
5. **Test Coverage**: Missing tests and edge cases
6. **Recommendations**: Prioritized action items

## Severity Levels

| Level | Description | Action Required |
|-------|-------------|-----------------|
| **CRITICAL** | Security vulnerabilities, data loss risks | Fix before merge |
| **HIGH** | Major quality issues, performance problems | Fix soon |
| **MEDIUM** | Code smells, maintainability concerns | Address in next sprint |
| **LOW** | Style issues, minor improvements | Nice to have |

## Quality Standards

Reviews check against `.claude/coding-standards.yaml`:

- ✅ **Naming**: snake_case functions, PascalCase classes
- ✅ **Type hints**: Present and correct
- ✅ **Docstrings**: Google style with examples
- ✅ **Function length**: < 50 lines preferred
- ✅ **Complexity**: Cyclomatic complexity < 10
- ✅ **DRY**: No code duplication
- ✅ **Error handling**: Actionable error messages

## Security Checks

All reviews scan for:

- Hardcoded secrets (API keys, passwords, tokens)
- Input validation on user inputs
- SQL injection vulnerabilities
- Path traversal risks
- Authentication/authorization enforcement
- HTTPS for external connections
- Dependency vulnerabilities

## Error Handling

### Path Not Found

```bash
/review nonexistent/path.py
```markdown

**Response**:
```markdown
Error: Path not found: nonexistent/path.py
Did you mean: /review src/path.py?
```markdown

### No Changes Detected

```bash
/review
```bash

(when working directory is clean)

**Response**:
```bash
No changes detected in git diff.
Specify a path: /review <path>
Or use: /review src/ to review specific directory
```python

## Configuration

Review behavior controlled in `.claude/config.yaml`:

```yaml
orchestration:
  agents:
    code_reviewer:
      model: sonnet                  # Fast, thorough reviews
      max_complexity: 10             # Cyclomatic complexity threshold
      min_test_coverage: 80          # Minimum test coverage %
```markdown

## Related Commands

| Command | Purpose | When to Use Instead |
|---------|---------|---------------------|
| `/review` | Code quality review | Pre-merge quality gate |
| `/ai write code` | Implement features | Need code written |
| `/git` | Create commits | Commit reviewed code |
| `/route code_reviewer <task>` | Manual routing | Force specific review |

## Workflow Integration

Common patterns:

1. **Code → Review → Commit**:
   ```bash
   /ai implement user authentication
   # Review implementation
   /review src/auth/
   # Fix critical issues
   /git "feat: add user authentication"
```bash

2. **PR Review**:
   ```bash
   /review
   # Address findings
   /review  # Re-review after fixes
   /git "fix: address review findings"
```markdown

3. **Security Audit**:
   ```bash
   /review src/ --security
   # Fix vulnerabilities
   /review src/ --security  # Verify fixes
```bash

## Pro Tips

### 1. Review Before Commit

Always review significant changes before committing:

```bash
/review && /git "feat: new feature"
```python

### 2. Focus Reviews

Use flags to narrow scope:

```bash
/review src/auth/ --security       # Security-only
/review src/complex.py --verbose   # Detailed analysis
```markdown

### 3. Incremental Reviews

Review small chunks frequently rather than large batches:

```bash
# Good: Review one module
/review src/module.py

# Avoid: Review entire codebase
/review src/  # Too broad, less actionable
```markdown

## Comparison

| Approach | Speed | Depth | Best For |
|----------|-------|-------|----------|
| `/review` | Medium | High | Pre-merge quality gate |
| `/review --security` | Fast | Security | Security audits |
| `/review --verbose` | Slow | Maximum | Complex modules |

## See Also

- `.claude/agents/code_reviewer.md` - Full agent capabilities
- `.claude/coding-standards.yaml` - Project coding standards
- `.claude/commands/git.md` - Commit reviewed code
- `.claude/commands/route.md` - Manual routing control
- `CONTRIBUTING.md` - Development workflow
- `CLAUDE.md` - Project overview

## Version History

- **v1.0.0** (2026-01-22): Initial creation with routing to code_reviewer agent
````
