# /review - Code Quality Review

Force code review workflow for quality, security, and best practices auditing.

## Purpose

Bypass intelligent routing and go directly to `code_reviewer` agent for comprehensive code quality analysis, security auditing, and architectural feedback.

## Usage

```bash
/review [path]                              # Review file or directory
/review src/auth/jwt.py                     # Review specific file
/review src/api/                            # Review directory
/review                                     # Review recent changes
/review --security src/auth/                # Security-focused review
```

## When to Use

✅ **Use /review when**:

- Checking code quality before commit/PR
- Security audit of authentication/authorization code
- Architectural review of new features
- Finding potential bugs or issues
- Getting feedback on code style
- Pre-deployment quality check

❌ **Don't use /review when**:

- Just want to write code → Use `/code` or `/feature`
- Need documentation → Use `/doc`
- Need research → Use `/research`
- Want to implement something → Use `/code` or `/feature`

## How It Works

```
/review [path]
  ↓
Force route to code_reviewer (bypass orchestrator)
  ↓
code_reviewer analyzes code for:
  - Code quality
  - Security vulnerabilities
  - Best practices
  - Performance issues
  - Architectural concerns
  ↓
Returns: Detailed review with recommendations
```

## Examples

### Example 1: File Review

```bash
/review src/auth/jwt.py
```

Returns:

```markdown
# Code Review: src/auth/jwt.py

## Summary

Overall quality: B+ (Good, minor improvements needed)

## Security Analysis ⚠️

1. **MEDIUM**: JWT secret stored in config file
   - Recommendation: Use environment variable or secrets manager
   - Line 15: `SECRET_KEY = config['jwt_secret']`

## Code Quality ✓

1. **Good**: Type hints used consistently
2. **Good**: Comprehensive docstrings
3. **Minor**: Function too long (45 lines)
   - Recommendation: Split `validate_token()` into smaller functions

## Performance

1. **Minor**: Unnecessary regex compilation in loop
   - Line 67: Move `re.compile()` outside loop

## Best Practices ✓

1. **Good**: Error handling is comprehensive
2. **Good**: Follows project coding standards

## Recommendations

1. Move JWT secret to environment variable (HIGH PRIORITY)
2. Refactor `validate_token()` for better readability
3. Cache regex compilation

## Grade: B+ (85/100)
```

### Example 2: Directory Review

```bash
/review src/api/
```

Returns overview of all files with:

- Critical issues summary
- Per-file grades
- Common patterns (good and bad)
- Architectural recommendations

### Example 3: Security-Focused Review

```bash
/review --security src/
```

Returns focused security audit:

- SQL injection risks
- XSS vulnerabilities
- Authentication/authorization issues
- Secrets in code
- Insecure dependencies

## Review Categories

code_reviewer analyzes:

### 1. Security ⚠️

- Injection vulnerabilities (SQL, XSS, command)
- Authentication/authorization flaws
- Secrets in code
- Insecure cryptography
- OWASP Top 10 issues

### 2. Code Quality

- Type safety
- Error handling
- Code organization
- Naming conventions
- Documentation quality

### 3. Performance

- Algorithmic efficiency
- Database query optimization
- Caching opportunities
- Resource leaks
- Bottlenecks

### 4. Best Practices

- Design patterns
- SOLID principles
- DRY violations
- Code duplication
- Project standards compliance

### 5. Architecture

- Module organization
- Dependency management
- Separation of concerns
- Scalability concerns

## After Using /review

What you can do next:

1. **Fix issues** - `/code fix [issue]`
2. **Refactor** - `/code refactor [file] according to review`
3. **Commit** - `/git "address code review findings"`
4. **Re-review** - `/review [path]` after changes
5. **Document** - `/doc document [complex logic]`

## Review Flags

Customize review focus:

```bash
/review --security <path>     # Security-focused
/review --performance <path>  # Performance-focused
/review --style <path>        # Style/standards focused
/review --architecture <path> # Architectural review
/review --all <path>          # Comprehensive (default)
```

## Review Output Format

```markdown
# Code Review: <File/Directory>

## Summary

Overall assessment and grade

## Critical Issues 🔴

Must-fix issues (security, bugs)

## Major Issues 🟡

Should-fix issues (quality, performance)

## Minor Issues 🟢

Nice-to-have improvements (style, refactoring)

## Positive Highlights ✓

What's done well

## Recommendations

Prioritized action items

## Detailed Analysis

Per-file or per-section breakdown

## Grade: X (score/100)
```

## Severity Levels

|Level|Icon|Description|Action|
|---|---|---|---|
|**Critical**|🔴|Security vulnerabilities, bugs|Fix immediately|
|**Major**|🟡|Quality issues, performance|Fix soon|
|**Minor**|🟢|Style, optimization|Fix when convenient|
|**Info**|ℹ️|Suggestions, best practices|Consider|

## Integration with Workflow

### Before Commit

```
/code implement feature
  ↓
/review src/feature.py
  ↓
Fix issues
  ↓
/git "add feature with review feedback"
```

### Before PR

```
/feature implement authentication
  ↓
/review src/auth/
  ↓
Address findings
  ↓
Create PR
```

### Security Audit

```
/review --security src/
  ↓
Identify vulnerabilities
  ↓
/code fix [security issues]
  ↓
/review --security src/  (verify fixed)
```

## What Gets Reviewed

code_reviewer checks:

✅ **Source Code**:

- Python files (.py)
- Configuration files
- SQL queries
- API endpoints

✅ **Patterns**:

- Common anti-patterns
- Security vulnerabilities
- Performance issues
- Code smells

❌ **NOT Reviewed**:

- Binary files
- Generated code
- Third-party libraries
- Test fixtures (unless requested)

## Comparison with Other Tools

|Tool|Purpose|Focus|Output|
|---|---|---|---|
|`/review`|Code quality|Comprehensive|Review report|
|`ruff`|Linting|Syntax/style|Error list|
|`mypy`|Type checking|Type safety|Type errors|
|`pytest`|Testing|Functionality|Test results|

`/review` complements these tools with human-like analysis and architectural feedback.

## Configuration

Review behavior in `.claude/config.yaml`:

```yaml
agents:
  code_reviewer:
    strictness: normal # or "lenient", "strict"
    focus_areas:
      - security
      - quality
      - performance
    ignore_patterns:
      - 'tests/**' # Don't review tests
      - '**/__init__.py' # Skip empty inits
    minimum_grade: B # Warn if below
```

## When to Request Formal Review

Use `/review` before:

- **Committing** sensitive code (auth, payments, etc.)
- **Creating PR** for complex features
- **Deploying** to production
- **Merging** to main branch
- **Refactoring** critical systems

## Workflow

```
/review → code_reviewer (direct, no routing)
        → Read files
        → Analyze patterns
        → Check security
        → Assess quality
        → Generate report with grades
```

## Aliases

The following aliases work identically:

- `/audit` → `/review`
- `/check` → `/review`
- `/analyze` → `/review`

## Best Practices

### Good Review Requests:

```bash
/review src/auth/jwt.py                    # Specific file
/review src/api/ --security                 # Security focus
/review                                     # Recent changes
/review src/ --architecture                 # Architectural review
```

### After Implementing Code:

```bash
/code implement user validation
  ↓
/review [generated file]
  ↓
Fix issues from review
  ↓
/git "add user validation"
```

## Agent

Routes to: **code_reviewer** (always, no routing logic)

For normal routing behavior, see `.claude/docs/routing-concepts.md`

## See Also

- `.claude/docs/routing-concepts.md` - How routing works
- `.claude/agents/code_reviewer.md` - Agent details
- `.claude/commands/agents.md` - All available agents

## Version

v1.0.0 (2026-01-09) - Initial creation as part of workflow command system
v1.1.0 (2026-01-16) - Added cross-references to routing concepts
