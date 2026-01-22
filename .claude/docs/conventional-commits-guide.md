# Conventional Commits Guide

Comprehensive guide to conventional commit format and examples for the Oscura project.

## Format Specification

```python
<type>[optional scope][optional !]: <description>

[optional body]

[optional footer(s)]
```python

**Rules**:
- Subject line ≤ 50 characters (enforced)
- Body lines wrapped at 72 characters (enforced)
- Imperative mood: "add" not "added" or "adds"
- Lowercase description (no capital first letter)
- No trailing period on subject line
- Body separated from subject by blank line
- Footer separated from body by blank line

## Type Reference

| Type | Usage | When to Use |
|------|-------|-------------|
| `feat` | New feature | Adding new functionality, capabilities, modules |
| `fix` | Bug fix | Correcting errors, fixing issues |
| `docs` | Documentation | README, guides, docstrings, comments |
| `test` | Tests | Adding or modifying tests |
| `refactor` | Code change (no behavior change) | Restructuring without changing external API |
| `perf` | Performance improvement | Optimizations, efficiency improvements |
| `build` | Build system | Dependencies, build scripts, tooling |
| `ci` | CI/CD changes | GitHub Actions, workflows, automation |
| `chore` | Maintenance | Routine tasks, cleanup, tooling updates |
| `style` | Code style | Formatting, whitespace (no logic change) |
| `revert` | Revert previous commit | Undoing changes |

## Scope Examples

Common scopes by area:

**Source code**:
- `loaders` - Signal loaders
- `analyzers` - Signal analyzers
- `protocols` - Protocol decoders
- `core` - Core abstractions (Signal, Frame, etc.)
- `cli` - Command-line interface

**Infrastructure**:
- `agents` - Claude agent definitions
- `commands` - Slash commands
- `docs` - Documentation
- `tests` - Test suite
- `scripts` - Development scripts
- `ci` - CI/CD workflows

**Configuration**:
- `deps` - Dependencies
- `config` - Configuration files
- `hooks` - Git hooks

## Breaking Changes

Indicate breaking changes in two ways:

1. **Exclamation mark** after type/scope:
```markdown
   feat(api)!: change parameter order in process_signal
```markdown

2. **BREAKING CHANGE footer**:
```markdown
   feat(api): change parameter order in process_signal

   BREAKING CHANGE: process_signal now takes sample_rate as first param
```markdown

## Comprehensive Examples

### feat (New Feature)

**Simple feature**:
```bash
feat(loaders): add support for CSV file format

Implements CSV loader with configurable delimiter and header options.
Includes comprehensive tests and documentation.
```bash

**Feature with breaking change**:
```bash
feat(core)!: change Signal constructor signature

Signal now requires explicit channel names for clarity.
Old: Signal(data, sample_rate)
New: Signal(data, sample_rate, channel_names)

BREAKING CHANGE: All Signal instantiations must provide channel_names
```bash

**Complex feature with body**:
```markdown
feat(protocols): add I2C protocol decoder

Implements I2C protocol decoder supporting:
- Standard mode (100 kbit/s)
- Fast mode (400 kbit/s)
- Start/stop condition detection
- ACK/NACK handling
- Address and data parsing

Includes 15 tests covering all modes and edge cases.
Documented in docs/protocols/i2c.md with usage examples.
```markdown

### fix (Bug Fix)

**Simple fix**:
```bash
fix(analyzers): correct FFT frequency bin calculation

Off-by-one error in bin indexing causing incorrect peak detection.
Fixes #123
```bash

**Fix with detailed explanation**:
```bash
fix(loaders): handle malformed VCD timestamp entries

VCD files with missing timescale directive now default to 1ns
instead of crashing. Added validation for all timestamp formats.

Fixes #456
```markdown

**Performance fix**:
```markdown
fix(core): optimize Signal slicing memory usage

Reduced memory consumption by 40% using views instead of copies
for slicing operations. Maintains same API semantics.
```python

### docs (Documentation)

**Simple documentation**:
```python
docs(guides): add API authentication tutorial

Step-by-step guide for implementing JWT authentication
with practical examples and security best practices.
```python

**API documentation**:
```python
docs(api): improve Signal class docstrings

Added examples for all public methods and clarified
parameter types. Includes usage patterns for common scenarios.
```python

**Documentation fix**:
```python
docs(readme): fix installation instructions

Corrected Python version requirement from 3.11 to 3.12.
Updated dependency installation command to use uv.
```python

### test (Testing)

**New tests**:
```python
test(protocols): add edge case tests for UART decoder

Tests for malformed frames, parity errors, and buffer overflows.
Increases coverage from 85% to 95%.
```python

**Test refactoring**:
```python
test(analyzers): refactor FFT tests to use fixtures

Converted 12 tests to use parametrized fixtures reducing
duplication by 60% and improving maintainability.
```python

**Integration tests**:
```python
test(integration): add end-to-end workflow tests

Tests complete workflow from loading VCD file through
protocol decoding to result validation. Covers 5 common
use cases from demos/.
```markdown

### refactor (Code Refactoring)

**Simple refactoring**:
```python
refactor(core): simplify data processing pipeline

Reduces complexity without changing external API.
Improves performance by 15% and readability.
```python

**Major refactoring**:
```python
refactor(analyzers): extract common FFT functionality

Created shared FFTAnalyzer base class to reduce duplication
across spectral analyzers. All existing tests pass unchanged.
```markdown

**Architecture improvement**:
```markdown
refactor(loaders): standardize error handling

Unified error handling across all loaders to raise
consistent exception types with actionable messages.
```bash

### chore (Maintenance)

**Dependency updates**:
```bash
chore(deps): update dependencies to latest versions

- numpy 1.24 -> 1.26
- scipy 1.10 -> 1.11
- pytest 7.4 -> 8.0

All tests pass. No breaking changes.
```bash

**Tooling updates**:
```bash
chore(tools): update ruff configuration for 0.2.0

Added new lint rules for Python 3.12 compatibility.
Fixed 8 new violations in existing code.
```bash

**Cleanup**:
```bash
chore(scripts): remove deprecated test helper functions

Removed legacy test utilities replaced by pytest fixtures.
No tests reference these functions anymore.
```python

### build (Build System)

**Build configuration**:
```python
build(pyproject): configure package for PyPI publication

Added project metadata, classifiers, and entry points.
Configured build backend for wheel generation.
```python

**Build tooling**:
```python
build(uv): migrate from pip to uv for faster installs

Updated all scripts to use uv. Installation time
reduced from 45s to 8s on CI.
```python

### ci (CI/CD)

**Workflow updates**:
```python
ci(tests): add parallel test execution

Tests now run in parallel across 4 workers reducing
CI time from 5min to 2min. All tests remain isolated.
```python

**CI fixes**:
```python
ci(coverage): fix coverage reporting for integration tests

Configured coverage to properly track integration test
execution. Coverage increased from 85% to 92%.
```markdown

**New workflow**:
```markdown
ci(release): add automated release workflow

Workflow triggered on version tag push:
- Runs full test suite
- Builds wheels
- Publishes to PyPI
- Creates GitHub release
```markdown

### Multiple Files Example

**Grouped by domain**:
```bash
feat(agents): add code_reviewer agent with command

Implements comprehensive code review agent supporting:
- Quality assessment (complexity, duplication)
- Security scanning (OWASP Top 10)
- Standards enforcement
- Severity-based findings (CRITICAL/HIGH/MEDIUM/LOW)

Added /review command for direct invocation.
Includes 8 tests and complete documentation.

Files changed:
- .claude/agents/code_reviewer.md (new)
- .claude/commands/review.md (new)
```markdown

## Multi-Commit Strategy

For complex changes spanning multiple domains:

**Commit 1 - Core implementation**:
```bash
feat(loaders): add HDF5 loader with compression support
```bash

**Commit 2 - Tests**:
```bash
test(loaders): add comprehensive HDF5 loader tests

Tests for compressed/uncompressed files, corrupt data,
and all supported HDF5 dataset formats.
```bash

**Commit 3 - Documentation**:
```markdown
docs(loaders): document HDF5 loader usage and limitations

Added guide covering supported formats, compression options,
and performance characteristics.
```markdown

**Commit 4 - Integration**:
```markdown
feat(cli): add HDF5 format to CLI loader options

Updated CLI to auto-detect HDF5 files and use new loader.
```markdown

## Anti-Patterns

### Generic Messages (BAD)
```markdown
❌ update files
❌ fix stuff
❌ changes
❌ work in progress
❌ misc updates
```markdown

### Multiple Concerns (BAD)
```markdown
❌ feat: add CSV loader, fix UART bug, update docs

Should be 3 separate commits:
✓ feat(loaders): add CSV loader
✓ fix(protocols): correct UART parity calculation
✓ docs(protocols): update UART documentation
```markdown

### No Scope When Needed (BAD)
```markdown
❌ feat: add new feature
❌ fix: fix bug

Should specify scope:
✓ feat(loaders): add VCD loader
✓ fix(analyzers): correct FFT windowing
```markdown

### Subject Line Too Long (BAD)
```markdown
❌ feat(loaders): add comprehensive CSV loader with support for multiple delimiters and header configurations

Should be concise (≤50 chars):
✓ feat(loaders): add CSV loader with configurable options

Details go in body:
feat(loaders): add CSV loader with configurable options

Supports multiple delimiters, header detection, and custom
column types. Includes validation and error handling.
```markdown

## Commit Message Checklist

Before committing, verify:

- [ ] Type is appropriate for change (feat/fix/docs/etc.)
- [ ] Scope accurately identifies affected area
- [ ] Subject line ≤ 50 characters
- [ ] Subject uses imperative mood ("add" not "added")
- [ ] Subject starts with lowercase
- [ ] No period at end of subject line
- [ ] Body explains "why" not just "what" (if needed)
- [ ] Body lines wrapped at 72 characters
- [ ] Breaking changes marked with `!` or `BREAKING CHANGE:`
- [ ] Issue references included if applicable (`Fixes #123`)
- [ ] Commit represents single logical change

## Tools and Validation

### Pre-commit Hooks

Automated validation enforces:
- Conventional commit format
- Subject line length (≤50)
- Body line wrapping (≤72)
- No trailing whitespace
- Valid type and scope

### Manual Validation

```bash
# Check recent commit messages
git log --oneline -10

# Check specific commit format
git log -1 --pretty=format:"%s"

# Lint commit messages
npx commitlint --from HEAD~1
```bash

## See Also

- **Specification**: [Conventional Commits](https://www.conventionalcommits.org/)
- **Tool**: [commitlint](https://commitlint.js.org/)
- **Project Guide**: `CONTRIBUTING.md` - Full development workflow
- **Agent**: `git_commit_manager` - AI assistant for git commits
