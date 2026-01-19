# Oscura

Unified hardware reverse engineering framework. Extract all information from any system through signals and data.

## Tech Stack

Python 3.12+, numpy, pytest, ruff, uv, hypothesis

## What is Oscura

Hardware reverse engineering framework for security researchers, right-to-repair advocates, defense analysts, and commercial intelligence teams. Four core capabilities:

1. **Unknown protocol discovery** - Automatic protocol inference, state machine extraction
2. **System replication** - Reverse engineer proprietary devices for understanding/repair
3. **Security analysis** - Vulnerability discovery, CRC/checksum recovery, exploitation
4. **Signal analysis** - IEEE-compliant measurements, comprehensive protocol decoding

**Reverse Engineering**: Unknown protocols • State machines • CRC recovery • Device replication • Security vulnerabilities
**Signal Analysis**: Waveform/spectral/power analysis • 16+ protocol decoders • IEEE standards (181/1241/1459/2414)
**Built For**: Exploitation • Replication • Defense • Commercial intelligence • Right-to-repair

## Project Structure

```
src/oscura/          # Source code
tests/                 # Test suite (unit, integration)
demos/                 # Working demonstrations with validation
docs/                  # User documentation
scripts/               # Development utilities (test, check, fix)
.claude/               # Claude agents, commands, config
```

## Core Abstractions

- **Signal**: Time-series container (channels, sample rate, metadata)
- **Loader**: Parse file formats → Signal objects
- **Analyzer**: Signal → Measurements (dict of named values)
- **Protocol Decoder**: Signal → Frame objects (decoded data)

**Pattern**: Inherit from base classes, implement required methods. See existing implementations for patterns.

## Development Workflow

### Setup

```bash
uv sync --all-extras                 # Install all dependencies
./scripts/setup/install-hooks.sh     # Install git hooks (REQUIRED)
uv run pytest tests/unit -x          # Verify installation
```

### Quality Checks

```bash
./scripts/test.sh                    # Run tests (auto-parallel, optimal config)
./scripts/test.sh --fast             # Quick tests without coverage
./scripts/check.sh                   # All quality checks (ruff, mypy, markers)
./scripts/fix.sh                     # Auto-fix issues
./scripts/pre-push.sh --full         # Full CI validation
```

**IMPORTANT**: Use these scripts (SSOT for test configuration). Don't run pytest/ruff/mypy manually.

**Quality Check Strategy**:

- **Pre-commit hooks** (`.pre-commit-config.yaml`): Run on `git commit` for fast, essential checks (yaml, json, trailing whitespace, etc.)
- **./scripts/check.sh**: Manual quality verification (ruff, mypy, shellcheck, markdown, etc.) - run during development
- **./scripts/pre-push.sh**: Comprehensive CI simulation - run before pushing (includes all checks + tests + build)
- **Purpose**: Pre-commit catches basic issues early; check.sh for development iteration; pre-push for final verification

### Commit Workflow

- **Format**: Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`)
- **MUST** update `CHANGELOG.md` under `[Unreleased]` section (see protocol below)
- Pre-commit hooks run automatically (lint, format, type check)

### Changelog and Versioning Protocol

**CRITICAL**: Follow this protocol for ALL development work.

#### Version Management

- **Current version**: Check `pyproject.toml` [project.version]
- **Versioning scheme**: Semantic Versioning (MAJOR.MINOR.PATCH)
  - MAJOR: Breaking API changes
  - MINOR: New features (backward compatible)
  - PATCH: Bug fixes only
- **DO NOT bump version per PR** - accumulate changes in [Unreleased]
- **Bump version only when preparing a release** (separate commit + git tag)

#### CHANGELOG.md Update (Required for Every PR)

**Format**: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

Every PR MUST update CHANGELOG.md under `## [Unreleased]`:

**Sections**:

- `### Added` - New features, capabilities, files
- `### Changed` - Changes to existing functionality
- `### Fixed` - Bug fixes
- `### Removed` - Removed features/files
- `### Infrastructure` - CI/CD, tooling, dependencies

**Entry format**:

```markdown
- **Feature Name** (`path/to/file.py`):
  - Concise description of what it does
  - Key capabilities (bullet points)
  - Test count: X/X tests passing
  - Example location (if applicable)
```

**What to document**:

- ✅ New modules, classes, major functions
- ✅ Breaking changes, API modifications
- ✅ Fixed bugs, resolved issues
- ✅ New dependencies, CI/CD changes
- ❌ Refactoring (unless affects API)
- ❌ Minor code cleanup, comment updates
- ❌ Test-only changes (unless infrastructure)

**Release process**:

1. Accumulate changes in `[Unreleased]` during development
2. When ready: rename `[Unreleased]` → `[X.Y.Z] - YYYY-MM-DD`
3. Bump version in `pyproject.toml`
4. Create git tag `vX.Y.Z`
5. Push tag to trigger release workflow
6. Deploy documentation: `mike deploy X.Y.Z latest --update-aliases && mike set-default latest`

#### Documentation Versioning

- **Tool**: Mike (MkDocs versioning plugin)
- **Strategy**: Documentation is versioned per release (e.g., 0.1.2, 0.2.0)
- **Latest**: The most recent stable release is aliased as 'latest'
- **Deployment**: Automated via GitHub Actions on tag push
- **Manual deploy**: `mike deploy <version> latest --update-aliases`
- **Site URL**: https://oscura-re.github.io/oscura

### Test Strategy

- **Unit tests**: Algorithm correctness, use `SignalBuilder` fixtures
- **Integration tests**: Edge cases ONLY (not workflows covered by demos)
- **Demo validation**: Demos serve as living integration tests (run `demos/validate_all_demos.py`)

See `docs/testing/test-suite-guide.md` for complete strategy.

## Key Conventions

- **Tests**: Use fixtures from `tests/conftest.py`, synthetic data only (<100KB)
- **Naming**: `snake_case` files/functions, `PascalCase` classes, `SCREAMING_SNAKE_CASE` constants
- **Standards**: Follow IEEE standards where applicable (181, 1241, 1459, 2414)
- **Commits**: Update CHANGELOG.md in every PR (see protocol above)

## Workspace File Creation Policy

**CRITICAL**: Do NOT create intermediate reports, summaries, or analysis files in version-controlled workspace.

### Allowed File Creation

✅ **User-facing documentation**: README, CONTRIBUTING, docs/, tutorials
✅ **Source code**: src/, tests/, configuration files
✅ **CI/CD**: .github/workflows/, scripts/

### Forbidden File Patterns

❌ **Intermediate files**: `*_ANALYSIS*.md`, `*_REPORT*.md`, `*_AUDIT*.md`, `*_FIXES*.md`, `*_SUMMARY*.md`
❌ **Working papers**: `COMPREHENSIVE_*.md`, `FINAL_*.md`, `COMPLETE_*.md`

### Where to Put Working Papers

- **Analysis/research**: `.claude/reports/YYYY-MM-DD-topic.md` (gitignored, auto-archived)
- **Agent outputs**: `.claude/agent-outputs/[id]-complete.json` (gitignored)
- **Coordination**: `.coordination/` (gitignored, auto-cleaned)
- **User communication**: Communicate directly, do NOT create file

### Single Source of Truth

| Information Type | SSOT Location | NOT Here |
|-----------------|---------------|----------|
| What changed when | `CHANGELOG.md` | `*_CHANGES.md` |
| How to develop | `CONTRIBUTING.md`, `CLAUDE.md` | `DEVELOPMENT_GUIDE.md` |
| Project info | `README.md` | `PROJECT_SUMMARY.md` |
| Badge maintenance | `.github/BADGE_MAINTENANCE.md` | `BADGE_GUIDE.md` |

**See**: `.claude/WORKSPACE_POLICY.md` for complete policy and decision trees.

## Quick Reference

| Task | Command/Location |
|------|------------------|
| Run tests | `./scripts/test.sh` |
| Quality checks | `./scripts/check.sh` |
| Auto-fix issues | `./scripts/fix.sh` |
| Add loader | Check `src/oscura/loaders/vcd.py` for pattern |
| Add analyzer | Check `src/oscura/analyzers/` + tests |
| Add protocol | Check `src/oscura/analyzers/protocols/uart.py` |
| Test fixtures | `tests/conftest.py`, `tests/fixtures/signal_builders.py` |
| Code style | `.claude/coding-standards.yaml` (SSOT) |
| Git workflow | `CONTRIBUTING.md` |
| Version info | `pyproject.toml` [project.version] |

## Extended Documentation

- **Testing**: `docs/testing/test-suite-guide.md` - Comprehensive test strategy
- **Contributing**: `CONTRIBUTING.md` - Git workflow, PR process, changelog protocol
- **Examples**: `examples/` - Working code organized by category
- **User docs**: `docs/` - Getting started, guides, API reference
- **Claude agents**: `.claude/agents/` - Available specialized agents
- **Claude commands**: `.claude/commands/` - Slash commands reference
- **Project config**: `.claude/config.yaml` - Agent orchestration settings
- **Glossary**: `.claude/docs/glossary.md` - Terminology definitions

## Where Things Live

| Need | Location |
|------|----------|
| Add file format | `src/oscura/loaders/` |
| Add measurement | `src/oscura/analyzers/` |
| Add protocol decoder | `src/oscura/analyzers/protocols/` |
| Working examples | `examples/` (organized by category) |
| Test data generation | `scripts/test-data/generate_comprehensive_test_data.py` |
| Coding standards | `.claude/coding-standards.yaml` |

When uncertain about implementation, **examine existing similar code** in `src/oscura/` for established patterns.
