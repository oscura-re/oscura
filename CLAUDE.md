# Oscura - Hardware Reverse Engineering Framework

**Tech Stack**: Python 3.12+, numpy, pytest, ruff, mypy, uv, hypothesis

---

## CRITICAL - QUALITY ENFORCEMENT

### Pre-Commit Requirements (MANDATORY)

BEFORE ANY git commit, MUST execute in order:

1. `python3 .claude/hooks/validate_all.py` → MUST show 5/5 passing
2. `./scripts/check.sh` → MUST pass all quality checks
3. IF any validation fails → BLOCK commit, fix errors first
4. NEVER commit with failing validators or tests

### Code Quality Gates (AUTOMATIC)

AFTER writing ANY code (>20 lines), MUST:

1. Run linter: `uv run ruff check <file>`
2. Run type checker: `uv run mypy <file> --strict`
3. IF Python file → MUST show 0 errors before proceeding
4. IF shell script → run `shellcheck <file>`
5. NEVER skip validation "because it's a small change"

---

## AUTOMATIC BEHAVIORS

### Agent Orchestration (SPAWN PROACTIVELY)

AUTOMATICALLY spawn agents when:

- **After writing >50 lines code** → spawn `code_reviewer` (MANDATORY)
- **After completing feature** → spawn `code_reviewer` for security/quality review
- **When creating documentation** → spawn `technical_writer` for clarity
- **When research needed** → spawn `knowledge_researcher` for investigation
- **Multi-step task detected** → spawn `orchestrator` for coordination

DO NOT spawn agents for:

- Simple file edits (<20 lines)
- Reading/exploring codebase (use Glob/Grep directly)
- Answering questions
- Running existing commands

### Changelog Updates (MANDATORY EVERY CHANGE)

EVERY code change MUST update `CHANGELOG.md`:

1. Add entry under `## [Unreleased]` section
2. Use category: `### Added`, `### Changed`, `### Fixed`, or `### Removed`
3. Format: `- **Component** (path/to/file.py): Description with test count`
4. NEVER skip changelog - even for "minor" changes
5. IF unsure of category → ask user first

### Validation Timing (AUTOMATIC)

Run validators AUTOMATICALLY:

- ALWAYS before any git commit
- ALWAYS after modifying `.claude/` directory
- ALWAYS after writing >20 lines of code
- WHEN uncertain → run validators (no harm in extra validation)

---

## WORKFLOW PATTERNS

### Pattern: Implementing New Feature

1. Read existing code in relevant directory for established patterns
2. Implement following project conventions (see CONVENTIONS section)
3. Write tests in `tests/unit/` or `tests/integration/`
4. AUTOMATICALLY spawn `code_reviewer` when code >50 lines
5. Update `CHANGELOG.md` under `### Added`
6. Run `./scripts/test.sh` (SSOT for test config)
7. Run `python3 .claude/hooks/validate_all.py` before commit
8. MUST achieve 5/5 passing validators

### Pattern: Fixing Bug

1. Identify root cause, create minimal reproduction
2. Fix with minimal code changes
3. Add regression test if coverage <80%
4. Update `CHANGELOG.md` under `### Fixed`
5. Run `./scripts/test.sh` to verify fix
6. Run validators before commit

### Pattern: Adding Documentation

1. AUTOMATICALLY spawn `technical_writer` agent for clarity
2. Follow Google docstring style (see coding-standards.yaml)
3. Include code examples in docstrings
4. Validate all markdown links before commit
5. Update `CHANGELOG.md` if user-facing

### Pattern: Code Review

1. ALWAYS spawn `code_reviewer` after writing >50 lines
2. Review security, performance, maintainability
3. Address all findings before commit
4. Re-run validators after changes

---

## PROJECT LAYOUT

```text
src/oscura/              # Source code (loaders, analyzers, protocols)
  loaders/               # File format parsers (VCD, WAV, etc.)
  analyzers/             # Signal analysis (waveform, spectral, protocols)
    protocols/           # Protocol decoders (UART, SPI, I2C, etc.)
tests/                   # Test suite
  unit/                  # Unit tests (algorithm correctness)
  integration/           # Integration tests (edge cases only)
  conftest.py            # Shared fixtures (MUST use these)
  fixtures/              # Test data builders
demos/                   # Working demonstrations with validation (33+ demos)
  01_waveform_analysis/  # Basic oscilloscope file loading (2 demos)
  02_file_format_io/     # VCD, CSV, HDF5 formats (1 demo)
  03_custom_daq/         # Memory-efficient loaders (3 demos)
  04_serial_protocols/   # Manchester, I2S, JTAG, USB (6 demos)
  05_protocol_decoding/  # Multi-protocol decoder (1 demo)
  06_udp_packet_analysis/ # PCAP network analysis (1 demo)
  07_protocol_inference/ # CRC, state machines, dissectors (3 demos)
  08_automotive_protocols/ # FlexRay, LIN (2 demos)
  09_automotive/         # OBD-II, UDS, J1939 (1 demo)
  10_timing_measurements/ # IEEE 181 pulse (1 demo)
  11_mixed_signal/       # Analog + digital (1 demo)
  12_spectral_compliance/ # IEEE 1241 FFT/THD (1 demo)
  13_jitter_analysis/    # IEEE 2414 jitter (2 demos)
  14_power_analysis/     # IEEE 1459 power (2 demos)
  15_signal_integrity/   # TDR, S-params (3 demos)
  16_emc_compliance/     # CISPR/IEC EMC (1 demo)
  17_signal_reverse_engineering/ # Complete RE workflow (3 demos)
  18_advanced_inference/ # ML/Bayesian (3 demos)
  19_complete_workflows/ # End-to-end pipelines (3 demos)
  common/                # Shared utilities (BaseDemo, ValidationSuite)
examples/                # High-level workflow examples (6 files)
docs/                    # User documentation
scripts/                 # Development utilities
  test.sh                # Run tests (SSOT for pytest config)
  check.sh               # All quality checks (MUST use before commit)
  fix.sh                 # Auto-fix linting issues
  pre-push.sh            # Full CI validation
.claude/                 # Claude Code orchestration
  agents/                # 6 available agents
  commands/              # 10 slash commands
  hooks/                 # Enforcement and validation
  config.yaml            # Orchestration settings (max_concurrent: 3)
  coding-standards.yaml  # Code style SSOT
```

---

## TOOL COMMANDS

### Testing (SSOT: ./scripts/test.sh)

```bash
./scripts/test.sh                    # Run tests (auto-parallel, coverage)
./scripts/test.sh --fast             # Quick tests without coverage
uv run pytest tests/unit -x          # Run unit tests only
```

**CRITICAL**: NEVER run `pytest` directly - always use scripts (SSOT for config)

### Quality Checks

```bash
./scripts/check.sh                   # All quality checks (MUST before commit)
./scripts/fix.sh                     # Auto-fix linting issues
uv run ruff check <file>             # Lint specific file
uv run mypy <file> --strict          # Type check specific file
shellcheck <file>.sh                 # Check shell scripts
```

### Validation

```bash
python3 .claude/hooks/validate_all.py  # MUST run before every commit
```

**MUST show**: `5/5 validators passing` before commit allowed

### Setup

```bash
./scripts/setup.sh                   # Complete setup (dependencies + hooks)
uv sync --all-extras                 # Install all dependencies
```

---

## CONVENTIONS

### Code Style (SSOT: .claude/coding-standards.yaml)

- **Files/functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Line length**: 100 characters
- **Docstrings**: Google style
- **Type hints**: Required (mypy --strict)

### Testing

- MUST use fixtures from `tests/conftest.py`
- Synthetic data only (<100KB)
- ALWAYS run via `./scripts/test.sh` (SSOT)
- DO NOT run `pytest` directly (wrong config)

### Commits

- **Format**: Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`)
- **MUST** update `CHANGELOG.md` every commit
- Pre-commit hooks run automatically

### CI/CD Pipeline

- **Pre-commit hooks**: Run automatically on `git commit` (format, lint, type check)
- **PR CI**: Runs comprehensive checks (tests, quality, security) on every pull request
- **Merge queue**: Validates final merge commit before landing on main (prevents untested commits)
- **Nightly tests**: Comprehensive test suite with chunked execution
- **Release automation**: Triggered by git tags (`vX.Y.Z`) - publishes to PyPI, deploys docs

**DO NOT** manually modify GitHub Actions workflows without understanding the full CI pipeline.
Workflows are in `.github/workflows/` - examine existing workflows before creating new ones.

### Domain-Specific

- Follow IEEE standards where applicable: 181, 1241, 1459, 2414
- Protocol decoders: inherit from base classes
- Signal analyzers: return `dict[str, Any]` measurements

---

## DECISION TREES

### When to Spawn Agent vs Handle Directly

```text
IF writing >50 lines code
  → AUTOMATICALLY spawn code_reviewer
ELSE IF creating documentation
  → AUTOMATICALLY spawn technical_writer
ELSE IF research/investigation needed
  → spawn knowledge_researcher
ELSE IF multi-step coordination needed
  → spawn orchestrator
ELSE IF simple edit (<20 lines) OR exploring codebase
  → Handle directly (no agent)
```

### When to Run Validators

```text
ALWAYS before git commit (MANDATORY)
ALWAYS after modifying .claude/ directory
ALWAYS after writing >20 lines
IF uncertain → run validators
NEVER skip "because it's a small change"
```

### Tool Selection

```text
File search → Glob (NOT find/ls via Bash)
Content search → Grep (NOT grep/rg via Bash)
Read files → Read tool (NOT cat via Bash)
Edit files → Edit tool (NOT sed via Bash)
Write files → Write tool (NOT echo via Bash)
Type check → uv run mypy --strict
Lint → uv run ruff check
Format → uv run ruff format
Tests → ./scripts/test.sh (SSOT)
```

---

## ANTI-PATTERNS (DO NOT)

### File Operations

- ❌ DO NOT use Bash for file operations (cat, sed, grep, find)
- ❌ DO NOT run pytest directly (use ./scripts/test.sh)
- ❌ DO NOT create intermediate report files in workspace
- ❌ DO NOT skip validators "because changes are small"
- ❌ DO NOT commit without CHANGELOG update

### Code Quality

- ❌ DO NOT write code without reading existing patterns first
- ❌ DO NOT skip type hints (mypy --strict required)
- ❌ DO NOT exceed 100 character line length
- ❌ DO NOT add features beyond what was requested (YAGNI)
- ❌ DO NOT create abstractions for one-time operations

### Git Workflow

- ❌ DO NOT commit with failing tests
- ❌ DO NOT commit with failing validators
- ❌ DO NOT skip CHANGELOG updates
- ❌ DO NOT force push to main/master
- ❌ DO NOT amend commits from other developers

---

## SSOT LOCATIONS

| Information      | Authoritative Source                    | NOT Here                 |
| ---------------- | --------------------------------------- | ------------------------ |
| Version          | `pyproject.toml` [project.version]      | README, docs             |
| Changes history  | `CHANGELOG.md`                          | git log, commit messages |
| Dependencies     | `pyproject.toml` [project.dependencies] | requirements.txt         |
| Test config      | `pyproject.toml` [tool.pytest]          | pytest.ini               |
| Code style       | `.claude/coding-standards.yaml`         | inline comments          |
| Project metadata | `.claude/project-metadata.yaml`         | multiple files           |
| Orchestration    | `.claude/config.yaml`                   | agent files              |

**Rule**: NEVER duplicate SSOT content - always reference authoritative source

---

## CLAUDE CODE INTEGRATION

### Available Agents (6)

1. **code_assistant** - Ad-hoc code writing
2. **code_reviewer** - Quality/security review (spawn after writing code)
3. **git_commit_manager** - Conventional commits
4. **knowledge_researcher** - Research and investigation
5. **orchestrator** - Multi-agent coordination
6. **technical_writer** - Documentation creation

### Slash Commands (10)

- `/status` - System health and running agents
- `/agents` - List available agents
- `/help` - Show all commands
- `/research` - Web research with citations → knowledge_researcher
- `/review` - Code review → code_reviewer
- `/git` - Smart commits → git_commit_manager
- `/cleanup` - Clean orchestration artifacts
- `/context` - Display context usage
- `/route` - Force route to specific agent
- `/swarm` - Parallel agent coordination

### Configuration (.claude/config.yaml)

- **max_concurrent**: 3 agents simultaneously
- **fuzzy_matching**: 80% threshold (handles typos)
- **auto_checkpoint**: 65% context usage
- **validators**: 5 comprehensive checks

### Enforcement (Automatic via Hooks)

- PreToolUse: Blocks agent spawning if ≥3 running
- Path validation: Blocks denied reads/writes (see config.yaml)
- SSOT validation: Prevents configuration duplication

---

## CONTEXT MANAGEMENT

When context usage reaches:

- **60%** → WARNING (consider spawning agents to offload)
- **65%** → AUTO-CHECKPOINT (automatic)
- **75%** → CRITICAL (minimize reads, spawn agents immediately)

Monitor via: `/context` command

---

## WHERE THINGS LIVE

| Need                     | Location                                                |
| ------------------------ | ------------------------------------------------------- |
| Add file format loader   | `src/oscura/loaders/`                                   |
| Add measurement analyzer | `src/oscura/analyzers/`                                 |
| Add protocol decoder     | `src/oscura/analyzers/protocols/`                       |
| Working demonstrations   | `demos/` (19 categories, 33+ comprehensive demos)       |
| High-level examples      | `examples/` (6 workflow examples)                       |
| Test fixtures            | `tests/conftest.py`, `tests/fixtures/`                  |
| Test data generation     | `scripts/test-data/generate_comprehensive_test_data.py` |
| Coding standards         | `.claude/coding-standards.yaml`                         |

**When uncertain**: Examine existing similar code for established patterns

---

## QUICK REFERENCE

| Task            | Command                                   |
| --------------- | ----------------------------------------- |
| Run all tests   | `./scripts/test.sh`                       |
| Quality checks  | `./scripts/check.sh` (MUST before commit) |
| Auto-fix issues | `./scripts/fix.sh`                        |
| Validate all    | `python3 .claude/hooks/validate_all.py`   |
| Type check file | `uv run mypy <file> --strict`             |
| Lint file       | `uv run ruff check <file>`                |
| Format file     | `uv run ruff format <file>`               |

---

**Remember**: This file is read ONCE at session start. All directives marked CRITICAL, MUST, ALWAYS, NEVER are mandatory and enforced automatically.
