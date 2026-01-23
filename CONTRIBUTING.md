# Contributing to Oscura

First off, thank you for considering contributing to Oscura! It's people like you that make Oscura such a great tool.

## Documentation Philosophy

Oscura follows a "demos as documentation" approach:

- **Learn by Example**: All capabilities are demonstrated in [demos/](demos/) with working code
- **Demo READMEs**: Each demo category has a comprehensive README explaining concepts
- **API Reference**: Generated documentation at [docs/api/](docs/api/)

When adding new capabilities:

1. Implement the feature
2. Add working demo in appropriate category
3. Update demo README to explain the capability
4. Ensure docstrings are complete (for API docs generation)

Do NOT create separate user guides or tutorials - they will drift out of sync. Demos are the source of truth.

## Versioning and Compatibility

Oscura follows [Semantic Versioning](https://semver.org/) (SemVer):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

**Stability Commitment:**

- Backwards compatibility - Maintained within major versions
- Deprecation warnings - Added before removing features
- Migration guides - Provided for major version upgrades
- Semantic versioning - Strictly followed for all releases

## Code of Conduct

This project and everyone participating in it is governed by the [Oscura Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, waveform files, etc.)
- **Describe the behavior you observed and what you expected**
- **Include your environment details** (Python version, OS, Oscura version)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List any alternatives you've considered**

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes using [conventional commits](https://www.conventionalcommits.org/)
   - Pre-commit hooks run automatically (lint, format, type check)
5. **Run local CI verification** (`./scripts/pre-push.sh`)
   - This is AUTOMATIC if you installed hooks via `./scripts/setup/install-hooks.sh`
   - Verifies your code against the full CI pipeline locally
6. Push to your branch (`git push origin feature/amazing-feature`)
   - Pre-push hook runs automatically (if installed)
7. Open a Pull Request

**CRITICAL:** Steps 4-5 must pass BEFORE step 6. If you skip verification, your PR will fail CI.

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Node.js 18+ (for markdownlint)

### Installation

````bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/oscura.git
cd oscura

# Complete setup (RECOMMENDED - does everything)
./scripts/setup.sh

# Verify setup
./scripts/verify-setup.sh
```bash

**Alternative (manual steps):**

```bash
# Install all dependencies
uv sync --all-extras

# Install git hooks (REQUIRED - prevents CI failures)
./scripts/setup/install-hooks.sh

# Verify setup
uv run pytest tests/unit -x --maxfail=5
```bash

**IMPORTANT:** The `install-hooks.sh` script installs BOTH pre-commit and pre-push hooks.
These hooks are **REQUIRED** to prevent CI failures. Do not skip this step!

## Development Workflow

### Quick Reference

```bash
# 1. Make your changes

# 2. Quick verification during development
./scripts/setup/verify.sh         # Fast: lint + format check
./scripts/setup/verify.sh --fix   # Auto-fix issues

# 3. Run comprehensive checks
./scripts/check.sh                # Lint + typecheck
./scripts/fix.sh                  # Auto-fix all issues
./scripts/test.sh                 # Run tests (optimal config)
./scripts/test.sh --fast          # Quick tests without coverage

# 4. Before pushing - full CI verification
./scripts/pre-push.sh             # Full CI verification (10-15 min)
./scripts/pre-push.sh --quick     # Fast mode (2 min)

# 5. Commit and push
git add .
git commit -m "feat: your change"  # Pre-commit hook runs automatically
git push                           # Pre-push hook runs automatically
```bash

### Verification Scripts

|Script|Purpose|Duration|
|---|---|---|
|`./scripts/setup/verify.sh`|Quick lint/format check|~30s|
|`./scripts/check.sh`|Lint + typecheck|~1m|
|`./scripts/fix.sh`|Auto-fix all issues|~30s|
|`./scripts/test.sh`|Optimal test execution|~8-10m|
|`./scripts/test.sh --fast`|Quick tests (no coverage)|~5-7m|
|`./scripts/pre-push.sh`|Full CI verification|~10-15m|

**Always use these validated scripts instead of manual commands.** They provide optimal configuration and are battle-tested through CI/CD.

### What Pre-Push Verifies

The `pre-push.sh` script mirrors the GitHub Actions CI pipeline:

**Stage 1 - Fast Checks:**

- Pre-commit hooks (ruff, format, yaml, markdown, etc.)
- Ruff lint and format verification
- MyPy type checking
- Config validation (SSOT, orchestration)

**Stage 2 - Tests:**

- Test marker validation
- Unit tests (parallelized)
- Integration tests
- Compliance tests

**Stage 3 - Build Verification:**

- MkDocs documentation build (--strict)
- Package build (uv build)
- CLI command verification
- Docstring coverage check

### Git Hooks

Oscura uses two types of git hooks to prevent CI failures:

1. **Pre-commit hooks** (via pre-commit framework) - Run quality checks on every commit
2. **Pre-push hooks** (custom) - Run comprehensive CI verification before push

#### Bypassing Git Hooks (Use Sparingly)

In rare cases, you may need to bypass git hooks:

```bash
git commit --no-verify    # Skip pre-commit hooks
git push --no-verify      # Skip pre-push hook
````

**When to bypass:**

✅ **Acceptable reasons:**

- Creating a WIP (work-in-progress) commit on a feature branch
- Emergency hotfix needed immediately (fix CI in next commit)
- Hook has a bug preventing legitimate work
- Rebasing/amending commits (hooks already ran before)

❌ **NOT acceptable:**

- "Hooks are too slow" (use `--quick` mode instead)
- "I'll fix it later" (fix it now before committing)
- Pushing to main/develop (hooks are there to protect these branches)
- Avoiding test failures (tests exist for a reason)

**Important notes:**

- **Branch protection still applies**: Even with `--no-verify`, failing code CANNOT merge to main
- **You're not circumventing CI**: GitHub CI will still run all checks
- **You're bypassing local validation**: This means pushing untested code, which will fail CI
- **Use pre-push `--quick` instead**: For faster feedback during development

**Better alternatives:**

```bash
# Instead of --no-verify, use quick mode:
./scripts/pre-push.sh --quick        # Fast checks (2 min)

# Or auto-fix issues first:
./scripts/pre-push.sh --fix          # Auto-fix then verify

# For feature branches, hooks run quick mode automatically
git push  # Quick verification for feature branches
```

### Running Tests

For comprehensive test documentation, see **[docs/testing/test-suite-guide.md](docs/testing/test-suite-guide.md)**.

**Recommended:** Use the optimized test script:

````bash
./scripts/test.sh              # Full tests with coverage (8-10 min)
./scripts/test.sh --fast       # Quick tests without coverage (5-7 min)
```bash

**Manual test commands** (only if needed for specific scenarios):

```bash
# Run unit tests
uv run pytest tests/unit -v --timeout=90

# Run tests with coverage
uv run pytest tests/unit --cov=src/oscura --cov-report=term-missing

# Run specific module tests
uv run pytest tests/unit/analyzers -v
uv run pytest tests/unit/protocols -v

# Run in parallel
uv run pytest tests/unit -n auto
```markdown

## Coding Standards

### Style Guide

- Follow PEP 8 (enforced by ruff)
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and small

### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
<type>(<scope>): <subject>

<body>

<footer>
```bash

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, missing semi-colons, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests
- `chore`: Maintenance tasks

**Examples:**

```markdown
feat(protocols): add FlexRay decoder support
fix(loaders): correct Tektronix WFM channel parsing
docs(api): add spectral analysis examples
test(analyzers): increase rise time test coverage
```markdown

### Testing

- Write tests for all new features
- Maintain or improve test coverage
- Use descriptive test names that explain the scenario
- Follow the existing test structure

### Documentation

- Update demo README for any user-facing changes
- Add docstrings to all public functions
- Include code examples where helpful
- Demos are the primary documentation

## IEEE Compliance Guidelines

When implementing measurement functions, follow IEEE standards:

- **IEEE 181-2011**: Pulse measurements (rise/fall time, slew rate)
- **IEEE 1241-2010**: ADC testing (SNR, SINAD, ENOB)
- **IEEE 2414-2020**: Jitter measurements (TIE, period jitter)

Include references to specific standard sections in docstrings:

```python
def rise_time(trace: TraceData, low: float = 0.1, high: float = 0.9) -> float:
    """Calculate rise time per IEEE 181-2011 Section 5.2.

    The rise time is the interval between the reference level instants
    when the signal crosses the specified low and high percentage levels.

    Args:
        trace: Input waveform trace.
        low: Low reference level (0-1). Default 10%.
        high: High reference level (0-1). Default 90%.

    Returns:
        Rise time in seconds.

    References:
        IEEE 181-2011 Section 5.2 "Rise Time and Fall Time"
    """
```python

## Documentation Checklist

Before submitting a PR that includes new code, ensure:

- [ ] All new public functions have docstrings
- [ ] Docstrings follow NumPy style format
- [ ] Args section lists all parameters with descriptions
- [ ] Returns section describes the return value
- [ ] Raises section documents all exceptions (if applicable)
- [ ] Examples are included for complex functionality
- [ ] Demo README updated if behavior changes
- [ ] CHANGELOG.md updated for user-visible changes
- [ ] **IEEE references included** for measurement functions
- [ ] **Local verification passes** (`./scripts/pre-push.sh`)

### Docstring Format

Use Google-style docstrings (as configured in pyproject.toml):

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """Brief one-line description.

    Extended description if needed. Can span multiple lines
    and provide additional context.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter. Defaults to 0.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is empty.
        TypeError: When param2 is not an integer.

    Examples:
        >>> example_function("value", 10)
        True

    Note:
        Additional implementation notes if needed.

    References:
        IEEE 181-2011 Section X.X (if applicable)
    """
```markdown

## Project Structure

```bash
oscura/
├── demos/                  # Primary documentation (working examples)
│   ├── 01_waveform_analysis/       # Waveform loading and analysis
│   ├── 02_file_format_io/          # CSV, HDF5, NumPy formats
│   ├── 03_custom_daq/              # Custom DAQ loaders
│   ├── 04_serial_protocols/        # UART, SPI, I2C basics
│   ├── 05_protocol_decoding/       # Multi-protocol decoding
│   ├── 06_udp_packet_analysis/     # Network packet analysis
│   ├── 07_protocol_inference/      # Unknown protocol inference
│   ├── 08_automotive_protocols/    # CAN, OBD-II, J1939
│   ├── 09_automotive/              # Advanced automotive
│   ├── 10_timing_measurements/     # Timing and jitter
│   ├── 11_mixed_signal/            # Mixed signal validation
│   ├── 12_spectral_compliance/     # Spectral analysis (IEEE 1241)
│   ├── 13_jitter_analysis/         # Jitter (IEEE 2414)
│   ├── 14_power_analysis/          # Power quality (IEEE 1459)
│   ├── 15_signal_integrity/        # Signal integrity metrics
│   ├── 16_emc_compliance/          # EMC compliance testing
│   ├── 17_signal_reverse_engineering/  # Signal RE workflows
│   ├── 18_advanced_inference/      # Advanced inference
│   └── 19_complete_workflows/      # End-to-end workflows
├── docs/                   # API reference & technical docs
│   ├── api/                    # Generated API reference
│   └── testing/                # Testing documentation
├── src/oscura/           # Source code
│   ├── core/               # Data types, exceptions, configuration
│   ├── loaders/            # File format loaders
│   ├── analyzers/          # Signal analysis modules
│   ├── protocols/          # Protocol decoders
│   ├── inference/          # Protocol inference
│   ├── exporters/          # Data export formats
│   └── visualization/      # Plotting utilities
├── scripts/                # Development utilities
│   ├── setup/              # Setup and installation
│   │   ├── install-hooks.sh    # Install git hooks (REQUIRED)
│   │   └── verify.sh           # Quick verification
│   ├── quality/            # Quality checks
│   │   ├── lint.sh             # Linting only
│   │   └── format.sh           # Formatting only
│   ├── testing/            # Test utilities
│   │   └── run_coverage.sh     # Coverage report
│   ├── pre-push.sh         # Full CI verification (use before push)
│   ├── check.sh            # Lint + typecheck (use frequently)
│   ├── fix.sh              # Auto-fix all issues
│   └── test.sh             # Optimal test execution (SSOT)
└── tests/                  # Test suite
    ├── unit/               # Unit tests
    ├── integration/        # Integration tests
    └── conftest.py         # Test fixtures and configuration
```bash

## Troubleshooting CI Failures

If CI fails after push, here's how to debug:

### 1. Run Local Verification First

```bash
# This should catch most issues
./scripts/pre-push.sh

# If tests fail locally, get detailed output
uv run pytest <failing_test> -v --tb=long
```bash

### 2. Common CI Failure Causes

|Failure|Local Check|Fix|
|---|---|---|
|Ruff lint|`./scripts/quality/lint.sh`|`./scripts/fix.sh`|
|Ruff format|`./scripts/quality/format.sh`|`./scripts/fix.sh`|
|MyPy|`uv run mypy src/`|Fix type errors|
|Pre-commit|`pre-commit run --all-files`|Follow error messages|
|MkDocs|`uv run mkdocs build --strict`|Fix broken links/warnings|
|Docstrings|`uv run interrogate src/oscura -f 95`|Add missing docstrings|

### 3. Environment Differences

CI runs on Ubuntu with Python 3.12 and 3.13. If tests pass locally but fail in CI:

```bash
# Check Python version
python --version

# Run tests with strict markers (like CI)
uv run pytest tests/unit -v --strict-markers --strict-config
```markdown

## Getting Help

- **GitHub Discussions**: For questions and discussions
- **GitHub Issues**: For bugs and feature requests

## Recognition

Contributors are recognized in:

- The CHANGELOG.md for significant contributions
- The GitHub contributors page
- Special thanks in release notes for major features

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Oscura!
````
