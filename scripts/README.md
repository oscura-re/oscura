# Scripts Directory

**Comprehensive automation scripts for Oscura development.**

---

## Essential Workflow (Start Here)

These 8 scripts handle 99% of daily development tasks:

```bash
./scripts/setup.sh              # Initial setup & dependencies
./scripts/check.sh              # Quick quality check (30s)
./scripts/fix.sh                # Auto-fix formatting/lint issues
./scripts/test.sh               # Run test suite (~8-10 min)
./scripts/doctor.sh             # Diagnose environment problems
./scripts/clean.sh              # Clean build artifacts
./scripts/pre-push.sh           # Full CI verification (~10-15 min)
./scripts/audit_scripts.sh      # Audit script consistency
```

**That's it!** Everything else is in specialized subdirectories.

---

## Subdirectories (By Purpose)

### setup/ - Installation & Configuration

```bash
setup/install-hooks.sh          # Install git hooks
setup/verify.sh                 # Verify installation
setup/check_dependencies.sh     # Check all dependencies
setup/install_extensions.sh     # Install VS Code extensions
setup/setup-git-hooks.sh        # Git hook setup helper
```

### testing/ - Test Utilities

```bash
testing/run_coverage.sh         # Memory-safe batched coverage
testing/check_test_isolation.py # Verify test independence
testing/validate_test_markers.py # Check pytest markers
testing/verify_test_data.sh     # Verify test data integrity
```

### test-data/ - Test Data Generation

```bash
test-data/generate_comprehensive_test_data.py  # Generate complete test suite
test-data/generate_synthetic_wfm.py            # Generate WFM files
test-data/verify_synthetic_test_data.py        # Verify generated data
test-data/download_test_data.sh                # Download external test data
```

### quality/ - Additional QA Tools

```bash
quality/lint.sh                 # Lint only (no format)
quality/format.sh               # Format only (no lint)
quality/pre-push-validation.sh  # Pre-push validation helper
quality/codebase_health.py      # Comprehensive health check
quality/validate_tools.sh       # Validate tool configs
quality/validate_vscode.sh      # Validate VS Code config
```

### docs/ - Documentation Tools

```bash
docs/validate_docs.py           # Check doc links/structure
docs/validate_api_docs.py       # Verify API docs completeness
docs/generate_diagrams.py       # Generate architecture diagrams
docs/generate_tree.sh           # Generate directory tree
docs/check_version.sh           # Check version consistency
```

### deployment/ - Publishing

```bash
deployment/publish-to-pypi.sh   # Publish to PyPI
deployment/verify-pypi-setup.sh # Verify PyPI configuration
deployment/setup-testpypi-token.sh # Configure TestPyPI token
```

### git/ - Git Utilities

```bash
git/git_sync.sh                 # Git sync helper
git/git_reset_clean_all.sh      # Nuclear reset (DANGEROUS)
git/git_reset_preserve_ignored.sh # Reset preserving .gitignore
git/validate_workflows.sh       # Validate GitHub workflows
```

### tools/ - Tool Wrappers

Low-level wrappers for individual tools (ruff, mypy, shellcheck, etc.).
Generally called by higher-level scripts, not directly.

### hooks/ - Git Hooks

Git hooks installed by `setup/install-hooks.sh`.

### lib/ - Common Library

`lib/common.sh` - Shared functions for all shell scripts (colors, counters, JSON output).

---

## Quick Examples

### Daily Development

```bash
# Morning routine
./scripts/setup.sh --check-only  # Verify environment
./scripts/doctor.sh              # If issues found

# Before committing
./scripts/check.sh               # Quick check (30s)
./scripts/fix.sh                 # Auto-fix issues

# Before pushing
./scripts/pre-push.sh --quick    # Quick check (2 min)
# OR
./scripts/pre-push.sh            # Full check (10-15 min)
```

### Test Data

```bash
# Generate comprehensive test data
cd scripts/test-data
uv run python generate_comprehensive_test_data.py ../../test_data/comprehensive/

# Generate custom WFM file
uv run python generate_synthetic_wfm.py \
  --signal sine \
  --frequency 1000 \
  --output test.wfm
```

### Documentation

```bash
# Validate all documentation
cd scripts/docs
uv run python validate_docs.py
uv run python validate_api_docs.py
uv run python generate_diagrams.py
```

### Specialized Testing

```bash
# Memory-safe coverage (for very large test suites)
./scripts/testing/run_coverage.sh

# Check test isolation
cd scripts/testing
uv run python check_test_isolation.py --sample 10

# Validate test markers
uv run python validate_test_markers.py
```

---

## Documentation

- **CLAUDE.md** - Project context and complete workflow guide
- **CONTRIBUTING.md** - Git workflow and PR process
- **docs/testing/index.md** - Testing strategy
- **.claude/coding-standards.yaml** - Coding standards (SSOT)

---

## Design Philosophy

### Top-Level (8 scripts)

**Essential workflow scripts used daily by everyone.**

- Simple, memorable names
- Clear single purpose
- Well-documented with --help

### Subdirectories (31 scripts)

**Specialized scripts organized by purpose.**

- Grouped by use case (setup, testing, docs, etc.)
- Used less frequently or for specific tasks
- Self-contained within their domain

### Benefits

- ✅ **Clear hierarchy** - Essential vs specialized
- ✅ **Easy discovery** - "Where do I find X?" is obvious
- ✅ **Maintainable** - Related scripts grouped together
- ✅ **Scalable** - Easy to add new specialized scripts

---

## Script Standards

All scripts follow these conventions:

### Shell Scripts

```bash
#!/usr/bin/env bash
set -euo pipefail

# Source common library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"  # Adjust path as needed

# Support standard flags
--json      # JSON output (for CI)
-v          # Verbose output
-h, --help  # Show help
```

### Python Scripts

```python
#!/usr/bin/env python3
"""Brief description."""
# Use argparse for CLI
# Use pathlib for paths
# Return proper exit codes (0=success, 1=failure, 2=error)
```

### Exit Codes

- **0** - Success
- **1** - Failure (tests failed, checks failed)
- **2** - Error (bad arguments, missing dependencies)

---

## Troubleshooting

### "Script not found"

```bash
# All top-level scripts are in scripts/
ls scripts/*.sh

# Specialized scripts are in subdirectories
ls scripts/*/

# Make sure scripts are executable
./scripts/setup.sh  # This fixes permissions
```

### "Command not found: uv"

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# OR
pip install uv
```

### "Pre-commit hook failed"

```bash
# Auto-fix issues first
./scripts/fix.sh

# Then check again
./scripts/check.sh

# Reinstall hooks if needed
./scripts/setup/install-hooks.sh
```

### "Tests timing out"

```bash
# Use fast mode (no coverage)
./scripts/test.sh --fast

# Or memory-safe batched coverage
./scripts/testing/run_coverage.sh
```

---

## Script Statistics

- **Top-level**: 8 essential scripts
- **Subdirectories**: 10 categorized directories
- **Total scripts**: 39 active scripts (47 including tools/)
- **Lines of code**: ~15,000 lines (including lib/common.sh)

---

## Maintenance

### Adding New Scripts

1. **Determine category** - Setup? Testing? Docs? Quality?
2. **Place in appropriate subdirectory**
3. **Follow script standards** (see above)
4. **Update this README** with one-line description
5. **Test in isolation** and with core workflow

### Deprecating Scripts

1. **Mark as deprecated** in this README
2. **Update references** in other scripts/docs
3. **Wait one release cycle**
4. **Delete** and update this README

### Regular Audits

```bash
# Check script consistency
./scripts/audit_scripts.sh

# Verify all scripts are executable
./scripts/setup.sh

# Check for broken references
grep -r "scripts/" . --include="*.sh" --include="*.py" | grep -v "^Binary"
```

---

## Questions?

- **Development workflow?** See `CLAUDE.md`
- **Environment issues?** Run `./scripts/doctor.sh`
- **Script not working?** Run with `-v` flag for verbose output
- **CI failures?** Run `./scripts/pre-push.sh` locally first

---

**Last Updated**: 2026-01-15
**Status**: Reorganized - 8 essential, 31 specialized
