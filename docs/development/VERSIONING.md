# Oscura Version Management - Complete Guide

## TL;DR: Version Bumping Strategy

**DO NOT** bump version with every commit. Accumulate changes in CHANGELOG.md `[Unreleased]`, then bump version when releasing.

---

## Current Version

Check `pyproject.toml` [project.version]:

```bash
grep "^version = " pyproject.toml
# Current: version = "0.3.0"
```

---

## Semantic Versioning (SemVer)

Oscura follows [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH
  0  .  3  .  0
```

### When to Bump

- **MAJOR (0 → 1)**: Breaking API changes (incompatible)
  - Example: Removing public functions, changing function signatures
  - Note: We're still in 0.x (pre-1.0), so breaking changes are MINOR bumps

- **MINOR (0.3 → 0.4)**: New features (backward compatible)
  - Example: Adding new analyzers, new protocol decoders
  - Example: Adding new parameters with defaults (non-breaking)

- **PATCH (0.3.0 → 0.3.1)**: Bug fixes only (backward compatible)
  - Example: Fixing calculation errors, correcting behavior
  - Example: Performance improvements without API changes

---

## Development Workflow

### ❌ WRONG: Bump version per commit

```bash
git commit -m "fix: resolve bug"
# Edit pyproject.toml: 0.3.0 → 0.3.1
git commit -m "chore: bump version to 0.3.1"
```

**Problems:**

- Version drift between commits
- Confusing history
- Breaks CI/CD automation
- Makes releases messy

### ✅ CORRECT: Accumulate changes, release later

```bash
# Commit 1
git commit -m "fix: resolve bug X"
# Edit CHANGELOG.md under [Unreleased] → ### Fixed

# Commit 2  
git commit -m "feat: add new analyzer Y"
# Edit CHANGELOG.md under [Unreleased] → ### Added

# Commit 3
git commit -m "fix: correct calculation in Z"
# Edit CHANGELOG.md under [Unreleased] → ### Fixed

# ... more commits ...

# When ready to release:
./scripts/release.sh 0.3.1  # Bumps version, tags, pushes
```

---

## Per-Commit Workflow

### What to Do with EVERY Commit

1. **Make your changes** (code, tests, docs)
2. **Update CHANGELOG.md** under `[Unreleased]` section:

   ```markdown
   ## [Unreleased]

   ### Added
   - **New Analyzer** (`src/oscura/analyzers/new.py`): Description

   ### Fixed
   - **Bug in X** (`src/oscura/core/x.py:42`): Fixed issue where...
   ```

3. **Commit with conventional format**:

   ```bash
   git commit -m "fix: resolve issue in X analyzer"
   ```

4. **DO NOT touch pyproject.toml version**

### CHANGELOG.md Format

Required sections under `[Unreleased]`:

```markdown
## [Unreleased]

### Added
- New features, capabilities, files

### Changed  
- Changes to existing functionality

### Fixed
- Bug fixes

### Removed
- Removed features/files

### Infrastructure
- CI/CD, tooling, dependencies
```

**Entry format:**

```markdown
- **Feature Name** (`path/to/file.py`):
  - Concise description
  - Key capabilities
  - Test count: X/X tests passing
  - Example location (if applicable)
```

---

## Release Workflow

### When to Release

Release when:

- Accumulated meaningful changes in [Unreleased]
- Critical bug fixes that users need
- Planned milestone/sprint completion
- Before breaking changes (to establish rollback point)

### Release Steps

**Option 1: Automated (Recommended)**

```bash
# Use release script (if exists)
./scripts/release.sh 0.3.1

# Or manually:
# 1. Update CHANGELOG.md
sed -i 's/## \[Unreleased\]/## [0.3.1] - 2026-01-21/' CHANGELOG.md

# 2. Add new [Unreleased] section
cat > CHANGELOG_HEADER.md << 'HEADER'
## [Unreleased]

### Added

### Changed

### Fixed

### Removed

### Infrastructure

HEADER
cat CHANGELOG_HEADER.md CHANGELOG.md > CHANGELOG_NEW.md
mv CHANGELOG_NEW.md CHANGELOG.md

# 3. Bump version in pyproject.toml
sed -i 's/version = "0.3.0"/version = "0.3.1"/' pyproject.toml

# 4. Commit version bump
git add CHANGELOG.md pyproject.toml
git commit -m "chore: release v0.3.1"

# 5. Create git tag
git tag -a v0.3.1 -m "Release v0.3.1"

# 6. Push (triggers release workflow)
git push origin main --tags
```

**Option 2: GitHub Release**

1. Go to GitHub → Releases → Draft new release
2. Choose tag: v0.3.1 (create new)
3. Generate release notes (auto-populates from commits)
4. Publish release
5. CI automatically builds and deploys

### Post-Release

After tagging, CI automatically:

- Builds package
- Publishes to PyPI (if configured)
- Deploys documentation: `mike deploy 0.3.1 latest`
- Updates GitHub release notes

---

## Version Checking

### For Users

```python
import oscura
print(oscura.__version__)  # "0.3.0"
```

### For Scripts

```bash
# Get current version
python -c "import oscura; print(oscura.__version__)"

# Check pyproject.toml
grep "^version = " pyproject.toml | cut -d'"' -f2
```

### In CI

```bash
# Verify version matches tag
TAG_VERSION=${GITHUB_REF#refs/tags/v}
PKG_VERSION=$(python -c "import oscura; print(oscura.__version__)")

if [ "$TAG_VERSION" != "$PKG_VERSION" ]; then
  echo "Version mismatch!"
  exit 1
fi
```

---

## Version Synchronization

The `sync_versions.py` pre-commit hook ensures versions stay synchronized across:

- `pyproject.toml` (SSOT)
- `src/oscura/__init__.py` (`__version__`)
- `src/oscura/automotive/__init__.py` (if applicable)
- Documentation files

**Hook runs automatically on commit** - no manual action needed.

---

## Examples

### Example: Bug Fix PR

```markdown
PR: "fix: correct FFT calculation for odd-length signals"

Changes:
1. Fix bug in src/oscura/analyzers/spectral/fft.py
2. Add test case
3. Update CHANGELOG.md:

## [Unreleased]

### Fixed
- **FFT Analyzer** (`src/oscura/analyzers/spectral/fft.py:87`):
  - Fixed incorrect FFT calculation for odd-length signals
  - Added test coverage: 15/15 tests passing

Result:
- Version stays 0.3.0 (no bump)
- Change documented for next release
```

### Example: New Feature PR

```markdown
PR: "feat: add CAN-FD protocol decoder"

Changes:
1. Implement CAN-FD decoder
2. Add comprehensive tests  
3. Add demo example
4. Update CHANGELOG.md:

## [Unreleased]

### Added
- **CAN-FD Protocol Decoder** (`src/oscura/analyzers/protocols/can_fd.py`):
  - Full CAN-FD frame decoding with BRS support
  - Flexible data rate handling
  - Test coverage: 42/42 tests passing
  - Example: `demos/protocols/canfd_decode.py`

Result:
- Version stays 0.3.0 (no bump)  
- Change documented for next release (0.4.0)
```

### Example: Release

```markdown
Releasing v0.3.1 (bug fix release)

Steps:
1. Rename [Unreleased] → [0.3.1] - 2026-01-21
2. Bump version in pyproject.toml
3. Commit: "chore: release v0.3.1"
4. Tag: git tag -a v0.3.1
5. Push: git push origin main --tags

Result:
- Version bumped: 0.3.0 → 0.3.1
- Tag created and pushed
- CI deploys release
```

---

## FAQ

### Q: Should I bump version for my PR?

**A: NO.** Only update CHANGELOG.md under [Unreleased].

### Q: When does version get bumped?

**A: During release process** by maintainer or release automation.

### Q: What if I made breaking changes?

**A: Document in CHANGELOG.md** under ### Changed or ### Removed. Mark as BREAKING. Maintainer will bump MINOR (0.3 → 0.4) when releasing.

### Q: Can I create a patch release myself?

**A: Only if you're a maintainer.** Contributors update CHANGELOG.md, maintainers handle releases.

### Q: What if version gets out of sync?

**A: Pre-commit hook fixes it automatically.** The sync_versions.py hook ensures all version references stay synchronized.

### Q: How do I know what version my changes will be in?

**A: Check CHANGELOG.md [Unreleased] section.** When maintainer releases, those changes move to the versioned section.

---

## Summary

### For Every Commit

✅ Update CHANGELOG.md under [Unreleased]
✅ Use conventional commit format
❌ DO NOT bump version in pyproject.toml

### For Releases (Maintainers Only)

1. Rename [Unreleased] → [X.Y.Z] - DATE
2. Bump version in pyproject.toml
3. Commit + tag + push
4. CI handles the rest

### Key Principle

**Accumulate changes, release intentionally.**
