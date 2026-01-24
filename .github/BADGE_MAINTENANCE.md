# Badge Maintenance Guide

This document tracks all README badges and their auto-update mechanisms to ensure they never drift or become stale.

## Auto-Updating Badge Configuration

All badges in README.md use APIs that automatically update without manual intervention. This document serves as reference for understanding each badge's update mechanism.

---

## Build Status Badges (4)

### 1. CI Workflow Badge

```markdown
[![CI](https://github.com/oscura-re/oscura/actions/workflows/ci.yml/badge.svg?branch=main)](...)
```

- **Update Mechanism**: GitHub Actions API (real-time)
- **Updates When**: Any commit pushed to main that triggers CI workflow
- **Source**: `.github/workflows/ci.yml`
- **Status Check**: Automatically reflects pass/fail of latest CI run
- **Cache**: None (always fresh)

### 2. Code Quality Workflow Badge

```markdown
[![Code Quality](https://github.com/oscura-re/oscura/actions/workflows/code-quality.yml/badge.svg?branch=main)](...)
```

- **Update Mechanism**: GitHub Actions API (real-time)
- **Updates When**: Any commit pushed to main that triggers code-quality workflow
- **Source**: `.github/workflows/code-quality.yml`
- **Status Check**: Reflects pass/fail of quality checks
- **Cache**: None (always fresh)

### 3. Documentation Workflow Badge

```markdown
[![Documentation](https://github.com/oscura-re/oscura/actions/workflows/docs.yml/badge.svg?branch=main)](...)
```

- **Update Mechanism**: GitHub Actions API (real-time)
- **Updates When**: Any commit pushed to main that triggers docs workflow
- **Source**: `.github/workflows/docs.yml`
- **Status Check**: Reflects docs build status
- **Cache**: None (always fresh)

### 4. Test Quality Workflow Badge

```markdown
[![Test Quality](https://github.com/oscura-re/oscura/actions/workflows/test-quality.yml/badge.svg?branch=main)](...)
```

- **Update Mechanism**: GitHub Actions API (real-time)
- **Updates When**: Any commit pushed to main that triggers test-quality workflow
- **Source**: `.github/workflows/test-quality.yml`
- **Status Check**: Reflects test quality gates status
- **Cache**: None (always fresh)

---

## Package Badges (4)

### 5. PyPI Version Badge

```markdown
[![PyPI version](https://img.shields.io/pypi/v/oscura)](...)
```

- **Update Mechanism**: Shields.io + PyPI API (15-minute cache)
- **Updates When**: New version published to PyPI
- **Source**: https://pypi.org/project/oscura/
- **Version Source**: `pyproject.toml` [project.version]
- **Cache**: 15 minutes (shields.io default)
- **Manual Trigger**: Publishing new release to PyPI

### 6. Python Version Badge

```markdown
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](...)
```

- **Update Mechanism**: Static badge (manual update only)
- **Updates When**: Minimum Python version changes in `pyproject.toml`
- **Source**: Manually configured
- **Update Required**: When bumping minimum Python requirement
- **Last Updated**: 2026-01-18 (Python 3.12+)
- **Next Review**: When considering Python 3.14 support

### 7. License Badge

```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](...)
```

- **Update Mechanism**: Static badge (should never change)
- **Updates When**: License change (extremely rare)
- **Source**: `LICENSE` file
- **Note**: MIT license, no update expected

### 8. PyPI Downloads Badge

```markdown
[![PyPI Downloads](https://img.shields.io/pypi/dm/oscura)](...)
```

- **Update Mechanism**: Shields.io + PyPI API (daily refresh)
- **Updates When**: Download stats refresh (daily)
- **Source**: PyPI download statistics
- **Cache**: ~24 hours
- **Metric**: Downloads per month

---

## Code Quality Badges (3)

### 9. Codecov Coverage Badge

```markdown
[![codecov](https://codecov.io/gh/oscura-re/oscura/branch/main/graph/badge.svg)](...)
```

- **Update Mechanism**: Codecov API (after each coverage upload)
- **Updates When**: CI runs and uploads coverage to Codecov
- **Source**: `.github/workflows/ci.yml` (codecov-action@v5)
- **Coverage Upload**: Every CI run on main branch (Python 3.12 jobs)
- **Cache**: Real-time after upload processing (~1-2 minutes)
- **Configuration**: Public repo, no token required for badge
- **Note**: If badge shows 404, verify Codecov has received at least one upload

### 10. Ruff Code Style Badge

```markdown
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](...)
```

- **Update Mechanism**: Shields.io endpoint + Ruff repository JSON
- **Updates When**: Ruff project updates their badge
- **Source**: https://github.com/astral-sh/ruff (upstream)
- **Cache**: 5 minutes (shields.io endpoint)
- **Note**: Shows we use Ruff, not project-specific metrics

### 11. Docstring Coverage Badge

```markdown
[![Docstring Coverage](https://raw.githubusercontent.com/oscura-re/oscura/main/docs/badges/interrogate_badge.svg)](...)
```

- **Update Mechanism**: Git commit + CI workflow generation
- **Updates When**: `.github/workflows/docs.yml` generates new badge
- **Source**: `docs/badges/interrogate_badge.svg` (committed to repo)
- **Generation**: `interrogate --generate-badge` in docs workflow
- **Trigger**: Any docs workflow run (commits to main)
- **Cache**: GitHub raw content (~5 minutes)
- **Note**: Badge is generated and committed by CI

---

## Project Status Badges (2)

### 12. Maintenance Status Badge

```markdown
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](...)
```

- **Update Mechanism**: Static badge (manual semantic update)
- **Updates When**: Project maintenance status changes
- **Source**: Manually configured
- **Current Status**: "yes" (actively maintained)
- **Update Required**: If project enters maintenance-only or archived state
- **Review Frequency**: Annually or when development pace changes significantly

### 13. Last Commit Badge

```markdown
[![Last Commit](https://img.shields.io/github/last-commit/oscura-re/oscura)](...)
```

- **Update Mechanism**: Shields.io + GitHub API (real-time)
- **Updates When**: Any commit pushed to any branch
- **Source**: GitHub repository commits
- **Cache**: 5 minutes (shields.io default)
- **Note**: Auto-updates, shows project activity

---

## Badge Update Verification Checklist

Run this checklist quarterly or when adding new badges:

### Automated Badges (No Action Required)

- [ ] All GitHub Actions workflow badges reflect latest run status
- [ ] PyPI version badge matches latest release
- [ ] PyPI downloads badge shows recent statistics
- [ ] Codecov badge shows coverage percentage (verify uploads working)
- [ ] Last commit badge shows recent date
- [ ] Docstring coverage badge shows current percentage

### Manual Review Required

- [ ] Python version badge matches `pyproject.toml` minimum requirement
- [ ] License badge matches `LICENSE` file
- [ ] Maintenance status badge reflects current project state
- [ ] Ruff badge link is valid (upstream project active)

### Troubleshooting

**Codecov Badge Shows 404:**

1. Verify Codecov has received coverage uploads: https://app.codecov.io/gh/oscura-re/oscura
2. Check CI logs for successful `codecov/codecov-action@v5` execution
3. Ensure public repository (or add CODECOV_TOKEN secret for private)
4. Wait 2-3 minutes after first upload for badge generation

**GitHub Workflow Badges Not Updating:**

1. Verify workflow file exists in `.github/workflows/`
2. Check workflow has run at least once on main branch
3. Ensure workflow name in badge URL matches workflow filename exactly
4. Clear browser cache (badges are cached for ~5 minutes)

**PyPI Badge Not Updating:**

1. Verify package exists on PyPI: https://pypi.org/project/oscura/
2. Check shields.io status: https://status.shields.io/
3. Wait 15 minutes for cache to expire
4. Manually clear shields.io cache: append `?cache=0` to badge URL temporarily

**Interrogate Badge Not Updating:**

1. Check docs workflow ran successfully
2. Verify `docs/badges/interrogate_badge.svg` exists in repo
3. Check raw GitHub URL is accessible
4. Ensure interrogate command in docs workflow includes `--generate-badge`

---

## Badge Best Practices

### Do's ✅

- Use shields.io for consistent styling
- Link badges to relevant pages (workflow runs, PyPI, docs)
- Group badges by category with section headers
- Prefer auto-updating badges over static
- Document manual badges that need periodic review

### Don'ts ❌

- Don't use badges that require manual version updates
- Don't mix badge styles (shields.io vs custom)
- Don't add badges without clear value
- Don't forget to test badge URLs after adding
- Don't use badges with long cache times for frequently changing data

### Badge Addition Workflow

1. Identify metric to display
2. Find auto-updating badge source (shields.io, provider API)
3. Add badge to README in appropriate section
4. Document in this file (update mechanism, cache behavior)
5. Test badge URL loads correctly
6. Commit changes

---

## Maintenance Schedule

- **Weekly**: None (all badges auto-update)
- **Monthly**: None (all badges auto-update)
- **Quarterly**: Review manual badges (Python version, maintenance status)
- **Annually**: Full audit of all badges (remove stale, add relevant new ones)
- **Ad-hoc**: Update when underlying configuration changes (Python version, license)

---

## Badge URLs Reference

Quick reference for all badge URLs (for copy-paste or CI automation):

```markdown
# Build Status

https://github.com/oscura-re/oscura/actions/workflows/ci.yml/badge.svg?branch=main
https://github.com/oscura-re/oscura/actions/workflows/code-quality.yml/badge.svg?branch=main
https://github.com/oscura-re/oscura/actions/workflows/docs.yml/badge.svg?branch=main
https://github.com/oscura-re/oscura/actions/workflows/test-quality.yml/badge.svg?branch=main

# Package

https://img.shields.io/pypi/v/oscura
https://img.shields.io/badge/python-3.12+-blue.svg
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/pypi/dm/oscura

# Code Quality

https://codecov.io/gh/oscura-re/oscura/branch/main/graph/badge.svg
https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
https://raw.githubusercontent.com/oscura-re/oscura/main/docs/badges/interrogate_badge.svg

# Project Status

https://img.shields.io/badge/Maintained%3F-yes-green.svg
https://img.shields.io/github/last-commit/oscura-re/oscura
```

---

**Last Updated**: 2026-01-19
**Next Review**: 2026-04-19 (quarterly)
**Maintained By**: Project maintainers
