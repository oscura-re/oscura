# Pull Request

## Description

<!-- Brief description of what this PR does. Be specific. -->

Fixes #<!-- issue number -->

## Type of Change

<!-- Check all that apply -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring (no functional changes)
- [ ] Test coverage improvement
- [ ] CI/CD or tooling update

## Changes Made

<!-- Detailed list of changes. Be specific about what was modified. -->

- Change 1
- Change 2
- Change 3

## Testing

### Automated Tests

- [ ] Added tests for new functionality
- [ ] All existing tests pass locally (`uv run pytest tests/`)
- [ ] Tested with Python 3.12 and 3.13
- [ ] Test coverage maintained or improved

### Manual Testing

<!-- Describe any manual testing performed -->

```python
# Example code showing how you tested this change
import oscura

# Test scenario
```

## Quality Checklist

### Code Quality

- [ ] Code follows project style (ruff format passes)
- [ ] Type hints added/updated (mypy passes)
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] No new warnings generated

### Documentation

- [ ] Docstrings added/updated for public APIs
- [ ] README updated if needed
- [ ] CHANGELOG.md updated (for user-facing changes)
- [ ] Example code tested and working

### Standards Compliance

<!-- If this change relates to IEEE/JEDEC standards -->

- [ ] N/A - Not standards-related
- [ ] Implementation follows relevant standard(s)
- [ ] References added to docstrings
- [ ] Standard test cases included (if applicable)

**Standards addressed**: <!-- e.g., IEEE 181, JEDEC JESD22-A122 -->

## Breaking Changes

<!-- If this is a breaking change, describe the migration path -->

<details>
<summary>Migration Guide</summary>

**Before**:

```python
# Old usage
```

**After**:

```python
# New usage
```

</details>

## Performance Impact

- [ ] No performance impact expected
- [ ] Performance improved (describe below)
- [ ] Performance may be affected (justify below)

<!-- If performance is affected, provide benchmarks or reasoning -->

## Screenshots / Visualizations

<!-- If applicable, add screenshots or plots (e.g., eye diagrams, waveforms) -->

## Dependencies

- [ ] No new dependencies
- [ ] New dependencies added (list below with justification)

<!-- List any new dependencies and why they're needed -->

## Reviewer Notes

<!-- Any specific areas you'd like reviewers to focus on -->

---

## Pre-merge Checklist

<!-- For maintainers -->

- [ ] PR title follows conventional commit format
- [ ] All CI checks pass
- [ ] At least one approval received
- [ ] No merge conflicts
- [ ] Ready to merge
