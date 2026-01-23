# Stress Tests

## Overview

Stress tests in this directory are **intentionally NOT run** in regular CI pipelines. They are designed to validate system behavior under extreme conditions and resource constraints.

## Purpose

Stress tests serve distinct purposes from regular unit and integration tests:

- **Regular tests**: Verify correctness of functionality under normal conditions
- **Stress tests**: Verify behavior under extreme load, resource limits, or edge cases

## Test Categories

### 1. Configuration Validation (`test_config_validation.py`)

Tests edge cases in YAML/JSON parsing and configuration loading:

- Empty configurations
- Deeply nested structures (100+ levels)
- Large configurations (1MB+ files, 1000+ items)
- Unicode and special characters
- Circular dependencies
- Invalid/corrupted files

**Why separate**: These tests are slow and memory-intensive, not needed for every commit.

### 2. Hook Execution (`test_hook_execution.py`)

Tests hook system behavior under stress:

- Cascading failures and error propagation
- Resource exhaustion (memory limits, large outputs)
- Timeouts and cleanup
- Concurrent hook execution
- Emergency bypass scenarios

**Why separate**: These tests apply resource limits and can cause system instability in CI environments.

### 3. Agent Orchestration (`test_agent_orchestration.py`)

Tests agent coordination under extreme conditions:

- Long-running agents (24+ hours)
- Orphaned process cleanup
- Registry corruption recovery
- Stale agent detection

**Why separate**: Long-running tests not suitable for CI feedback loops.

## Running Stress Tests

### Locally

Run all stress tests:

```bash
pytest -m stress -v
```

Run specific stress test file:

```bash
pytest tests/stress/test_config_validation.py -v
```

Run with increased timeouts:

```bash
pytest -m stress -v --timeout=600
```

### In CI

Stress tests run in a separate workflow:

- **Workflow**: `.github/workflows/stress-tests.yml`
- **Schedule**: Weekly (Sundays at 00:00 UTC)
- **Trigger**: Manual via workflow_dispatch or PR label `stress-test`

To trigger manually:

```bash
gh workflow run stress-tests.yml
```

To trigger via PR label:

```bash
gh pr edit <PR_NUMBER> --add-label stress-test
```

## Test Markers

All stress tests are marked with `@pytest.mark.stress` or have module-level marker:

```python
# Module-level marker (applies to all tests in file)
pytestmark = pytest.mark.stress

# Individual test marker
@pytest.mark.stress
def test_extreme_scenario():
    ...
```

## Development Guidelines

### When to Add Stress Tests

Add stress tests when testing:

- **High resource usage**: Memory >100MB, CPU >50% sustained
- **Long duration**: Tests taking >5 seconds
- **Extreme inputs**: 1000+ items, deeply nested structures
- **Error conditions**: Timeouts, resource exhaustion, cascading failures
- **System limits**: OS limits, file descriptors, process limits

### When NOT to Add Stress Tests

Do not add stress tests for:

- Normal functionality (use unit tests)
- Integration between components (use integration tests)
- Performance benchmarks (use `@pytest.mark.performance`)
- Slow but not resource-intensive tests (use `@pytest.mark.slow`)

### Best Practices

1. **Always mark tests**: Use `@pytest.mark.stress` on all stress tests
2. **Document why**: Add docstring explaining what stress condition is tested
3. **Set timeouts**: Use `@pytest.mark.timeout(seconds)` for long-running tests
4. **Clean up resources**: Always clean up in `finally` blocks or fixtures
5. **Parametrize scales**: Use `pytest.mark.parametrize` for different stress levels

Example:

```python
@pytest.mark.stress
@pytest.mark.timeout(300)  # 5 minutes max
def test_large_configuration_parsing(tmp_path):
    """Test parsing 1MB YAML with 10,000+ entries.

    Stress condition: Large file size + high item count
    Expected: Parses successfully within timeout
    """
    # Generate large config
    config_path = tmp_path / "large.yaml"
    generate_large_config(config_path, num_items=10000)

    # Parse and validate
    try:
        config = load_config(config_path)
        assert len(config.items) == 10000
    finally:
        config_path.unlink()  # Clean up
```

## Interpreting Results

### Expected Behavior

- **Slow execution**: Stress tests can take minutes
- **High resource usage**: Memory/CPU spikes are expected
- **Occasional failures**: Stress tests may fail on resource-constrained systems

### Failure Patterns

**Timeout**: Test exceeded time limit

- **Action**: Review if timeout is appropriate, increase if testing long-running scenarios
- **Investigate**: Check if test is stuck in infinite loop

**MemoryError**: Out of memory

- **Action**: This may be expected for memory stress tests
- **Investigate**: Verify test cleans up properly, check for memory leaks

**OSError: Too many open files**: File descriptor limit reached

- **Action**: This may be the stress condition being tested
- **Investigate**: Ensure proper resource cleanup with context managers

## CI/CD Integration

### Regular CI (`.github/workflows/ci.yml`)

Stress tests are **excluded** from regular CI:

```yaml
- name: Run tests
  run: |
    pytest tests/ \
      -m "not slow and not performance and not stress" \
      -v
```

### Stress Test Workflow (`.github/workflows/stress-tests.yml`)

Dedicated workflow for stress testing:

```yaml
name: Stress Tests

on:
  schedule:
    - cron: '0 0 * * 0' # Weekly on Sundays
  workflow_dispatch: # Manual trigger
  pull_request:
    types: [labeled] # When PR labeled with 'stress-test'

jobs:
  stress:
    runs-on: ubuntu-latest
    timeout-minutes: 120 # 2 hours for stress tests

    steps:
      - name: Run stress tests
        run: pytest -m stress -v --timeout=600
```

## Troubleshooting

### Stress Tests Failing Locally

If stress tests fail on your local machine:

1. **Check resources**: Ensure sufficient RAM/CPU available
2. **Close applications**: Free up system resources
3. **Increase limits**: Use `ulimit` to increase file descriptor limits
4. **Run individually**: Run tests one at a time to isolate issues

### Stress Tests Pass Locally, Fail in CI

This is **expected** for some stress tests:

- CI runners have limited resources
- Network/disk I/O may be slower
- Concurrent builds may compete for resources

**Action**: Review test expectations, adjust timeouts or skip in CI if truly environment-dependent.

## Maintenance

### Quarterly Review

Review stress tests every quarter:

- Remove obsolete stress conditions no longer relevant
- Update resource limits as hardware/cloud instances improve
- Add new stress scenarios for new features
- Verify stress tests still catch real issues

### Adding New Stress Tests

When adding stress tests:

1. Add to appropriate file (`test_config_validation.py`, `test_hook_execution.py`, etc.)
2. Mark with `@pytest.mark.stress`
3. Add timeout if >60 seconds
4. Document stress condition in docstring
5. Update this README if adding new category

## Related Documentation

- `docs/testing/test-suite-guide.md` - Overall test strategy
- `.github/workflows/stress-tests.yml` - Stress test CI workflow
- `tests/stress/test_config_validation.py` - Configuration stress tests
- `tests/stress/test_hook_execution.py` - Hook system stress tests
- `tests/stress/test_agent_orchestration.py` - Agent orchestration stress tests
