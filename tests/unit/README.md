# Unit Tests

This directory contains unit tests for Oscura modules. Unit tests verify individual functions and classes in isolation.

## Organization

Tests are organized by module structure, mirroring `src/oscura/`:

```
unit/
├── analyzers/          # Analyzer module tests
│   ├── digital/       # Digital signal analysis
│   ├── spectral/      # Spectral analysis
│   ├── protocols/     # Protocol decoders
│   └── ...
├── loaders/           # File format loaders
├── core/              # Core functionality
└── ...
```

## Running Unit Tests

```bash
# Run all unit tests
uv run pytest tests/unit/

# Run specific module tests
uv run pytest tests/unit/analyzers/

# Run with coverage
uv run pytest tests/unit/ --cov=oscura --cov-report=html

# Run with verbose output
uv run pytest tests/unit/ -v
```

## Test Markers

Unit tests use these markers:

- `@pytest.mark.unit` - Standard unit test
- `@pytest.mark.fast` - Fast-running test (<0.1s)
- `@pytest.mark.slow` - Slow-running test (>1s)
- `@pytest.mark.parametrize` - Parameterized test cases

## Writing Unit Tests

### Test File Naming

- Name: `test_<module>.py`
- Location: Match source module path
- Example: `src/oscura/core/types.py` → `tests/unit/core/test_types.py`

### Test Function Naming

```python
def test_<function>_<scenario>():
    """Test <function> when <scenario>."""
    # Arrange
    input_data = ...

    # Act
    result = function(input_data)

    # Assert
    assert result == expected
```

### Fixtures

Use fixtures from `tests/conftest.py`:

```python
def test_with_trace(sample_trace):
    """Test using the sample_trace fixture."""
    assert len(sample_trace.data) > 0
```

## Guidelines

1. **Isolation**: Unit tests should not depend on external resources (files, network, databases)
2. **Speed**: Keep tests fast (<100ms per test)
3. **Independence**: Tests should not depend on each other
4. **Coverage**: Aim for >80% code coverage
5. **Assertions**: Use descriptive assertion messages

## See Also

- [Testing Guide](../../docs/testing/test-suite-guide.md) - Complete testing documentation
- [Test Suite Overview](../README.md) - Test suite organization
- [Integration Tests](../integration/README.md) - Integration test suite
- [Performance Tests](../performance/README.md) - Performance benchmarks
