"""Tests for version fallback coverage (improves diff coverage)."""
from unittest.mock import patch
import pytest

pytestmark = pytest.mark.unit


def test_oscura_main_version_fallback():
    """Test oscura.__init__.py version fallback when package not installed."""
    with patch('importlib.metadata.version', side_effect=Exception("Not installed")):
        import importlib
        import oscura
        importlib.reload(oscura)
        # Should fall back to "0.8.0"
        assert oscura.__version__ == "0.8.0"


def test_automotive_version_fallback():
    """Test automotive.__init__.py version fallback when package not installed."""  
    with patch('importlib.metadata.version', side_effect=Exception("Not installed")):
        import importlib
        from oscura import automotive
        importlib.reload(automotive)
        # Should have version attribute from fallback
        assert hasattr(automotive, '__version__')
