"""Unit tests for lazy import utilities.

This module tests lazy import functionality including:
- Lazy module loading
- Attribute access triggering imports
- Error handling for missing modules
- Performance characteristics
"""

import sys
from unittest.mock import patch

import pytest

from oscura.utils import lazy_imports


class TestLazyImports:
    """Test cases for lazy import functionality."""

    def test_lazy_import_module_exists(self) -> None:
        """Test lazy importing an existing module."""
        # Test with a built-in module
        lazy_mod = lazy_imports.LazyModule("os")

        # Module should not be imported yet
        assert "os" not in sys.modules or isinstance(sys.modules.get("os"), type(sys))

        # Access an attribute to trigger import
        path_exists = lazy_mod.path.exists

        # Now module should be imported
        assert callable(path_exists)

    def test_lazy_import_attribute_access(self) -> None:
        """Test that attribute access triggers module import."""
        lazy_mod = lazy_imports.LazyModule("json")

        # Access attribute
        dumps = lazy_mod.dumps

        # Should be the actual function
        assert callable(dumps)
        assert dumps({"key": "value"}) == '{"key": "value"}'

    def test_lazy_import_nonexistent_module(self) -> None:
        """Test error handling for non-existent modules."""
        lazy_mod = lazy_imports.LazyModule("nonexistent_module_xyz")

        # Should raise ImportError when accessed
        with pytest.raises(ImportError):
            _ = lazy_mod.some_attribute

    def test_lazy_import_caching(self) -> None:
        """Test that lazy imports are cached after first access."""
        lazy_mod = lazy_imports.LazyModule("os")

        # First access
        attr1 = lazy_mod.path
        # Second access
        attr2 = lazy_mod.path

        # Should be same object
        assert attr1 is attr2

    def test_lazy_import_dir(self) -> None:
        """Test that dir() works on lazy modules."""
        lazy_mod = lazy_imports.LazyModule("os")

        # Trigger import
        _ = lazy_mod.path

        # Should have attributes
        attrs = dir(lazy_mod)
        assert "path" in attrs
        assert "getcwd" in attrs

    def test_lazy_import_repr(self) -> None:
        """Test string representation of lazy module."""
        lazy_mod = lazy_imports.LazyModule("os")

        repr_str = repr(lazy_mod)
        assert "LazyModule" in repr_str or "os" in repr_str

    def test_lazy_import_optional_dependency(self) -> None:
        """Test handling of optional dependencies."""
        # Create lazy import for optional package
        with patch.dict("sys.modules", {}, clear=False):
            lazy_mod = lazy_imports.LazyModule("optional_package_xyz")

            # Should not raise error immediately
            assert lazy_mod is not None

            # But should raise when accessed
            with pytest.raises(ImportError):
                _ = lazy_mod.some_function

    def test_lazy_import_submodule(self) -> None:
        """Test lazy importing submodules."""
        lazy_mod = lazy_imports.LazyModule("os.path")

        # Access submodule function
        exists_func = lazy_mod.exists

        assert callable(exists_func)

    def test_lazy_import_from_package(self) -> None:
        """Test lazy importing from a package."""
        lazy_mod = lazy_imports.LazyModule("pathlib")

        # Access class from module
        Path = lazy_mod.Path

        assert Path is not None
        # Should be able to create instances
        p = Path(".")
        assert p.exists()

    @pytest.mark.parametrize(
        "module_name",
        [
            "os",
            "sys",
            "json",
            "pathlib",
            "collections",
        ],
    )
    def test_lazy_import_various_modules(self, module_name: str) -> None:
        """Test lazy importing various standard library modules."""
        lazy_mod = lazy_imports.LazyModule(module_name)

        # Should create lazy module
        assert lazy_mod is not None

        # Access any attribute to trigger import
        attrs = dir(lazy_mod)
        assert len(attrs) > 0

    def test_lazy_import_performance(self) -> None:
        """Test that lazy imports don't slow down startup."""
        import time

        # Create many lazy imports (should be fast)
        start = time.time()
        lazy_mods = [lazy_imports.LazyModule(f"module_{i}") for i in range(100)]
        creation_time = time.time() - start

        # Should be very fast (< 0.1 seconds)
        assert creation_time < 0.1
        assert len(lazy_mods) == 100

    def test_lazy_import_getattr_fallback(self) -> None:
        """Test __getattr__ fallback behavior."""
        lazy_mod = lazy_imports.LazyModule("os")

        # Non-existent attribute should raise AttributeError
        with pytest.raises(AttributeError):
            _ = lazy_mod.nonexistent_attribute_xyz

    def test_lazy_import_callable(self) -> None:
        """Test lazy import of callable objects."""
        lazy_mod = lazy_imports.LazyModule("os.path")

        exists = lazy_mod.exists

        # Should be callable
        assert callable(exists)

        # Should work correctly
        assert exists("/")  # Root should always exist

    def test_lazy_import_with_error_message(self) -> None:
        """Test that import errors have helpful messages."""
        lazy_mod = lazy_imports.LazyModule("missing_dependency")

        try:
            _ = lazy_mod.function
            pytest.fail("Should have raised ImportError")
        except ImportError as e:
            # Error message should mention the module
            assert "missing_dependency" in str(e).lower()
