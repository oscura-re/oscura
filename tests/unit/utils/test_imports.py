"""Tests for lazy import utilities."""

import sys
from unittest import mock

import pytest

from oscura.utils.imports import (
    MissingOptionalDependency,
    has_jinja2,
    has_matplotlib,
    has_pandas,
    has_psutil,
    require_jinja2,
    require_matplotlib,
    require_pandas,
    require_psutil,
)


class TestMissingOptionalDependency:
    """Test MissingOptionalDependency exception."""

    def test_exception_inheritance(self) -> None:
        """Verify exception inherits from ImportError."""
        assert issubclass(MissingOptionalDependency, ImportError)

    def test_exception_can_be_raised(self) -> None:
        """Verify exception can be raised and caught."""
        with pytest.raises(MissingOptionalDependency):
            raise MissingOptionalDependency("Test error")


class TestRequireMatplotlib:
    """Test require_matplotlib function."""

    def test_require_matplotlib_success(self) -> None:
        """Test successful matplotlib import."""
        import matplotlib

        result = require_matplotlib()
        assert result is matplotlib

    def test_require_matplotlib_failure(self) -> None:
        """Test matplotlib import failure."""
        with mock.patch.dict(sys.modules, {"matplotlib": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(MissingOptionalDependency) as exc_info:
                    require_matplotlib()

                assert "matplotlib" in str(exc_info.value).lower()
                assert "oscura[visualization]" in str(exc_info.value)


class TestRequirePandas:
    """Test require_pandas function."""

    def test_require_pandas_success(self) -> None:
        """Test successful pandas import."""
        import pandas

        result = require_pandas()
        assert result is pandas

    def test_require_pandas_failure(self) -> None:
        """Test pandas import failure."""
        with mock.patch.dict(sys.modules, {"pandas": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(MissingOptionalDependency) as exc_info:
                    require_pandas()

                assert "pandas" in str(exc_info.value).lower()
                assert "oscura[dataframes]" in str(exc_info.value)


class TestRequirePsutil:
    """Test require_psutil function."""

    def test_require_psutil_success(self) -> None:
        """Test successful psutil import."""
        import psutil

        result = require_psutil()
        assert result is psutil

    def test_require_psutil_failure(self) -> None:
        """Test psutil import failure."""
        with mock.patch.dict(sys.modules, {"psutil": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(MissingOptionalDependency) as exc_info:
                    require_psutil()

                assert "psutil" in str(exc_info.value).lower()
                assert "oscura[system]" in str(exc_info.value)


class TestRequireJinja2:
    """Test require_jinja2 function."""

    def test_require_jinja2_success(self) -> None:
        """Test successful jinja2 import."""
        import jinja2

        result = require_jinja2()
        assert result is jinja2

    def test_require_jinja2_failure(self) -> None:
        """Test jinja2 import failure."""
        with mock.patch.dict(sys.modules, {"jinja2": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(MissingOptionalDependency) as exc_info:
                    require_jinja2()

                assert "jinja2" in str(exc_info.value).lower()
                assert "oscura[reporting]" in str(exc_info.value)


class TestHasMatplotlib:
    """Test has_matplotlib check function."""

    def test_has_matplotlib_when_available(self) -> None:
        """Test has_matplotlib returns True when available."""
        assert has_matplotlib() is True

    def test_has_matplotlib_when_unavailable(self) -> None:
        """Test has_matplotlib returns False when unavailable."""
        with mock.patch.dict(sys.modules, {"matplotlib": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError):
                assert has_matplotlib() is False


class TestHasPandas:
    """Test has_pandas check function."""

    def test_has_pandas_when_available(self) -> None:
        """Test has_pandas returns True when available."""
        assert has_pandas() is True

    def test_has_pandas_when_unavailable(self) -> None:
        """Test has_pandas returns False when unavailable."""
        with mock.patch.dict(sys.modules, {"pandas": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError):
                assert has_pandas() is False


class TestHasPsutil:
    """Test has_psutil check function."""

    def test_has_psutil_when_available(self) -> None:
        """Test has_psutil returns True when available."""
        assert has_psutil() is True

    def test_has_psutil_when_unavailable(self) -> None:
        """Test has_psutil returns False when unavailable."""
        with mock.patch.dict(sys.modules, {"psutil": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError):
                assert has_psutil() is False


class TestHasJinja2:
    """Test has_jinja2 check function."""

    def test_has_jinja2_when_available(self) -> None:
        """Test has_jinja2 returns True when available."""
        assert has_jinja2() is True

    def test_has_jinja2_when_unavailable(self) -> None:
        """Test has_jinja2 returns False when unavailable."""
        with mock.patch.dict(sys.modules, {"jinja2": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError):
                assert has_jinja2() is False
