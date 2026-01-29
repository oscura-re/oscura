"""Tests for correlation ID management.

Tests the correlation ID generation and propagation for request tracing (LOG-004).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import uuid

import pytest

from oscura.core.correlation import (
    CorrelationContext,
    _correlation_id,
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    with_correlation_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


@pytest.fixture(autouse=True)
def clean_correlation_state() -> None:
    """Clean correlation ID state before each test."""
    # Reset to default (None)
    try:
        _correlation_id.set(None)  # type: ignore[arg-type]
    except (LookupError, AttributeError):
        pass


class TestGetCorrelationId:
    """Test get_correlation_id() function."""

    def test_get_without_context_returns_none(self) -> None:
        """Test getting correlation ID without context returns None."""
        corr_id = get_correlation_id()
        assert corr_id is None

    def test_get_after_set(self) -> None:
        """Test getting correlation ID after setting it."""
        test_id = "test-correlation-id"
        set_correlation_id(test_id)
        corr_id = get_correlation_id()
        assert corr_id == test_id

    def test_get_with_context(self) -> None:
        """Test getting correlation ID within context."""
        with CorrelationContext() as corr_id:
            retrieved_id = get_correlation_id()
            assert retrieved_id == corr_id
            assert retrieved_id is not None


class TestSetCorrelationId:
    """Test set_correlation_id() function."""

    def test_set_simple_string(self) -> None:
        """Test setting correlation ID with simple string."""
        set_correlation_id("my-id")
        assert get_correlation_id() == "my-id"

    def test_set_uuid_string(self) -> None:
        """Test setting correlation ID with UUID string."""
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        set_correlation_id(uuid_str)
        assert get_correlation_id() == uuid_str

    def test_set_overwrites_previous(self) -> None:
        """Test that setting overwrites previous correlation ID."""
        set_correlation_id("first-id")
        set_correlation_id("second-id")
        assert get_correlation_id() == "second-id"

    def test_set_empty_string(self) -> None:
        """Test setting empty string as correlation ID."""
        set_correlation_id("")
        assert get_correlation_id() == ""


class TestCorrelationContext:
    """Test CorrelationContext class."""

    def test_context_auto_generates_id(self) -> None:
        """Test that context auto-generates UUID if not provided."""
        with CorrelationContext() as corr_id:
            assert corr_id is not None
            assert len(corr_id) == 36  # UUID format length
            # Verify it's a valid UUID
            uuid.UUID(corr_id)  # Should not raise

    def test_context_with_explicit_id(self) -> None:
        """Test context with explicit correlation ID."""
        explicit_id = "my-custom-id"
        with CorrelationContext(explicit_id) as corr_id:
            assert corr_id == explicit_id
            assert get_correlation_id() == explicit_id

    def test_context_sets_id_on_enter(self) -> None:
        """Test that entering context sets the ID."""
        with CorrelationContext("test-id"):
            assert get_correlation_id() == "test-id"

    def test_context_clears_id_on_exit(self) -> None:
        """Test that exiting context clears the ID."""
        with CorrelationContext("test-id"):
            pass  # ID set inside
        # After exiting, should be None (or previous value)
        assert get_correlation_id() is None

    def test_context_restores_previous_id(self) -> None:
        """Test that context restores previous correlation ID."""
        set_correlation_id("outer-id")
        with CorrelationContext("inner-id"):
            assert get_correlation_id() == "inner-id"
        assert get_correlation_id() == "outer-id"

    def test_nested_contexts(self) -> None:
        """Test nested correlation contexts."""
        with CorrelationContext("outer") as outer_id:
            assert get_correlation_id() == "outer"
            with CorrelationContext("inner") as inner_id:
                assert get_correlation_id() == "inner"
            assert get_correlation_id() == "outer"

    def test_context_with_exception(self) -> None:
        """Test that context cleans up even with exception."""
        set_correlation_id("before")
        try:
            with CorrelationContext("during"):
                raise ValueError("Test error")
        except ValueError:
            pass
        # Should restore to "before", not "during"
        assert get_correlation_id() == "before"

    def test_context_returns_id(self) -> None:
        """Test that context manager returns the correlation ID."""
        with CorrelationContext() as returned_id:
            current_id = get_correlation_id()
            assert returned_id == current_id


class TestWithCorrelationIdDecorator:
    """Test with_correlation_id() decorator."""

    def test_decorator_without_id(self) -> None:
        """Test decorator without explicit ID (auto-generates)."""

        @with_correlation_id()
        def test_func() -> str | None:
            return get_correlation_id()

        corr_id = test_func()
        assert corr_id is not None
        assert len(corr_id) == 36  # UUID format

    def test_decorator_with_explicit_id(self) -> None:
        """Test decorator with explicit correlation ID."""

        @with_correlation_id("my-decorator-id")
        def test_func() -> str | None:
            return get_correlation_id()

        corr_id = test_func()
        assert corr_id == "my-decorator-id"

    def test_decorator_preserves_function_args(self) -> None:
        """Test that decorator preserves function arguments."""

        @with_correlation_id()
        def test_func(x: int, y: int) -> int:
            return x + y

        result = test_func(5, 3)
        assert result == 8

    def test_decorator_preserves_function_kwargs(self) -> None:
        """Test that decorator preserves keyword arguments."""

        @with_correlation_id()
        def test_func(a: int, b: int = 10) -> int:
            return a * b

        result = test_func(5, b=7)
        assert result == 35

    def test_decorator_preserves_function_name(self) -> None:
        """Test that decorator preserves function name."""

        @with_correlation_id()
        def my_function() -> None:
            pass

        assert my_function.__name__ == "my_function"

    def test_decorator_preserves_docstring(self) -> None:
        """Test that decorator preserves docstring."""

        @with_correlation_id()
        def documented_func() -> None:
            """This is a docstring."""

        assert documented_func.__doc__ == "This is a docstring."

    def test_decorator_clears_id_after_function(self) -> None:
        """Test that decorator clears correlation ID after function."""
        set_correlation_id("before")

        @with_correlation_id("during")
        def test_func() -> None:
            assert get_correlation_id() == "during"

        test_func()
        assert get_correlation_id() == "before"

    def test_decorator_with_return_value(self) -> None:
        """Test decorator preserves return value."""

        @with_correlation_id()
        def test_func() -> dict[str, int]:
            return {"result": 42}

        result = test_func()
        assert result == {"result": 42}

    def test_decorator_with_exception(self) -> None:
        """Test decorator handles exceptions properly."""

        @with_correlation_id("test-id")
        def test_func() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            test_func()

        # ID should be cleared even after exception
        assert get_correlation_id() is None

    def test_decorator_nested_calls(self) -> None:
        """Test nested decorated function calls."""

        @with_correlation_id("outer-func")
        def outer() -> str | None:
            return inner()

        @with_correlation_id("inner-func")
        def inner() -> str | None:
            return get_correlation_id()

        result = outer()
        # Inner should have its own ID
        assert result == "inner-func"


class TestGenerateCorrelationId:
    """Test generate_correlation_id() function."""

    def test_generates_valid_uuid(self) -> None:
        """Test that generated ID is a valid UUID."""
        corr_id = generate_correlation_id()
        # Should not raise
        uuid.UUID(corr_id)

    def test_generates_unique_ids(self) -> None:
        """Test that multiple calls generate unique IDs."""
        id1 = generate_correlation_id()
        id2 = generate_correlation_id()
        id3 = generate_correlation_id()
        assert id1 != id2
        assert id2 != id3
        assert id1 != id3

    def test_generates_string_format(self) -> None:
        """Test that generated ID is a string."""
        corr_id = generate_correlation_id()
        assert isinstance(corr_id, str)

    def test_generates_uuid4_format(self) -> None:
        """Test that generated ID follows UUID4 format."""
        corr_id = generate_correlation_id()
        assert len(corr_id) == 36
        assert corr_id[14] == "4"  # UUID4 version indicator


class TestThreadSafety:
    """Test thread-safety of correlation context."""

    def test_isolation_between_threads(self) -> None:
        """Test that correlation IDs are isolated between threads."""
        results: dict[str, str | None] = {}

        def thread_func(name: str, corr_id: str) -> None:
            with CorrelationContext(corr_id):
                # Simulate some work
                import time

                time.sleep(0.01)
                # Each thread should see its own ID
                results[name] = get_correlation_id()

        threads = [
            threading.Thread(target=thread_func, args=("thread1", "id-1")),
            threading.Thread(target=thread_func, args=("thread2", "id-2")),
            threading.Thread(target=thread_func, args=("thread3", "id-3")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["thread1"] == "id-1"
        assert results["thread2"] == "id-2"
        assert results["thread3"] == "id-3"

    def test_concurrent_set_and_get(self) -> None:
        """Test concurrent set and get operations."""
        results: list[str | None] = []

        def worker(worker_id: str) -> None:
            set_correlation_id(worker_id)
            results.append(get_correlation_id())

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, f"worker-{i}") for i in range(10)]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        # Each worker should have set and retrieved its own ID
        assert len(results) == 10


class TestAsyncSupport:
    """Test async/await compatibility."""

    @pytest.mark.anyio
    async def test_correlation_in_async_function(self) -> None:
        """Test correlation context in async function."""
        with CorrelationContext("async-id") as corr_id:
            assert get_correlation_id() == "async-id"
            await asyncio.sleep(0.001)
            assert get_correlation_id() == "async-id"

    @pytest.mark.anyio
    async def test_correlation_across_await(self) -> None:
        """Test that correlation ID persists across await."""

        async def async_operation() -> str | None:
            await asyncio.sleep(0.001)
            return get_correlation_id()

        with CorrelationContext("persistent-id"):
            result = await async_operation()
            assert result == "persistent-id"

    @pytest.mark.anyio
    async def test_decorator_with_async_function(self) -> None:
        """Test decorator works with async functions."""

        @with_correlation_id("async-decorator-id")
        async def async_func() -> str | None:
            await asyncio.sleep(0.001)
            return get_correlation_id()

        result = await async_func()
        assert result == "async-decorator-id"

    @pytest.mark.anyio
    async def test_decorator_async_without_explicit_id(self) -> None:
        """Test async decorator auto-generates ID."""

        @with_correlation_id()
        async def async_func() -> str | None:
            await asyncio.sleep(0.001)
            return get_correlation_id()

        result = await async_func()
        assert result is not None
        assert len(result) == 36  # UUID format

    @pytest.mark.anyio
    async def test_decorator_async_preserves_function_metadata(self) -> None:
        """Test that async decorator preserves function name and docstring."""

        @with_correlation_id()
        async def my_async_function() -> None:
            """My async docstring."""

        assert my_async_function.__name__ == "my_async_function"
        assert my_async_function.__doc__ == "My async docstring."

    @pytest.mark.anyio
    async def test_decorator_async_with_exception(self) -> None:
        """Test async decorator handles exceptions properly."""

        @with_correlation_id("test-async-id")
        async def async_func() -> None:
            await asyncio.sleep(0.001)
            raise ValueError("Async test error")

        with pytest.raises(ValueError, match="Async test error"):
            await async_func()

        # ID should be cleared even after exception
        assert get_correlation_id() is None

    @pytest.mark.anyio
    async def test_decorator_async_restores_previous_id(self) -> None:
        """Test async decorator restores previous correlation ID."""
        set_correlation_id("before-async")

        @with_correlation_id("during-async")
        async def async_func() -> str | None:
            await asyncio.sleep(0.001)
            return get_correlation_id()

        result = await async_func()
        assert result == "during-async"
        assert get_correlation_id() == "before-async"

    @pytest.mark.anyio
    async def test_decorator_async_nested_calls(self) -> None:
        """Test nested async decorated function calls."""

        @with_correlation_id("outer-async")
        async def outer_async() -> str | None:
            await asyncio.sleep(0.001)
            return await inner_async()

        @with_correlation_id("inner-async")
        async def inner_async() -> str | None:
            await asyncio.sleep(0.001)
            return get_correlation_id()

        result = await outer_async()
        assert result == "inner-async"


class TestCoreCorrelationIntegration:
    """Integration tests for correlation ID workflow."""

    def test_full_workflow(self) -> None:
        """Test complete correlation ID workflow."""
        # Start with no correlation
        assert get_correlation_id() is None

        # Set an ID
        set_correlation_id("workflow-id")
        assert get_correlation_id() == "workflow-id"

        # Use context (should override)
        with CorrelationContext("context-id"):
            assert get_correlation_id() == "context-id"

        # Should restore to previous
        assert get_correlation_id() == "workflow-id"

    def test_decorator_in_workflow(self) -> None:
        """Test using decorator in a workflow."""

        @with_correlation_id()
        def operation1() -> str | None:
            return get_correlation_id()

        @with_correlation_id()
        def operation2() -> str | None:
            return get_correlation_id()

        id1 = operation1()
        id2 = operation2()

        # Each operation should have its own ID
        assert id1 is not None
        assert id2 is not None
        assert id1 != id2  # Different auto-generated IDs

    def test_manual_id_propagation(self) -> None:
        """Test manual correlation ID propagation."""
        # Generate ID at entry point
        entry_id = generate_correlation_id()

        # Set for entire workflow
        with CorrelationContext(entry_id):
            # Simulated function calls
            def step1() -> str | None:
                # Should see the same ID
                return get_correlation_id()

            def step2() -> str | None:
                return get_correlation_id()

            step1_id = step1()
            step2_id = step2()

            assert step1_id == entry_id
            assert step2_id == entry_id

    def test_context_nesting_workflow(self) -> None:
        """Test realistic nested context workflow."""
        # Top-level request
        with CorrelationContext("request-123") as request_id:
            assert get_correlation_id() == "request-123"

            # Sub-operation with its own ID
            with CorrelationContext("sub-operation-456"):
                assert get_correlation_id() == "sub-operation-456"

            # Back to request ID
            assert get_correlation_id() == "request-123"

            # Another sub-operation
            with CorrelationContext("sub-operation-789"):
                assert get_correlation_id() == "sub-operation-789"

            # Still request ID
            assert get_correlation_id() == "request-123"
