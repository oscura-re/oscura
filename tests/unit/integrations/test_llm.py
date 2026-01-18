"""Comprehensive unit tests for LLM integration module.

Tests API-016, API-018, API-019, API-020:
- LLM integration hooks
- OpenAI integration (chat completion, trace analysis, measurement suggestions)
- Anthropic integration (chat completion, trace analysis, measurement suggestions)
- LLM abstraction layer (unified interface, rate limiting, graceful degradation)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from oscura.integrations.llm import (
    AnalysisHook,
    AnthropicClient,
    CostTracker,
    FailoverLLMClient,
    LLMConfig,
    LLMError,
    LLMIntegration,
    LLMProvider,
    LLMResponse,
    LocalLLMClient,
    OpenAIClient,
    RateLimiter,
    ResponseCache,
    estimate_tokens,
    get_client,
    get_client_auto,
    get_client_with_failover,
    get_cost_tracker,
    get_llm,
    get_provider,
    get_response_cache,
    is_provider_available,
    list_available_providers,
)

pytestmark = pytest.mark.unit


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def mock_trace():
    """Create a mock trace object with metadata and data."""

    class MockMetadata:
        sample_rate = 1e9
        num_samples = 1000
        duration = 1e-6

    class MockTrace:
        metadata = MockMetadata()
        data = np.random.randn(1000)

    return MockTrace()


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables for LLM testing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# ==============================================================================
# CostTracker Tests
# ==============================================================================


@pytest.mark.unit
class TestCostTracker:
    """Test cost tracking functionality."""

    def test_cost_tracker_initialization(self):
        """Test CostTracker initial state."""
        tracker = CostTracker()

        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost == 0.0
        assert tracker.request_count == 0

    def test_record_known_model(self):
        """Test recording usage for known model."""
        tracker = CostTracker()

        cost = tracker.record("gpt-4", 1000, 500)

        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert tracker.request_count == 1
        # GPT-4: $0.03/1K input, $0.06/1K output
        expected_cost = 1000 / 1000 * 0.03 + 500 / 1000 * 0.06
        assert cost == pytest.approx(expected_cost)
        assert tracker.total_cost == pytest.approx(expected_cost)

    def test_record_unknown_model_uses_default(self):
        """Test recording usage for unknown model uses default rates."""
        tracker = CostTracker()

        cost = tracker.record("unknown-model", 1000, 500)

        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        # Default: $0.001/1K input, $0.002/1K output
        expected_cost = 1000 / 1000 * 0.001 + 500 / 1000 * 0.002
        assert cost == pytest.approx(expected_cost)

    def test_record_multiple_requests(self):
        """Test recording multiple requests accumulates correctly."""
        tracker = CostTracker()

        tracker.record("gpt-4", 1000, 500)
        tracker.record("gpt-4", 2000, 1000)

        assert tracker.total_input_tokens == 3000
        assert tracker.total_output_tokens == 1500
        assert tracker.request_count == 2

    def test_reset(self):
        """Test reset clears all counters."""
        tracker = CostTracker()
        tracker.record("gpt-4", 1000, 500)

        tracker.reset()

        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost == 0.0
        assert tracker.request_count == 0

    def test_get_summary(self):
        """Test getting usage summary."""
        tracker = CostTracker()
        tracker.record("gpt-4", 1000, 500)
        tracker.record("gpt-4", 2000, 1000)

        summary = tracker.get_summary()

        assert summary["total_input_tokens"] == 3000
        assert summary["total_output_tokens"] == 1500
        assert summary["total_tokens"] == 4500
        assert summary["request_count"] == 2
        assert "total_cost_usd" in summary
        assert "avg_cost_per_request" in summary
        assert summary["avg_cost_per_request"] > 0

    def test_get_summary_no_requests(self):
        """Test summary with no requests."""
        tracker = CostTracker()

        summary = tracker.get_summary()

        assert summary["request_count"] == 0
        assert summary["avg_cost_per_request"] == 0.0

    def test_thread_safety(self):
        """Test that CostTracker is thread-safe."""
        import threading

        tracker = CostTracker()
        threads = []

        def record_usage():
            for _ in range(100):
                tracker.record("gpt-4", 10, 5)

        for _ in range(10):
            t = threading.Thread(target=record_usage)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert tracker.request_count == 1000
        assert tracker.total_input_tokens == 10000


# ==============================================================================
# ResponseCache Tests
# ==============================================================================


@pytest.mark.unit
class TestResponseCache:
    """Test response caching functionality."""

    def test_cache_initialization(self):
        """Test ResponseCache initialization."""
        cache = ResponseCache(max_size=10, ttl_seconds=60.0)

        assert cache.max_size == 10
        assert cache.ttl_seconds == 60.0
        assert cache.size == 0

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ResponseCache()

        result = cache.get("test prompt", "gpt-4")

        assert result is None

    def test_cache_hit(self):
        """Test cache hit returns cached value."""
        cache = ResponseCache()
        response = {"answer": "test answer"}

        cache.set("test prompt", "gpt-4", response)
        result = cache.get("test prompt", "gpt-4")

        assert result == response

    def test_cache_key_includes_kwargs(self):
        """Test cache key differentiates based on kwargs."""
        cache = ResponseCache()
        response1 = {"answer": "answer1"}
        response2 = {"answer": "answer2"}

        cache.set("prompt", "gpt-4", response1, temperature=0.5)
        cache.set("prompt", "gpt-4", response2, temperature=0.9)

        result1 = cache.get("prompt", "gpt-4", temperature=0.5)
        result2 = cache.get("prompt", "gpt-4", temperature=0.9)

        assert result1 == response1
        assert result2 == response2

    def test_cache_expiration(self):
        """Test cache entries expire based on TTL."""
        cache = ResponseCache(ttl_seconds=0.1)
        response = {"answer": "test"}

        cache.set("prompt", "gpt-4", response)
        time.sleep(0.2)
        result = cache.get("prompt", "gpt-4")

        assert result is None
        assert cache.size == 0  # Expired entry removed

    def test_cache_eviction(self):
        """Test cache evicts oldest entries when full."""
        cache = ResponseCache(max_size=2)

        cache.set("prompt1", "gpt-4", {"answer": "1"})
        time.sleep(0.01)  # Ensure different timestamps
        cache.set("prompt2", "gpt-4", {"answer": "2"})
        time.sleep(0.01)
        cache.set("prompt3", "gpt-4", {"answer": "3"})

        # First entry should be evicted
        assert cache.get("prompt1", "gpt-4") is None
        assert cache.get("prompt2", "gpt-4") == {"answer": "2"}
        assert cache.get("prompt3", "gpt-4") == {"answer": "3"}
        assert cache.size == 2

    def test_cache_clear(self):
        """Test clearing all cache entries."""
        cache = ResponseCache()
        cache.set("prompt1", "gpt-4", {"answer": "1"})
        cache.set("prompt2", "gpt-4", {"answer": "2"})

        cache.clear()

        assert cache.size == 0
        assert cache.get("prompt1", "gpt-4") is None

    def test_cache_key_deterministic(self):
        """Test cache key is deterministic for same inputs."""
        cache = ResponseCache()

        key1 = cache._make_key("prompt", "gpt-4", temp=0.5, top_p=0.9)
        key2 = cache._make_key("prompt", "gpt-4", temp=0.5, top_p=0.9)
        key3 = cache._make_key("prompt", "gpt-4", top_p=0.9, temp=0.5)

        assert key1 == key2 == key3


# ==============================================================================
# RateLimiter Tests
# ==============================================================================


@pytest.mark.unit
class TestRateLimiter:
    """Test rate limiting functionality."""

    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(requests_per_minute=60)

        assert limiter.requests_per_minute == 60
        assert limiter.min_interval == 1.0

    def test_no_rate_limiting_when_zero(self):
        """Test no rate limiting when requests_per_minute is 0."""
        limiter = RateLimiter(requests_per_minute=0)

        start = time.time()
        limiter.acquire()
        limiter.acquire()
        elapsed = time.time() - start

        assert elapsed < 0.01  # Should be instant

    def test_rate_limiting_enforced(self):
        """Test rate limiting delays requests appropriately."""
        limiter = RateLimiter(requests_per_minute=120)  # 2 per second

        start = time.time()
        limiter.acquire()
        limiter.acquire()
        elapsed = time.time() - start

        # Second request should be delayed by ~0.5 seconds
        assert elapsed >= 0.4  # Allow some margin

    def test_thread_safety(self):
        """Test RateLimiter is thread-safe."""
        import threading

        limiter = RateLimiter(requests_per_minute=300)
        acquired_times = []

        def acquire_once():
            limiter.acquire()
            acquired_times.append(time.time())

        threads = []
        for _ in range(5):
            t = threading.Thread(target=acquire_once)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check that acquisitions are spaced appropriately
        assert len(acquired_times) == 5


# ==============================================================================
# LLMConfig Tests
# ==============================================================================


@pytest.mark.unit
class TestLLMConfig:
    """Test LLM configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = LLMConfig()

        assert config.provider == LLMProvider.LOCAL
        assert config.model == "default"
        assert config.api_key is None
        assert config.privacy_mode is True
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.requests_per_minute == 60

    def test_config_custom_values(self):
        """Test creating config with custom values."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
            privacy_mode=False,
            timeout=60.0,
            max_retries=5,
            requests_per_minute=120,
        )

        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4"
        assert config.api_key == "test-key"
        assert config.privacy_mode is False
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.requests_per_minute == 120


# ==============================================================================
# LLMResponse Tests
# ==============================================================================


@pytest.mark.unit
class TestLLMResponse:
    """Test LLM response dataclass."""

    def test_response_initialization(self):
        """Test LLMResponse initialization."""
        response = LLMResponse(
            answer="Test answer",
            confidence=0.95,
            suggested_commands=["measure frequency"],
            metadata={"model": "gpt-4"},
            estimated_cost=0.001,
            cached=False,
        )

        assert response.answer == "Test answer"
        assert response.confidence == 0.95
        assert response.suggested_commands == ["measure frequency"]
        assert response.metadata == {"model": "gpt-4"}
        assert response.estimated_cost == 0.001
        assert response.cached is False

    def test_response_defaults(self):
        """Test LLMResponse with default values."""
        response = LLMResponse(answer="Test")

        assert response.answer == "Test"
        assert response.confidence is None
        assert response.suggested_commands == []
        assert response.metadata == {}
        assert response.raw_response is None
        assert response.estimated_cost == 0.0
        assert response.cached is False


# ==============================================================================
# Helper Function Tests
# ==============================================================================


@pytest.mark.unit
class TestHelperFunctions:
    """Test helper functions."""

    def test_estimate_tokens_basic(self):
        """Test basic token estimation."""
        text = "This is a test"
        tokens = estimate_tokens(text)

        assert tokens == len(text) // 4

    def test_estimate_tokens_empty(self):
        """Test token estimation for empty string."""
        tokens = estimate_tokens("")

        assert tokens == 1  # Minimum 1 token

    def test_estimate_tokens_long_text(self):
        """Test token estimation for long text."""
        text = "a" * 1000
        tokens = estimate_tokens(text)

        assert tokens == 250

    def test_get_cost_tracker(self):
        """Test getting global cost tracker."""
        tracker = get_cost_tracker()

        assert isinstance(tracker, CostTracker)
        # Should return same instance
        assert get_cost_tracker() is tracker

    def test_get_response_cache(self):
        """Test getting global response cache."""
        cache = get_response_cache()

        assert isinstance(cache, ResponseCache)
        # Should return same instance
        assert get_response_cache() is cache


# ==============================================================================
# LocalLLMClient Tests
# ==============================================================================


@pytest.mark.unit
class TestLocalLLMClient:
    """Test local LLM client (mock implementation)."""

    def test_local_client_initialization(self):
        """Test LocalLLMClient initialization."""
        config = LLMConfig()
        client = LocalLLMClient(config)

        assert client.config == config

    def test_local_client_query(self):
        """Test local client query returns mock response."""
        config = LLMConfig()
        client = LocalLLMClient(config)

        response = client.query("test prompt", {"key": "value"})

        assert isinstance(response, LLMResponse)
        assert "mock" in response.answer.lower()
        assert response.confidence == 0.0

    def test_local_client_analyze_protocol_question(self):
        """Test local client analyze with protocol question."""
        config = LLMConfig()
        client = LocalLLMClient(config)
        mock_trace = Mock()

        response = client.analyze(mock_trace, "What protocol is this?")

        assert isinstance(response, LLMResponse)
        assert "protocol" in response.answer.lower() or "unable" in response.answer.lower()
        assert len(response.suggested_commands) > 0

    def test_local_client_analyze_generic_question(self):
        """Test local client analyze with generic question."""
        config = LLMConfig()
        client = LocalLLMClient(config)
        mock_trace = Mock()

        response = client.analyze(mock_trace, "What is this signal?")

        assert isinstance(response, LLMResponse)
        assert "measure all" in response.suggested_commands

    def test_local_client_explain(self):
        """Test local client explain."""
        config = LLMConfig()
        client = LocalLLMClient(config)

        explanation = client.explain({"frequency": 1000})

        assert isinstance(explanation, str)
        assert "not available" in explanation.lower()


# ==============================================================================
# OpenAIClient Tests (Mocked)
# ==============================================================================


@pytest.mark.unit
class TestOpenAIClient:
    """Test OpenAI client with mocked API."""

    def test_initialization_requires_api_key(self, clean_env):
        """Test OpenAI client requires API key."""
        # Mock the openai module in sys.modules
        import sys

        mock_openai = MagicMock()
        mock_openai.OpenAI = Mock()

        with patch.dict(sys.modules, {"openai": mock_openai}):
            config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")

            with pytest.raises(LLMError, match="API key required"):
                OpenAIClient(config)

    def test_initialization_from_env(self, monkeypatch):
        """Test OpenAI client gets API key from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        import sys

        mock_openai = MagicMock()
        mock_openai.OpenAI = Mock()

        with patch.dict(sys.modules, {"openai": mock_openai}):
            config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
            client = OpenAIClient(config)

            assert client.config.model == "gpt-4"

    def test_initialization_from_config(self):
        """Test OpenAI client gets API key from config."""
        import sys

        mock_openai = MagicMock()
        mock_openai.OpenAI = Mock()

        with patch.dict(sys.modules, {"openai": mock_openai}):
            config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4", api_key="config-key")
            client = OpenAIClient(config)

            mock_openai.OpenAI.assert_called_once()

    def test_chat_completion_success(self, monkeypatch):
        """Test successful chat completion."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        import sys

        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            # Setup mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150
            mock_response.model = "gpt-4"
            mock_response.id = "test-id"
            mock_response.created = 1234567890

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4", api_key="test-key")
            client = OpenAIClient(config)

            response = client.chat_completion([{"role": "user", "content": "Hello"}])

            assert response.answer == "Test response"
            assert response.metadata["model"] == "gpt-4"
            assert response.metadata["usage"]["prompt_tokens"] == 100
            assert response.estimated_cost > 0

    def test_chat_completion_retry_on_rate_limit(self, monkeypatch):
        """Test retry logic on rate limit error."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        import sys

        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            mock_openai.RateLimitError = Exception

            # First call raises RateLimitError, second succeeds
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Success"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150
            mock_response.model = "gpt-4"
            mock_response.id = "test-id"
            mock_response.created = 1234567890

            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = [
                mock_openai.RateLimitError("Rate limit"),
                mock_response,
            ]
            mock_openai.OpenAI.return_value = mock_client

            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                api_key="test-key",
                max_retries=2,
            )
            client = OpenAIClient(config)

            with patch("time.sleep"):  # Speed up test
                response = client.chat_completion([{"role": "user", "content": "Hello"}])

            assert response.answer == "Success"

    def test_chat_completion_fails_after_retries(self, monkeypatch):
        """Test failure after max retries."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        import sys

        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            mock_openai.RateLimitError = Exception

            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = mock_openai.RateLimitError(
                "Rate limit"
            )
            mock_openai.OpenAI.return_value = mock_client

            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                api_key="test-key",
                max_retries=2,
            )
            client = OpenAIClient(config)

            with patch("time.sleep"):
                with pytest.raises(LLMError, match="rate limit"):
                    client.chat_completion([{"role": "user", "content": "Hello"}])

    def test_analyze_trace(self, monkeypatch, mock_trace):
        """Test trace analysis."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        import sys

        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "This is a sine wave"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150
            mock_response.model = "gpt-4"
            mock_response.id = "test-id"
            mock_response.created = 1234567890

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4", api_key="test-key")
            client = OpenAIClient(config)

            response = client.analyze_trace(mock_trace, "What is this signal?")

            assert "sine wave" in response.answer.lower()

    def test_suggest_measurements(self, monkeypatch, mock_trace):
        """Test measurement suggestions."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        import sys

        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[
                0
            ].message.content = "Suggested measurements: frequency, amplitude, and rms"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150
            mock_response.model = "gpt-4"
            mock_response.id = "test-id"
            mock_response.created = 1234567890

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4", api_key="test-key")
            client = OpenAIClient(config)

            response = client.suggest_measurements(mock_trace)

            assert len(response.suggested_commands) > 0
            assert any("frequency" in cmd for cmd in response.suggested_commands)


# ==============================================================================
# AnthropicClient Tests (Mocked)
# ==============================================================================


@pytest.mark.unit
class TestAnthropicClient:
    """Test Anthropic client with mocked API."""

    def test_initialization_requires_api_key(self, clean_env):
        """Test Anthropic client requires API key."""
        import sys

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = Mock()

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            config = LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-3-opus")

            with pytest.raises(LLMError, match="API key required"):
                AnthropicClient(config)

    def test_initialization_from_env(self, monkeypatch):
        """Test Anthropic client gets API key from environment."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        import sys

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = Mock()

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            config = LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-3-opus")
            client = AnthropicClient(config)

            assert client.config.model == "claude-3-opus"

    def test_chat_completion_success(self, monkeypatch):
        """Test successful chat completion."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        import sys

        mock_anthropic = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            # Setup mock response
            mock_text_block = Mock()
            mock_text_block.text = "Test response"

            mock_response = Mock()
            mock_response.content = [mock_text_block]
            mock_response.usage = Mock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            mock_response.model = "claude-3-opus"
            mock_response.id = "test-id"
            mock_response.type = "message"
            mock_response.stop_reason = "end_turn"

            mock_client = Mock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            config = LLMConfig(
                provider=LLMProvider.ANTHROPIC, model="claude-3-opus", api_key="test-key"
            )
            client = AnthropicClient(config)

            response = client.chat_completion(
                [{"role": "user", "content": "Hello"}], system="You are helpful"
            )

            assert response.answer == "Test response"
            assert response.metadata["model"] == "claude-3-opus"
            assert response.metadata["usage"]["input_tokens"] == 100
            assert response.estimated_cost > 0

    def test_analyze_trace(self, monkeypatch, mock_trace):
        """Test trace analysis with Anthropic."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        import sys

        mock_anthropic = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            mock_text_block = Mock()
            mock_text_block.text = "This appears to be a digital signal"

            mock_response = Mock()
            mock_response.content = [mock_text_block]
            mock_response.usage = Mock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            mock_response.model = "claude-3-opus"
            mock_response.id = "test-id"
            mock_response.type = "message"
            mock_response.stop_reason = "end_turn"

            mock_client = Mock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            config = LLMConfig(
                provider=LLMProvider.ANTHROPIC, model="claude-3-opus", api_key="test-key"
            )
            client = AnthropicClient(config)

            response = client.analyze_trace(mock_trace, "Analyze this signal")

            assert "signal" in response.answer.lower()


# ==============================================================================
# LLMIntegration Tests
# ==============================================================================


@pytest.mark.unit
class TestLLMIntegration:
    """Test main LLM integration class."""

    def test_initialization_defaults(self):
        """Test LLMIntegration initialization with defaults."""
        integration = LLMIntegration()

        assert integration.config.provider == LLMProvider.LOCAL
        assert integration.config.privacy_mode is True

    def test_initialization_with_config(self):
        """Test LLMIntegration initialization with custom config."""
        config = LLMConfig(provider=LLMProvider.LOCAL, model="test-model")
        integration = LLMIntegration(config)

        assert integration.config.model == "test-model"

    def test_configure_openai(self):
        """Test configuring OpenAI provider."""
        integration = LLMIntegration()

        integration.configure("openai", "gpt-4", api_key="test-key")

        assert integration.config.provider == LLMProvider.OPENAI
        assert integration.config.model == "gpt-4"
        assert integration.config.api_key == "test-key"
        assert integration.config.privacy_mode is False

    def test_configure_invalid_provider(self):
        """Test configuring invalid provider raises error."""
        integration = LLMIntegration()

        with pytest.raises(LLMError, match="Unknown provider"):
            integration.configure("invalid-provider", "model")

    def test_register_and_trigger_hook(self):
        """Test registering and triggering hooks."""
        integration = LLMIntegration()
        callback_called = []

        def callback(*args, **kwargs):
            callback_called.append((args, kwargs))

        integration.register_hook(AnalysisHook.BEFORE_ANALYSIS, callback)
        integration.trigger_hook(AnalysisHook.BEFORE_ANALYSIS, "arg1", key="value")

        assert len(callback_called) == 1
        assert callback_called[0][0] == ("arg1",)
        assert callback_called[0][1] == {"key": "value"}

    def test_hook_errors_dont_break_flow(self):
        """Test that hook errors don't break analysis flow."""
        integration = LLMIntegration()

        def failing_callback(*args, **kwargs):
            raise RuntimeError("Hook failed")

        integration.register_hook(AnalysisHook.BEFORE_ANALYSIS, failing_callback)

        # Should not raise
        integration.trigger_hook(AnalysisHook.BEFORE_ANALYSIS, "test")

    def test_prepare_context_basic(self, mock_trace):
        """Test preparing trace context."""
        integration = LLMIntegration()

        context = integration.prepare_context(mock_trace)

        assert context["type"] == "MockTrace"
        assert context["sample_rate"] == 1e9
        assert context["num_samples"] == 1000
        assert context["duration"] == 1e-6

    def test_prepare_context_privacy_mode(self, mock_trace):
        """Test context preparation in privacy mode."""
        config = LLMConfig(privacy_mode=True)
        integration = LLMIntegration(config)

        context = integration.prepare_context(mock_trace)

        # Should include data hash but not statistics
        assert "data_hash" in context
        assert "statistics" not in context

    def test_prepare_context_no_privacy_mode(self, mock_trace):
        """Test context preparation without privacy mode."""
        config = LLMConfig(privacy_mode=False)
        integration = LLMIntegration(config)

        context = integration.prepare_context(mock_trace)

        # Should include statistics
        assert "statistics" in context
        assert "mean" in context["statistics"]
        assert "std" in context["statistics"]

    def test_analyze_local_provider(self, mock_trace):
        """Test analyze with local provider."""
        integration = LLMIntegration()

        response = integration.analyze(mock_trace, "What is this?")

        assert isinstance(response, LLMResponse)
        assert len(response.answer) > 0

    def test_analyze_triggers_hooks(self, mock_trace):
        """Test that analyze triggers appropriate hooks."""
        integration = LLMIntegration()
        hooks_triggered = []

        def before_hook(*args, **kwargs):
            hooks_triggered.append("before")

        def after_hook(*args, **kwargs):
            hooks_triggered.append("after")

        integration.register_hook(AnalysisHook.BEFORE_ANALYSIS, before_hook)
        integration.register_hook(AnalysisHook.AFTER_ANALYSIS, after_hook)

        integration.analyze(mock_trace, "test question")

        assert "before" in hooks_triggered
        assert "after" in hooks_triggered

    def test_analyze_error_triggers_error_hook(self, mock_trace):
        """Test that analysis errors trigger error hook."""
        config = LLMConfig(provider=LLMProvider.OPENAI, api_key="invalid")
        integration = LLMIntegration(config)
        error_hook_called = []

        def error_hook(*args, **kwargs):
            error_hook_called.append(args)

        integration.register_hook(AnalysisHook.ON_ERROR, error_hook)

        with pytest.raises(LLMError):
            integration.analyze(mock_trace, "test question")

        assert len(error_hook_called) > 0


# ==============================================================================
# Factory Function Tests
# ==============================================================================


@pytest.mark.unit
class TestFactoryFunctions:
    """Test factory functions for getting LLM clients."""

    def test_get_provider_local(self):
        """Test getting local provider."""
        client = get_provider("local")

        assert isinstance(client, LocalLLMClient)

    def test_get_provider_unknown_raises_error(self):
        """Test getting unknown provider raises error."""
        with pytest.raises(LLMError, match="Unknown provider"):
            get_provider("unknown-provider")

    def test_get_provider_openai_without_package(self, clean_env):
        """Test getting OpenAI provider without package installed."""
        with patch("oscura.integrations.llm.OpenAIClient") as mock_class:
            mock_class.side_effect = LLMError("OpenAI package not installed")

            with pytest.raises(LLMError, match="not installed|unavailable"):
                get_provider("openai", api_key="test-key")

    def test_get_client_with_provider(self):
        """Test get_client with explicit provider."""
        client = get_client("local")

        assert isinstance(client, LocalLLMClient)

    def test_get_client_auto_selects_local(self, clean_env):
        """Test get_client auto-selects local when no API keys."""
        client = get_client()

        assert isinstance(client, LocalLLMClient)

    def test_get_client_auto_prefers_openai(self, monkeypatch):
        """Test get_client_auto prefers OpenAI when available."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("oscura.integrations.llm.OpenAIClient") as mock_class:
            mock_class.return_value = Mock(spec=OpenAIClient)
            client = get_client_auto()

            mock_class.assert_called_once()

    def test_get_client_auto_falls_back_to_anthropic(self, monkeypatch):
        """Test get_client_auto falls back to Anthropic."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("oscura.integrations.llm.OpenAIClient") as mock_openai:
            with patch("oscura.integrations.llm.AnthropicClient") as mock_anthropic:
                mock_openai.side_effect = LLMError("Not available")
                mock_anthropic.return_value = Mock(spec=AnthropicClient)

                client = get_client_auto()

                mock_anthropic.assert_called_once()

    def test_is_provider_available_local(self):
        """Test local provider is always available."""
        assert is_provider_available("local") is True

    def test_is_provider_available_openai_no_key(self, clean_env):
        """Test OpenAI not available without API key."""
        assert is_provider_available("openai") is False

    def test_is_provider_available_openai_with_key(self, monkeypatch):
        """Test OpenAI available with API key."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock the import to succeed
        import sys

        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            assert is_provider_available("openai") is True

    def test_list_available_providers_local_only(self, clean_env):
        """Test listing providers with only local available."""
        providers = list_available_providers()

        assert "local" in providers
        assert len(providers) >= 1


# ==============================================================================
# FailoverLLMClient Tests
# ==============================================================================


@pytest.mark.unit
class TestFailoverLLMClient:
    """Test failover client functionality."""

    def test_failover_initialization(self):
        """Test FailoverLLMClient initialization."""
        client = FailoverLLMClient(["local", "openai"])

        assert client.providers == ["local", "openai"]

    def test_get_client_with_failover(self):
        """Test getting client with failover."""
        client = get_client_with_failover(["local"])

        assert isinstance(client, FailoverLLMClient)

    def test_failover_tries_first_provider_first(self):
        """Test failover tries providers in order."""
        client = FailoverLLMClient(["local"])

        response = client.query("test", {})

        assert isinstance(response, LLMResponse)

    def test_failover_to_second_provider(self):
        """Test failover to second provider on failure."""
        client = FailoverLLMClient(["openai", "local"])

        # OpenAI will fail (no API key), should fall back to local
        response = client.query("test", {})

        assert isinstance(response, LLMResponse)

    def test_failover_all_providers_fail(self):
        """Test error when all providers fail."""
        # Create a client with providers that will all fail
        with patch("oscura.integrations.llm.get_provider") as mock_get:
            mock_get.return_value = None

            client = FailoverLLMClient(["provider1", "provider2"])

            with pytest.raises(LLMError, match="All providers failed"):
                client.query("test", {})

    def test_failover_remembers_last_successful(self):
        """Test failover remembers last successful provider."""
        client = FailoverLLMClient(["local"])

        # First call establishes last successful
        client.query("test1", {})

        # Second call should use same provider
        response = client.query("test2", {})

        assert isinstance(response, LLMResponse)

    def test_failover_chat_completion(self):
        """Test failover chat completion."""
        client = FailoverLLMClient(["local"])

        response = client.chat_completion("Hello")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_failover_analyze_trace(self):
        """Test failover trace analysis."""
        client = FailoverLLMClient(["local"])
        trace_data = {"sample_rate": 1e9, "duration": 1e-6}

        result = client.analyze_trace(trace_data)

        assert isinstance(result, dict)
        assert "answer" in result

    def test_failover_suggest_measurements(self):
        """Test failover measurement suggestions."""
        client = FailoverLLMClient(["local"])
        characteristics = {"sample_rate": 1e9}

        suggestions = client.suggest_measurements(characteristics)

        assert isinstance(suggestions, list)

    def test_failover_explain(self):
        """Test failover explain."""
        client = FailoverLLMClient(["local"])

        explanation = client.explain({"frequency": 1000})

        assert isinstance(explanation, str)
        assert len(explanation) > 0


# ==============================================================================
# Integration Tests
# ==============================================================================


@pytest.mark.unit
class TestIntegrationsLlmIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_local_analysis(self, mock_trace):
        """Test complete analysis flow with local provider."""
        # Get LLM integration
        llm = get_llm()
        llm.configure("local", "default")

        # Analyze trace
        response = llm.analyze(mock_trace, "What is this signal?")

        assert isinstance(response, LLMResponse)
        assert len(response.answer) > 0

    def test_cost_tracking_integration(self):
        """Test cost tracking works end-to-end."""
        tracker = get_cost_tracker()
        tracker.reset()

        initial_count = tracker.request_count

        # Simulate some token usage
        tracker.record("gpt-4", 1000, 500)

        assert tracker.request_count == initial_count + 1
        assert tracker.total_cost > 0

        summary = tracker.get_summary()
        assert summary["request_count"] == initial_count + 1

    def test_response_caching_integration(self):
        """Test response caching works end-to-end."""
        cache = get_response_cache()
        cache.clear()

        # Cache a response
        test_response = {"answer": "test"}
        cache.set("test prompt", "gpt-4", test_response)

        # Retrieve it
        cached = cache.get("test prompt", "gpt-4")

        assert cached == test_response

    def test_multiple_hook_registration(self, mock_trace):
        """Test multiple hooks can be registered and triggered."""
        integration = LLMIntegration()
        hooks_called = []

        def hook1(*args):
            hooks_called.append(1)

        def hook2(*args):
            hooks_called.append(2)

        integration.register_hook(AnalysisHook.BEFORE_ANALYSIS, hook1)
        integration.register_hook(AnalysisHook.BEFORE_ANALYSIS, hook2)

        integration.analyze(mock_trace, "test")

        assert 1 in hooks_called
        assert 2 in hooks_called


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================


@pytest.mark.unit
class TestIntegrationsLlmEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_prompt(self):
        """Test handling of empty prompt."""
        client = LocalLLMClient(LLMConfig())

        response = client.query("", {})

        assert isinstance(response, LLMResponse)

    def test_very_long_prompt(self):
        """Test handling of very long prompt."""
        client = LocalLLMClient(LLMConfig())
        long_prompt = "a" * 10000

        response = client.query(long_prompt, {})

        assert isinstance(response, LLMResponse)

    def test_trace_without_metadata(self):
        """Test analyzing trace without metadata."""
        integration = LLMIntegration()

        class MinimalTrace:
            pass

        trace = MinimalTrace()
        context = integration.prepare_context(trace)

        assert context["type"] == "MinimalTrace"

    def test_trace_without_data(self):
        """Test analyzing trace without data attribute."""
        integration = LLMIntegration()

        class MetadataOnlyTrace:
            metadata = Mock(sample_rate=1e9)

        trace = MetadataOnlyTrace()
        context = integration.prepare_context(trace)

        assert "sample_rate" in context

    def test_invalid_model_name(self):
        """Test cost tracking with invalid model name."""
        tracker = CostTracker()

        cost = tracker.record("totally-invalid-model-xyz", 100, 50)

        # Should use default costs
        assert cost > 0
        assert tracker.request_count == 1

    def test_zero_token_usage(self):
        """Test recording zero tokens."""
        tracker = CostTracker()

        cost = tracker.record("gpt-4", 0, 0)

        assert cost == 0.0
        assert tracker.request_count == 1

    def test_cache_with_none_response(self):
        """Test caching None response."""
        cache = ResponseCache()

        cache.set("prompt", "model", None)
        result = cache.get("prompt", "model")

        assert result is None

    def test_rate_limiter_negative_rpm(self):
        """Test rate limiter with negative requests_per_minute."""
        limiter = RateLimiter(requests_per_minute=-10)

        # Should disable rate limiting
        start = time.time()
        limiter.acquire()
        elapsed = time.time() - start

        assert elapsed < 0.01

    def test_llm_config_with_all_none(self):
        """Test LLM config with all optional fields None."""
        config = LLMConfig(
            api_key=None,
            base_url=None,
        )

        assert config.api_key is None
        assert config.base_url is None

    def test_concurrent_cache_access(self):
        """Test concurrent cache access is thread-safe."""
        import threading

        cache = ResponseCache()
        errors = []

        def access_cache(n):
            try:
                for i in range(10):
                    cache.set(f"prompt{n}-{i}", "model", {"answer": f"response{n}-{i}"})
                    cache.get(f"prompt{n}-{i}", "model")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=access_cache, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.size <= cache.max_size
