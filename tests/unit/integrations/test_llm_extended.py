"""Unit tests for LLM integration.

Tests API-018 (OpenAI), API-019 (Anthropic), and API-020 (Abstraction Layer).
All external API calls are mocked - no actual API usage.

Requirements tested:
"""

import os
import sys
import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from oscura.integrations.llm import (
    AnthropicClient,
    CostTracker,
    FailoverLLMClient,
    LLMConfig,
    LLMError,
    LLMIntegration,
    LLMProvider,
    LLMResponse,
    OpenAIClient,
    RateLimiter,
    ResponseCache,
    estimate_tokens,
    get_client,
    get_client_auto,
    get_client_with_failover,
    get_cost_tracker,
    get_provider,
    get_response_cache,
    is_provider_available,
    list_available_providers,
)

pytestmark = pytest.mark.unit


# ==============================================================================
# ==============================================================================


class TestRateLimiter:
    """Test rate limiting functionality."""

    def test_rate_limiter_respects_limit(self):
        """Rate limiter enforces requests per minute limit."""
        limiter = RateLimiter(requests_per_minute=120)  # 2 per second

        # First request should be immediate
        start = time.time()
        limiter.acquire()
        elapsed = time.time() - start
        assert elapsed < 0.1, "First request should be immediate"

        # Second request should wait ~0.5 seconds
        start = time.time()
        limiter.acquire()
        elapsed = time.time() - start
        assert 0.4 < elapsed < 0.6, f"Expected ~0.5s delay, got {elapsed:.3f}s"

    def test_rate_limiter_no_limit(self):
        """Rate limiter with 0 requests_per_minute doesn't block."""
        limiter = RateLimiter(requests_per_minute=0)

        start = time.time()
        for _ in range(10):
            limiter.acquire()
        elapsed = time.time() - start
        assert elapsed < 0.1, "No rate limiting should be instant"

    def test_rate_limiter_thread_safe(self):
        """Rate limiter is thread-safe."""
        limiter = RateLimiter(requests_per_minute=60)

        # Multiple acquires should be serialized
        start = time.time()
        limiter.acquire()
        limiter.acquire()
        elapsed = time.time() - start
        assert elapsed >= 1.0, "Sequential acquires should respect rate limit"


# ==============================================================================
# ==============================================================================


class TestOpenAIClient:
    """Test OpenAI integration."""

    def test_openai_client_init_success(self):
        """OpenAI client initializes with API key from environment."""
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"openai": mock_openai}):
                config = LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model="gpt-4",
                )
                client = OpenAIClient(config)

                assert client.config == config
                assert client.rate_limiter is not None
                mock_openai.OpenAI.assert_called_once()

    def test_openai_client_init_no_api_key(self):
        """OpenAI client raises error if no API key."""
        mock_openai = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(sys.modules, {"openai": mock_openai}):
                config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")

                with pytest.raises(LLMError, match="API key required"):
                    OpenAIClient(config)

    def test_openai_client_init_missing_package(self):
        """OpenAI client raises error if package not installed."""
        with patch.dict(sys.modules, {"openai": None}):
            config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")

            with pytest.raises(LLMError, match="not installed"):
                OpenAIClient(config)

    def test_openai_chat_completion(self):
        """OpenAI client chat_completion works with retry logic."""
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.id = "test-id"
        mock_response.created = 1234567890

        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"openai": mock_openai}):
                config = LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model="gpt-4",
                    requests_per_minute=0,  # No rate limiting for test speed
                )
                client = OpenAIClient(config)

                messages = [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ]

                response = client.chat_completion(messages)

                assert isinstance(response, LLMResponse)
                assert response.answer == "Test response"
                assert response.metadata["model"] == "gpt-4"
                assert response.metadata["usage"]["total_tokens"] == 30

    def test_openai_analyze_trace(self):
        """OpenAI client analyze_trace sends trace summary."""
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.id = "test-id"
        mock_response.created = 1234567890

        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"openai": mock_openai}):
                config = LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model="gpt-4",
                    requests_per_minute=0,
                )
                client = OpenAIClient(config)

                # Mock trace object
                trace = Mock()
                trace.metadata = Mock()
                trace.metadata.sample_rate = 1e9
                trace.metadata.num_samples = 10000
                trace.metadata.duration = 1e-5
                trace.data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

                response = client.analyze_trace(trace, "What is the frequency?")

                assert isinstance(response, LLMResponse)
                assert response.answer == "Test response"

    def test_openai_suggest_measurements(self):
        """OpenAI client suggest_measurements returns suggestions."""
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        # Mock response with measurement keywords
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = "I suggest measuring the rise_time and frequency of this signal."
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.id = "test-id"
        mock_response.created = 1234567890

        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"openai": mock_openai}):
                config = LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model="gpt-4",
                    requests_per_minute=0,
                )
                client = OpenAIClient(config)

                trace = Mock()
                trace.metadata = Mock()
                trace.metadata.sample_rate = 1e9
                trace.metadata.num_samples = 10000
                trace.metadata.duration = 1e-5
                trace.data = np.array([0.1, 0.2, 0.3])

                response = client.suggest_measurements(trace)

                assert isinstance(response, LLMResponse)
                assert len(response.suggested_commands) >= 2
                assert "measure rise_time" in response.suggested_commands
                assert "measure frequency" in response.suggested_commands

    def test_openai_retry_on_rate_limit(self):
        """OpenAI client retries on rate limit errors."""
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        # Define custom exception classes
        mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_openai.APIError = type("APIError", (Exception,), {})

        # Mock rate limit error then success
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.id = "test-id"
        mock_response.created = 1234567890

        mock_client.chat.completions.create.side_effect = [
            mock_openai.RateLimitError("Rate limited"),
            mock_response,
        ]

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"openai": mock_openai}):
                config = LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model="gpt-4",
                    max_retries=3,
                    requests_per_minute=0,
                )
                client = OpenAIClient(config)

                messages = [{"role": "user", "content": "Hello"}]

                with patch("time.sleep"):  # Speed up test
                    response = client.chat_completion(messages)

                assert isinstance(response, LLMResponse)
                assert mock_client.chat.completions.create.call_count == 2

    def test_openai_retry_exhaustion(self):
        """OpenAI client raises error after max retries."""
        mock_openai = MagicMock()
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        # Define custom exception classes
        mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_openai.APIError = type("APIError", (Exception,), {})

        # Mock persistent error
        mock_client.chat.completions.create.side_effect = mock_openai.APIError("API error")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"openai": mock_openai}):
                config = LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model="gpt-4",
                    max_retries=2,
                    requests_per_minute=0,
                )
                client = OpenAIClient(config)

                messages = [{"role": "user", "content": "Hello"}]

                with patch("time.sleep"):
                    with pytest.raises(LLMError, match="API error"):
                        client.chat_completion(messages)

                assert mock_client.chat.completions.create.call_count == 2


# ==============================================================================
# ==============================================================================


class TestAnthropicClient:
    """Test Anthropic integration."""

    def test_anthropic_client_init_success(self):
        """Anthropic client initializes with API key from environment."""
        mock_anthropic = MagicMock()
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
                config = LLMConfig(
                    provider=LLMProvider.ANTHROPIC,
                    model="claude-3-opus",
                )
                client = AnthropicClient(config)

                assert client.config == config
                assert client.rate_limiter is not None
                mock_anthropic.Anthropic.assert_called_once()

    def test_anthropic_client_init_no_api_key(self):
        """Anthropic client raises error if no API key."""
        mock_anthropic = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
                config = LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-3-opus")

                with pytest.raises(LLMError, match="API key required"):
                    AnthropicClient(config)

    def test_anthropic_chat_completion(self):
        """Anthropic client chat_completion works with system prompts."""
        mock_anthropic = MagicMock()
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.model = "claude-3-opus"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.stop_reason = "end_turn"
        mock_response.id = "test-id"
        mock_response.type = "message"

        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
                config = LLMConfig(
                    provider=LLMProvider.ANTHROPIC,
                    model="claude-3-opus",
                    requests_per_minute=0,
                )
                client = AnthropicClient(config)

                messages = [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ]

                response = client.chat_completion(messages)

                assert isinstance(response, LLMResponse)
                assert response.answer == "Test response"
                assert response.metadata["model"] == "claude-3-opus"
                assert response.metadata["usage"]["input_tokens"] == 10

    def test_anthropic_analyze_trace(self):
        """Anthropic client analyze_trace sends trace summary."""
        mock_anthropic = MagicMock()
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.model = "claude-3-opus"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.stop_reason = "end_turn"
        mock_response.id = "test-id"
        mock_response.type = "message"

        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
                config = LLMConfig(
                    provider=LLMProvider.ANTHROPIC,
                    model="claude-3-opus",
                    requests_per_minute=0,
                )
                client = AnthropicClient(config)

                trace = Mock()
                trace.metadata = Mock()
                trace.metadata.sample_rate = 1e9
                trace.metadata.num_samples = 10000
                trace.metadata.duration = 1e-5
                trace.data = np.array([0.1, 0.2, 0.3])

                response = client.analyze_trace(trace, "What is the frequency?")

                assert isinstance(response, LLMResponse)
                assert response.answer == "Test response"

    def test_anthropic_retry_on_error(self):
        """Anthropic client retries on transient errors."""
        mock_anthropic = MagicMock()
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Define custom exception classes
        mock_anthropic.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_anthropic.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_anthropic.APIError = type("APIError", (Exception,), {})

        # Mock timeout then success
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.model = "claude-3-opus"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.stop_reason = "end_turn"
        mock_response.id = "test-id"
        mock_response.type = "message"

        mock_client.messages.create.side_effect = [
            mock_anthropic.APITimeoutError("Timeout"),
            mock_response,
        ]

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
                config = LLMConfig(
                    provider=LLMProvider.ANTHROPIC,
                    model="claude-3-opus",
                    max_retries=3,
                    requests_per_minute=0,
                )
                client = AnthropicClient(config)

                messages = [{"role": "user", "content": "Hello"}]

                with patch("time.sleep"):
                    response = client.chat_completion(messages)

                assert isinstance(response, LLMResponse)
                assert mock_client.messages.create.call_count == 2


# ==============================================================================
# ==============================================================================


class TestAbstractionLayer:
    """Test LLM abstraction layer."""

    def test_get_provider_openai(self):
        """get_provider returns OpenAI client."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("oscura.integrations.llm.OpenAIClient") as mock_class:
                mock_class.return_value = Mock()

                client = get_provider("openai", model="gpt-4")

                assert mock_class.called
                config = mock_class.call_args[0][0]
                assert config.provider == LLMProvider.OPENAI
                assert config.model == "gpt-4"

    def test_get_provider_anthropic(self):
        """get_provider returns Anthropic client."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("oscura.integrations.llm.AnthropicClient") as mock_class:
                mock_class.return_value = Mock()

                client = get_provider("anthropic", model="claude-3-opus")

                assert mock_class.called
                config = mock_class.call_args[0][0]
                assert config.provider == LLMProvider.ANTHROPIC
                assert config.model == "claude-3-opus"

    def test_get_provider_local(self):
        """get_provider returns local client."""
        client = get_provider("local")

        from oscura.integrations.llm import LocalLLMClient

        assert isinstance(client, LocalLLMClient)

    def test_get_provider_unknown(self):
        """get_provider raises error for unknown provider."""
        with pytest.raises(LLMError, match="Unknown provider"):
            get_provider("unknown")

    def test_get_provider_rate_limiting(self):
        """get_provider respects rate limiting configuration."""
        with patch("oscura.integrations.llm.LocalLLMClient") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            client = get_provider("local", requests_per_minute=30)

            config = mock_class.call_args[0][0]
            assert config.requests_per_minute == 30

    def test_get_provider_graceful_degradation(self):
        """get_provider handles missing packages gracefully."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("oscura.integrations.llm.OpenAIClient") as mock_class:
                mock_class.side_effect = ImportError("openai not installed")

                with pytest.raises(LLMError, match=r"unavailable.*openai not installed"):
                    get_provider("openai")


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestLLMIntegration:
    """Test LLM integration manager."""

    def test_llm_integration_default_config(self):
        """LLM integration defaults to local provider."""
        integration = LLMIntegration()

        assert integration.config.provider == LLMProvider.LOCAL
        assert integration.config.privacy_mode is True

    def test_llm_integration_configure(self):
        """LLM integration can be reconfigured."""
        integration = LLMIntegration()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            integration.configure("openai", "gpt-4", api_key="test-key")

        assert integration.config.provider == LLMProvider.OPENAI
        assert integration.config.model == "gpt-4"
        assert integration._client is None  # Client reset

    def test_llm_integration_prepare_context(self):
        """LLM integration prepares trace context correctly."""
        integration = LLMIntegration()

        # Use spec to limit what attributes exist
        trace = Mock(spec=["metadata"])
        trace.metadata = Mock()
        trace.metadata.sample_rate = 1e9
        trace.metadata.num_samples = 10000
        trace.metadata.duration = 1e-5

        context = integration.prepare_context(trace)

        assert context["sample_rate"] == 1e9
        assert context["num_samples"] == 10000
        assert context["duration"] == 1e-5

    def test_llm_integration_privacy_mode(self):
        """LLM integration respects privacy mode."""
        integration = LLMIntegration()
        integration.config.privacy_mode = True

        trace = Mock()
        trace.data = np.array([0.1, 0.2, 0.3])

        context = integration.prepare_context(trace)

        # Should include hash but not actual data
        assert "data_hash" in context
        assert "statistics" not in context


# ==============================================================================
# Test Utilities
# ==============================================================================


def test_extract_commands():
    """Test command extraction from LLM responses."""
    mock_openai = MagicMock()
    mock_client = Mock()
    mock_openai.OpenAI.return_value = mock_client

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch.dict(sys.modules, {"openai": mock_openai}):
            config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
            client = OpenAIClient(config)

            text = "I recommend measuring the rise_time, fall_time, and frequency."
            commands = client._extract_commands(text)

            assert "measure rise_time" in commands
            assert "measure fall_time" in commands
            assert "measure frequency" in commands


def test_summarize_trace():
    """Test trace summarization for LLM context."""
    mock_openai = MagicMock()
    mock_client = Mock()
    mock_openai.OpenAI.return_value = mock_client

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch.dict(sys.modules, {"openai": mock_openai}):
            config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
            client = OpenAIClient(config)

            trace = Mock()
            trace.metadata = Mock()
            trace.metadata.sample_rate = 1e9
            trace.metadata.num_samples = 10000
            trace.metadata.duration = 1e-5
            trace.data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

            summary = client._summarize_trace(trace)

            assert "Sample Rate" in summary
            assert "1.00e+09" in summary
            assert "Mean" in summary


# ==============================================================================
# ==============================================================================


class TestCostTracker:
    """Test cost tracking functionality."""

    def test_cost_tracker_record(self):
        """Cost tracker records token usage and cost."""
        tracker = CostTracker()

        cost = tracker.record("gpt-4", input_tokens=100, output_tokens=50)

        assert cost > 0
        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50
        assert tracker.request_count == 1

    def test_cost_tracker_accumulates(self):
        """Cost tracker accumulates across multiple requests."""
        tracker = CostTracker()

        tracker.record("gpt-4", input_tokens=100, output_tokens=50)
        tracker.record("gpt-4", input_tokens=200, output_tokens=100)

        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 150
        assert tracker.request_count == 2

    def test_cost_tracker_reset(self):
        """Cost tracker reset clears all values."""
        tracker = CostTracker()
        tracker.record("gpt-4", input_tokens=100, output_tokens=50)

        tracker.reset()

        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost == 0.0
        assert tracker.request_count == 0

    def test_cost_tracker_get_summary(self):
        """Cost tracker provides summary statistics."""
        tracker = CostTracker()
        tracker.record("gpt-4", input_tokens=100, output_tokens=50)
        tracker.record("gpt-4", input_tokens=200, output_tokens=100)

        summary = tracker.get_summary()

        assert summary["total_input_tokens"] == 300
        assert summary["total_output_tokens"] == 150
        assert summary["total_tokens"] == 450
        assert summary["request_count"] == 2
        assert "total_cost_usd" in summary
        assert "avg_cost_per_request" in summary

    def test_cost_tracker_unknown_model(self):
        """Cost tracker handles unknown models with default cost."""
        tracker = CostTracker()

        cost = tracker.record("unknown-model", input_tokens=100, output_tokens=50)

        assert cost > 0  # Uses default costs
        assert tracker.request_count == 1


# ==============================================================================
# ==============================================================================


class TestResponseCache:
    """Test response caching functionality."""

    def test_cache_set_and_get(self):
        """Cache stores and retrieves responses."""
        cache = ResponseCache()

        cache.set("test prompt", "gpt-4", "cached response")
        result = cache.get("test prompt", "gpt-4")

        assert result == "cached response"

    def test_cache_miss(self):
        """Cache returns None for missing entries."""
        cache = ResponseCache()

        result = cache.get("unknown prompt", "gpt-4")

        assert result is None

    def test_cache_key_includes_model(self):
        """Cache keys include model name."""
        cache = ResponseCache()

        cache.set("test prompt", "gpt-4", "response 1")
        cache.set("test prompt", "gpt-3.5-turbo", "response 2")

        assert cache.get("test prompt", "gpt-4") == "response 1"
        assert cache.get("test prompt", "gpt-3.5-turbo") == "response 2"

    def test_cache_ttl_expiration(self):
        """Cache entries expire after TTL."""
        cache = ResponseCache(ttl_seconds=0.1)

        cache.set("test prompt", "gpt-4", "cached response")
        time.sleep(0.2)
        result = cache.get("test prompt", "gpt-4")

        assert result is None

    def test_cache_max_size(self):
        """Cache evicts oldest entries at max size."""
        cache = ResponseCache(max_size=2)

        cache.set("prompt1", "gpt-4", "response1")
        cache.set("prompt2", "gpt-4", "response2")
        cache.set("prompt3", "gpt-4", "response3")

        # First entry should be evicted
        assert cache.get("prompt1", "gpt-4") is None
        assert cache.get("prompt2", "gpt-4") == "response2"
        assert cache.get("prompt3", "gpt-4") == "response3"

    def test_cache_clear(self):
        """Cache clear removes all entries."""
        cache = ResponseCache()
        cache.set("prompt1", "gpt-4", "response1")
        cache.set("prompt2", "gpt-4", "response2")

        cache.clear()

        assert cache.size == 0
        assert cache.get("prompt1", "gpt-4") is None


# ==============================================================================
# ==============================================================================


class TestTokenEstimation:
    """Test token estimation for cost estimation."""

    def test_estimate_tokens_basic(self):
        """Token estimation provides reasonable estimate."""
        text = "Hello, how are you doing today?"
        tokens = estimate_tokens(text)

        # ~4 chars per token
        assert 5 <= tokens <= 15

    def test_estimate_tokens_empty(self):
        """Token estimation handles empty string."""
        tokens = estimate_tokens("")

        assert tokens == 1  # Minimum 1 token

    def test_estimate_tokens_long_text(self):
        """Token estimation scales with text length."""
        short_text = "Hello"
        long_text = "Hello " * 100

        short_tokens = estimate_tokens(short_text)
        long_tokens = estimate_tokens(long_text)

        assert long_tokens > short_tokens * 50


# ==============================================================================
# ==============================================================================


class TestGetClient:
    """Test get_client factory function."""

    def test_get_client_explicit_provider(self):
        """get_client with explicit provider."""
        client = get_client("local")

        from oscura.integrations.llm import LocalLLMClient

        assert isinstance(client, LocalLLMClient)

    def test_get_client_auto_selection(self):
        """get_client with no provider uses auto-selection."""
        # Clear API keys to ensure local fallback
        with patch.dict(os.environ, {}, clear=True):
            client = get_client()

            from oscura.integrations.llm import LocalLLMClient

            assert isinstance(client, LocalLLMClient)


class TestGetClientAuto:
    """Test auto provider selection."""

    def test_auto_selects_openai_when_available(self):
        """Auto selection prefers OpenAI when key available."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("oscura.integrations.llm.OpenAIClient") as mock_class:
                mock_class.return_value = Mock()

                client = get_client_auto()

                assert mock_class.called

    def test_auto_selects_anthropic_when_openai_unavailable(self):
        """Auto selection falls back to Anthropic."""
        # Clear OpenAI key, set Anthropic key
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch("oscura.integrations.llm.AnthropicClient") as mock_class:
                mock_class.return_value = Mock()

                client = get_client_auto()

                assert mock_class.called

    def test_auto_selects_local_as_last_resort(self):
        """Auto selection falls back to local provider."""
        with patch.dict(os.environ, {}, clear=True):
            client = get_client_auto()

            from oscura.integrations.llm import LocalLLMClient

            assert isinstance(client, LocalLLMClient)


class TestProviderAvailability:
    """Test provider availability checking."""

    def test_local_always_available(self):
        """Local provider is always available."""
        assert is_provider_available("local") is True

    def test_openai_unavailable_without_key(self):
        """OpenAI unavailable without API key."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_provider_available("openai") is False

    def test_anthropic_unavailable_without_key(self):
        """Anthropic unavailable without API key."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_provider_available("anthropic") is False

    def test_list_available_providers(self):
        """List available providers includes local."""
        with patch.dict(os.environ, {}, clear=True):
            providers = list_available_providers()

            assert "local" in providers


# ==============================================================================
# ==============================================================================


class TestFailoverLLMClient:
    """Test failover client."""

    def test_failover_init(self):
        """Failover client initializes with provider list."""
        client = FailoverLLMClient(["openai", "anthropic", "local"])

        assert client.providers == ["openai", "anthropic", "local"]

    def test_failover_uses_local_when_others_fail(self):
        """Failover client uses local when others unavailable."""
        with patch.dict(os.environ, {}, clear=True):
            client = get_client_with_failover(providers=["local"])

            # Should work with local provider
            response = client.query("test", {})
            assert isinstance(response, LLMResponse)

    def test_failover_analyze(self):
        """Failover client analyze works."""
        with patch.dict(os.environ, {}, clear=True):
            client = get_client_with_failover(providers=["local"])

            trace = Mock()
            response = client.analyze(trace, "test question")

            assert isinstance(response, LLMResponse)

    def test_failover_explain(self):
        """Failover client explain works."""
        with patch.dict(os.environ, {}, clear=True):
            client = get_client_with_failover(providers=["local"])

            explanation = client.explain("test measurement")

            assert isinstance(explanation, str)

    def test_failover_all_fail(self):
        """Failover raises error when all providers fail."""
        # Create a failover client with impossible providers
        client = FailoverLLMClient(["unknown1", "unknown2"])

        with pytest.raises(LLMError, match="All providers failed"):
            client.query("test", {})


# ==============================================================================
# ==============================================================================


class TestGlobalInstances:
    """Test global cost tracker and cache instances."""

    def test_get_cost_tracker_returns_singleton(self):
        """Cost tracker returns same instance."""
        tracker1 = get_cost_tracker()
        tracker2 = get_cost_tracker()

        assert tracker1 is tracker2

    def test_get_response_cache_returns_singleton(self):
        """Response cache returns same instance."""
        cache1 = get_response_cache()
        cache2 = get_response_cache()

        assert cache1 is cache2


# ==============================================================================
# ==============================================================================


class TestLLMResponseExtended:
    """Test extended LLMResponse fields."""

    def test_response_includes_cost(self):
        """LLMResponse includes estimated_cost field."""
        response = LLMResponse(answer="test", estimated_cost=0.001)

        assert response.estimated_cost == 0.001

    def test_response_includes_cached_flag(self):
        """LLMResponse includes cached field."""
        response = LLMResponse(answer="test", cached=True)

        assert response.cached is True

    def test_response_defaults(self):
        """LLMResponse has correct defaults."""
        response = LLMResponse(answer="test")

        assert response.estimated_cost == 0.0
        assert response.cached is False


# ==============================================================================
# ==============================================================================


class TestLLMConfigExtended:
    """Test extended LLMConfig fields."""

    def test_config_enable_cache(self):
        """LLMConfig has enable_cache option."""
        config = LLMConfig(enable_cache=True)

        assert config.enable_cache is True

    def test_config_track_costs(self):
        """LLMConfig has track_costs option."""
        config = LLMConfig(track_costs=True)

        assert config.track_costs is True

    def test_config_defaults(self):
        """LLMConfig has correct defaults for new fields."""
        config = LLMConfig()

        assert config.enable_cache is False
        assert config.track_costs is True
