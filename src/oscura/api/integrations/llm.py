"""LLM Integration for Oscura.

Provides hooks for Large Language Model integration to enable natural language
analysis and assistance.


Examples:
    Basic usage with auto-selection:

    >>> from oscura.api.integrations import llm
    >>> client = llm.get_client()  # Auto-selects available provider
    >>> response = client.chat_completion("What is signal rise time?")

    Provider-specific usage:

    >>> client = llm.get_client("openai", model="gpt-4")
    >>> analysis = client.analyze_trace({"sample_rate": 1e9, "mean": 0.5})

    With failover:

    >>> client = llm.get_client_with_failover(
    ...     providers=["openai", "anthropic", "local"]
    ... )
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Protocol

from oscura.core.exceptions import OscuraError

# ==============================================================================
# Cost Constants (API-020: Cost Tracking)
# ==============================================================================

# Pricing per 1K tokens (approximate, as of 2024)
TOKEN_COSTS: dict[str, dict[str, float]] = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "default": {"input": 0.001, "output": 0.002},
}


@dataclass
class CostTracker:
    """Tracks API usage costs.

    Attributes:
        total_input_tokens: Total input tokens used across all requests
        total_output_tokens: Total output tokens used across all requests
        total_cost: Total estimated cost in USD
        request_count: Number of API requests made
    """

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    request_count: int = 0
    _lock: Lock = field(default_factory=Lock)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record token usage and return estimated cost.

        Args:
            model: Model name for cost lookup
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens

        Returns:
            Estimated cost in USD for this request
        """
        # Get cost rates for model, fall back to default
        rates = TOKEN_COSTS.get(model, TOKEN_COSTS["default"])

        cost = input_tokens / 1000 * rates["input"] + output_tokens / 1000 * rates["output"]

        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self.request_count += 1

        return cost

    def reset(self) -> None:
        """Reset all tracking counters."""
        with self._lock:
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_cost = 0.0
            self.request_count = 0

    def get_summary(self) -> dict[str, Any]:
        """Get summary of usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        with self._lock:
            return {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "total_cost_usd": round(self.total_cost, 6),
                "request_count": self.request_count,
                "avg_cost_per_request": (
                    round(self.total_cost / self.request_count, 6)
                    if self.request_count > 0
                    else 0.0
                ),
            }


class ResponseCache:
    """Simple LRU cache for LLM responses.

    Caches responses based on prompt hash to avoid repeated API calls
    for identical queries. Thread-safe implementation.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 3600.0):
        """Initialize response cache.

        Args:
            max_size: Maximum number of cached responses
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = Lock()

    def _make_key(self, prompt: str, model: str, **kwargs: Any) -> str:
        """Create cache key from request parameters.

        Args:
            prompt: The prompt text
            model: Model name
            **kwargs: Additional parameters affecting response

        Returns:
            Hash key for cache lookup
        """
        key_data = json.dumps(
            {"prompt": prompt, "model": model, "kwargs": sorted(kwargs.items())}, sort_keys=True
        )
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, prompt: str, model: str, **kwargs: Any) -> Any | None:
        """Get cached response if available and not expired.

        Args:
            prompt: The prompt text
            model: Model name
            **kwargs: Additional parameters

        Returns:
            Cached response or None if not found/expired
        """
        key = self._make_key(prompt, model, **kwargs)

        with self._lock:
            if key in self._cache:
                response, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    return response
                # Expired entry
                del self._cache[key]
            return None

    def set(self, prompt: str, model: str, response: Any, **kwargs: Any) -> None:
        """Cache a response.

        Args:
            prompt: The prompt text
            model: Model name
            response: Response to cache
            **kwargs: Additional parameters
        """
        key = self._make_key(prompt, model, **kwargs)

        with self._lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[key] = (response, time.time())

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        with self._lock:
            return len(self._cache)


# Global instances for tracking
_global_cost_tracker = CostTracker()
_global_response_cache = ResponseCache()


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance.

    Returns:
        Global CostTracker for monitoring API costs
    """
    return _global_cost_tracker


def get_response_cache() -> ResponseCache:
    """Get global response cache instance.

    Returns:
        Global ResponseCache for caching LLM responses
    """
    return _global_response_cache


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    CUSTOM = "custom"


class AnalysisHook(Enum):
    """Hook points for LLM integration."""

    BEFORE_ANALYSIS = "before_analysis"
    AFTER_ANALYSIS = "after_analysis"
    ON_ERROR = "on_error"


class RateLimiter:
    """Rate limiter for API requests.

        Implements token bucket algorithm for rate limiting.
    .: Rate limiting (configurable requests/minute).
    """

    def __init__(self, requests_per_minute: int = 60):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.last_request_time = 0.0
        self.lock = Lock()

    def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        if self.requests_per_minute <= 0:
            return  # No rate limiting

        with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            self.last_request_time = time.time()


@dataclass
class LLMConfig:
    """Configuration for LLM integration.

    Attributes:
        provider: LLM provider to use
        model: Model identifier (e.g., 'gpt-4', 'claude-3-opus')
        api_key: API key for cloud providers (optional)
        base_url: Custom API endpoint (for local/custom providers)
        privacy_mode: If True, no data sent to cloud (local only)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts for failed requests
        requests_per_minute: Rate limit for API requests (API-020)
        enable_cache: If True, cache responses for repeated queries (API-020)
        track_costs: If True, track token usage and costs (API-020)
    """

    provider: LLMProvider = LLMProvider.LOCAL
    model: str = "default"
    api_key: str | None = None
    base_url: str | None = None
    privacy_mode: bool = True
    timeout: float = 30.0
    max_retries: int = 3
    requests_per_minute: int = 60
    enable_cache: bool = False
    track_costs: bool = True


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (API-019: token counting).

    Uses approximate character-to-token ratio. Actual count varies by model.

    Args:
        text: Input text to estimate tokens for

    Returns:
        Estimated token count (roughly 4 characters per token)
    """
    # Average ~4 chars per token for English text
    return max(1, len(text) // 4)


@dataclass
class LLMResponse:
    """Response from LLM query.

    Attributes:
        answer: Main text response
        confidence: Confidence score (0-1) if available
        suggested_commands: List of suggested Oscura commands
        metadata: Additional metadata from LLM
        raw_response: Raw response data for debugging
        estimated_cost: Estimated cost in USD for this request (API-020)
        cached: Whether this response was served from cache (API-020)
    """

    answer: str
    confidence: float | None = None
    suggested_commands: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] | None = None
    estimated_cost: float = 0.0
    cached: bool = False


class LLMClient(Protocol):
    """Protocol for LLM client implementations."""

    def query(self, prompt: str, context: dict[str, Any]) -> LLMResponse:
        """Send query to LLM.

        Args:
            prompt: User prompt
            context: Analysis context (trace metadata, etc.)
        """
        ...

    def analyze(self, trace: Any, question: str) -> LLMResponse:
        """Analyze trace with natural language question.

        Args:
            trace: Trace object
            question: Natural language question
        """
        ...

    def explain(self, measurement: Any) -> str:
        """Explain a measurement result.

        Args:
            measurement: Measurement result
        """
        ...


class LLMError(OscuraError):
    """LLM integration error."""


class LLMIntegration:
    """LLM integration manager.

    Provides hooks for LLM-assisted analysis and natural language interfaces.
    """

    def __init__(self, config: LLMConfig | None = None):
        """Initialize LLM integration.

        Args:
            config: LLM configuration (defaults to privacy mode)
        """
        self.config = config or LLMConfig()
        self._client: LLMClient | None = None
        self._hooks: dict[AnalysisHook, list[Callable]] = {  # type: ignore[type-arg]
            AnalysisHook.BEFORE_ANALYSIS: [],
            AnalysisHook.AFTER_ANALYSIS: [],
            AnalysisHook.ON_ERROR: [],
        }

    def configure(
        self, provider: str, model: str, api_key: str | None = None, **kwargs: Any
    ) -> None:
        """Configure LLM provider.

        Args:
            provider: Provider name ('openai', 'anthropic', 'local', 'custom')
            model: Model identifier
            api_key: API key for cloud providers
            **kwargs: Additional configuration options

        Raises:
            LLMError: If provider is unknown
        """
        try:
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            raise LLMError(f"Unknown provider: {provider}")

        self.config = LLMConfig(
            provider=provider_enum,
            model=model,
            api_key=api_key,
            base_url=kwargs.get("base_url"),
            privacy_mode=kwargs.get("privacy_mode", provider_enum == LLMProvider.LOCAL),
            timeout=kwargs.get("timeout", 30.0),
            max_retries=kwargs.get("max_retries", 3),
            requests_per_minute=kwargs.get("requests_per_minute", 60),
        )

        # Reset client to force reinitialization
        self._client = None

    def _get_client(self) -> LLMClient:
        """Get or create LLM client.

        Returns:
            LLM client instance
        """
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> LLMClient:
        """Create LLM client based on configuration.

        Returns:
            LLM client instance

        Raises:
            LLMError: If client cannot be created
        """
        if self.config.provider == LLMProvider.OPENAI:
            return self._create_openai_client()
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return self._create_anthropic_client()
        elif self.config.provider == LLMProvider.LOCAL:
            return self._create_local_client()
        else:
            raise LLMError(f"Provider not implemented: {self.config.provider.value}")

    def _create_openai_client(self) -> LLMClient:
        """Create OpenAI client.

        Returns:
            OpenAI client

        Raises:
            LLMError: If OpenAI package not available or configuration invalid
        """
        try:
            import openai  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            raise LLMError("OpenAI package not installed. Install with: pip install openai")

        if not self.config.api_key:
            raise LLMError("OpenAI API key required")

        if self.config.privacy_mode:
            raise LLMError("Privacy mode not compatible with OpenAI (cloud provider)")

        return OpenAIClient(self.config)

    def _create_anthropic_client(self) -> LLMClient:
        """Create Anthropic client.

        Returns:
            Anthropic client

        Raises:
            LLMError: If Anthropic package not available or configuration invalid
        """
        try:
            import anthropic  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            raise LLMError("Anthropic package not installed. Install with: pip install anthropic")

        if not self.config.api_key:
            raise LLMError("Anthropic API key required")

        if self.config.privacy_mode:
            raise LLMError("Privacy mode not compatible with Anthropic (cloud provider)")

        return AnthropicClient(self.config)

    def _create_local_client(self) -> LLMClient:
        """Create local LLM client.

        Returns:
            Local client (mock/stub for now)
        """
        return LocalLLMClient(self.config)

    def register_hook(self, hook: AnalysisHook, callback: Callable) -> None:  # type: ignore[type-arg]
        """Register callback for analysis hook.

        Args:
            hook: Hook point
            callback: Callback function
        """
        self._hooks[hook].append(callback)

    def trigger_hook(self, hook: AnalysisHook, *args: Any, **kwargs: Any) -> None:
        """Trigger all callbacks for a hook.

        Args:
            hook: Hook point
            *args: Positional arguments for callbacks
            **kwargs: Keyword arguments for callbacks
        """
        for callback in self._hooks[hook]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                # Don't let hook errors break analysis
                print(f"Warning: Hook {hook.value} failed: {e}")

    def prepare_context(self, trace: Any) -> dict[str, Any]:
        """Prepare trace metadata for LLM context.

        Args:
            trace: Trace object

        Returns:
            Context dictionary with trace metadata
        """
        context = {
            "type": type(trace).__name__,
        }

        # Extract common metadata
        if hasattr(trace, "metadata"):
            meta = trace.metadata
            context.update(
                {
                    "sample_rate": getattr(meta, "sample_rate", None),  # type: ignore[dict-item]
                    "num_samples": getattr(meta, "num_samples", None),  # type: ignore[dict-item]
                    "duration": getattr(meta, "duration", None),  # type: ignore[dict-item]
                }
            )

        # Data statistics (without sending actual data in privacy mode)
        if hasattr(trace, "data") and not self.config.privacy_mode:
            import numpy as np

            data = trace.data
            context["statistics"] = {  # type: ignore[assignment]
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
            }
        elif self.config.privacy_mode:
            # Compute hash of data for change detection without sending data
            if hasattr(trace, "data"):
                import numpy as np

                data_bytes = trace.data.tobytes()
                context["data_hash"] = hashlib.sha256(data_bytes).hexdigest()[:16]

        return context

    def analyze(self, trace: Any, question: str) -> LLMResponse:
        """Analyze trace with natural language question.

        Args:
            trace: Trace object
            question: Natural language question

        Returns:
            LLM response with answer and suggestions

        Raises:
            LLMError: If analysis fails
        """
        self.trigger_hook(AnalysisHook.BEFORE_ANALYSIS, trace, question)

        try:
            client = self._get_client()
            response = client.analyze(trace, question)
            self.trigger_hook(AnalysisHook.AFTER_ANALYSIS, trace, response)
            return response

        except Exception as e:
            self.trigger_hook(AnalysisHook.ON_ERROR, trace, question, e)
            raise LLMError(f"LLM analysis failed: {e}")

    def explain(self, measurement: Any) -> str:
        """Explain a measurement result.

        Args:
            measurement: Measurement result to explain

        Returns:
            Explanation text
        """
        client = self._get_client()
        return client.explain(measurement)


# Stub implementations for different providers


class OpenAIClient:
    """OpenAI client implementation.

    Full implementation.:
    - chat_completion() with retry logic
    - analyze_trace() for trace analysis
    - suggest_measurements() for measurement recommendations
    - Error handling for API failures, rate limits, timeouts
    - API key from OPENAI_API_KEY environment variable
    """

    def __init__(self, config: LLMConfig):
        """Initialize OpenAI client.

        Args:
            config: LLM configuration

        Raises:
            LLMError: If openai package not available
        """
        self.config = config
        self.rate_limiter = RateLimiter(config.requests_per_minute)

        # Import and initialize OpenAI client
        try:
            import openai

            self._openai = openai
        except ImportError:
            raise LLMError("OpenAI package not installed. Install with: pip install openai")

        # Get API key from config or environment
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key to configure()"
            )

        # Initialize OpenAI client
        self.client = self._openai.OpenAI(api_key=api_key, timeout=config.timeout)

    def chat_completion(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Send chat completion request with retry logic.

        Full implementation with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for OpenAI API

        Returns:
            LLM response with answer and metadata

        Raises:
            LLMError: If API request fails after retries
        """
        self.rate_limiter.acquire()

        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model, messages=messages, **kwargs
                )

                # Extract response content
                answer = response.choices[0].message.content or ""

                # Track costs.
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                estimated_cost = 0.0

                if self.config.track_costs:
                    estimated_cost = _global_cost_tracker.record(
                        response.model, input_tokens, output_tokens
                    )

                return LLMResponse(
                    answer=answer,
                    confidence=None,  # OpenAI doesn't provide confidence scores
                    suggested_commands=[],
                    metadata={
                        "model": response.model,
                        "usage": {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": response.usage.total_tokens if response.usage else 0,
                        },
                        "finish_reason": response.choices[0].finish_reason,
                    },
                    raw_response={
                        "id": response.id,
                        "created": response.created,
                    },
                    estimated_cost=estimated_cost,
                )

            except self._openai.RateLimitError as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff for rate limits
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                    continue
                raise LLMError(f"OpenAI rate limit exceeded: {e}")

            except self._openai.APITimeoutError as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)
                    continue
                raise LLMError(f"OpenAI request timeout: {e}")

            except self._openai.APIError as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)
                    continue
                raise LLMError(f"OpenAI API error: {e}")

            except Exception as e:
                last_exception = e
                raise LLMError(f"OpenAI request failed: {e}")

        raise LLMError(
            f"OpenAI request failed after {self.config.max_retries} retries: {last_exception}"
        )

    def analyze_trace(self, trace: Any, question: str) -> LLMResponse:
        """Analyze trace with question.

        Send trace summary, get insights.

        Args:
            trace: Trace object
            question: Natural language question about the trace

        Returns:
            LLM response with analysis
        """
        # Prepare trace summary
        trace_summary = self._summarize_trace(trace)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in signal analysis and oscilloscope data. "
                    "Analyze the provided trace data and answer questions accurately. "
                    "Provide specific, actionable insights."
                ),
            },
            {
                "role": "user",
                "content": f"Trace Summary:\n{trace_summary}\n\nQuestion: {question}",
            },
        ]

        return self.chat_completion(messages)

    def suggest_measurements(self, trace: Any) -> LLMResponse:
        """Suggest measurements based on trace characteristics.

        Recommend measurements based on trace.

        Args:
            trace: Trace object

        Returns:
            LLM response with measurement suggestions
        """
        trace_summary = self._summarize_trace(trace)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in signal analysis. Based on trace characteristics, "
                    "suggest relevant measurements. Provide 3-5 specific measurement recommendations "
                    "with brief explanations."
                ),
            },
            {
                "role": "user",
                "content": f"Trace Summary:\n{trace_summary}\n\nWhat measurements would be most informative for this trace?",
            },
        ]

        response = self.chat_completion(messages)

        # Try to extract suggested commands from the response
        suggested_commands = self._extract_commands(response.answer)
        response.suggested_commands = suggested_commands

        return response

    def _summarize_trace(self, trace: Any) -> str:
        """Create a text summary of trace for LLM context.

        Args:
            trace: Trace object

        Returns:
            Text summary of trace characteristics
        """
        summary_parts = [f"Trace Type: {type(trace).__name__}"]

        # Extract metadata
        if hasattr(trace, "metadata"):
            meta = trace.metadata
            if hasattr(meta, "sample_rate"):
                summary_parts.append(f"Sample Rate: {meta.sample_rate:.2e} Hz")
            if hasattr(meta, "num_samples"):
                summary_parts.append(f"Number of Samples: {meta.num_samples:,}")
            if hasattr(meta, "duration"):
                summary_parts.append(f"Duration: {meta.duration:.6f} s")

        # Data statistics
        if hasattr(trace, "data"):
            import numpy as np

            data = trace.data
            summary_parts.extend(
                [
                    f"Mean: {np.mean(data):.6e}",
                    f"Std Dev: {np.std(data):.6e}",
                    f"Min: {np.min(data):.6e}",
                    f"Max: {np.max(data):.6e}",
                    f"Peak-to-Peak: {np.ptp(data):.6e}",
                ]
            )

        return "\n".join(summary_parts)

    def _extract_commands(self, text: str) -> list[str]:
        """Extract suggested Oscura commands from LLM response.

        Args:
            text: LLM response text

        Returns:
            List of extracted command strings
        """
        commands = []
        # Look for common measurement names
        measurement_keywords = [
            "rise_time",
            "fall_time",
            "frequency",
            "period",
            "amplitude",
            "rms",
            "thd",
            "snr",
            "fft",
            "psd",
            "peak",
            "duty_cycle",
        ]

        text_lower = text.lower()
        for keyword in measurement_keywords:
            if keyword in text_lower:
                commands.append(f"measure {keyword}")

        return commands

    def query(self, prompt: str, context: dict[str, Any]) -> LLMResponse:
        """Send query to LLM with context.

        Args:
            prompt: User prompt
            context: Analysis context

        Returns:
            LLM response
        """
        context_str = json.dumps(context, indent=2)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for signal analysis.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuery: {prompt}",
            },
        ]
        return self.chat_completion(messages)

    def analyze(self, trace: Any, question: str) -> LLMResponse:
        """Analyze trace with natural language question.

        Args:
            trace: Trace object
            question: Natural language question

        Returns:
            Analysis response
        """
        return self.analyze_trace(trace, question)

    def explain(self, measurement: Any) -> str:
        """Explain a measurement result.

        Args:
            measurement: Measurement result

        Returns:
            Explanation text
        """
        messages = [
            {
                "role": "system",
                "content": "You are an expert in signal measurement interpretation. Explain measurement results clearly and concisely.",
            },
            {
                "role": "user",
                "content": f"Explain this measurement result: {measurement}",
            },
        ]
        response = self.chat_completion(messages)
        return response.answer


def _convert_anthropic_messages(
    messages: list[dict[str, str]], system: str | None
) -> tuple[list[dict[str, str]], str | None]:
    """Convert messages to Anthropic format (separate system from user messages).

    Args:
        messages: Original messages with mixed roles.
        system: Optional system prompt.

    Returns:
        Tuple of (user_messages, system_message).
    """
    user_messages = []
    system_message = system

    for msg in messages:
        if msg["role"] == "system" and not system_message:
            system_message = msg["content"]
        elif msg["role"] in ["user", "assistant"]:
            user_messages.append(msg)

    return (user_messages, system_message)


def _extract_anthropic_answer(response: Any) -> str:
    """Extract text answer from Anthropic response.

    Args:
        response: Anthropic API response.

    Returns:
        Concatenated text from all content blocks.
    """
    answer = ""
    for block in response.content:
        if hasattr(block, "text"):
            answer += block.text
    return answer


def _build_anthropic_response(response: Any, answer: str, estimated_cost: float) -> LLMResponse:
    """Build LLMResponse from Anthropic API response.

    Args:
        response: Anthropic API response.
        answer: Extracted answer text.
        estimated_cost: Estimated API cost.

    Returns:
        LLMResponse with metadata.
    """
    return LLMResponse(
        answer=answer,
        confidence=None,  # Anthropic doesn't provide confidence scores
        suggested_commands=[],
        metadata={
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "stop_reason": response.stop_reason,
        },
        raw_response={
            "id": response.id,
            "type": response.type,
        },
        estimated_cost=estimated_cost,
    )


class AnthropicClient:
    """Anthropic client implementation.

    Full implementation.:
    - chat_completion() with retry logic
    - analyze_trace() for trace analysis
    - suggest_measurements() for measurement recommendations
    - API key from ANTHROPIC_API_KEY environment variable
    """

    def __init__(self, config: LLMConfig):
        """Initialize Anthropic client.

        Args:
            config: LLM configuration

        Raises:
            LLMError: If anthropic package not available
        """
        self.config = config
        self.rate_limiter = RateLimiter(config.requests_per_minute)

        # Import and initialize Anthropic client
        try:
            import anthropic

            self._anthropic = anthropic
        except ImportError:
            raise LLMError("Anthropic package not installed. Install with: pip install anthropic")

        # Get API key from config or environment
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key to configure()"
            )

        # Initialize Anthropic client
        self.client = self._anthropic.Anthropic(api_key=api_key, timeout=config.timeout)

    def chat_completion(
        self, messages: list[dict[str, str]], system: str | None = None, **kwargs: Any
    ) -> LLMResponse:
        """Send chat completion request with retry logic.

        Full implementation with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system: System prompt (optional)
            **kwargs: Additional parameters for Anthropic API

        Returns:
            LLM response with answer and metadata

        Raises:
            LLMError: If API request fails after retries
        """
        self.rate_limiter.acquire()

        # Convert messages format for Anthropic
        user_messages, system_message = _convert_anthropic_messages(messages, system)

        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                # Build and send request
                response = self._send_anthropic_request(user_messages, system_message, kwargs)

                # Extract answer from response
                answer = _extract_anthropic_answer(response)

                # Track token usage and costs
                estimated_cost = self._track_anthropic_costs(response)

                # Build and return LLM response
                return _build_anthropic_response(response, answer, estimated_cost)

            except self._anthropic.RateLimitError as e:
                last_exception = e
                if not self._handle_rate_limit_retry(attempt):
                    raise LLMError(f"Anthropic rate limit exceeded: {e}")

            except self._anthropic.APITimeoutError as e:
                last_exception = e
                if not self._handle_timeout_retry(attempt):
                    raise LLMError(f"Anthropic request timeout: {e}")

            except self._anthropic.APIError as e:
                last_exception = e
                if not self._handle_api_error_retry(attempt):
                    raise LLMError(f"Anthropic API error: {e}")

            except Exception as e:
                last_exception = e
                raise LLMError(f"Anthropic request failed: {e}")

        raise LLMError(
            f"Anthropic request failed after {self.config.max_retries} retries: {last_exception}"
        )

    def _send_anthropic_request(
        self,
        user_messages: list[dict[str, str]],
        system_message: str | None,
        kwargs: dict[str, Any],
    ) -> Any:
        """Send request to Anthropic API.

        Args:
            user_messages: Filtered user/assistant messages.
            system_message: System prompt.
            kwargs: Additional API parameters.

        Returns:
            Anthropic API response.
        """
        request_params = {
            "model": self.config.model,
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
        }
        if system_message:
            request_params["system"] = system_message

        # Add optional parameters
        for key in ["temperature", "top_p", "top_k"]:
            if key in kwargs:
                request_params[key] = kwargs[key]

        return self.client.messages.create(**request_params)

    def _track_anthropic_costs(self, response: Any) -> float:
        """Track token usage and return estimated cost.

        Args:
            response: Anthropic API response.

        Returns:
            Estimated cost in USD.
        """
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        if self.config.track_costs:
            return _global_cost_tracker.record(response.model, input_tokens, output_tokens)
        return 0.0

    def _handle_rate_limit_retry(self, attempt: int) -> bool:
        """Handle rate limit error with exponential backoff.

        Args:
            attempt: Current attempt number.

        Returns:
            True if should continue retry, False if should raise.
        """
        if attempt < self.config.max_retries - 1:
            wait_time = 2**attempt
            time.sleep(wait_time)
            return True
        return False

    def _handle_timeout_retry(self, attempt: int) -> bool:
        """Handle timeout error with retry.

        Args:
            attempt: Current attempt number.

        Returns:
            True if should continue retry, False if should raise.
        """
        if attempt < self.config.max_retries - 1:
            time.sleep(1)
            return True
        return False

    def _handle_api_error_retry(self, attempt: int) -> bool:
        """Handle API error with retry.

        Args:
            attempt: Current attempt number.

        Returns:
            True if should continue retry, False if should raise.
        """
        if attempt < self.config.max_retries - 1:
            time.sleep(1)
            return True
        return False

    def analyze_trace(self, trace: Any, question: str) -> LLMResponse:
        """Analyze trace with question.

        Trace analysis with Anthropic.

        Args:
            trace: Trace object
            question: Natural language question about the trace

        Returns:
            LLM response with analysis
        """
        # Prepare trace summary
        trace_summary = self._summarize_trace(trace)

        system_prompt = (
            "You are an expert in signal analysis and oscilloscope data. "
            "Analyze the provided trace data and answer questions accurately. "
            "Provide specific, actionable insights."
        )

        messages = [
            {
                "role": "user",
                "content": f"Trace Summary:\n{trace_summary}\n\nQuestion: {question}",
            },
        ]

        return self.chat_completion(messages, system=system_prompt)

    def suggest_measurements(self, trace: Any) -> LLMResponse:
        """Suggest measurements based on trace characteristics.

        Measurement recommendations.

        Args:
            trace: Trace object

        Returns:
            LLM response with measurement suggestions
        """
        trace_summary = self._summarize_trace(trace)

        system_prompt = (
            "You are an expert in signal analysis. Based on trace characteristics, "
            "suggest relevant measurements. Provide 3-5 specific measurement recommendations "
            "with brief explanations."
        )

        messages = [
            {
                "role": "user",
                "content": f"Trace Summary:\n{trace_summary}\n\nWhat measurements would be most informative for this trace?",
            },
        ]

        response = self.chat_completion(messages, system=system_prompt)

        # Try to extract suggested commands from the response
        suggested_commands = self._extract_commands(response.answer)
        response.suggested_commands = suggested_commands

        return response

    def _summarize_trace(self, trace: Any) -> str:
        """Create a text summary of trace for LLM context.

        Args:
            trace: Trace object

        Returns:
            Text summary of trace characteristics
        """
        summary_parts = [f"Trace Type: {type(trace).__name__}"]

        # Extract metadata
        if hasattr(trace, "metadata"):
            meta = trace.metadata
            if hasattr(meta, "sample_rate"):
                summary_parts.append(f"Sample Rate: {meta.sample_rate:.2e} Hz")
            if hasattr(meta, "num_samples"):
                summary_parts.append(f"Number of Samples: {meta.num_samples:,}")
            if hasattr(meta, "duration"):
                summary_parts.append(f"Duration: {meta.duration:.6f} s")

        # Data statistics
        if hasattr(trace, "data"):
            import numpy as np

            data = trace.data
            summary_parts.extend(
                [
                    f"Mean: {np.mean(data):.6e}",
                    f"Std Dev: {np.std(data):.6e}",
                    f"Min: {np.min(data):.6e}",
                    f"Max: {np.max(data):.6e}",
                    f"Peak-to-Peak: {np.ptp(data):.6e}",
                ]
            )

        return "\n".join(summary_parts)

    def _extract_commands(self, text: str) -> list[str]:
        """Extract suggested Oscura commands from LLM response.

        Args:
            text: LLM response text

        Returns:
            List of extracted command strings
        """
        commands = []
        # Look for common measurement names
        measurement_keywords = [
            "rise_time",
            "fall_time",
            "frequency",
            "period",
            "amplitude",
            "rms",
            "thd",
            "snr",
            "fft",
            "psd",
            "peak",
            "duty_cycle",
        ]

        text_lower = text.lower()
        for keyword in measurement_keywords:
            if keyword in text_lower:
                commands.append(f"measure {keyword}")

        return commands

    def query(self, prompt: str, context: dict[str, Any]) -> LLMResponse:
        """Send query to LLM with context.

        Args:
            prompt: User prompt
            context: Analysis context

        Returns:
            LLM response
        """
        context_str = json.dumps(context, indent=2)
        system_prompt = "You are a helpful assistant for signal analysis."
        messages = [
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuery: {prompt}",
            },
        ]
        return self.chat_completion(messages, system=system_prompt)

    def analyze(self, trace: Any, question: str) -> LLMResponse:
        """Analyze trace with natural language question.

        Args:
            trace: Trace object
            question: Natural language question

        Returns:
            Analysis response
        """
        return self.analyze_trace(trace, question)

    def explain(self, measurement: Any) -> str:
        """Explain a measurement result.

        Args:
            measurement: Measurement result

        Returns:
            Explanation text
        """
        system_prompt = "You are an expert in signal measurement interpretation. Explain measurement results clearly and concisely."
        messages = [
            {
                "role": "user",
                "content": f"Explain this measurement result: {measurement}",
            },
        ]
        response = self.chat_completion(messages, system=system_prompt)
        return response.answer


class LocalLLMClient:
    """Local LLM client (mock implementation)."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def query(self, prompt: str, context: dict[str, Any]) -> LLMResponse:
        """Mock query implementation."""
        return LLMResponse(
            answer="Local LLM not configured. This is a mock response.",
            confidence=0.0,
            suggested_commands=[],
            metadata={"mock": True},
        )

    def analyze(self, trace: Any, question: str) -> LLMResponse:
        """Mock analysis implementation."""
        # Simple heuristic-based responses
        question_lower = question.lower()

        if "protocol" in question_lower:
            return LLMResponse(
                answer="Unable to determine protocol without LLM. Try manual inspection.",
                confidence=0.0,
                suggested_commands=[
                    "measure frequency",
                    "plot $trace",
                ],
            )

        return LLMResponse(
            answer=f"Local LLM analysis not available. Question was: {question}",
            confidence=0.0,
            suggested_commands=["measure all"],
        )

    def explain(self, measurement: Any) -> str:
        """Mock explanation implementation."""
        return f"Measurement result: {measurement}. Local LLM explanation not available."


def get_provider(name: str, **config_kwargs: Any) -> LLMClient:
    """Get LLM provider by name with unified interface.

    get_provider(name: str) factory function.

    Args:
        name: Provider name ('openai', 'anthropic', 'local')
        **config_kwargs: Configuration parameters for the provider

    Returns:
        LLM client instance

    Raises:
        LLMError: If provider unknown or configuration invalid

    Examples:
        >>> # Get OpenAI provider
        >>> client = get_provider('openai', model='gpt-4', api_key='...')
        >>> response = client.analyze(trace, "What is the frequency?")
        >>>
        >>> # Get Anthropic provider with rate limiting
        >>> client = get_provider('anthropic', model='claude-3-opus-20240229',
        ...                       requests_per_minute=30)
        >>> response = client.suggest_measurements(trace)
        >>>
        >>> # Get local provider (no API key needed)
        >>> client = get_provider('local')
        >>> response = client.analyze(trace, "Analyze this signal")
    """
    try:
        provider_enum = LLMProvider(name.lower())
    except ValueError:
        raise LLMError(f"Unknown provider: {name}. Available: {[p.value for p in LLMProvider]}")

    # Build config with sensible defaults
    config = LLMConfig(
        provider=provider_enum,
        model=config_kwargs.get("model", "default"),
        api_key=config_kwargs.get("api_key"),
        base_url=config_kwargs.get("base_url"),
        privacy_mode=config_kwargs.get("privacy_mode", provider_enum == LLMProvider.LOCAL),
        timeout=config_kwargs.get("timeout", 30.0),
        max_retries=config_kwargs.get("max_retries", 3),
        requests_per_minute=config_kwargs.get("requests_per_minute", 60),
    )

    # Create appropriate client with graceful degradation
    try:
        if provider_enum == LLMProvider.OPENAI:
            return OpenAIClient(config)
        elif provider_enum == LLMProvider.ANTHROPIC:
            return AnthropicClient(config)
        elif provider_enum == LLMProvider.LOCAL:
            return LocalLLMClient(config)
        else:
            # .: Graceful degradation
            raise LLMError(
                f"Provider {name} not yet implemented. "
                "Falling back to local provider is recommended."
            )
    except ImportError as e:
        # .: Graceful degradation when API unavailable
        raise LLMError(
            f"Provider {name} unavailable: {e}. "
            "Install the required package or use 'local' provider."
        )


# Global LLM integration instance
_global_llm: LLMIntegration | None = None


def get_llm() -> LLMIntegration:
    """Get global LLM integration instance.

    Returns:
        Global LLM integration instance
    """
    global _global_llm
    if _global_llm is None:
        _global_llm = LLMIntegration()
    return _global_llm


def configure(provider: str, model: str, **kwargs: Any) -> None:
    """Configure global LLM integration.

    Args:
        provider: Provider name
        model: Model identifier
        **kwargs: Additional configuration
    """
    llm = get_llm()
    llm.configure(provider, model, **kwargs)


def analyze(trace: Any, question: str) -> LLMResponse:
    """Analyze trace with LLM.

    Args:
        trace: Trace object
        question: Natural language question

    Returns:
        LLM response
    """
    llm = get_llm()
    return llm.analyze(trace, question)


def explain(measurement: Any) -> str:
    """Explain measurement with LLM.

    Args:
        measurement: Measurement result

    Returns:
        Explanation text
    """
    llm = get_llm()
    return llm.explain(measurement)


# ==============================================================================
# ==============================================================================


def get_client(provider: str | None = None, **config_kwargs: Any) -> LLMClient:
    """Get LLM client with optional auto-selection.

    get_client(provider: str) -> LLMClient.
    Alias for get_provider() with auto-selection support.

    Args:
        provider: Provider name ('openai', 'anthropic', 'local'), or None for auto-select
        **config_kwargs: Configuration parameters for the provider

    Returns:
        LLM client instance

    Examples:
        >>> # Auto-select based on available API keys
        >>> client = get_client()
        >>>
        >>> # Explicit provider selection
        >>> client = get_client("openai", model="gpt-4")
    """
    if provider is not None:
        return get_provider(provider, **config_kwargs)

    # Auto-selection: try providers in preference order
    return get_client_auto(**config_kwargs)


def get_client_auto(**config_kwargs: Any) -> LLMClient:
    """Automatically select an available LLM provider.

    Automatic provider selection based on availability.

    Checks for API keys in environment and returns first available provider:
    1. OpenAI (if OPENAI_API_KEY set)
    2. Anthropic (if ANTHROPIC_API_KEY set)
    3. Local (fallback, always available)

    Args:
        **config_kwargs: Configuration parameters for the provider

    Returns:
        LLM client instance for the first available provider

    Examples:
        >>> client = get_client_auto(model="gpt-4")  # Uses OpenAI if key available
    """
    # Check for OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return get_provider("openai", **config_kwargs)
        except LLMError:
            pass  # Fall through to next provider

    # Check for Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return get_provider("anthropic", **config_kwargs)
        except LLMError:
            pass  # Fall through to next provider

    # Default to local
    return get_provider("local", **config_kwargs)


def get_client_with_failover(
    providers: list[str] | None = None, **config_kwargs: Any
) -> FailoverLLMClient:
    """Get LLM client with automatic failover between providers.

    Failover logic (try OpenAI, fallback to Anthropic).

    Args:
        providers: List of provider names in preference order.
                   Default: ["openai", "anthropic", "local"]
        **config_kwargs: Configuration parameters for providers

    Returns:
        FailoverLLMClient that tries providers in order

    Examples:
        >>> client = get_client_with_failover(
        ...     providers=["openai", "anthropic"],
        ...     model="gpt-4"
        ... )
        >>> response = client.chat_completion("Hello")  # Tries OpenAI, then Anthropic
    """
    if providers is None:
        providers = ["openai", "anthropic", "local"]

    return FailoverLLMClient(providers, **config_kwargs)


class FailoverLLMClient:
    """LLM client wrapper with automatic failover between providers.

    .: Failover logic for provider availability.

        Attempts each provider in order until one succeeds. Useful for
        handling API outages or rate limiting gracefully.
    """

    def __init__(self, providers: list[str], **config_kwargs: Any):
        """Initialize failover client.

        Args:
            providers: List of provider names in preference order
            **config_kwargs: Configuration parameters for providers
        """
        self.providers = providers
        self.config_kwargs = config_kwargs
        self._clients: dict[str, LLMClient] = {}
        self._last_successful_provider: str | None = None

    def _get_or_create_client(self, provider: str) -> LLMClient | None:
        """Get or create client for provider.

        Args:
            provider: Provider name

        Returns:
            LLM client or None if unavailable
        """
        if provider not in self._clients:
            try:
                self._clients[provider] = get_provider(provider, **self.config_kwargs)
            except LLMError:
                return None
        return self._clients.get(provider)

    def _try_providers(self, operation: Callable[[LLMClient], Any]) -> Any:
        """Try operation on each provider until one succeeds.

        Args:
            operation: Callable that takes a client and returns result

        Returns:
            Result from first successful provider

        Raises:
            LLMError: If all providers fail
        """
        errors = []

        # Try last successful provider first for efficiency
        if self._last_successful_provider:
            reordered = [self._last_successful_provider] + [
                p for p in self.providers if p != self._last_successful_provider
            ]
        else:
            reordered = self.providers

        for provider in reordered:
            client = self._get_or_create_client(provider)
            if client is None:
                errors.append(f"{provider}: not available")
                continue

            try:
                result = operation(client)
                self._last_successful_provider = provider
                return result
            except Exception as e:
                errors.append(f"{provider}: {e}")
                continue

        raise LLMError(f"All providers failed: {'; '.join(errors)}")

    def chat_completion(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Send chat completion with failover.

        Args:
            prompt: User prompt
            model: Model name (optional, uses config default)
            **kwargs: Additional parameters

        Returns:
            Response text from first successful provider
        """

        def operation(client: LLMClient) -> str:
            if hasattr(client, "chat_completion"):
                messages = [{"role": "user", "content": prompt}]
                response = client.chat_completion(messages, **kwargs)
                return response.answer  # type: ignore[no-any-return]
            else:
                response = client.query(prompt, {})
                return response.answer

        return self._try_providers(operation)  # type: ignore[no-any-return]

    def analyze_trace(self, trace_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze trace data with failover.

        Args:
            trace_data: Dictionary containing trace information

        Returns:
            Analysis results dictionary
        """

        def operation(client: LLMClient) -> dict[str, Any]:
            # Create mock trace object from dict
            class DictTrace:
                def __init__(self, data: dict[str, Any]):
                    self._data = data
                    for k, v in data.items():
                        setattr(self, k, v)

            trace = DictTrace(trace_data)

            if hasattr(client, "analyze_trace"):
                response = client.analyze_trace(trace, "Analyze this signal")
            else:
                response = client.analyze(trace, "Analyze this signal")

            return {
                "answer": response.answer,
                "suggested_commands": response.suggested_commands,
                "metadata": response.metadata,
            }

        return self._try_providers(operation)  # type: ignore[no-any-return]

    def suggest_measurements(self, signal_characteristics: dict[str, Any]) -> list[str]:
        """Suggest measurements based on signal characteristics.

        Args:
            signal_characteristics: Dictionary describing the signal

        Returns:
            List of suggested measurement names
        """

        def operation(client: LLMClient) -> list[str]:
            # Create mock trace from characteristics
            class CharTrace:
                def __init__(self, chars: dict[str, Any]):
                    self.metadata = type("Meta", (), chars)()
                    self.data = None

            trace = CharTrace(signal_characteristics)

            if hasattr(client, "suggest_measurements"):
                response = client.suggest_measurements(trace)
            else:
                response = client.analyze(trace, "What measurements should I perform?")

            return response.suggested_commands  # type: ignore[no-any-return]

        return self._try_providers(operation)  # type: ignore[no-any-return]

    def query(self, prompt: str, context: dict[str, Any]) -> LLMResponse:
        """Send query with failover.

        Args:
            prompt: User prompt
            context: Analysis context

        Returns:
            LLM response
        """
        return self._try_providers(lambda c: c.query(prompt, context))  # type: ignore[no-any-return]

    def analyze(self, trace: Any, question: str) -> LLMResponse:
        """Analyze trace with failover.

        Args:
            trace: Trace object
            question: Natural language question

        Returns:
            Analysis response
        """
        return self._try_providers(lambda c: c.analyze(trace, question))  # type: ignore[no-any-return]

    def explain(self, measurement: Any) -> str:
        """Explain measurement with failover.

        Args:
            measurement: Measurement result

        Returns:
            Explanation text
        """
        return self._try_providers(lambda c: c.explain(measurement))  # type: ignore[no-any-return]


def is_provider_available(provider: str) -> bool:
    """Check if a provider is available (API key set, package installed).

    Check provider availability.

    Args:
        provider: Provider name to check

    Returns:
        True if provider can be initialized

    Examples:
        >>> if is_provider_available("openai"):
        ...     client = get_client("openai")
    """
    if provider == "local":
        return True

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401

            return True
        except ImportError:
            return False

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401

            return True
        except ImportError:
            return False

    return False


def list_available_providers() -> list[str]:
    """List all currently available LLM providers.

    Discover available providers.

    Returns:
        List of provider names that can be used

    Examples:
        >>> providers = list_available_providers()
        >>> print(providers)  # ['openai', 'local'] if OpenAI key is set
    """
    return [provider.value for provider in LLMProvider if is_provider_available(provider.value)]
