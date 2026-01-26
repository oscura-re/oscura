"""Oscura Integrations.

Third-party and external system integrations.
"""

from oscura.api.integrations.llm import (
    AnalysisHook,
    CostTracker,
    FailoverLLMClient,
    LLMConfig,
    LLMError,
    LLMIntegration,
    LLMProvider,
    LLMResponse,
    ResponseCache,
    analyze,
    configure,
    estimate_tokens,
    explain,
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

__all__ = [
    # Core classes
    "AnalysisHook",
    "CostTracker",
    "FailoverLLMClient",
    "LLMConfig",
    "LLMError",
    "LLMIntegration",
    "LLMProvider",
    "LLMResponse",
    "ResponseCache",
    # High-level functions
    "analyze",
    "configure",
    # Cost and caching utilities (API-020)
    "estimate_tokens",
    "explain",
    # Client factory functions (API-020)
    "get_client",
    "get_client_auto",
    "get_client_with_failover",
    "get_cost_tracker",
    "get_llm",
    "get_provider",
    "get_response_cache",
    # Provider discovery (API-020)
    "is_provider_available",
    "list_available_providers",
]
