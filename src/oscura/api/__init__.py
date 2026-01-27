"""Expert API module for Oscura.

This module provides advanced APIs for power users including DSL,
fluent interfaces, performance profiling, REST API server, and advanced workflow control.
"""

from oscura.api.dsl import (
    Expression,
)
from oscura.api.dsl import (
    Interpreter as DSLParser,
)
from oscura.api.dsl import (
    execute_dsl as analyze,
)
from oscura.api.dsl import (
    parse_dsl as parse_expression,
)

# Alias for backward compatibility
DSLExpression = Expression
from oscura.api.fluent import (
    FluentResult,
    FluentTrace,
    trace,
)
from oscura.api.operators import (
    TimeIndex,
    UnitConverter,
    convert_units,
    make_pipeable,
)
from oscura.api.optimization import (
    GridSearch,
    OptimizationResult,
    ParameterSpace,
    optimize_parameters,
)
from oscura.api.profiling import (
    OperationProfile,
    Profiler,
    ProfileReport,
    profile,
)
from oscura.api.rest_server import (
    AnalysisRequest,
    AnalysisResponse,
    ErrorResponse,
    ProtocolResponse,
    RESTAPIServer,
    SessionManager,
    SessionResponse,
)

__all__ = [
    # REST API Server
    "AnalysisRequest",
    "AnalysisResponse",
    # DSL (API-010)
    "DSLExpression",
    "DSLParser",
    "ErrorResponse",
    # Fluent (API-019)
    "FluentResult",
    "FluentTrace",
    # Optimization (API-014)
    "GridSearch",
    # Profiling (API-012)
    "OperationProfile",
    "OptimizationResult",
    "ParameterSpace",
    "ProfileReport",
    "Profiler",
    "ProtocolResponse",
    "RESTAPIServer",
    "SessionManager",
    "SessionResponse",
    # Operators (API-015, API-016, API-018)
    "TimeIndex",
    "UnitConverter",
    "analyze",
    "convert_units",
    "make_pipeable",
    "optimize_parameters",
    "parse_expression",
    "profile",
    "trace",
]
