"""Extension point registry and management system.

This module implements a central registry for extension points that allows
plugins and custom code to extend Oscura functionality at well-defined
integration points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")


class HookErrorPolicy(Enum):
    """Policy for handling hook errors.

    Attributes:
        CONTINUE: Continue executing remaining hooks after error
        ABORT: Stop execution immediately on error
        IGNORE: Ignore error silently
    """

    CONTINUE = auto()
    ABORT = auto()
    IGNORE = auto()


@dataclass
class ExtensionPointSpec:
    """Specification for an extension point.

    Defines the contract that implementations must follow including
    required and optional methods, version info, and documentation.

    Attributes:
        name: Unique name for the extension point
        version: API version (semver format)
        description: Human-readable description
        required_methods: List of method names that must be implemented
        optional_methods: List of optional method names
        interface: Optional interface class that implementations should inherit from

    Example:
        >>> spec = ExtensionPointSpec(
        ...     name="protocol_decoder",
        ...     version="1.0.0",
        ...     description="Decode protocol from waveform",
        ...     required_methods=["decode", "get_metadata"],
        ...     optional_methods=["configure", "reset"]
        ... )

    References:
        EXT-001: Extension Point Registry
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    required_methods: list[str] = field(default_factory=list)
    optional_methods: list[str] = field(default_factory=list)
    interface: type | None = None

    def validate_implementation(self, impl: Any) -> tuple[bool, list[str]]:
        """Validate that implementation matches interface.

        Args:
            impl: Implementation to validate

        Returns:
            Tuple of (is_valid, list of missing methods)

        Example:
            >>> is_valid, missing = spec.validate_implementation(MyDecoder())
            >>> if not is_valid:
            ...     print(f"Missing methods: {missing}")
        """
        missing = []
        for method in self.required_methods:
            if not hasattr(impl, method) or not callable(getattr(impl, method)):
                missing.append(method)
        return len(missing) == 0, missing


@dataclass
class RegisteredAlgorithm:
    """Metadata for a registered algorithm.

    Attributes:
        name: Algorithm name
        category: Algorithm category
        func: Algorithm implementation
        priority: Execution priority (higher = first)
        performance: Performance characteristics
        supports: Supported data types
        description: Human-readable description
        complexity: Time complexity string
        capabilities: Algorithm capabilities
        memory_usage: Memory usage characteristics
        registration_order: Order in which algorithm was registered

    References:
        EXT-002: Algorithm Registration (capability queries, performance metadata)
        EXT-004: Priority System (registration order for tie-breaking)
    """

    name: str
    category: str
    func: Callable[..., Any]
    priority: int = 50
    performance: dict[str, str] = field(default_factory=dict)
    supports: list[str] = field(default_factory=list)
    description: str = ""
    complexity: str = "O(n)"
    capabilities: dict[str, Any] = field(default_factory=dict)
    memory_usage: str = "unknown"
    registration_order: int = 0

    def can(self, capability: str) -> bool:
        """Check if algorithm has a specific capability.

        Args:
            capability: Capability name to check

        Returns:
            True if algorithm supports the capability

        Example:
            >>> algo.can("multi_channel")
            True

        References:
            EXT-002: Algorithm Registration (capability queries)
        """
        return self.capabilities.get(capability, False)  # type: ignore[no-any-return]

    def get_capabilities(self) -> dict[str, Any]:
        """Get all capabilities of this algorithm.

        Returns:
            Dictionary of capability names to values

        Example:
            >>> caps = algo.get_capabilities()
            >>> print(caps)
            {'multi_channel': True, 'real_time': False, 'max_sample_rate': 1000000}

        References:
            EXT-002: Algorithm Registration (capability queries)
        """
        return self.capabilities.copy()


@dataclass
class HookContext:
    """Context passed to hook functions.

    Attributes:
        data: Primary data being processed
        metadata: Additional context metadata
        abort: Set to True to abort operation
        abort_reason: Reason for abort

    Example:
        >>> @osc.hooks.register("pre_decode")
        >>> def validate_waveform(context):
        ...     if context.data.sample_rate < 1000:
        ...         context.abort = True
        ...         context.abort_reason = "Sample rate too low"
        ...     return context

    References:
        EXT-005: Hook System
    """

    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    abort: bool = False
    abort_reason: str = ""

    def __post_init__(self):  # type: ignore[no-untyped-def]
        """Initialize metadata if None."""
        if self.metadata is None:
            self.metadata = {}  # type: ignore[unreachable]


@dataclass
class RegisteredHook:
    """Registered hook function.

    Attributes:
        hook_point: Name of hook point
        func: Hook function
        priority: Execution priority (higher = first)
        name: Optional hook name
    """

    hook_point: str
    func: Callable[[HookContext], HookContext]
    priority: int = 50
    name: str = ""


class ExtensionPointRegistry:
    """Central registry of all extension points in Oscura.

    Manages registration and lookup of extension points, algorithms,
    and hooks throughout the system.

    Example:
        >>> # List all extension points
        >>> extension_points = osc.extensions.list()
        >>> for ep in extension_points:
        ...     print(f"{ep.name} v{ep.version}")

        >>> # Get specific extension point
        >>> decoder_ep = osc.extensions.get("protocol_decoder")
        >>> print(f"Required methods: {decoder_ep.required_methods}")

    References:
        EXT-001: Extension Point Registry
        EXT-002: Algorithm Registration
        EXT-003: Algorithm Selection
        EXT-004: Priority System
        EXT-005: Hook System
        EXT-006: Custom Decoder Registration
    """

    _instance: ExtensionPointRegistry | None = None

    def __new__(cls) -> ExtensionPointRegistry:
        """Ensure singleton instance.

        Returns:
            Singleton ExtensionPointRegistry instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._extension_points: dict[str, ExtensionPointSpec] = {}  # type: ignore[misc, attr-defined]
            cls._instance._algorithms: dict[str, dict[str, RegisteredAlgorithm]] = {}  # type: ignore[misc, attr-defined]
            cls._instance._hooks: dict[str, list[RegisteredHook]] = {}  # type: ignore[misc, attr-defined]
            cls._instance._hook_error_policy = HookErrorPolicy.CONTINUE
            cls._instance._log_hook_errors = True
            cls._instance._initialized = False  # type: ignore[has-type]
            cls._instance._registration_counter: int = 0  # type: ignore[misc, attr-defined]
        return cls._instance

    def initialize(self) -> None:
        """Initialize built-in extension points.

        Registers the standard extension points that come with Oscura.
        """
        if self._initialized:  # type: ignore[has-type]
            return

        # Register standard extension points
        self.register_point(
            ExtensionPointSpec(
                name="protocol_decoder",
                version="1.0.0",
                description="Decode protocol from waveform or digital trace",
                required_methods=["decode", "get_metadata"],
                optional_methods=["configure", "reset", "validate_config"],
            )
        )

        self.register_point(
            ExtensionPointSpec(
                name="file_loader",
                version="1.0.0",
                description="Load trace data from file format",
                required_methods=["load", "can_load"],
                optional_methods=["get_metadata", "get_channels"],
            )
        )

        self.register_point(
            ExtensionPointSpec(
                name="measurement",
                version="1.0.0",
                description="Compute measurement from trace",
                required_methods=["measure"],
                optional_methods=["validate_input", "get_units"],
            )
        )

        self.register_point(
            ExtensionPointSpec(
                name="exporter",
                version="1.0.0",
                description="Export trace data to file format",
                required_methods=["export"],
                optional_methods=["get_supported_formats"],
            )
        )

        self.register_point(
            ExtensionPointSpec(
                name="algorithm",
                version="1.0.0",
                description="Signal processing algorithm",
                required_methods=["process"],
                optional_methods=["configure", "get_parameters"],
            )
        )

        self._initialized = True
        logger.debug("Extension point registry initialized with built-in points")

    # =========================================================================
    # Extension Point Management (EXT-001)
    # =========================================================================

    def register_point(self, spec: ExtensionPointSpec) -> None:
        """Register an extension point.

        Args:
            spec: Extension point specification

        Raises:
            ValueError: If extension point already exists

        Example:
            >>> spec = ExtensionPointSpec(
            ...     name="my_extension",
            ...     version="1.0.0",
            ...     required_methods=["process"]
            ... )
            >>> registry.register_point(spec)
        """
        if spec.name in self._extension_points:  # type: ignore[attr-defined]
            raise ValueError(f"Extension point '{spec.name}' already registered")
        self._extension_points[spec.name] = spec  # type: ignore[attr-defined]
        logger.debug(f"Registered extension point: {spec.name} v{spec.version}")

    def get_point(self, name: str) -> ExtensionPointSpec:
        """Get extension point specification.

        Args:
            name: Extension point name

        Returns:
            Extension point specification

        Raises:
            KeyError: If extension point not found
        """
        if name not in self._extension_points:  # type: ignore[attr-defined]
            raise KeyError(
                f"Extension point '{name}' not found. "
                f"Available: {list(self._extension_points.keys())}"  # type: ignore[attr-defined]
            )
        return self._extension_points[name]  # type: ignore[no-any-return, attr-defined]

    def list_points(self) -> list[ExtensionPointSpec]:
        """List all registered extension points.

        Returns:
            List of extension point specifications
        """
        return list(self._extension_points.values())  # type: ignore[attr-defined]

    def exists(self, name: str) -> bool:
        """Check if extension point exists.

        Args:
            name: Extension point name

        Returns:
            True if exists
        """
        return name in self._extension_points  # type: ignore[attr-defined]

    # =========================================================================
    # Algorithm Management (EXT-002, EXT-003, EXT-004)
    # =========================================================================

    def register_algorithm(
        self,
        name: str,
        func: Callable[..., Any],
        category: str,
        priority: int = 50,
        performance: dict[str, str] | None = None,
        supports: list[str] | None = None,
        description: str = "",
        complexity: str = "O(n)",
        capabilities: dict[str, Any] | None = None,
        memory_usage: str = "unknown",
    ) -> None:
        """Register a custom algorithm implementation.

        Args:
            name: Algorithm name
            func: Algorithm function
            category: Algorithm category
            priority: Execution priority (0-100, higher = first)
            performance: Performance characteristics dict (speed/accuracy/memory)
            supports: List of supported data types
            description: Human-readable description
            complexity: Time complexity string (e.g., "O(n)", "O(n log n)")
            capabilities: Algorithm capabilities dict (e.g., {'multi_channel': True})
            memory_usage: Memory usage characteristics (low/medium/high/unknown)

        Raises:
            ValueError: If algorithm already registered
            TypeError: If func is not callable

        Example:
            >>> def my_edge_detector(data, threshold=0.5):
            ...     return find_edges(data, threshold)
            >>> registry.register_algorithm(
            ...     name="my_detector",
            ...     func=my_edge_detector,
            ...     category="edge_detection",
            ...     priority=75,
            ...     performance={"speed": "fast", "accuracy": "medium", "memory": "low"},
            ...     capabilities={"multi_channel": True, "max_sample_rate": 1000000},
            ...     memory_usage="low"
            ... )

        References:
            EXT-002: Algorithm Registration (capability queries, performance metadata)
        """
        if not callable(func):
            raise TypeError(f"Algorithm must be callable, got {type(func).__name__}")

        if category not in self._algorithms:  # type: ignore[attr-defined]
            self._algorithms[category] = {}  # type: ignore[attr-defined]

        if name in self._algorithms[category]:  # type: ignore[attr-defined]
            raise ValueError(f"Algorithm '{name}' already registered in category '{category}'")

        # Increment registration counter
        self._registration_counter += 1  # type: ignore[attr-defined]

        algo = RegisteredAlgorithm(
            name=name,
            category=category,
            func=func,
            priority=priority,
            performance=performance or {},
            supports=supports or [],
            description=description,
            complexity=complexity,
            capabilities=capabilities or {},
            memory_usage=memory_usage,
            registration_order=self._registration_counter,  # type: ignore[attr-defined]
        )

        self._algorithms[category][name] = algo  # type: ignore[attr-defined]
        logger.debug(f"Registered algorithm: {name} in category {category}")

    def get_algorithm(self, category: str, name: str) -> RegisteredAlgorithm:
        """Get algorithm by category and name.

        Args:
            category: Algorithm category
            name: Algorithm name

        Returns:
            Registered algorithm metadata

        Raises:
            KeyError: If not found
        """
        if category not in self._algorithms:  # type: ignore[attr-defined]
            raise KeyError(f"Category '{category}' not found")
        if name not in self._algorithms[category]:  # type: ignore[attr-defined]
            raise KeyError(f"Algorithm '{name}' not found in category '{category}'")
        return self._algorithms[category][name]  # type: ignore[no-any-return, attr-defined]

    def select_algorithm(
        self,
        category: str,
        name: str | None = None,
        *,
        optimize_for: str = "speed",
        constraints: dict[str, Any] | None = None,
        required_capabilities: list[str] | None = None,
    ) -> RegisteredAlgorithm:
        """Select algorithm implementation at runtime.

        Selects by name if provided, otherwise auto-selects based on
        optimization criteria and capability matching.

        Args:
            category: Algorithm category
            name: Specific algorithm name (optional)
            optimize_for: Optimization target: "speed", "accuracy", "memory"
            constraints: Filter constraints on performance/supports
            required_capabilities: List of required capabilities for auto-selection

        Returns:
            Selected algorithm

        Raises:
            KeyError: If category not found or no matching algorithm

        Example:
            >>> # Select by name
            >>> algo = registry.select_algorithm("edge_detection", "fast_detector")

            >>> # Auto-select for speed
            >>> algo = registry.select_algorithm(
            ...     "edge_detection",
            ...     optimize_for="speed"
            ... )

            >>> # Auto-select by capability matching
            >>> algo = registry.select_algorithm(
            ...     "edge_detection",
            ...     required_capabilities=["multi_channel", "real_time"]
            ... )

        References:
            EXT-003: Algorithm Selection (auto-selection by capability matching)
        """
        if category not in self._algorithms:  # type: ignore[attr-defined]
            raise KeyError(f"Category '{category}' not found")

        if name:
            return self.get_algorithm(category, name)

        # Get and filter candidates
        candidates = list(self._algorithms[category].values())  # type: ignore[attr-defined]
        if not candidates:
            raise KeyError(f"No algorithms registered in category '{category}'")

        candidates = self._filter_by_capabilities(candidates, required_capabilities, category)
        candidates = self._filter_by_constraints(candidates, constraints, category)

        # Select best match
        sort_key = self._get_sort_key(optimize_for)
        candidates.sort(key=sort_key)
        return candidates[0]

    def _filter_by_capabilities(
        self,
        candidates: list[RegisteredAlgorithm],
        required_capabilities: list[str] | None,
        category: str,
    ) -> list[RegisteredAlgorithm]:
        """Filter algorithms by required capabilities.

        Args:
            candidates: List of candidate algorithms.
            required_capabilities: Required capabilities.
            category: Category name for error messages.

        Returns:
            Filtered list of algorithms.

        Raises:
            KeyError: If no algorithms match capabilities.
        """
        if not required_capabilities:
            return candidates

        filtered = [
            algo for algo in candidates if all(algo.can(cap) for cap in required_capabilities)
        ]

        if not filtered:
            raise KeyError(f"No algorithms match required capabilities in category '{category}'")

        return filtered

    def _filter_by_constraints(
        self,
        candidates: list[RegisteredAlgorithm],
        constraints: dict[str, Any] | None,
        category: str,
    ) -> list[RegisteredAlgorithm]:
        """Filter algorithms by constraints.

        Args:
            candidates: List of candidate algorithms.
            constraints: Constraint dictionary.
            category: Category name for error messages.

        Returns:
            Filtered list of algorithms.

        Raises:
            KeyError: If no algorithms match constraints.
        """
        if not constraints:
            return candidates

        filtered = [algo for algo in candidates if self._matches_constraints(algo, constraints)]

        if not filtered:
            raise KeyError(f"No algorithms match constraints in category '{category}'")

        return filtered

    def _matches_constraints(
        self,
        algo: RegisteredAlgorithm,
        constraints: dict[str, Any],
    ) -> bool:
        """Check if algorithm matches all constraints.

        Args:
            algo: Algorithm to check.
            constraints: Constraint dictionary.

        Returns:
            True if all constraints match.
        """
        for key, value in constraints.items():
            if not self._check_single_constraint(algo, key, value):
                return False
        return True

    def _check_single_constraint(
        self,
        algo: RegisteredAlgorithm,
        key: str,
        value: Any,
    ) -> bool:
        """Check if algorithm matches a single constraint.

        Args:
            algo: Algorithm to check.
            key: Constraint key.
            value: Expected value.

        Returns:
            True if constraint matches.
        """
        if key.startswith("performance."):
            return self._check_performance_constraint(algo, key, value)
        elif key.startswith("capabilities."):
            return self._check_capabilities_constraint(algo, key, value)
        elif key == "supports":
            return self._check_supports_constraint(algo, value)
        elif key == "memory_usage":
            memory_check: bool = bool(algo.memory_usage == value)
            return memory_check
        return True

    def _check_performance_constraint(
        self, algo: RegisteredAlgorithm, key: str, value: Any
    ) -> bool:
        """Check performance constraint."""
        perf_key = key.split(".", 1)[1]
        result: bool = bool(algo.performance.get(perf_key) == value)
        return result

    def _check_capabilities_constraint(
        self, algo: RegisteredAlgorithm, key: str, value: Any
    ) -> bool:
        """Check capabilities constraint."""
        cap_key = key.split(".", 1)[1]
        result: bool = bool(algo.capabilities.get(cap_key) == value)
        return result

    def _check_supports_constraint(self, algo: RegisteredAlgorithm, value: Any) -> bool:
        """Check supports constraint."""
        if isinstance(value, list):
            result: bool = bool(any(s in algo.supports for s in value))
            return result
        else:
            in_supports: bool = bool(value in algo.supports)
            return in_supports

    def _get_sort_key(self, optimize_for: str) -> Any:
        """Get sort key function for optimization criterion.

        Args:
            optimize_for: Optimization target.

        Returns:
            Sort key function.
        """
        if optimize_for == "speed":

            def sort_key(a: RegisteredAlgorithm) -> tuple[int, int]:
                speed = a.performance.get("speed")
                rank = 0 if speed == "fast" else 1 if speed == "medium" else 2
                return (rank, -a.priority)
        elif optimize_for == "accuracy":

            def sort_key(a: RegisteredAlgorithm) -> tuple[int, int]:
                acc = a.performance.get("accuracy")
                rank = 0 if acc == "high" else 1 if acc == "medium" else 2
                return (rank, -a.priority)
        elif optimize_for == "memory":

            def sort_key(a: RegisteredAlgorithm) -> tuple[int, int]:
                mem = a.performance.get("memory")
                rank = 0 if mem == "low" else 1 if mem == "medium" else 2
                return (rank, -a.priority)
        else:

            def sort_key(a: RegisteredAlgorithm) -> tuple[int, int]:
                return (-a.priority, 0)

        return sort_key

    def list_algorithms(
        self,
        category: str,
        ordered: bool = False,
        tie_break: str = "name",
    ) -> list[RegisteredAlgorithm]:
        """List all algorithms in a category.

        Args:
            category: Algorithm category
            ordered: If True, sort by priority (highest first)
            tie_break: Tie-breaking rule: "name" (alphabetical) or "registration" (order registered)

        Returns:
            List of registered algorithms

        Raises:
            KeyError: If category not found

        Example:
            >>> # Get algorithms sorted by priority, ties broken by name
            >>> algos = registry.list_algorithms("edge_detection", ordered=True, tie_break="name")

            >>> # Get algorithms sorted by priority, ties broken by registration order
            >>> algos = registry.list_algorithms("edge_detection", ordered=True, tie_break="registration")

        References:
            EXT-004: Priority System (tie-breaking rules by name or registration order)
        """
        if category not in self._algorithms:  # type: ignore[attr-defined]
            raise KeyError(f"Category '{category}' not found")

        algos = list(self._algorithms[category].values())  # type: ignore[attr-defined]

        if ordered:
            if tie_break == "registration":
                # Sort by priority (highest first), then by registration order for ties
                algos.sort(key=lambda a: (-a.priority, a.registration_order))
            else:
                # Sort by priority (highest first), then by name for ties (default)
                algos.sort(key=lambda a: (-a.priority, a.name))

        return algos

    def list_categories(self) -> list[str]:
        """List all algorithm categories.

        Returns:
            List of category names
        """
        return list(self._algorithms.keys())  # type: ignore[attr-defined]

    def benchmark_algorithms(
        self,
        category: str,
        test_data: Any,
        *,
        metrics: list[str] | None = None,
        iterations: int = 10,
    ) -> dict[str, dict[str, float]]:
        """Benchmark all algorithms in a category.

        Runs performance tests on all registered algorithms and measures
        execution time, memory usage, and optionally custom metrics.

        Args:
            category: Algorithm category to benchmark
            test_data: Test data to pass to algorithms
            metrics: List of metrics to measure (defaults to ["execution_time"])
            iterations: Number of iterations to average over

        Returns:
            Dict mapping algorithm names to metric results

        Raises:
            KeyError: If category is not found.

        Example:
            >>> import numpy as np
            >>> test_signal = np.random.randn(1000)
            >>> results = registry.benchmark_algorithms(
            ...     "edge_detection",
            ...     test_signal,
            ...     metrics=["execution_time", "memory_usage"],
            ...     iterations=100
            ... )
            >>> for name, metrics in results.items():
            ...     print(f"{name}: {metrics['execution_time']:.3f}s")

        References:
            EXT-003: Algorithm Selection (benchmarking support)
        """
        import time
        import tracemalloc

        if category not in self._algorithms:  # type: ignore[attr-defined]
            raise KeyError(f"Category '{category}' not found")

        if metrics is None:
            metrics = ["execution_time"]

        results = {}

        for name, algo in self._algorithms[category].items():  # type: ignore[attr-defined]
            algo_results = {}

            if "execution_time" in metrics:
                times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    try:
                        algo.func(test_data)
                    except Exception as e:
                        logger.warning(f"Algorithm {name} failed during benchmark: {e}")
                        times.append(float("inf"))
                        continue
                    end = time.perf_counter()
                    times.append(end - start)

                algo_results["execution_time"] = sum(times) / len(times)
                algo_results["min_time"] = min(times)
                algo_results["max_time"] = max(times)

            if "memory_usage" in metrics:
                tracemalloc.start()
                try:
                    algo.func(test_data)
                    current, peak = tracemalloc.get_traced_memory()
                    algo_results["memory_current"] = current / 1024 / 1024  # MB
                    algo_results["memory_peak"] = peak / 1024 / 1024  # MB
                except Exception as e:
                    logger.warning(f"Algorithm {name} failed during benchmark: {e}")
                    algo_results["memory_current"] = float("inf")
                    algo_results["memory_peak"] = float("inf")
                finally:
                    tracemalloc.stop()

            results[name] = algo_results

        return results

    def configure_priorities(self, config: dict[str, dict[str, int]]) -> None:
        """Override algorithm priorities via configuration.

        Args:
            config: Dict mapping category -> {algorithm_name: new_priority}

        Example:
            >>> registry.configure_priorities({
            ...     "edge_detection": {
            ...         "fast_detector": 100,
            ...         "accurate_detector": 50
            ...     }
            ... })

        References:
            EXT-004: Priority System
        """
        for category, priorities in config.items():
            if category not in self._algorithms:  # type: ignore[attr-defined]
                continue
            for name, priority in priorities.items():
                if name in self._algorithms[category]:  # type: ignore[attr-defined]
                    self._algorithms[category][name].priority = priority  # type: ignore[attr-defined]
                    logger.debug(f"Set priority for {category}/{name} to {priority}")

    # =========================================================================
    # Hook System (EXT-005)
    # =========================================================================

    def register_hook(
        self,
        hook_point: str,
        func: Callable[[HookContext], HookContext],
        priority: int = 50,
        name: str = "",
    ) -> None:
        """Register a hook function.

        Args:
            hook_point: Name of hook point (e.g., "pre_decode", "post_decode")
            func: Hook function accepting and returning HookContext
            priority: Execution priority (higher = first)
            name: Optional hook name for identification

        Example:
            >>> @osc.hooks.register("pre_decode")
            >>> def validate_waveform(context):
            ...     if context.data.sample_rate < 1000:
            ...         raise ValueError("Sample rate too low")
            ...     return context

        References:
            EXT-005: Hook System
        """
        if hook_point not in self._hooks:  # type: ignore[attr-defined]
            self._hooks[hook_point] = []  # type: ignore[attr-defined]

        hook = RegisteredHook(
            hook_point=hook_point,
            func=func,
            priority=priority,
            name=name or func.__name__,
        )

        self._hooks[hook_point].append(hook)  # type: ignore[attr-defined]
        # Sort by priority (highest first)
        self._hooks[hook_point].sort(key=lambda h: -h.priority)  # type: ignore[attr-defined]

        logger.debug(f"Registered hook '{hook.name}' at point '{hook_point}'")

    def execute_hooks(self, hook_point: str, context: HookContext) -> HookContext:
        """Execute all hooks at a hook point with chaining and error isolation.

        Hooks are executed in priority order (highest first). Each hook receives
        the context from the previous hook (chaining). If a hook fails, the error
        is isolated based on the configured error policy, preventing one hook's
        failure from stopping other hooks.

        Args:
            hook_point: Hook point name
            context: Hook context to pass through

        Returns:
            Modified context after all hooks

        Raises:
            Exception: If error policy is ABORT and a hook fails.

        Example:
            >>> context = HookContext(data=trace)
            >>> context = registry.execute_hooks("pre_decode", context)
            >>> if context.abort:
            ...     raise ValueError(context.abort_reason)

        References:
            EXT-005: Hook System (hook chaining, error isolation)
        """
        if hook_point not in self._hooks:  # type: ignore[attr-defined]
            return context

        # Execute hooks in priority order (hook chaining - EXT-005)
        for hook in self._hooks[hook_point]:  # type: ignore[attr-defined]
            try:
                context = hook.func(context)
                if context.abort:
                    logger.info(f"Hook '{hook.name}' requested abort: {context.abort_reason}")
                    break
            except Exception as e:
                # Error isolation - EXT-005: one hook failure doesn't stop others
                if self._log_hook_errors:
                    logger.error(f"Hook '{hook.name}' at '{hook_point}' failed: {e}")

                if self._hook_error_policy == HookErrorPolicy.ABORT:
                    raise
                elif self._hook_error_policy == HookErrorPolicy.CONTINUE:
                    continue  # Continue to next hook despite error
                # IGNORE falls through

        return context

    def configure_hooks(self, on_error: str = "continue", log_errors: bool = True) -> None:
        """Configure hook error handling behavior.

        Args:
            on_error: Error policy: "continue", "abort", "ignore"
            log_errors: Whether to log hook errors

        References:
            EXT-005: Hook System
        """
        policy_map = {
            "continue": HookErrorPolicy.CONTINUE,
            "abort": HookErrorPolicy.ABORT,
            "ignore": HookErrorPolicy.IGNORE,
        }
        self._hook_error_policy = policy_map.get(on_error, HookErrorPolicy.CONTINUE)
        self._log_hook_errors = log_errors

    def list_hooks(self, hook_point: str | None = None) -> dict[str, list[str]]:
        """List registered hooks.

        Args:
            hook_point: Specific hook point, or None for all

        Returns:
            Dict mapping hook points to list of hook names
        """
        if hook_point:
            if hook_point not in self._hooks:  # type: ignore[attr-defined]
                return {hook_point: []}
            return {hook_point: [h.name for h in self._hooks[hook_point]]}  # type: ignore[attr-defined]

        return {point: [h.name for h in hooks] for point, hooks in self._hooks.items()}  # type: ignore[attr-defined]

    def clear_hooks(self, hook_point: str | None = None) -> None:
        """Clear registered hooks.

        Args:
            hook_point: Specific hook point to clear, or None for all
        """
        if hook_point:
            self._hooks.pop(hook_point, None)  # type: ignore[attr-defined]
        else:
            self._hooks.clear()  # type: ignore[attr-defined]

    # =========================================================================
    # Custom Decoder Registration (EXT-006)
    # =========================================================================

    def register_decoder(self, protocol: str, decoder_class: type, priority: int = 50) -> None:
        """Register a custom protocol decoder.

        Args:
            protocol: Protocol name (e.g., "uart", "spi", "my_custom")
            decoder_class: Decoder class implementing ProtocolDecoder interface
            priority: Registration priority

        Raises:
            ValueError: If decoder doesn't implement required interface or lacks documentation

        Example:
            >>> class MyDecoder:
            ...     '''Custom decoder for my protocol.'''
            ...     def decode(self, trace):
            ...         return []
            ...     def get_metadata(self):
            ...         return {"name": "my_decoder"}
            >>> registry.register_decoder("my_protocol", MyDecoder)

        References:
            EXT-006: Custom Decoder Registration (validation of decoder interface, documentation requirements)
        """
        # Validate decoder implements required interface
        spec = self.get_point("protocol_decoder")
        instance = decoder_class()
        is_valid, missing = spec.validate_implementation(instance)

        if not is_valid:
            raise ValueError(f"Decoder '{protocol}' missing required methods: {missing}")

        # Check documentation requirements (EXT-006)
        if not decoder_class.__doc__ or not decoder_class.__doc__.strip():
            raise ValueError(
                f"Decoder '{protocol}' must have a docstring documenting its purpose and usage"
            )

        # Register as algorithm in protocol_decoder category
        self.register_algorithm(
            name=protocol,
            func=decoder_class,
            category="protocol_decoder",
            priority=priority,
            description=decoder_class.__doc__.strip().split("\n")[0]
            if decoder_class.__doc__
            else f"Protocol decoder for {protocol}",
        )

        logger.info(f"Registered custom decoder for protocol: {protocol}")

    def get_decoder(self, protocol: str) -> type:
        """Get decoder class for a protocol.

        Args:
            protocol: Protocol name

        Returns:
            Decoder class
        """
        algo = self.get_algorithm("protocol_decoder", protocol)
        return algo.func  # type: ignore[return-value]

    def list_decoders(self) -> list[str]:
        """List all registered protocol decoders.

        Returns:
            List of protocol names
        """
        if "protocol_decoder" not in self._algorithms:  # type: ignore[attr-defined]
            return []
        return list(self._algorithms["protocol_decoder"].keys())  # type: ignore[attr-defined]


# Global registry instance
_registry = ExtensionPointRegistry()


# =========================================================================
# Module-Level Convenience Functions
# =========================================================================


def get_registry() -> ExtensionPointRegistry:
    """Get the global extension point registry.

    Returns:
        Global ExtensionPointRegistry instance
    """
    _registry.initialize()
    return _registry


def list_extension_points() -> list[ExtensionPointSpec]:
    """List all registered extension points.

    Returns:
        List of extension point specifications

    References:
        EXT-001: Extension Point Registry
    """
    return get_registry().list_points()


def get_extension_point(name: str) -> ExtensionPointSpec:
    """Get extension point by name.

    Args:
        name: Extension point name

    Returns:
        Extension point specification

    References:
        EXT-001: Extension Point Registry
    """
    return get_registry().get_point(name)


def extension_point_exists(name: str) -> bool:
    """Check if extension point exists.

    Args:
        name: Extension point name

    Returns:
        True if exists

    References:
        EXT-001: Extension Point Registry
    """
    return get_registry().exists(name)


def register_extension_point(spec: ExtensionPointSpec) -> None:
    """Register a new extension point.

    Args:
        spec: Extension point specification

    References:
        EXT-001: Extension Point Registry
    """
    get_registry().register_point(spec)


# Hook decorator
def hook(hook_point: str, priority: int = 50, name: str = ""):  # type: ignore[no-untyped-def]
    """Decorator for registering hook functions.

    Args:
        hook_point: Hook point name
        priority: Execution priority
        name: Optional hook name

    Returns:
        Decorator function that registers the hook.

    Example:
        >>> @hook("pre_decode", priority=100)
        >>> def validate_input(context):
        ...     # validation logic
        ...     return context

    References:
        EXT-005: Hook System
    """

    def decorator(func: Callable[[HookContext], HookContext]):  # type: ignore[no-untyped-def]
        get_registry().register_hook(hook_point, func, priority, name or func.__name__)
        return func

    return decorator


__all__ = [
    "ExtensionPointRegistry",
    "ExtensionPointSpec",
    "HookContext",
    "HookErrorPolicy",
    "RegisteredAlgorithm",
    "RegisteredHook",
    "extension_point_exists",
    "get_extension_point",
    "get_registry",
    "hook",
    "list_extension_points",
    "register_extension_point",
]
