"""Extensibility framework for plugins and custom algorithms.

This package provides registries, plugin management, and custom measurement
frameworks for extending Oscura functionality.
"""

from .docs import (
    ClassDoc,
    ExtensionDocs,
    FunctionDoc,
    ModuleDoc,
    extract_plugin_metadata,
    generate_decoder_docs,
    generate_extension_docs,
)
from .extensions import (
    ExtensionPointRegistry,
    ExtensionPointSpec,
    HookContext,
    HookErrorPolicy,
    RegisteredAlgorithm,
    RegisteredHook,
    extension_point_exists,
    get_extension_point,
    get_registry,
    hook,
    list_extension_points,
    register_extension_point,
)
from .logging import (
    PluginLoggerAdapter,
    configure_plugin_logging,
    get_plugin_log_level,
    get_plugin_logger,
    list_plugin_loggers,
    log_plugin_lifecycle,
    set_plugin_log_level,
)
from .measurements import (
    MeasurementDefinition,
    MeasurementRegistry,
    get_measurement_registry,
    list_measurements,
    measure,
    register_measurement,
)
from .plugins import (
    PluginError,
    PluginManager,
    PluginMetadata,
    get_plugin_manager,
    list_plugins,
    load_plugin,
)
from .registry import (
    AlgorithmRegistry,
    get_algorithm,
    get_algorithms,
    register_algorithm,
)
from .templates import (
    PluginTemplate,
    PluginType,
    generate_plugin_template,
)
from .validation import (
    ValidationIssue,
    ValidationResult,
    validate_decoder_interface,
    validate_extension,
    validate_hook_function,
)

__all__ = [
    # Registry
    "AlgorithmRegistry",
    # Documentation (EXT-006)
    "ClassDoc",
    # Extension Points (EXT-001 through EXT-006)
    "ExtensionDocs",
    "ExtensionPointRegistry",
    "ExtensionPointSpec",
    "FunctionDoc",
    "HookContext",
    "HookErrorPolicy",
    # Measurements
    "MeasurementDefinition",
    "MeasurementRegistry",
    "ModuleDoc",
    # Plugins
    "PluginError",
    "PluginLoggerAdapter",
    "PluginManager",
    "PluginMetadata",
    "PluginTemplate",
    "PluginType",
    "RegisteredAlgorithm",
    "RegisteredHook",
    # Validation (EXT-005)
    "ValidationIssue",
    "ValidationResult",
    "configure_plugin_logging",
    "extension_point_exists",
    "extract_plugin_metadata",
    "generate_decoder_docs",
    "generate_extension_docs",
    "generate_plugin_template",
    "get_algorithm",
    "get_algorithms",
    "get_extension_point",
    "get_measurement_registry",
    "get_plugin_log_level",
    "get_plugin_logger",
    "get_plugin_manager",
    "get_registry",
    "hook",
    "list_extension_points",
    "list_measurements",
    "list_plugin_loggers",
    "list_plugins",
    "load_plugin",
    "log_plugin_lifecycle",
    "measure",
    "register_algorithm",
    "register_extension_point",
    "register_measurement",
    "set_plugin_log_level",
    "validate_decoder_interface",
    "validate_extension",
    "validate_hook_function",
]
