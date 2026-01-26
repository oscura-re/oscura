"""Oscura configuration and schema validation system.

This package provides:
- JSON Schema-based configuration validation
- YAML/JSON configuration loading
- Protocol definition loading
- Threshold configuration management
- Pipeline configuration
- User preferences


Example:
    >>> from oscura.core.config import load_config, validate_config
    >>> config = load_config("pipeline.yaml")
    >>> validate_config(config, schema="pipeline")
"""

from oscura.core.config.defaults import (
    DEFAULT_CONFIG,
    get_effective_config,
    inject_defaults,
)
from oscura.core.config.loader import (
    get_config_value,
    load_config,
    load_config_file,
    save_config,
)
from oscura.core.config.memory import (
    MemoryConfiguration,
    configure_from_environment,
    enable_auto_degrade,
    get_memory_config,
    reset_to_defaults,
    set_memory_limit,
    set_memory_reserve,
    set_memory_thresholds,
)
from oscura.core.config.migration import (
    Migration,
    MigrationFunction,
    SchemaMigration,
    get_config_version,  # noqa: F401
    get_migration_registry,
    list_migrations,
    migrate_config,
    register_migration,
)
from oscura.core.config.pipeline import (
    Pipeline,
    PipelineDefinition,
    PipelineExecutionError,
    PipelineResult,
    PipelineStep,
    PipelineTemplate,
    PipelineValidationError,
    load_pipeline,
    resolve_includes,
)
from oscura.core.config.preferences import (
    DefaultsPreferences,
    EditorPreferences,
    ExportPreferences,
    LoggingPreferences,
    PreferencesManager,
    UserPreferences,
    VisualizationPreferences,
    get_preferences,
    get_preferences_manager,
    save_preferences,
)
from oscura.core.config.protocol import (
    ProtocolCapabilities,
    ProtocolDefinition,
    ProtocolRegistry,
    ProtocolWatcher,
    get_protocol_registry,
    load_protocol,
    resolve_inheritance,
)
from oscura.core.config.schema import (
    ConfigSchema,
    SchemaRegistry,
    ValidationError,
    get_schema_registry,
    register_schema,
    validate_against_schema,
)

# Alias for backward compatibility with oscura.core.config module
validate_config = validate_against_schema

# Legacy imports for backward compatibility
from oscura.core.config.legacy import SmartDefaults, _deep_merge
from oscura.core.config.settings import (
    AnalysisSettings,
    CLIDefaults,
    OutputSettings,
    Settings,
    get_settings,
    load_settings,
    reset_settings,
    save_settings,
    set_settings,
)
from oscura.core.config.thresholds import (
    LogicFamily,
    ThresholdProfile,
    ThresholdRegistry,
    get_threshold_registry,
    get_user_logic_families_dir,
    load_logic_family,
    load_user_logic_families,
)

__all__ = [
    "DEFAULT_CONFIG",
    # Settings
    "AnalysisSettings",
    "CLIDefaults",
    # Schema
    "ConfigSchema",
    # Preferences
    "DefaultsPreferences",
    "EditorPreferences",
    "ExportPreferences",
    "LoggingPreferences",
    # Thresholds
    "LogicFamily",
    # Memory configuration
    "MemoryConfiguration",
    #
    "Migration",
    "MigrationFunction",
    "OutputSettings",
    # Pipeline
    "Pipeline",
    "PipelineDefinition",
    "PipelineExecutionError",
    "PipelineResult",
    "PipelineStep",
    "PipelineTemplate",
    "PipelineValidationError",
    "PreferencesManager",
    # Protocol
    "ProtocolCapabilities",
    "ProtocolDefinition",
    "ProtocolRegistry",
    "ProtocolWatcher",
    "SchemaMigration",
    "SchemaRegistry",
    "Settings",
    "SmartDefaults",
    "ThresholdProfile",
    "ThresholdRegistry",
    "UserPreferences",
    "ValidationError",
    "VisualizationPreferences",
    "_deep_merge",
    "configure_from_environment",
    "enable_auto_degrade",
    "get_config_value",
    "get_effective_config",
    "get_memory_config",
    "get_migration_registry",
    "get_preferences",
    "get_preferences_manager",
    "get_protocol_registry",
    "get_schema_registry",
    "get_settings",
    "get_threshold_registry",
    "get_user_logic_families_dir",
    # Defaults
    "inject_defaults",
    "list_migrations",
    # Loading
    "load_config",
    "load_config_file",
    "load_logic_family",
    "load_pipeline",
    "load_protocol",
    "load_settings",
    "load_user_logic_families",
    "migrate_config",
    "register_migration",
    "register_schema",
    "reset_settings",
    "reset_to_defaults",
    "resolve_includes",
    "resolve_inheritance",
    "save_config",
    "save_preferences",
    "save_settings",
    "set_memory_limit",
    "set_memory_reserve",
    "set_memory_thresholds",
    "set_settings",
    "validate_against_schema",
    "validate_config",
]
