"""Plugin Basics: Introduction to Oscura's plugin system

Demonstrates:
- oscura.get_plugin_manager() - Access the global plugin manager
- oscura.list_plugins() - Discover available plugins by group
- oscura.load_plugin() - Load a plugin from a specific group
- Plugin metadata inspection (name, version, capabilities, dependencies)
- Plugin discovery and enumeration
- Plugin health checking

IEEE Standards: N/A
Related Demos:
- 08_extensibility/02_custom_measurement.py
- 08_extensibility/03_custom_analyzer.py

The plugin system allows Oscura to be extended with custom functionality.
This demonstration shows how to discover, inspect, and load plugins without
needing to write your own - perfect for understanding how extensibility works.

This is a P0 CRITICAL feature - demonstrates plugin architecture to users.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import oscura as osc
from demonstrations.common import BaseDemo


class PluginBasicsDemo(BaseDemo):
    """Demonstrates plugin system basics and discovery."""

    def __init__(self) -> None:
        """Initialize plugin basics demonstration."""
        super().__init__(
            name="plugin_basics",
            description="Discover, inspect, and load plugins",
            capabilities=[
                "oscura.get_plugin_manager",
                "oscura.list_plugins",
                "oscura.load_plugin",
                "oscura.PluginMetadata",
                "oscura.PluginManager",
            ],
            related_demos=[
                "08_extensibility/02_custom_measurement.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """No test data needed for plugin discovery.

        Returns:
            Empty dictionary (plugin system doesn't require signals)
        """
        return {}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run plugin basics demonstration."""
        results: dict[str, Any] = {}

        # ===== Section 1: Getting the Plugin Manager =====
        self.section("Part 1: Accessing the Plugin Manager")
        self.subsection("Get Global Plugin Manager")

        manager = osc.get_plugin_manager()
        self.result("Manager type", type(manager).__name__)
        self.info("oscura.get_plugin_manager() returns the global plugin manager")
        self.info("This is a singleton - all calls return the same instance")
        self.result("Manager instance ID", id(manager))

        results["manager"] = manager
        results["manager_type"] = type(manager).__name__

        # ===== Section 2: Discovering Plugins =====
        self.section("Part 2: Plugin Discovery")
        self.subsection("List Available Plugins")

        available_plugins = osc.list_plugins()
        self.info("oscura.list_plugins() discovers all available plugins")
        self.result("Groups found", len(available_plugins))

        if available_plugins:
            self.subsection("Plugin Groups")
            has_plugins = False
            for group_name, plugin_names in available_plugins.items():
                if plugin_names:
                    has_plugins = True
                self.info(f"{group_name}: {len(plugin_names)} plugins")
                for plugin_name in plugin_names[:3]:  # Show first 3
                    self.info(f"  - {plugin_name}")
                if len(plugin_names) > 3:
                    self.info(f"  ... and {len(plugin_names) - 3} more")

            results["available_plugins"] = available_plugins
            results["total_groups"] = len(available_plugins)

            if not has_plugins:
                self.info("No plugins currently loaded (this is normal for a fresh installation)")
                self.info("Plugins can be installed via: pip install oscura-plugin-*")

        else:
            self.warning("No plugins discovered (this is normal for a fresh installation)")
            self.info("Plugins can be installed via: pip install oscura-plugin-*")
            results["available_plugins"] = {}
            results["total_groups"] = 0

        # ===== Section 3: Plugin Metadata =====
        self.section("Part 3: Plugin Metadata Inspection")
        self.subsection("Understanding Plugin Information")

        self.info("Each plugin has metadata describing its capabilities and requirements:")
        self.info("  - name: Unique identifier")
        self.info("  - version: Semantic version (e.g., 1.0.0)")
        self.info("  - api_version: Required Oscura API version")
        self.info("  - author: Plugin author or organization")
        self.info("  - description: Human-readable summary")
        self.info("  - capabilities: Features the plugin provides")
        self.info("  - dependencies: Required plugins or packages")
        self.info("  - enabled: Whether plugin is active")

        # If plugins are available, show metadata for one
        first_plugin = None
        first_group = None

        if available_plugins:
            # Find first group with actual plugins
            for group, plugins in available_plugins.items():
                if plugins:
                    first_group = group
                    first_plugin = plugins[0]
                    break

        if first_plugin and first_group:
            self.subsection(f"Example: {first_plugin} from {first_group}")

            try:
                # Load the plugin to get its metadata
                plugin_metadata = osc.load_plugin(first_group, first_plugin)

                self.info(f"Name: {plugin_metadata.name}")
                self.info(f"Entry Point: {plugin_metadata.entry_point}")
                if plugin_metadata.version:
                    self.info(f"Version: {plugin_metadata.version}")
                if plugin_metadata.module:
                    self.info(f"Module: {plugin_metadata.module}")
                if plugin_metadata.dependencies:
                    deps = ", ".join(plugin_metadata.dependencies)
                    self.info(f"Dependencies: {deps}")

                results["example_plugin"] = first_plugin
                results["example_metadata"] = {
                    "name": plugin_metadata.name,
                    "entry_point": plugin_metadata.entry_point,
                    "version": plugin_metadata.version,
                }

            except Exception as e:
                self.warning(f"Could not inspect plugin metadata: {e}")
        else:
            self.info("No plugins currently loaded to inspect")
            self.info("After installing plugins, metadata will be available here")

        # ===== Section 4: Loading Plugins =====
        self.section("Part 4: Loading Plugins")
        self.subsection("Plugin Loading Process")

        self.info("oscura.load_plugin(group, name) loads a specific plugin")
        self.info("Loading involves:")
        self.info("  1. Locating the plugin in the specified group")
        self.info("  2. Verifying API version compatibility")
        self.info("  3. Resolving dependencies")
        self.info("  4. Initializing the plugin")
        self.info("  5. Returning metadata and plugin instance")

        if first_plugin and first_group:
            self.subsection(f"Loading {first_plugin}")
            self.info(f"Attempting: osc.load_plugin('{first_group}', '{first_plugin}')")

            try:
                metadata = osc.load_plugin(first_group, first_plugin)
                self.success(f"Successfully loaded {first_plugin}")
                self.result("Loaded plugin name", metadata.name)
                self.result("Loaded plugin version", metadata.version)
                results["loaded_plugin"] = first_plugin
                results["loaded_metadata"] = {
                    "name": metadata.name,
                    "version": metadata.version,
                }

            except Exception as e:
                self.warning(f"Could not load plugin: {e}")
                self.info("This is normal if plugin dependencies are not installed")
        else:
            self.subsection("Plugin Loading Example")
            self.info("No plugins available to load in this demonstration")
            self.info("Example usage with available plugin:")
            self.info("  metadata = osc.load_plugin('oscura.decoders', 'uart')")
            self.info("  print(f'Loaded {metadata.name} v{metadata.version}')")

        # ===== Section 5: Plugin Manager Methods =====
        self.section("Part 5: Plugin Manager API")
        self.subsection("Available Manager Methods")

        self.info("The plugin manager provides methods for:")
        self.info("  - discover_plugins(group) → Dict[str, List[str]]")
        self.info("    Discover all available plugins in specified group (or all)")
        self.info("")
        self.info("  - load_plugin(group, name) → PluginMetadata")
        self.info("    Load a plugin and return its metadata")
        self.info("")
        self.info("  - get_plugin(group, name) → Any")
        self.info("    Get the loaded plugin instance from cache")
        self.info("")
        self.info("  - is_loaded(group, name) → bool")
        self.info("    Check if a plugin is currently loaded")
        self.info("")
        self.info("  - list_loaded_plugins() → List[PluginMetadata]")
        self.info("    List all currently loaded plugins")
        self.info("")
        self.info("  - unload_plugin(group, name) → None")
        self.info("    Unload a plugin and clear it from cache")
        self.info("")
        self.info("Entry point groups for organizing plugins:")
        self.info("  - oscura.decoders: Protocol decoders (UART, SPI, I2C, CAN, etc.)")
        self.info("  - oscura.measurements: Custom measurements")
        self.info("  - oscura.loaders: File format loaders")
        self.info("  - oscura.exporters: Export format handlers")

        results["manager_methods_available"] = True

        # ===== Section 6: Use Cases =====
        self.section("Part 6: Common Use Cases")
        self.subsection("Why Use the Plugin System?")

        self.info("1. EXTENSIBILITY - Add custom protocol decoders")
        self.info("   Example: Load custom UART decoder for specific hardware")
        self.info("")
        self.info("2. MODULARITY - Keep optional features separate")
        self.info("   Example: Load advanced analysis only when needed")
        self.info("")
        self.info("3. INTEGRATION - Connect external tools")
        self.info("   Example: Load plugin that bridges to simulation software")
        self.info("")
        self.info("4. DOMAIN-SPECIFIC - Tailor Oscura for your domain")
        self.info("   Example: Load power electronics measurement plugins")
        self.info("")
        self.info("5. UPGRADABILITY - Update features without restarting")
        self.info("   Example: Hot reload plugin after bug fix")

        results["use_cases_documented"] = True

        self.success("Plugin basics demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate plugin demonstration results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        # Must have manager
        if "manager" not in results:
            self.error("Plugin manager not obtained")
            return False

        # Plugin discovery should work (even if no plugins available)
        if "available_plugins" not in results:
            self.error("Plugin discovery failed")
            return False

        # Manager methods should be documented
        if not results.get("manager_methods_available", False):
            self.error("Manager API not documented")
            return False

        return True


if __name__ == "__main__":
    demo = PluginBasicsDemo()
    success = demo.execute()
    exit(0 if success else 1)
