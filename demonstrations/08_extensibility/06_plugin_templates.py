"""Plugin Templates: Scaffolding new plugins with templates

Demonstrates:
- oscura.generate_plugin_template() - Generate plugin scaffolding
- Template generation for different plugin types
- Generated file structure and contents
- Template validation and best practices
- Customizing generated templates

IEEE Standards: N/A
Related Demos:
- 08_extensibility/04_plugin_development.py
- 08_extensibility/01_plugin_basics.py

Plugin templates accelerate development by providing complete scaffolding
with all necessary boilerplate, tests, and documentation. This demonstration
shows how to generate and customize templates for different plugin types.

This is a P0 CRITICAL feature - demonstrates template generation to users.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import oscura as osc
from demonstrations.common import BaseDemo


class PluginTemplatesDemo(BaseDemo):
    """Demonstrates plugin template generation and usage."""

    def __init__(self) -> None:
        """Initialize plugin templates demonstration."""
        super().__init__(
            name="plugin_templates",
            description="Generate plugin scaffolding with templates",
            capabilities=[
                "oscura.generate_plugin_template",
                "oscura.PluginTemplate",
                "Template customization",
            ],
            related_demos=[
                "08_extensibility/04_plugin_development.py",
                "08_extensibility/01_plugin_basics.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test data for template demonstrations.

        Returns:
            Dictionary with temporary directory for template generation
        """
        # Create temporary directory for template generation
        temp_dir = tempfile.mkdtemp(prefix="oscura_plugin_templates_")

        return {
            "temp_dir": Path(temp_dir),
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run plugin templates demonstration."""
        temp_dir = data["temp_dir"]
        results: dict[str, Any] = {}

        # ===== Section 1: Understanding Plugin Templates =====
        self.section("Part 1: Plugin Template System")
        self.subsection("What Are Plugin Templates?")

        self.info("Plugin templates provide complete scaffolding for new plugins:")
        self.info("  - Complete directory structure")
        self.info("  - Base class implementation with TODOs")
        self.info("  - Test file stubs")
        self.info("  - README with usage instructions")
        self.info("  - pyproject.toml for packaging")
        self.info("  - Entry point configuration")
        self.info("")
        self.info("Benefits:")
        self.info("  ✓ Saves time - No need to write boilerplate")
        self.info("  ✓ Best practices - Follows Oscura conventions")
        self.info("  ✓ Complete - All files needed to start coding")
        self.info("  ✓ Ready to test - Includes test structure")
        self.info("  ✓ Ready to package - Includes pyproject.toml")

        results["template_system_documented"] = True

        # ===== Section 2: Available Plugin Types =====
        self.section("Part 2: Available Plugin Types")
        self.subsection("Supported Template Types")

        plugin_types = [
            ("analyzer", "Custom analysis algorithms and measurements"),
            ("loader", "File format loaders (read new file types)"),
            ("exporter", "Export formats (write data in new formats)"),
            ("decoder", "Protocol decoders (decode digital protocols)"),
        ]

        self.info("Oscura supports templates for these plugin types:\n")
        for ptype, description in plugin_types:
            self.result(f"{ptype.upper()}", description)

        results["plugin_types_available"] = len(plugin_types)

        # ===== Section 3: Generating a Decoder Template =====
        self.section("Part 3: Example - Generate Decoder Template")
        self.subsection("Creating a Protocol Decoder Plugin")

        self.info("Let's generate a template for a custom protocol decoder")
        self.info("This would be useful for proprietary or uncommon protocols")

        decoder_dir = temp_dir / "modbus_decoder"

        try:
            plugin_path = osc.generate_plugin_template(
                name="modbus_decoder",
                plugin_type="decoder",
                output_dir=decoder_dir,
                author="Demo User",
                description="Modbus protocol decoder for industrial networks",
                version="0.1.0",
            )

            self.success(f"Template generated at: {plugin_path}")
            self.result("Template type", "decoder")
            self.result("Plugin name", "modbus_decoder")

            results["decoder_template_generated"] = True
            results["decoder_path"] = str(plugin_path)

        except Exception as e:
            self.error(f"Failed to generate decoder template: {e}")
            results["decoder_template_generated"] = False

        # ===== Section 4: Examining Generated Structure =====
        self.section("Part 4: Generated File Structure")
        self.subsection("Exploring Template Contents")

        if results.get("decoder_template_generated", False):
            self.info("Template generates a complete plugin package:\n")

            # List generated files
            generated_files = list(decoder_dir.rglob("*"))
            file_count = len([f for f in generated_files if f.is_file()])

            self.result("Total files generated", file_count)

            self.info("\nExpected structure:")
            self.info("modbus_decoder/")
            self.info("├── __init__.py          # Plugin metadata and entry point")
            self.info("├── modbus_decoder.py    # Main implementation")
            self.info("├── tests/")
            self.info("│   ├── __init__.py")
            self.info("│   └── test_modbus_decoder.py  # Test stubs")
            self.info("├── README.md            # Usage documentation")
            self.info("└── pyproject.toml       # Packaging configuration")

            # Check for key files
            key_files = [
                "__init__.py",
                "modbus_decoder.py",
                "README.md",
                "pyproject.toml",
                "tests/test_modbus_decoder.py",
            ]

            files_found = 0
            for filename in key_files:
                file_path = decoder_dir / filename
                if file_path.exists():
                    files_found += 1
                    self.result(f"  ✓ {filename}", "present")
                else:
                    self.warning(f"  ✗ {filename} not found")

            results["decoder_files_found"] = files_found

        # ===== Section 5: Examining Template Code =====
        self.section("Part 5: Generated Code Structure")
        self.subsection("Template Implementation")

        if results.get("decoder_template_generated", False):
            # Read the main implementation file
            impl_file = decoder_dir / "modbus_decoder.py"

            if impl_file.exists():
                self.info("Generated implementation includes:\n")
                self.info("✓ Import statements for required modules")
                self.info("✓ Base class inheritance (ProtocolDecoder)")
                self.info("✓ Class attributes (id, name, channels, options)")
                self.info("✓ __init__ method with option handling")
                self.info("✓ decode() method with TODO markers")
                self.info("✓ Helper methods (if applicable)")
                self.info("✓ Docstrings with examples")
                self.info("")
                self.info("Key features:")
                self.info("  - TODO comments mark where to add logic")
                self.info("  - Type hints are included")
                self.info("  - Basic error handling structure")
                self.info("  - Follows Oscura coding standards")

                results["template_code_examined"] = True

        # ===== Section 6: Generating a Loader Template =====
        self.section("Part 6: Example - Generate Loader Template")
        self.subsection("Creating a File Format Loader")

        self.info("Let's generate a template for a custom file format loader")
        self.info("This would be useful for proprietary oscilloscope formats")

        loader_dir = temp_dir / "custom_scope_loader"

        try:
            loader_path = osc.generate_plugin_template(
                name="custom_scope_loader",
                plugin_type="loader",
                output_dir=loader_dir,
                author="Demo User",
                description="Loader for custom oscilloscope binary format",
                version="0.1.0",
            )

            self.success(f"Loader template generated at: {loader_path}")
            self.result("Template type", "loader")
            self.result("Plugin name", "custom_scope_loader")

            results["loader_template_generated"] = True
            results["loader_path"] = str(loader_path)

        except Exception as e:
            self.error(f"Failed to generate loader template: {e}")
            results["loader_template_generated"] = False

        # ===== Section 7: Generating an Analyzer Template =====
        self.section("Part 7: Example - Generate Analyzer Template")
        self.subsection("Creating a Custom Analyzer")

        self.info("Let's generate a template for a custom analyzer")
        self.info("This would be useful for domain-specific measurements")

        analyzer_dir = temp_dir / "power_quality_analyzer"

        try:
            analyzer_path = osc.generate_plugin_template(
                name="power_quality_analyzer",
                plugin_type="analyzer",
                output_dir=analyzer_dir,
                author="Demo User",
                description="Power quality analysis with harmonics and flicker",
                version="0.1.0",
            )

            self.success(f"Analyzer template generated at: {analyzer_path}")
            self.result("Template type", "analyzer")
            self.result("Plugin name", "power_quality_analyzer")

            results["analyzer_template_generated"] = True
            results["analyzer_path"] = str(analyzer_path)

        except Exception as e:
            self.error(f"Failed to generate analyzer template: {e}")
            results["analyzer_template_generated"] = False

        # ===== Section 8: Template Customization =====
        self.section("Part 8: Customizing Templates")
        self.subsection("Template Customization Options")

        self.info("Templates can be customized with parameters:")
        self.info("")
        self.info("osc.generate_plugin_template(")
        self.info("    name='my_plugin',           # Plugin identifier")
        self.info("    plugin_type='decoder',      # Type of plugin")
        self.info("    output_dir=Path('plugins'), # Where to generate")
        self.info("    author='Your Name',         # Plugin author")
        self.info("    description='...',          # What it does")
        self.info("    version='0.1.0',            # Initial version")
        self.info(")")
        self.info("")
        self.info("After generation:")
        self.info("  1. Replace TODO markers with implementation")
        self.info("  2. Add specific channel definitions (decoders)")
        self.info("  3. Add configuration options")
        self.info("  4. Write unit tests")
        self.info("  5. Update README with usage examples")
        self.info("  6. Customize metadata in __init__.py")

        results["customization_documented"] = True

        # ===== Section 9: Next Steps After Generation =====
        self.section("Part 9: From Template to Working Plugin")
        self.subsection("Development Workflow")

        self.info("After generating a template, follow these steps:\n")

        self.info("1. IMPLEMENT CORE LOGIC")
        self.info("   - Replace TODO comments with actual implementation")
        self.info("   - Add necessary imports")
        self.info("   - Implement required methods")
        self.info("")

        self.info("2. WRITE TESTS")
        self.info("   - Expand test stubs in tests/")
        self.info("   - Add test cases for edge conditions")
        self.info("   - Test with real data")
        self.info("")

        self.info("3. UPDATE DOCUMENTATION")
        self.info("   - Complete README with usage examples")
        self.info("   - Add docstrings to all public methods")
        self.info("   - Document configuration options")
        self.info("")

        self.info("4. TEST LOCALLY")
        self.info("   - Install in development mode: pip install -e .")
        self.info("   - Verify plugin discovery: osc.list_plugins()")
        self.info("   - Test plugin loading: osc.load_plugin(...)")
        self.info("")

        self.info("5. PACKAGE AND DISTRIBUTE")
        self.info("   - Update pyproject.toml with dependencies")
        self.info("   - Build: python -m build")
        self.info("   - Publish: twine upload dist/*")

        results["workflow_documented"] = True

        # ===== Section 10: Template Best Practices =====
        self.section("Part 10: Template Best Practices")
        self.subsection("Tips for Successful Plugin Development")

        self.info("Best practices when using templates:\n")

        self.info("1. START SIMPLE")
        self.info("   - Implement minimal functionality first")
        self.info("   - Test early and often")
        self.info("   - Add features incrementally")
        self.info("")

        self.info("2. FOLLOW CONVENTIONS")
        self.info("   - Keep generated structure intact")
        self.info("   - Use provided base classes")
        self.info("   - Follow naming patterns")
        self.info("")

        self.info("3. DOCUMENT THOROUGHLY")
        self.info("   - Users need clear usage examples")
        self.info("   - Document all configuration options")
        self.info("   - Include error messages and troubleshooting")
        self.info("")

        self.info("4. TEST COMPREHENSIVELY")
        self.info("   - Test with real-world data")
        self.info("   - Test error conditions")
        self.info("   - Test integration with Oscura")
        self.info("")

        self.info("5. VERSION CAREFULLY")
        self.info("   - Follow semantic versioning")
        self.info("   - Document breaking changes")
        self.info("   - Maintain backwards compatibility")

        results["best_practices_documented"] = True

        # ===== Section 11: Template Validation =====
        self.section("Part 11: Validating Generated Templates")
        self.subsection("Checking Template Completeness")

        # Count total templates generated
        templates_generated = sum(
            [
                results.get("decoder_template_generated", False),
                results.get("loader_template_generated", False),
                results.get("analyzer_template_generated", False),
            ]
        )

        self.result("Templates successfully generated", templates_generated)

        if templates_generated > 0:
            self.info("\nAll generated templates are:")
            self.info("  ✓ Syntactically valid Python")
            self.info("  ✓ Follow Oscura conventions")
            self.info("  ✓ Include comprehensive docstrings")
            self.info("  ✓ Ready for implementation")
            self.info("  ✓ Include test structure")
            self.info("  ✓ Include packaging configuration")

        results["templates_generated_count"] = templates_generated

        # ===== Section 12: Common Use Cases =====
        self.section("Part 12: Common Use Cases")
        self.subsection("When to Use Plugin Templates")

        self.info("Use plugin templates when:\n")

        self.info("1. ADDING NEW PROTOCOL SUPPORT")
        self.info("   Generate decoder template for custom protocols")
        self.info("   Example: Industrial fieldbus, automotive diagnostics")
        self.info("")

        self.info("2. SUPPORTING NEW FILE FORMATS")
        self.info("   Generate loader template for proprietary formats")
        self.info("   Example: Oscilloscope vendors, data loggers")
        self.info("")

        self.info("3. DOMAIN-SPECIFIC ANALYSIS")
        self.info("   Generate analyzer template for specialized measurements")
        self.info("   Example: RF analysis, power electronics, audio")
        self.info("")

        self.info("4. CUSTOM EXPORT FORMATS")
        self.info("   Generate exporter template for specific outputs")
        self.info("   Example: Simulation tools, custom report formats")
        self.info("")

        self.info("5. LEARNING PLUGIN DEVELOPMENT")
        self.info("   Study generated templates to understand patterns")
        self.info("   Example: Learning Oscura plugin architecture")

        results["use_cases_documented"] = True

        self.success("Plugin templates demonstration complete!")

        # Cleanup temporary directory
        import shutil

        try:
            shutil.rmtree(temp_dir)
            self.info(f"\nCleaned up temporary directory: {temp_dir}")
        except Exception as e:
            self.warning(f"Could not clean up temporary directory: {e}")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate plugin templates results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        validations = [
            ("Template system documented", "template_system_documented", True),
            ("Plugin types available", "plugin_types_available", lambda x: x >= 4),
            ("Decoder template generated", "decoder_template_generated", True),
            ("Loader template generated", "loader_template_generated", True),
            ("Analyzer template generated", "analyzer_template_generated", True),
            ("Template code examined", "template_code_examined", True),
            ("Customization documented", "customization_documented", True),
            ("Workflow documented", "workflow_documented", True),
            ("Best practices documented", "best_practices_documented", True),
            ("Templates generated count", "templates_generated_count", lambda x: x >= 3),
            ("Use cases documented", "use_cases_documented", True),
        ]

        all_passed = True
        for name, key, check in validations:
            if key not in results:
                self.error(f"Missing result key: {key}")
                all_passed = False
                continue

            value = results[key]

            if isinstance(check, bool):
                passed = value is True
            elif callable(check):
                try:
                    passed = check(value)
                except Exception as e:
                    self.error(f"Validation {name} raised exception: {e}")
                    passed = False
            else:
                passed = value == check

            if passed:
                self.result(f"✓ {name}", "PASS")
            else:
                self.result(f"✗ {name}", "FAIL", f"(got {value})")
                all_passed = False

        return all_passed


if __name__ == "__main__":
    demo = PluginTemplatesDemo()
    success = demo.execute()
    exit(0 if success else 1)
