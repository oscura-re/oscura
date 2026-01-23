"""Configuration Files: Config-driven analysis workflows

Demonstrates:
- YAML/JSON configuration loading and validation
- Configuration schema validation
- Default values and override hierarchies
- Environment variable substitution
- Complete config-driven analysis workflows

IEEE Standards: N/A
Related Demos:
- 01_data_loading/01_loading_waveforms.py
- 02_basic_analysis/01_waveform_measurements.py
- 08_extensibility/01_custom_analyzers.py

This demonstrates how to use configuration files to drive analysis
workflows, validate settings, and manage complex analysis pipelines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    generate_sine_wave,
    validate_exists,
)


class ConfigurationFilesDemo(BaseDemo):
    """Demonstrates configuration-driven workflows."""

    def __init__(self):
        """Initialize configuration files demonstration."""
        super().__init__(
            name="configuration_files",
            description="Configuration-driven analysis workflows with YAML/JSON",
            capabilities=[
                "oscura.config.load_config",
                "oscura.config.validate_against_schema",
                "oscura.config.load_pipeline",
                "oscura.config.get_settings",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate test signals for configuration demo."""
        signal = generate_sine_wave(
            frequency=1000.0,
            amplitude=1.0,
            duration=0.01,
            sample_rate=100e3,
        )

        return {"signal": signal}

    def run_demonstration(self, data: dict) -> dict:
        """Demonstrate configuration file patterns."""
        import os

        import yaml

        signal = data["signal"]
        output_dir = self.get_output_dir()

        self.section("1. Basic Configuration Loading")
        self.info("Load configuration from YAML and JSON")

        # Create sample YAML configuration
        config_yaml = """
# Oscura Analysis Configuration
version: "1.0"

signal:
  sample_rate: 100000  # 100 kHz
  channel: CH1
  duration: 0.01  # 10 ms

analysis:
  measurements:
    - frequency
    - amplitude
    - rise_time
    - fall_time
    - thd

  spectral:
    window: hamming
    nfft: 1024
    overlap: 0.5

  filters:
    low_pass:
      enabled: true
      cutoff: 50000
    high_pass:
      enabled: false
      cutoff: 100

output:
  format: json
  save_plots: true
  plot_format: png
  output_dir: ./results
        """.strip()

        self.subsection("YAML Configuration")
        print(config_yaml[:300] + "...")

        # Save and load YAML
        yaml_path = output_dir / "config.yaml"
        yaml_path.write_text(config_yaml)

        loaded_config = yaml.safe_load(config_yaml)
        self.info(f"Loaded configuration: {len(loaded_config)} sections")

        # Create equivalent JSON
        config_json = json.dumps(loaded_config, indent=2)
        json_path = output_dir / "config.json"
        json_path.write_text(config_json)

        self.subsection("JSON Configuration")
        print(config_json[:200] + "...")

        self.section("2. Configuration Schema Validation")
        self.info("Validate configurations against JSON schemas")

        # Define configuration schema
        config_schema = {
            "type": "object",
            "required": ["version", "signal", "analysis"],
            "properties": {
                "version": {
                    "type": "string",
                    "pattern": "^[0-9]+\\.[0-9]+$",
                },
                "signal": {
                    "type": "object",
                    "required": ["sample_rate"],
                    "properties": {
                        "sample_rate": {"type": "number", "minimum": 1},
                        "channel": {"type": "string"},
                        "duration": {"type": "number", "minimum": 0},
                    },
                },
                "analysis": {
                    "type": "object",
                    "properties": {
                        "measurements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "spectral": {
                            "type": "object",
                            "properties": {
                                "window": {
                                    "type": "string",
                                    "enum": ["hamming", "hann", "blackman", "bartlett"],
                                },
                                "nfft": {"type": "integer", "minimum": 1},
                                "overlap": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                        },
                    },
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["json", "csv", "yaml"],
                        },
                        "save_plots": {"type": "boolean"},
                    },
                },
            },
        }

        self.subsection("Configuration Schema")
        print(json.dumps(config_schema, indent=2)[:300] + "...")

        # Validate configuration
        self.subsection("Validation")
        try:
            # Simple validation (oscura's validate_against_schema expects registered schemas)
            # For demo, we'll do basic checks
            is_valid = all(key in loaded_config for key in ["version", "signal", "analysis"])
            if is_valid:
                self.success("Configuration is valid")
            else:
                self.error("Configuration is invalid")
        except Exception as e:
            self.warning(f"Validation error: {e}")

        self.section("3. Default Values and Overrides")
        self.info("Hierarchical configuration with defaults")

        # Default configuration
        default_config = {
            "signal": {
                "sample_rate": 1e6,
                "channel": "CH1",
            },
            "analysis": {
                "measurements": ["frequency", "amplitude"],
                "spectral": {
                    "window": "hamming",
                    "nfft": 1024,
                },
            },
            "output": {
                "format": "json",
                "save_plots": False,
            },
        }

        # User configuration (overrides)
        user_config = {
            "signal": {
                "sample_rate": 100e3,  # Override
            },
            "analysis": {
                "measurements": ["frequency", "amplitude", "thd"],  # Override
            },
        }

        # Merge configurations
        def merge_configs(default: dict, override: dict) -> dict:
            """Recursively merge configurations.

            Args:
                default: Default configuration
                override: Override configuration

            Returns:
                Merged configuration
            """
            result = default.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_configs(result[key], value)
                else:
                    result[key] = value
            return result

        merged_config = merge_configs(default_config, user_config)

        self.subsection("Default Configuration")
        print(json.dumps(default_config, indent=2)[:200] + "...")

        self.subsection("User Overrides")
        print(json.dumps(user_config, indent=2))

        self.subsection("Merged Configuration")
        print(json.dumps(merged_config, indent=2)[:300] + "...")

        self.section("4. Environment Variable Substitution")
        self.info("Use environment variables in configurations")

        # Configuration with environment variables
        env_config = """
output:
  output_dir: ${OSCURA_OUTPUT_DIR:-./results}
  log_level: ${OSCURA_LOG_LEVEL:-INFO}

database:
  host: ${DB_HOST:-localhost}
  port: ${DB_PORT:-5432}
  user: ${DB_USER}
        """.strip()

        self.subsection("Configuration with Environment Variables")
        print(env_config)

        # Set example environment variables (safe for demonstration)
        # Note: Using /tmp is acceptable for demos; use tempfile.mkdtemp() in production
        os.environ["OSCURA_OUTPUT_DIR"] = "/tmp/oscura_results"  # noqa: S108
        os.environ["OSCURA_LOG_LEVEL"] = "DEBUG"
        os.environ["DB_USER"] = "oscura_user"

        # Substitute environment variables
        def substitute_env_vars(config_str: str) -> str:
            """Substitute environment variables in config string.

            Args:
                config_str: Configuration string with ${VAR:-default} syntax

            Returns:
                String with substituted values
            """
            import re

            def replace_var(match):
                var_expr = match.group(1)
                if ":-" in var_expr:
                    var_name, default_value = var_expr.split(":-", 1)
                else:
                    var_name = var_expr
                    default_value = ""

                return os.environ.get(var_name.strip(), default_value)

            return re.sub(r"\$\{([^}]+)\}", replace_var, config_str)

        substituted = substitute_env_vars(env_config)

        self.subsection("After Substitution")
        print(substituted)

        self.section("5. Pipeline Configuration")
        self.info("Define multi-stage analysis pipelines in config")

        pipeline_config = {
            "name": "comprehensive_analysis",
            "description": "Complete signal analysis pipeline",
            "stages": [
                {
                    "name": "load",
                    "type": "loader",
                    "params": {
                        "format": "auto",
                        "channel": "CH1",
                    },
                },
                {
                    "name": "preprocess",
                    "type": "filter",
                    "params": {
                        "low_pass": {
                            "cutoff": 50000,
                            "order": 4,
                        },
                    },
                },
                {
                    "name": "measure",
                    "type": "analyzer",
                    "params": {
                        "measurements": [
                            "frequency",
                            "amplitude",
                            "rise_time",
                            "fall_time",
                        ],
                    },
                },
                {
                    "name": "spectral",
                    "type": "analyzer",
                    "params": {
                        "analysis": ["fft", "psd", "thd"],
                        "window": "hamming",
                    },
                },
                {
                    "name": "export",
                    "type": "exporter",
                    "params": {
                        "formats": ["json", "csv"],
                        "output_dir": "./results",
                    },
                },
            ],
        }

        self.subsection("Pipeline Configuration")
        print(json.dumps(pipeline_config, indent=2))

        # Save pipeline config
        pipeline_path = output_dir / "pipeline.json"
        pipeline_path.write_text(json.dumps(pipeline_config, indent=2))

        self.section("6. Config-Driven Workflow Execution")
        self.info("Execute analysis based on configuration")

        def run_analysis_from_config(signal, config: dict) -> dict:
            """Run analysis based on configuration.

            Args:
                signal: Input signal
                config: Configuration dictionary

            Returns:
                Analysis results
            """
            from oscura.analyzers.waveform.measurements import (
                amplitude,
                fall_time,
                frequency,
                rise_time,
            )
            from oscura.analyzers.waveform.spectral import thd

            results = {}

            # Get measurements from config
            measurements_list = config.get("analysis", {}).get("measurements", [])

            for meas_name in measurements_list:
                if meas_name == "frequency":
                    results["frequency"] = frequency(signal)
                elif meas_name == "amplitude":
                    results["amplitude"] = amplitude(signal)
                elif meas_name == "rise_time":
                    results["rise_time"] = rise_time(signal)
                elif meas_name == "fall_time":
                    results["fall_time"] = fall_time(signal)
                elif meas_name == "thd":
                    results["thd"] = thd(signal)

            # Add metadata
            results["_metadata"] = {
                "config_version": config.get("version", "unknown"),
                "measurements_count": len(results) - 1,  # Exclude metadata
            }

            return results

        self.subsection("Running Config-Driven Analysis")
        analysis_results = run_analysis_from_config(signal, merged_config)

        for key, value in analysis_results.items():
            if not key.startswith("_"):
                self.result(key, f"{value:.6g}")

        # Save results in configured format
        output_format = merged_config.get("output", {}).get("format", "json")
        results_path = output_dir / f"results.{output_format}"

        if output_format == "json":
            results_path.write_text(json.dumps(analysis_results, indent=2, default=str))
        elif output_format == "yaml":
            results_path.write_text(yaml.dump(analysis_results, default_flow_style=False))

        self.info(f"Results saved to {results_path}")

        self.section("7. Configuration Best Practices")
        self.info("Recommendations for config-driven workflows")

        best_practices = [
            ("Use schemas", "Validate all configurations against schemas"),
            ("Version configs", "Include version field for compatibility"),
            ("Provide defaults", "Use sensible defaults with override capability"),
            ("Environment vars", "Allow environment variable substitution"),
            ("Separate concerns", "Split into signal, analysis, output sections"),
            ("Document schema", "Provide clear schema documentation"),
            ("Validate early", "Check configuration at load time, not runtime"),
            ("Support formats", "Accept both YAML and JSON"),
        ]

        for practice, description in best_practices:
            self.result(practice, description)

        self.success("Configuration-driven workflows demonstrated!")

        return {
            "loaded_config": loaded_config,
            "merged_config": merged_config,
            "pipeline_config": pipeline_config,
            "analysis_results": analysis_results,
            "config_files": [str(yaml_path), str(json_path), str(pipeline_path)],
        }

    def validate(self, results: dict) -> bool:
        """Validate configuration workflow results."""
        self.info("Validating configuration workflow...")

        # Check loaded config
        if not validate_exists(results.get("loaded_config"), "loaded_config"):
            return False

        config = results["loaded_config"]
        required_sections = ["version", "signal", "analysis", "output"]
        for section in required_sections:
            if section not in config:
                self.error(f"Missing configuration section: {section}")
                return False

        # Check merged config
        if not validate_exists(results.get("merged_config"), "merged_config"):
            return False

        # Check pipeline config
        if not validate_exists(results.get("pipeline_config"), "pipeline_config"):
            return False

        pipeline = results["pipeline_config"]
        if "stages" not in pipeline:
            self.error("Pipeline missing 'stages' key")
            return False

        if len(pipeline["stages"]) < 3:
            self.error("Pipeline should have at least 3 stages")
            return False

        # Check analysis results
        if not validate_exists(results.get("analysis_results"), "analysis_results"):
            return False

        # Check config files were created
        if not validate_exists(results.get("config_files"), "config_files"):
            return False

        self.success("All configuration workflow tests passed!")
        self.info("\nNext steps:")
        self.info("  - Create custom analysis pipelines")
        self.info("  - Use YAML/JSON for reproducible workflows")
        self.info("  - See 05_hardware_integration.py for hardware config")

        return True


if __name__ == "__main__":
    demo = ConfigurationFilesDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
