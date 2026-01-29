"""Oscura Validate Command - Protocol Specification Validation.

Provides CLI for validating protocol specifications and message structures.


Example:
    $ oscura validate protocol_spec.yaml
    $ oscura validate --spec uart --test-data capture.wfm
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger("oscura.cli.validate")


@click.command()
@click.argument("spec", type=click.Path(exists=True))
@click.option(
    "--test-data",
    type=click.Path(exists=True),
    default=None,
    help="Test data file for validation.",
)
@click.option(
    "--output",
    type=click.Choice(["json", "table"], case_sensitive=False),
    default="table",
    help="Output format.",
)
@click.pass_context
def validate(
    ctx: click.Context,
    spec: str,
    test_data: str | None,
    output: str,
) -> None:
    """Validate protocol specification.

    Validates protocol specification files (YAML/JSON) and optionally
    tests them against real data.

    Args:
        ctx: Click context object.
        spec: Path to specification file.
        test_data: Optional test data file.
        output: Output format.

    Examples:

        \b
        # Validate spec only
        $ oscura validate protocol_spec.yaml

        \b
        # Validate against test data
        $ oscura validate uart_spec.yaml --test-data capture.wfm
    """
    verbose = ctx.obj.get("verbose", 0)

    if verbose:
        logger.info(f"Validating specification: {spec}")

    try:
        import yaml

        # Load specification
        spec_path = Path(spec)
        with open(spec_path) as f:
            if spec_path.suffix in [".yaml", ".yml"]:
                spec_data = yaml.safe_load(f)
            else:
                import json

                spec_data = json.load(f)

        # Validate spec structure
        validation_results: dict[str, Any] = {
            "spec_file": str(spec_path.name),
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        _validate_spec_structure(spec_data, validation_results)

        # Validate against test data if provided
        if test_data:
            from oscura.loaders import load

            trace = load(test_data)
            _validate_against_data(spec_data, trace, validation_results)

        # Output results
        if output == "json":
            import json

            click.echo(json.dumps(validation_results, indent=2))
        else:
            _print_validation_results(validation_results)

        # Exit with error if validation failed
        if not validation_results["valid"]:
            ctx.exit(1)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        if verbose > 1:
            raise
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _validate_spec_structure(spec: dict[str, Any], results: dict[str, Any]) -> None:
    """Validate specification structure.

    Args:
        spec: Specification dictionary.
        results: Results dictionary to update.
    """
    required_fields = ["name", "version"]

    for field in required_fields:
        if field not in spec:
            results["errors"].append(f"Missing required field: {field}")
            results["valid"] = False

    # Check optional but recommended fields
    recommended_fields = ["description", "fields", "constraints"]
    for field in recommended_fields:
        if field not in spec:
            results["warnings"].append(f"Missing recommended field: {field}")

    # Validate fields if present
    if "fields" in spec and isinstance(spec["fields"], list):
        for i, field in enumerate(spec["fields"]):
            if not isinstance(field, dict):
                results["errors"].append(f"Field {i} is not a dictionary")
                results["valid"] = False
            elif "name" not in field:
                results["errors"].append(f"Field {i} missing 'name' attribute")
                results["valid"] = False


def _validate_against_data(spec: dict[str, Any], trace: Any, results: dict[str, Any]) -> None:
    """Validate specification against test data.

    Args:
        spec: Specification dictionary.
        trace: Test data trace.
        results: Results dictionary to update.
    """
    # Check sample rate requirements
    if "sample_rate_min" in spec:
        min_rate = spec["sample_rate_min"]
        if trace.metadata.sample_rate < min_rate:
            results["warnings"].append(
                f"Sample rate {trace.metadata.sample_rate} below minimum {min_rate}"
            )

    # Check data length
    if "min_samples" in spec:
        min_samples = spec["min_samples"]
        if len(trace.data) < min_samples:
            results["errors"].append(
                f"Data has {len(trace.data)} samples, need at least {min_samples}"
            )
            results["valid"] = False

    results["test_data_samples"] = len(trace.data)
    results["test_data_sample_rate"] = trace.metadata.sample_rate


def _print_validation_results(results: dict[str, Any]) -> None:
    """Print validation results.

    Args:
        results: Validation results dictionary.
    """
    click.echo("\n=== Validation Results ===\n")
    click.echo(f"Specification: {results['spec_file']}")
    click.echo(f"Status: {'PASS' if results['valid'] else 'FAIL'}\n")

    if results["errors"]:
        click.echo("Errors:")
        for err in results["errors"]:
            click.echo(f"  - {err}")
        click.echo()

    if results["warnings"]:
        click.echo("Warnings:")
        for warn in results["warnings"]:
            click.echo(f"  - {warn}")
        click.echo()

    if "test_data_samples" in results:
        click.echo("Test Data:")
        click.echo(f"  Samples: {results['test_data_samples']}")
        click.echo(f"  Sample Rate: {results['test_data_sample_rate']} Hz")
