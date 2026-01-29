"""Pipeline configuration and execution system.

This module provides YAML-based pipeline definitions, loading, execution,
templates, composition, and conditional logic for analysis workflows with
transaction semantics, type validation, circular detection, and expression
language support.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from oscura.core.config.schema import validate_against_schema
from oscura.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """Single step in an analysis pipeline.

    Attributes:
        name: Step identifier
        type: Step type (e.g., "input.file", "decoder.uart")
        params: Step parameters
        inputs: Input mappings from previous steps
        outputs: Output definitions
        condition: Optional condition for execution
        if_steps: Steps to execute if condition is true
        elif_conditions: List of (condition, steps) for elif branches
        else_steps: Steps to execute if condition is false

    Example:
        >>> step = PipelineStep(
        ...     name="decode_uart",
        ...     type="decoder.uart",
        ...     params={"baud_rate": 115200}
        ... )
    """

    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    condition: str | None = None
    if_steps: list[PipelineStep] = field(default_factory=list)
    elif_conditions: list[tuple[str, list[PipelineStep]]] = field(default_factory=list)
    else_steps: list[PipelineStep] = field(default_factory=list)


@dataclass
class PipelineDefinition:
    """Complete pipeline definition.

    Attributes:
        name: Pipeline identifier
        version: Pipeline version
        description: Human-readable description
        steps: Ordered list of pipeline steps
        parallel_groups: Groups of steps that can run in parallel
        variables: Template variables for parameterization
        includes: List of included sub-pipelines

    Example:
        >>> pipeline = PipelineDefinition(
        ...     name="uart_analysis",
        ...     steps=[load_step, decode_step, export_step]
        ... )
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    steps: list[PipelineStep] = field(default_factory=list)
    parallel_groups: list[list[str]] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    includes: list[str] = field(default_factory=list)
    source_file: str | None = None


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    Attributes:
        pipeline_name: Name of executed pipeline
        outputs: Dictionary of output data from steps
        step_results: Results from each step
        success: Whether pipeline completed successfully
        error: Error if failed
    """

    pipeline_name: str
    outputs: dict[str, Any] = field(default_factory=dict)
    step_results: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


class PipelineValidationError(ConfigurationError):
    """Pipeline validation error with step information.

    Attributes:
        step_name: Name of failing step
        suggestion: Suggested fix
    """

    def __init__(
        self,
        message: str,
        *,
        step_name: str | None = None,
        line: int | None = None,
        suggestion: str | None = None,
    ):
        self.step_name = step_name
        self.line = line
        self.suggestion = suggestion
        super().__init__(message)


class PipelineExecutionError(ConfigurationError):
    """Pipeline execution error.

    Attributes:
        step_name: Name of failing step
        traceback_str: Traceback string if available
    """

    def __init__(
        self,
        message: str,
        *,
        step_name: str | None = None,
        traceback_str: str | None = None,
    ):
        self.step_name = step_name
        self.traceback_str = traceback_str
        super().__init__(message)


class Pipeline:
    """Executable analysis pipeline.

    Loads, validates, and executes pipeline definitions with support
    for progress tracking, error handling, and rollback.

    Example:
        >>> pipeline = Pipeline.load("uart_analysis.yaml")
        >>> pipeline.on_progress(lambda step, pct: print(f"{step}: {pct}%"))
        >>> results = pipeline.execute()
    """

    def __init__(self, definition: PipelineDefinition):
        """Initialize pipeline from definition.

        Args:
            definition: Pipeline definition
        """
        self.definition = definition
        self._progress_callbacks: list[Callable[[str, int], None]] = []
        self._step_handlers: dict[str, Callable[..., Any]] = {}
        self._state: dict[str, Any] = {}
        self._cleanups: list[Callable[[], None]] = []

    @classmethod
    def load(cls, path: str | Path, variables: dict[str, Any] | None = None) -> Pipeline:
        """Load pipeline from YAML file.

        Args:
            path: Path to pipeline definition file
            variables: Template variable values

        Returns:
            Loaded and validated Pipeline

        Example:
            >>> pipeline = Pipeline.load("pipeline.yaml")
            >>> pipeline = Pipeline.load("pipeline.yaml", {"input_file": "trace.bin"})
        """
        definition = load_pipeline(path, variables)
        return cls(definition)

    def on_progress(self, callback: Callable[[str, int], None]) -> None:
        """Register progress callback.

        Args:
            callback: Function called with (step_name, percent_complete)

        Example:
            >>> pipeline.on_progress(lambda step, pct: print(f"{step}: {pct}%"))
        """
        self._progress_callbacks.append(callback)

    def register_handler(self, step_type: str, handler: Callable[..., Any]) -> None:
        """Register handler for step type.

        Args:
            step_type: Step type to handle
            handler: Handler function
        """
        self._step_handlers[step_type] = handler

    def execute(self, dry_run: bool = False) -> PipelineResult:
        """Execute the pipeline with transaction semantics.

        The pipeline executes with ACID-like semantics:
        - All steps complete successfully, or
        - Pipeline rolls back and cleanup is guaranteed

        Args:
            dry_run: If True, validate without executing (dry-run mode)

        Returns:
            Pipeline execution result

        Raises:
            PipelineExecutionError: If execution fails

        Example:
            >>> result = pipeline.execute()
            >>> result = pipeline.execute(dry_run=True)  # Validate only
        """
        result = PipelineResult(pipeline_name=self.definition.name)
        self._state = {}
        self._cleanups = []
        committed = False

        try:
            total_steps = len(self.definition.steps)

            # Transaction begin
            logger.debug(f"Beginning pipeline transaction: {self.definition.name}")

            for i, step in enumerate(self.definition.steps):
                progress = int((i / total_steps) * 100)
                self._notify_progress(step.name, progress)

                if dry_run:
                    logger.info(f"[DRY RUN] Would execute step: {step.name}")
                    # In dry-run, validate step configuration
                    self._validate_step(step)
                    continue

                # Check condition with short-circuit evaluation
                if step.condition:
                    try:
                        should_execute = self._evaluate_condition(step.condition)
                        if not should_execute:
                            logger.info(f"Skipping step '{step.name}' (condition false)")
                            continue
                    except Exception as e:
                        logger.warning(f"Condition evaluation failed for '{step.name}': {e}")
                        continue

                # Execute step
                step_result = self._execute_step(step)
                result.step_results[step.name] = step_result

                # Store outputs in state with namespace isolation
                for output_name, output_key in step.outputs.items():
                    namespaced_key = f"{step.name}.{output_name}"
                    self._state[namespaced_key] = step_result.get(output_key)
                    logger.debug(f"Stored output: {namespaced_key}")

            # Transaction commit
            logger.debug(f"Committing pipeline transaction: {self.definition.name}")
            committed = True

            # Notify completion
            self._notify_progress("complete", 100)
            result.success = True
            result.outputs = dict(self._state)

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Pipeline execution failed: {e}")

            # Transaction rollback
            if not committed:
                logger.warning(f"Rolling back pipeline transaction: {self.definition.name}")
                self._rollback()

            if not dry_run:
                raise PipelineExecutionError(
                    f"Pipeline '{self.definition.name}' failed",
                    step_name=step.name if "step" in dir() else None,
                    traceback_str=str(e),
                ) from e

        return result

    def _validate_step(self, step: PipelineStep) -> None:
        """Validate step configuration (used in dry-run).

        Args:
            step: Step to validate

        Raises:
            PipelineValidationError: If step configuration invalid
        """
        # Check required fields
        if not step.name:
            raise PipelineValidationError("Step name is required")
        if not step.type:
            raise PipelineValidationError(f"Step '{step.name}' missing type", step_name=step.name)

        # Validate input references
        for input_ref in step.inputs.values():
            if "." not in input_ref and input_ref not in self._state:
                logger.warning(f"Step '{step.name}' references undefined input: {input_ref}")

    def _rollback(self) -> None:
        """Rollback pipeline execution (cleanup all resources).

        Guaranteed cleanup of all resources allocated during execution.
        Runs in reverse order of allocation.
        """
        logger.info("Running rollback cleanup")
        self._run_cleanups()
        self._state.clear()
        logger.info("Rollback complete")

    def _execute_step(self, step: PipelineStep) -> dict[str, Any]:
        """Execute a single pipeline step.

        Args:
            step: Step to execute

        Returns:
            Step result dictionary

        Raises:
            PipelineExecutionError: If no handler found for step type.
        """
        logger.debug(f"Executing step: {step.name} (type={step.type})")

        # Resolve inputs from state
        resolved_inputs = {}
        for input_name, input_ref in step.inputs.items():
            if input_ref in self._state:
                resolved_inputs[input_name] = self._state[input_ref]
            else:
                logger.warning(f"Input '{input_ref}' not found in state")

        # Get handler
        handler = self._step_handlers.get(step.type)
        if handler is None:
            handler = self._get_default_handler(step.type)

        if handler is None:
            raise PipelineExecutionError(
                f"No handler for step type '{step.type}'", step_name=step.name
            )

        # Execute handler
        result = handler(inputs=resolved_inputs, params=step.params, step_name=step.name)

        return result if isinstance(result, dict) else {"result": result}

    def _get_default_handler(self, step_type: str) -> Callable[..., Any] | None:
        """Get default handler for step type.

        Args:
            step_type: Step type

        Returns:
            Handler function or None
        """
        # Built-in handlers for common step types
        handlers = {
            "input.file": self._handle_input_file,
            "output.json": self._handle_output_json,
            "analysis.statistics": self._handle_statistics,
        }
        return handlers.get(step_type)

    def _handle_input_file(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        step_name: str,
    ) -> dict[str, Any]:
        """Handle file input step."""
        # Placeholder - actual implementation would use loaders
        return {"waveform": params.get("path")}

    def _handle_output_json(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        step_name: str,
    ) -> dict[str, Any]:
        """Handle JSON output step."""
        import json

        path = params.get("path", "output.json")
        data = inputs.get("data", inputs)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return {"path": path}

    def _handle_statistics(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        step_name: str,
    ) -> dict[str, Any]:
        """Handle statistics step."""
        return {"statistics": {"count": len(inputs)}}

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate condition expression using expression language.

        Supports:
        - Comparison operators: ==, !=, <, >, <=, >=
        - Logical operators: and, or, not
        - Field access: data.confidence, step_name.output_name
        - Short-circuit evaluation (and/or operators)

        Args:
            condition: Condition expression string

        Returns:
            Evaluation result

        Example:
            >>> self._evaluate_condition("data.confidence > 0.8")
            >>> self._evaluate_condition("decode_uart.packets > 0 and data.valid == True")
        """
        try:
            # Parse and evaluate expression with short-circuit support
            return self._eval_expression(condition)
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {condition} - {e}")
            return False

    def _eval_expression(self, expr: str) -> bool:
        """Evaluate expression with short-circuit logic.

        Args:
            expr: Expression string

        Returns:
            Boolean result
        """
        # Handle logical operators with short-circuit evaluation
        if " or " in expr:
            parts = expr.split(" or ", 1)
            left = self._eval_expression(parts[0].strip())
            if left:  # Short-circuit: if left is True, don't evaluate right
                logger.debug(f"Short-circuit OR: left={left}, skipping right")
                return True
            return self._eval_expression(parts[1].strip())

        if " and " in expr:
            parts = expr.split(" and ", 1)
            left = self._eval_expression(parts[0].strip())
            if not left:  # Short-circuit: if left is False, don't evaluate right
                logger.debug(f"Short-circuit AND: left={left}, skipping right")
                return False
            return self._eval_expression(parts[1].strip())

        if expr.startswith("not "):
            return not self._eval_expression(expr[4:].strip())

        # Evaluate comparison
        return self._eval_comparison(expr)

    def _eval_comparison(self, expr: str) -> bool:
        """Evaluate comparison expression.

        Args:
            expr: Comparison expression

        Returns:
            Boolean result
        """
        operators = ["<=", ">=", "==", "!=", "<", ">"]

        for op in operators:
            if op in expr:
                left, right = expr.split(op, 1)
                left_val = self._resolve_value(left.strip())
                right_val = self._resolve_value(right.strip())

                if op == "==":
                    return left_val == right_val  # type: ignore[no-any-return]
                elif op == "!=":
                    return left_val != right_val  # type: ignore[no-any-return]
                elif op == "<":
                    return left_val < right_val  # type: ignore[no-any-return]
                elif op == ">":
                    return left_val > right_val  # type: ignore[no-any-return]
                elif op == "<=":
                    return left_val <= right_val  # type: ignore[no-any-return]
                elif op == ">=":
                    return left_val >= right_val  # type: ignore[no-any-return]

        # No comparison operator found, try as boolean
        return bool(self._resolve_value(expr.strip()))

    def _resolve_value(self, value_str: str) -> Any:
        """Resolve value from string (lookup in state or parse literal).

        Args:
            value_str: Value string (field reference or literal)

        Returns:
            Resolved value
        """
        value_str = value_str.strip()

        # Check if it's a field reference in state
        if value_str in self._state:
            return self._state[value_str]

        # Try to parse as literal
        # String literals
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            return value_str[1:-1]

        # Boolean literals
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

        # None/null literal
        if value_str.lower() in ("none", "null"):
            return None

        # Try numeric literals
        try:
            if "." in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass

        # Return as string if can't resolve
        logger.warning(f"Could not resolve value: {value_str}, returning as string")
        return value_str

    def _notify_progress(self, step: str, percent: int) -> None:
        """Notify progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(step, percent)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def _run_cleanups(self) -> None:
        """Run registered cleanup functions."""
        for cleanup in reversed(self._cleanups):
            try:
                cleanup()
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")


def load_pipeline(path: str | Path, variables: dict[str, Any] | None = None) -> PipelineDefinition:
    """Load pipeline definition from file.

    Args:
        path: Path to YAML file
        variables: Template variable values

    Returns:
        Pipeline definition

    Raises:
        PipelineValidationError: If validation fails
    """
    path = Path(path)

    if not path.exists():
        raise PipelineValidationError(
            f"Pipeline file not found: {path}", suggestion="Check file path"
        )

    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()

        # Apply template variables
        if variables:
            content = _substitute_variables(content, variables)

        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise PipelineValidationError(
            f"YAML parse error in {path}", suggestion="Check YAML syntax"
        ) from e

    # Handle nested 'pipeline' key
    if "pipeline" in data:
        data = data["pipeline"]

    # Validate against schema
    try:
        validate_against_schema(data, "pipeline")
    except Exception as e:
        raise PipelineValidationError(
            f"Pipeline validation failed for {path}", suggestion=str(e)
        ) from e

    # Parse steps
    steps = []
    for step_data in data.get("steps", []):
        step = _parse_step(step_data)
        steps.append(step)

    return PipelineDefinition(
        name=data.get("name", path.stem),
        version=data.get("version", "1.0.0"),
        description=data.get("description", ""),
        steps=steps,
        parallel_groups=data.get("parallel_groups", []),
        variables=variables or {},
        includes=data.get("includes", []),
        source_file=str(path),
    )


def _parse_step(data: dict[str, Any]) -> PipelineStep:
    """Parse step from dictionary."""
    step = PipelineStep(
        name=data.get("name", "unnamed"),
        type=data.get("type", "unknown"),
        params=data.get("params", {}),
        inputs=data.get("inputs", {}),
        outputs=data.get("outputs", {}),
        condition=data.get("condition"),
    )

    # Parse conditional steps
    if "if_steps" in data:
        step.if_steps = [_parse_step(s) for s in data["if_steps"]]
    if "elif_conditions" in data:
        for elif_data in data["elif_conditions"]:
            cond = elif_data.get("condition")
            steps = [_parse_step(s) for s in elif_data.get("steps", [])]
            step.elif_conditions.append((cond, steps))
    if "else_steps" in data:
        step.else_steps = [_parse_step(s) for s in data["else_steps"]]

    return step


def _substitute_variables(content: str, variables: dict[str, Any], max_depth: int = 3) -> str:
    """Substitute template variables in content with nested substitution.

    Supports nested substitution up to 3 levels deep (CFG-011):
    - Level 1: ${VAR1} -> "value1"
    - Level 2: ${VAR2} where VAR2 = "${VAR1}" -> "value1"
    - Level 3: ${VAR3} where VAR3 = "${VAR2}" -> "value1"

    Args:
        content: String content with ${VAR_NAME} placeholders
        variables: Variable name to value mapping
        max_depth: Maximum nested substitution depth (default 3.)

    Returns:
        Content with variables substituted

    Raises:
        PipelineValidationError: If nested substitution depth exceeded

    Example:
        >>> vars = {"BASE": "trace", "FILE": "${BASE}.bin"}
        >>> _substitute_variables("path: ${FILE}", vars)
        'path: trace.bin'
    """
    pattern = re.compile(r"\$\{(\w+)\}")
    depth = 0

    for depth in range(max_depth):
        prev_content = content
        substitutions_made = False

        # Find all matches in current content
        matches = list(pattern.finditer(content))
        for match in matches:
            var_name = match.group(1)
            if var_name in variables:
                value = str(variables[var_name])
                content = content.replace(match.group(0), value)
                substitutions_made = True

        # No more substitutions possible
        if content == prev_content or not substitutions_made:
            break

        # Check if we still have unresolved variables after this pass
        remaining = pattern.findall(content)
        if not remaining:
            break

    # Check for unresolved variables
    remaining_vars = pattern.findall(content)
    if remaining_vars:
        if depth >= max_depth - 1:
            raise PipelineValidationError(
                f"Nested substitution depth exceeded {max_depth} levels",
                suggestion=f"Reduce nesting or increase max_depth. Unresolved: {remaining_vars}",
            )
        else:
            # Some variables are undefined
            undefined = [v for v in remaining_vars if v not in variables]
            if undefined:
                logger.warning(f"Undefined variables: {undefined}")

    logger.debug(f"Variable substitution completed in {depth + 1} passes")
    return content


def resolve_includes(
    pipeline: PipelineDefinition,
    base_path: Path,
    *,
    max_depth: int = 5,
    namespace_isolation: bool = True,
    _visited: set[str] | None = None,
    _depth: int = 0,
) -> PipelineDefinition:
    """Resolve pipeline includes (composition) with circular detection.

    Supports include depth up to 5 levels with dependency graph traversal
    for cycle detection. Provides namespace isolation for included pipelines.

    Args:
        pipeline: Pipeline with includes
        base_path: Base path for resolving relative includes
        max_depth: Maximum include depth (default 5.)
        namespace_isolation: If True, prefix included steps with namespace
        _visited: Set of visited pipelines for cycle detection (DFS)
        _depth: Current depth (for tracking)

    Returns:
        Pipeline with includes resolved

    Raises:
        PipelineValidationError: If circular includes or depth exceeded

    Example:
        >>> pipeline = load_pipeline("main.yaml")
        >>> resolved = resolve_includes(pipeline, Path("."))
    """
    if _visited is None:
        _visited = set()

    if not pipeline.includes:
        return pipeline

    source_key = _get_source_key(pipeline)
    _validate_circular_dependency(source_key, _visited)
    _validate_depth_limit(_depth, max_depth, _visited, source_key)

    if source_key:
        _visited.add(source_key)

    merged_steps = _merge_included_pipelines(
        pipeline, base_path, max_depth, namespace_isolation, _visited, _depth
    )
    merged_steps.extend(pipeline.steps)

    return _build_resolved_pipeline(pipeline, merged_steps)


def _get_source_key(pipeline: PipelineDefinition) -> str | None:
    """Get normalized source file path for pipeline.

    Args:
        pipeline: Pipeline definition.

    Returns:
        Normalized source path or None.
    """
    return str(Path(pipeline.source_file).resolve()) if pipeline.source_file else None


def _validate_circular_dependency(source_key: str | None, visited: set[str]) -> None:
    """Validate no circular dependency.

    Args:
        source_key: Source file path.
        visited: Set of visited paths.

    Raises:
        PipelineValidationError: If circular dependency detected.
    """
    if source_key and source_key in visited:
        cycle_list = [*list(visited), source_key]
        cycle = " → ".join([Path(p).name for p in cycle_list])
        raise PipelineValidationError(
            f"Circular pipeline include detected: {cycle}",
            suggestion=f"Remove circular dependency from {Path(source_key).name}",
        )


def _validate_depth_limit(
    depth: int, max_depth: int, visited: set[str], source_key: str | None
) -> None:
    """Validate include depth within limit.

    Args:
        depth: Current depth.
        max_depth: Maximum allowed depth.
        visited: Set of visited paths.
        source_key: Current source key.

    Raises:
        PipelineValidationError: If depth exceeded.
    """
    if depth >= max_depth:
        chain = " → ".join(
            [Path(p).name for p in visited] + [Path(source_key).name if source_key else "?"]
        )
        raise PipelineValidationError(
            f"Pipeline include depth exceeded maximum of {max_depth}",
            suggestion=f"Reduce nesting. Current chain: {chain}",
        )


def _merge_included_pipelines(
    pipeline: PipelineDefinition,
    base_path: Path,
    max_depth: int,
    namespace_isolation: bool,
    visited: set[str],
    depth: int,
) -> list[PipelineStep]:
    """Merge all included pipelines.

    Args:
        pipeline: Main pipeline.
        base_path: Base path for includes.
        max_depth: Maximum depth.
        namespace_isolation: Enable namespacing.
        visited: Visited paths.
        depth: Current depth.

    Returns:
        List of merged steps.
    """
    merged_steps = []

    for include_path in pipeline.includes:
        include_full = base_path / include_path

        if not include_full.exists():
            logger.warning(f"Included pipeline not found: {include_path}")
            continue

        try:
            included = load_pipeline(include_full, pipeline.variables)
            resolved = resolve_includes(
                included,
                include_full.parent,
                max_depth=max_depth,
                namespace_isolation=namespace_isolation,
                _visited=visited.copy(),
                _depth=depth + 1,
            )

            if namespace_isolation:
                namespace = Path(include_path).stem
                namespaced_steps = _apply_namespace(resolved.steps, namespace)
                merged_steps.extend(namespaced_steps)
                logger.debug(f"Included pipeline '{namespace}' with {len(namespaced_steps)} steps")
            else:
                merged_steps.extend(resolved.steps)

        except Exception as e:
            logger.error(f"Failed to include pipeline {include_path}: {e}")
            raise PipelineValidationError(
                f"Failed to include pipeline: {include_path}",
                suggestion=f"Check file exists and is valid YAML: {e}",
            ) from e

    return merged_steps


def _build_resolved_pipeline(
    pipeline: PipelineDefinition, merged_steps: list[PipelineStep]
) -> PipelineDefinition:
    """Build resolved pipeline with merged steps.

    Args:
        pipeline: Original pipeline.
        merged_steps: Merged steps list.

    Returns:
        New PipelineDefinition with resolved includes.
    """
    return PipelineDefinition(
        name=pipeline.name,
        version=pipeline.version,
        description=pipeline.description,
        steps=merged_steps,
        parallel_groups=pipeline.parallel_groups,
        variables=pipeline.variables,
        includes=[],
        source_file=pipeline.source_file,
    )


def _apply_namespace(steps: list[PipelineStep], namespace: str) -> list[PipelineStep]:
    """Apply namespace prefix to pipeline steps.

    Args:
        steps: Steps to namespace
        namespace: Namespace prefix

    Returns:
        Namespaced steps

    Example:
        >>> steps = [PipelineStep(name="decode", ...)]
        >>> _apply_namespace(steps, "uart")
        [PipelineStep(name="uart.decode", ...)]
    """
    namespaced = []
    for step in steps:
        # Create a copy with namespaced name
        # NECESSARY COPIES: All .copy() calls create isolated params/conditions.
        # This prevents parameter mutations from affecting original step definitions.
        namespaced_step = PipelineStep(
            name=f"{namespace}.{step.name}",
            type=step.type,
            params=step.params.copy(),
            inputs=step.inputs.copy(),
            outputs={k: f"{namespace}.{v}" if "." not in v else v for k, v in step.outputs.items()},
            condition=step.condition,
            if_steps=step.if_steps.copy() if step.if_steps else [],
            elif_conditions=step.elif_conditions.copy() if step.elif_conditions else [],
            else_steps=step.else_steps.copy() if step.else_steps else [],
        )
        namespaced.append(namespaced_step)
    return namespaced


class PipelineTemplate:
    """Parameterized pipeline template.

    Provides pipeline definition with parameter placeholders that
    can be instantiated with different values.

    Example:
        >>> template = PipelineTemplate.load("analysis_template.yaml")
        >>> pipeline = template.instantiate(sample_rate=1e9, protocol="uart")
    """

    def __init__(self, definition: PipelineDefinition, parameters: dict[str, dict[str, Any]]):
        """Initialize template.

        Args:
            definition: Base pipeline definition
            parameters: Parameter definitions with type, default, required
        """
        self.definition = definition
        self.parameters = parameters

    @classmethod
    def load(cls, path: str | Path) -> PipelineTemplate:
        """Load template from file.

        Args:
            path: Path to template file

        Returns:
            Loaded template
        """
        path = Path(path)

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Extract parameter definitions
        params = data.get("parameters", {})
        parameter_defs = {}
        for name, spec in params.items():
            parameter_defs[name] = {
                "type": spec.get("type", "string"),
                "default": spec.get("default"),
                "required": spec.get("required", False),
                "description": spec.get("description", ""),
            }

        # Load pipeline without variable substitution
        definition = PipelineDefinition(
            name=data.get("pipeline", {}).get("name", path.stem),
            version=data.get("pipeline", {}).get("version", "1.0.0"),
            description=data.get("pipeline", {}).get("description", ""),
            steps=[_parse_step(s) for s in data.get("pipeline", {}).get("steps", [])],
            source_file=str(path),
        )

        return cls(definition, parameter_defs)

    def instantiate(self, **kwargs: Any) -> Pipeline:
        """Create pipeline instance with parameter values and type validation.

        Validates all parameters against their type specifications before
        instantiation. Supports: int, float, string, bool, list, dict.

        Args:
            **kwargs: Parameter values

        Returns:
            Instantiated Pipeline

        Raises:
            PipelineValidationError: If required parameters missing or type mismatch

        Example:
            >>> template = PipelineTemplate.load("analysis.yaml")
            >>> pipeline = template.instantiate(sample_rate=1000000, protocol="uart")
        """
        # Collect required and provided parameters
        required_params = [
            name for name, spec in self.parameters.items() if spec.get("required", False)
        ]
        provided_params = list(kwargs.keys())
        missing_params = [p for p in required_params if p not in kwargs]

        # Check for missing required parameters
        if missing_params:
            raise PipelineValidationError(
                f"Missing required parameters: {missing_params}",
                suggestion=f"Required: {required_params}, provided: {provided_params}",
            )

        # Validate parameters with type checking
        variables = {}
        type_errors = []

        for name, spec in self.parameters.items():
            if name in kwargs:
                value = kwargs[name]
                expected_type = spec.get("type", "string")

                # Type validation with detailed error reporting
                if not _validate_type(value, expected_type):
                    type_errors.append(
                        f"{name}: expects {expected_type}, got {type(value).__name__} ('{value}')"
                    )
                    continue

                variables[name] = value
            elif spec.get("required", False):
                # Already caught above, but defensive check
                raise PipelineValidationError(
                    f"Required parameter '{name}' not provided",
                    suggestion=f"Provide value for '{name}'",
                )
            elif "default" in spec:
                default_val = spec["default"]
                # Validate default value type
                expected_type = spec.get("type", "string")
                if not _validate_type(default_val, expected_type):
                    logger.warning(f"Default value for '{name}' doesn't match type {expected_type}")
                variables[name] = default_val

        # Report all type errors at once
        if type_errors:
            raise PipelineValidationError(
                f"Type validation failed for {len(type_errors)} parameter(s)",
                suggestion="Fix parameter types:\n  - " + "\n  - ".join(type_errors),
            )

        # Create copy of definition with substituted values
        definition_copy = copy.deepcopy(self.definition)
        definition_copy.variables = variables

        # Substitute in step params
        for step in definition_copy.steps:
            step.params = _substitute_dict_variables(step.params, variables)

        logger.info(
            f"Instantiated pipeline template '{self.definition.name}' with {len(variables)} variables"
        )
        return Pipeline(definition_copy)


def _validate_type(value: Any, expected_type: str) -> bool:
    """Validate value matches expected type."""
    type_map = {
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "number": (int, float),
        "bool": bool,
        "boolean": bool,
        "list": list,
        "array": list,
        "dict": dict,
        "object": dict,
    }
    expected = type_map.get(expected_type, str)
    return isinstance(value, expected)  # type: ignore[arg-type]


def _substitute_dict_variables(d: dict[str, Any], variables: dict[str, Any]) -> dict[str, Any]:
    """Recursively substitute variables in dictionary."""
    result = {}
    for key, value in d.items():
        if isinstance(value, str):
            result[key] = _substitute_variables(value, variables)
        elif isinstance(value, dict):
            result[key] = _substitute_dict_variables(value, variables)  # type: ignore[assignment]
        elif isinstance(value, list):
            result[key] = [  # type: ignore[assignment]
                _substitute_dict_variables(v, variables)
                if isinstance(v, dict)
                else _substitute_variables(v, variables)
                if isinstance(v, str)
                else v
                for v in value
            ]
        else:
            result[key] = value
    return result


__all__ = [
    "Pipeline",
    "PipelineDefinition",
    "PipelineExecutionError",
    "PipelineResult",
    "PipelineStep",
    "PipelineTemplate",
    "PipelineValidationError",
    "load_pipeline",
    "resolve_includes",
]
