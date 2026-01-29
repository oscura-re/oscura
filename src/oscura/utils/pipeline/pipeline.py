"""Pipeline architecture for chaining trace transformations.

This module implements sklearn-style pipeline composition for trace operations,
enabling declarative, reusable analysis workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import TraceTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from oscura.core.types import WaveformTrace


class Pipeline(TraceTransformer):
    """Chain multiple trace transformers into a single processing pipeline.

    Pipeline applies transformers sequentially: each stage transforms the output
    of the previous stage. Supports the fit/transform pattern and can be
    serialized with pickle or joblib.

    The pipeline is itself a TraceTransformer, so pipelines can be nested.

    Attributes:
        steps: List of (name, transformer) tuples defining the pipeline stages.
        named_steps: Dictionary mapping step names to transformers.

    Example:
        >>> import oscura as osc
        >>> pipeline = osc.Pipeline([
        ...     ('lowpass', osc.LowPassFilter(cutoff=1e6)),
        ...     ('resample', osc.Resample(rate=1e9)),
        ...     ('normalize', osc.Normalize())
        ... ])
        >>> result = pipeline.transform(trace)

    Advanced Example:
        >>> # Create analysis pipeline with fit/transform
        >>> pipeline = osc.Pipeline([
        ...     ('filter', osc.BandPassFilter(low=1e5, high=1e6)),
        ...     ('normalize', osc.Normalize(method='zscore')),
        ...     ('fft', osc.FFT(nfft=8192, window='hann')),
        ...     ('extract', osc.ExtractMeasurement('thd'))
        ... ])
        >>> # Fit on reference trace
        >>> pipeline.fit(reference_trace)
        >>> # Transform multiple traces
        >>> results = [pipeline.transform(t) for t in traces]
        >>> # Access intermediate results
        >>> filtered = pipeline.named_steps['filter'].transform(trace)
        >>> # Save for reuse
        >>> import joblib
        >>> joblib.dump(pipeline, 'analysis_pipeline.pkl')

    References:
        API-001: sklearn-style Pipeline Architecture
        sklearn.pipeline.Pipeline
        https://scikit-learn.org/stable/modules/compose.html
    """

    def __init__(self, steps: Sequence[tuple[str, TraceTransformer]]) -> None:
        """Initialize pipeline with sequence of transformers.

        Args:
            steps: Sequence of (name, transformer) tuples. Each transformer
                must be a TraceTransformer instance.

        Raises:
            TypeError: If any step is not a TraceTransformer.
            ValueError: If step names are not unique or empty.
        """
        if not steps:
            raise ValueError("Pipeline steps cannot be empty")

        # Validate steps
        names = []
        for name, transformer in steps:
            if not name:
                raise ValueError("Step name cannot be empty")
            if not isinstance(transformer, TraceTransformer):
                raise TypeError(
                    f"All pipeline steps must be TraceTransformer instances. "
                    f"Step '{name}' is {type(transformer).__name__}"
                )
            names.append(name)

        # Check for duplicate names
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate step names: {set(duplicates)}")

        self.steps = list(steps)
        self.named_steps = dict(steps)
        self._intermediate_results: dict[str, WaveformTrace] = {}

    def fit(self, trace: WaveformTrace) -> Pipeline:
        """Fit all transformers in the pipeline.

        Fits each transformer sequentially on the output of the previous stage.
        This allows stateful transformers to learn parameters from the trace.

        Args:
            trace: Reference WaveformTrace to fit to.

        Returns:
            Self for method chaining.

        Example:
            >>> pipeline = Pipeline([
            ...     ('normalize', AdaptiveNormalizer()),
            ...     ('filter', AdaptiveFilter())
            ... ])
            >>> pipeline.fit(reference_trace)
        """
        current = trace
        for _name, transformer in self.steps:
            # Fit transformer to current trace
            transformer.fit(current)
            # Transform for next stage
            current = transformer.transform(current)
        return self

    def transform(self, trace: WaveformTrace) -> WaveformTrace:
        """Transform trace through all pipeline stages.

        Applies each transformer sequentially, passing the output of each
        stage to the next. Optionally caches intermediate results.

        Args:
            trace: Input WaveformTrace to transform.

        Returns:
            Transformed WaveformTrace after passing through all stages.

        Example:
            >>> result = pipeline.transform(trace)
        """
        current = trace
        self._intermediate_results.clear()

        for name, transformer in self.steps:
            current = transformer.transform(current)
            # Cache intermediate result for introspection
            self._intermediate_results[name] = current

        return current

    def get_intermediate(self, step_name: str, key: str | None = None) -> Any:
        """Get intermediate result from a pipeline stage.

        Retrieves the cached output from a specific pipeline stage after
        transform() has been called. Can also access internal intermediate
        results from transformers that cache them (e.g., FFT coefficients).


        Args:
            step_name: Name of the pipeline step.
            key: Optional key for transformer-internal intermediate result.
                If None, returns the trace output from that stage.

        Returns:
            WaveformTrace output from that stage (if key=None), or
            specific intermediate result from the transformer.

        Raises:
            KeyError: If step name not found or transform() not yet called.

        Example:
            >>> pipeline = Pipeline([
            ...     ('filter', LowPassFilter(1e6)),
            ...     ('fft', FFT(nfft=8192)),
            ...     ('normalize', Normalize())
            ... ])
            >>> result = pipeline.transform(trace)
            >>> # Get trace output from filter stage
            >>> filtered = pipeline.get_intermediate('filter')
            >>> # Get FFT coefficients from FFT stage
            >>> fft_spectrum = pipeline.get_intermediate('fft', 'spectrum')
            >>> fft_frequencies = pipeline.get_intermediate('fft', 'frequencies')

        References:
            API-005: Intermediate Result Access
        """
        if step_name not in self._intermediate_results:
            if step_name not in self.named_steps:
                raise KeyError(f"Step '{step_name}' not found in pipeline")
            raise KeyError(
                f"No intermediate result for step '{step_name}'. Call transform() first."
            )

        # If no key specified, return the trace output from that stage
        if key is None:
            return self._intermediate_results[step_name]

        # Otherwise, try to get internal intermediate from the transformer
        transformer = self.named_steps[step_name]
        return transformer.get_intermediate_result(key)

    def has_intermediate(self, step_name: str, key: str | None = None) -> bool:
        """Check if intermediate result is available.

        Args:
            step_name: Name of the pipeline step.
            key: Optional key for transformer-internal intermediate result.

        Returns:
            True if intermediate result exists.

        Example:
            >>> if pipeline.has_intermediate('fft', 'spectrum'):
            ...     spectrum = pipeline.get_intermediate('fft', 'spectrum')

        References:
            API-005: Intermediate Result Access
        """
        if step_name not in self._intermediate_results:
            return False

        if key is None:
            return True

        transformer = self.named_steps[step_name]
        return transformer.has_intermediate_result(key)

    def list_intermediates(self, step_name: str | None = None) -> list[str] | dict[str, list[str]]:
        """List available intermediate results.

        Args:
            step_name: If specified, list intermediates for that step only.
                If None, return dict of all steps with their intermediates.

        Returns:
            List of intermediate keys for a step, or dict mapping step names
            to their available intermediates.

        Raises:
            KeyError: If step_name not found in pipeline.

        Example:
            >>> # List all intermediates
            >>> all_intermediates = pipeline.list_intermediates()
            >>> print(all_intermediates)
            {'filter': ['transfer_function', 'impulse_response'],
             'fft': ['spectrum', 'frequencies', 'power', 'phase']}
            >>> # List intermediates for specific step
            >>> fft_intermediates = pipeline.list_intermediates('fft')
            >>> print(fft_intermediates)
            ['spectrum', 'frequencies', 'power', 'phase']

        References:
            API-005: Intermediate Result Access
        """
        if step_name is not None:
            if step_name not in self.named_steps:
                raise KeyError(f"Step '{step_name}' not found in pipeline")
            transformer = self.named_steps[step_name]
            return transformer.list_intermediate_results()

        # Return all intermediates for all steps
        result = {}
        for name, transformer in self.steps:
            intermediates = transformer.list_intermediate_results()
            if intermediates:  # Only include steps with intermediates
                result[name] = intermediates
        return result

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for all transformers in the pipeline.

        Args:
            deep: If True, returns parameters for all nested transformers.

        Returns:
            Dictionary of parameters with step names as prefixes.

        Example:
            >>> params = pipeline.get_params()
            >>> print(params['filter__cutoff'])
            1000000.0
        """
        params: dict[str, Any] = {"steps": self.steps}

        if deep:
            for name, transformer in self.steps:
                transformer_params = transformer.get_params(deep=True)
                for key, value in transformer_params.items():
                    params[f"{name}__{key}"] = value

        return params

    def set_params(self, **params: Any) -> Pipeline:
        """Set parameters for transformers in the pipeline.

        Args:
            **params: Parameters to set, using step__param syntax.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If parameter format is invalid.

        Example:
            >>> pipeline.set_params(filter__cutoff=2e6, normalize__method='peak')
        """
        # Special case: setting steps directly
        if "steps" in params:
            self.steps = params["steps"]
            self.named_steps = dict(self.steps)
            return self

        # Parse step__param syntax
        for param_name, value in params.items():
            if "__" not in param_name:
                raise ValueError(
                    f"Pipeline parameter must use 'step__param' syntax, got '{param_name}'"
                )

            step_name, param = param_name.split("__", 1)
            if step_name not in self.named_steps:
                raise ValueError(
                    f"Step '{step_name}' not found in pipeline. "
                    f"Available steps: {list(self.named_steps.keys())}"
                )

            self.named_steps[step_name].set_params(**{param: value})

        return self

    def clone(self) -> Pipeline:
        """Create a copy of this pipeline.

        Returns:
            New Pipeline instance with cloned transformers.

        Example:
            >>> pipeline_copy = pipeline.clone()
        """
        cloned_steps = [(name, transformer.clone()) for name, transformer in self.steps]
        return Pipeline(cloned_steps)

    def __len__(self) -> int:
        """Return number of steps in the pipeline."""
        return len(self.steps)

    def __getitem__(self, index: int | str) -> TraceTransformer:
        """Get transformer by index or name.

        Args:
            index: Integer index or string name.

        Returns:
            TraceTransformer at that position.

        Example:
            >>> first_step = pipeline[0]
            >>> filter_step = pipeline['filter']
        """
        if isinstance(index, str):
            return self.named_steps[index]
        return self.steps[index][1]

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        step_strs = [
            f"('{name}', {transformer.__class__.__name__})" for name, transformer in self.steps
        ]
        return "Pipeline([\n  " + ",\n  ".join(step_strs) + "\n])"


__all__ = ["Pipeline"]
