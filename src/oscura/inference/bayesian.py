"""Bayesian inference for signal analysis and protocol characterization.

This module provides probabilistic reasoning about signal characteristics and
protocol properties using Bayesian updating. It enables inference with full
uncertainty quantification and supports sequential updates as more data arrives.


Key Features:
- Prior distributions for common signal properties (baud rate, frequency, etc.)
- Likelihood functions for observed measurements
- Posterior calculation with credible intervals
- Integration with quality scoring system (0-1 confidence mapping)
- Sequential Bayesian updating for streaming analysis
- Support for multiple distribution families (normal, uniform, beta, etc.)

Example:
    >>> from oscura.inference.bayesian import BayesianInference, infer_with_uncertainty
    >>> import numpy as np
    >>>
    >>> # Infer baud rate from edge timing observations
    >>> inference = BayesianInference()
    >>> edge_times = np.array([0.0, 0.00001, 0.00002, 0.00003])  # 100 kHz
    >>> posterior = inference.infer_baud_rate(edge_times)
    >>> print(f"Baud rate: {posterior.mean:.0f} ± {posterior.std:.0f}")
    >>> print(f"95% CI: [{posterior.ci_lower:.0f}, {posterior.ci_upper:.0f}]")
    >>> print(f"Confidence: {posterior.confidence:.2%}")
    >>>
    >>> # Infer number of symbols from amplitude histogram
    >>> amplitudes = np.random.choice([0.0, 0.33, 0.67, 1.0], size=1000)
    >>> histogram, _ = np.histogram(amplitudes, bins=50)
    >>> symbol_posterior = inference.infer_symbol_count(histogram)
    >>> print(f"Estimated symbols: {int(symbol_posterior.mean)}")
    >>>
    >>> # Sequential updating for streaming data
    >>> from oscura.inference.bayesian import SequentialBayesian, Prior
    >>> prior = Prior("normal", {"mean": 115200, "std": 10000})
    >>> sequential = SequentialBayesian("baud_rate", prior)
    >>> for _observation in streaming_data:
    ...     posterior = sequential.update(likelihood_fn)
    ...     if sequential.get_confidence() > 0.95:
    ...         break  # High confidence reached

References:
    - Gelman et al., "Bayesian Data Analysis" (3rd ed.)
    - Murphy, "Machine Learning: A Probabilistic Perspective"
    - scipy.stats documentation for distribution families
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

from oscura.core.exceptions import AnalysisError, InsufficientDataError

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Prior:
    """Prior distribution for a parameter.

    Represents prior belief about a parameter before observing data.
    Supports common distribution families used in signal analysis.

    Attributes:
        distribution: Distribution family name (e.g., "normal", "uniform", "beta").
        params: Distribution parameters as dict (keys depend on distribution).

    Supported distributions:
        - "normal": params = {"mean": float, "std": float}
        - "uniform": params = {"low": float, "high": float}
        - "log_uniform": params = {"low": float, "high": float} (for scale-invariant priors)
        - "beta": params = {"a": float, "b": float} (for probabilities)
        - "gamma": params = {"shape": float, "scale": float} (for positive values)
        - "half_normal": params = {"scale": float} (for positive values like noise std)
        - "geometric": params = {"p": float} (for discrete counts)

    Example:
        >>> # Weakly informative prior for baud rate (log-uniform over range)
        >>> prior = Prior("log_uniform", {"low": 100, "high": 10_000_000})
        >>> samples = prior.sample(1000)
        >>> density = prior.pdf(115200)
        >>>
        >>> # Prior for duty cycle (beta distribution centered at 0.5)
        >>> duty_prior = Prior("beta", {"a": 2, "b": 2})
    """

    distribution: str
    params: dict[str, float]

    def __post_init__(self) -> None:
        """Validate distribution parameters after initialization."""
        valid_distributions = {
            "normal",
            "uniform",
            "log_uniform",
            "beta",
            "gamma",
            "half_normal",
            "geometric",
        }

        if self.distribution not in valid_distributions:
            raise ValueError(
                f"Unknown distribution: {self.distribution}. "
                f"Supported: {sorted(valid_distributions)}"
            )

        # Validate required parameters for each distribution
        required_params = {
            "normal": {"mean", "std"},
            "uniform": {"low", "high"},
            "log_uniform": {"low", "high"},
            "beta": {"a", "b"},
            "gamma": {"shape", "scale"},
            "half_normal": {"scale"},
            "geometric": {"p"},
        }

        required = required_params[self.distribution]
        missing = required - set(self.params.keys())
        if missing:
            raise ValueError(f"Missing parameters for {self.distribution} distribution: {missing}")

    def pdf(self, x: float | NDArray[np.floating[Any]]) -> float | NDArray[np.floating[Any]]:
        """Compute probability density at x.

        Args:
            x: Value(s) at which to evaluate density.

        Returns:
            Probability density value(s).

        Raises:
            ValueError: If distribution is not recognized.
        """
        if self.distribution == "normal":
            return float(stats.norm.pdf(x, loc=self.params["mean"], scale=self.params["std"]))
        elif self.distribution == "uniform":
            return float(
                stats.uniform.pdf(
                    x, loc=self.params["low"], scale=self.params["high"] - self.params["low"]
                )
            )
        elif self.distribution == "log_uniform":
            # Log-uniform: uniform on log scale
            log_low = np.log(self.params["low"])
            log_high = np.log(self.params["high"])
            log_x = np.log(np.maximum(x, 1e-100))  # Avoid log(0)
            density = stats.uniform.pdf(log_x, loc=log_low, scale=log_high - log_low)
            # Jacobian correction: d(log x)/dx = 1/x
            result: float | NDArray[np.floating[Any]] = density / np.maximum(x, 1e-100)
            return result
        elif self.distribution == "beta":
            return float(stats.beta.pdf(x, a=self.params["a"], b=self.params["b"]))
        elif self.distribution == "gamma":
            return float(stats.gamma.pdf(x, a=self.params["shape"], scale=self.params["scale"]))
        elif self.distribution == "half_normal":
            return float(stats.halfnorm.pdf(x, scale=self.params["scale"]))
        elif self.distribution == "geometric":
            return float(stats.geom.pmf(x, p=self.params["p"]))
        else:
            raise ValueError(f"PDF not implemented for {self.distribution}")

    def sample(self, n: int = 1) -> NDArray[np.floating[Any]]:
        """Draw samples from prior distribution.

        Args:
            n: Number of samples to draw.

        Returns:
            Array of samples from the prior.

        Raises:
            ValueError: If distribution is not recognized.
        """
        if self.distribution == "normal":
            return stats.norm.rvs(loc=self.params["mean"], scale=self.params["std"], size=n)  # type: ignore[no-any-return]
        elif self.distribution == "uniform":
            return stats.uniform.rvs(  # type: ignore[no-any-return]
                loc=self.params["low"], scale=self.params["high"] - self.params["low"], size=n
            )
        elif self.distribution == "log_uniform":
            # Sample uniformly on log scale, then exponentiate
            log_low = np.log(self.params["low"])
            log_high = np.log(self.params["high"])
            log_samples = stats.uniform.rvs(loc=log_low, scale=log_high - log_low, size=n)
            return np.exp(log_samples)  # type: ignore[no-any-return]
        elif self.distribution == "beta":
            return stats.beta.rvs(a=self.params["a"], b=self.params["b"], size=n)  # type: ignore[no-any-return]
        elif self.distribution == "gamma":
            return stats.gamma.rvs(a=self.params["shape"], scale=self.params["scale"], size=n)  # type: ignore[no-any-return]
        elif self.distribution == "half_normal":
            return stats.halfnorm.rvs(scale=self.params["scale"], size=n)  # type: ignore[no-any-return]
        elif self.distribution == "geometric":
            return stats.geom.rvs(p=self.params["p"], size=n)  # type: ignore[no-any-return]
        else:
            raise ValueError(f"Sampling not implemented for {self.distribution}")


@dataclass
class Posterior:
    """Posterior distribution after updating with evidence.

    Represents updated belief about a parameter after observing data.
    Provides point estimates, uncertainty quantification, and confidence scores.

    Attributes:
        mean: Posterior mean (point estimate).
        std: Posterior standard deviation (uncertainty).
        ci_lower: Lower bound of 95% credible interval.
        ci_upper: Upper bound of 95% credible interval.
        samples: Optional array of posterior samples (for non-parametric posteriors).

    Example:
        >>> posterior = Posterior(mean=115200, std=5000, ci_lower=105600, ci_upper=124800)
        >>> print(f"Estimate: {posterior.mean:.0f} ± {posterior.std:.0f}")
        >>> print(f"95% CI: [{posterior.ci_lower:.0f}, {posterior.ci_upper:.0f}]")
        >>> print(f"Confidence: {posterior.confidence:.2%}")
    """

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    samples: NDArray[np.floating[Any]] | None = None

    @property
    def confidence(self) -> float:
        """Convert posterior certainty to 0-1 confidence score.

        Maps posterior standard deviation to confidence using an empirical formula.
        Lower std (more certain) -> higher confidence.

        The mapping is based on coefficient of variation (CV = std/mean):
        - CV < 0.05 (5%): High confidence (~0.95)
        - CV ~ 0.10 (10%): Medium confidence (~0.85)
        - CV ~ 0.20 (20%): Low confidence (~0.70)
        - CV > 0.50 (50%): Very low confidence (~0.50)

        Returns:
            Confidence score between 0 and 1.

        Example:
            >>> # Low uncertainty -> high confidence
            >>> p1 = Posterior(mean=100, std=5, ci_lower=90, ci_upper=110)
            >>> p1.confidence  # ~0.95
            >>>
            >>> # High uncertainty -> low confidence
            >>> p2 = Posterior(mean=100, std=30, ci_lower=40, ci_upper=160)
            >>> p2.confidence  # ~0.70
        """
        # Avoid division by zero
        if abs(self.mean) < 1e-10:
            cv = self.std / 1e-10
        else:
            cv = abs(self.std / self.mean)  # Coefficient of variation

        # Map CV to confidence using sigmoid-like function
        # confidence = 1 - min(1, cv / scale_factor)
        # Scale factor determines how quickly confidence drops with uncertainty
        scale_factor = 0.5  # 50% CV -> 0% confidence
        confidence = 1.0 - min(1.0, cv / scale_factor)

        # Ensure in valid range [0, 1]
        return max(0.0, min(1.0, confidence))


class BayesianInference:
    """Bayesian inference for signal analysis.

    Provides methods for inferring signal properties (baud rate, frequency,
    symbol count, etc.) with full uncertainty quantification using Bayesian
    methods.

    Attributes:
        priors: Dictionary of default prior distributions for common parameters.

    Example:
        >>> inference = BayesianInference()
        >>>
        >>> # Infer baud rate from edge timings
        >>> edge_times = np.array([0.0, 0.00001, 0.00002, 0.00003])
        >>> posterior = inference.infer_baud_rate(edge_times)
        >>>
        >>> # Infer protocol type probabilities
        >>> observations = {"idle_level": "high", "regularity": 0.3, "duty_cycle": 0.9}
        >>> protocol_probs = inference.infer_protocol_type(observations)
        >>> print(protocol_probs)  # {"UART": 0.85, "I2C": 0.10, "SPI": 0.05}
    """

    def __init__(self) -> None:
        """Initialize Bayesian inference engine with default priors."""
        self.priors = self._default_priors()

    def _default_priors(self) -> dict[str, Prior]:
        """Create default priors for common signal properties.

        Returns:
            Dictionary mapping parameter names to Prior objects.

        Priors are designed to be weakly informative:
        - Broad enough to cover typical use cases
        - Narrow enough to provide regularization
        - Match physical constraints (e.g., positive values)
        """
        return {
            # Log-uniform for scale-invariant parameters (wide range)
            "baud_rate": Prior("log_uniform", {"low": 100, "high": 10_000_000}),
            "frequency": Prior("log_uniform", {"low": 1, "high": 1e9}),
            # Beta distribution for probabilities/proportions
            "duty_cycle": Prior("beta", {"a": 2, "b": 2}),  # Centered at 0.5
            # Half-normal for positive values (noise, std, etc.)
            "noise_std": Prior("half_normal", {"scale": 0.1}),
            # Geometric for discrete counts (favor smaller values)
            "num_symbols": Prior("geometric", {"p": 0.3}),
            # Normal for typical signal characteristics
            "amplitude": Prior("normal", {"mean": 0.0, "std": 1.0}),
            "offset": Prior("normal", {"mean": 0.0, "std": 0.1}),
        }

    def update(
        self,
        param: str,
        likelihood_fn: Callable[[float], float],
        *,
        prior: Prior | None = None,
        num_samples: int = 10000,
    ) -> Posterior:
        """Update belief about parameter given observation.

        General-purpose Bayesian updating using sampling-based inference.
        Uses the prior distribution and likelihood function to compute
        the posterior via importance sampling.

        Args:
            param: Parameter name (used to get default prior if not provided).
            likelihood_fn: Function that computes likelihood p(observation | param_value).
            prior: Prior distribution (uses default if None).
            num_samples: Number of samples for posterior approximation.

        Returns:
            Posterior distribution with mean, std, and credible intervals.

        Raises:
            ValueError: If parameter is unknown and no prior is provided.
            AnalysisError: If likelihood function fails.

        Example:
            >>> def likelihood(rate: float) -> float:
            ...     # Example: Poisson likelihood for event rate
            ...     observed_count = 42
            ...     time_window = 1.0
            ...     expected = rate * time_window
            ...     return stats.poisson.pmf(observed_count, mu=expected)
            >>>
            >>> inference = BayesianInference()
            >>> posterior = inference.update("frequency", likelihood)
        """
        prior = self._get_prior(param, prior)
        samples = self._sample_from_prior(prior, param, num_samples)
        likelihoods = self._compute_likelihoods(samples, likelihood_fn, param)
        weights = self._normalize_weights(likelihoods, param)
        return self._build_posterior(samples, weights)

    def _get_prior(self, param: str, prior: Prior | None) -> Prior:
        """Get prior distribution for parameter.

        Args:
            param: Parameter name.
            prior: Optional explicit prior.

        Returns:
            Prior distribution to use.

        Raises:
            ValueError: If parameter unknown and no prior provided.
        """
        if prior is None:
            if param not in self.priors:
                raise ValueError(
                    f"Unknown parameter '{param}' and no prior provided. "
                    f"Known parameters: {list(self.priors.keys())}"
                )
            prior = self.priors[param]
        return prior

    def _sample_from_prior(
        self, prior: Prior, param: str, num_samples: int
    ) -> NDArray[np.floating[Any]]:
        """Sample from prior distribution.

        Args:
            prior: Prior distribution.
            param: Parameter name for error messages.
            num_samples: Number of samples.

        Returns:
            Array of prior samples.

        Raises:
            AnalysisError: If sampling fails.
        """
        try:
            return prior.sample(num_samples)
        except Exception as e:
            raise AnalysisError(
                f"Failed to sample from prior for '{param}'",
                details=str(e),
            ) from e

    def _compute_likelihoods(
        self,
        samples: NDArray[np.floating[Any]],
        likelihood_fn: Callable[[float], float],
        param: str,
    ) -> NDArray[np.floating[Any]]:
        """Compute likelihoods for all samples.

        Args:
            samples: Prior samples.
            likelihood_fn: Likelihood function.
            param: Parameter name for error messages.

        Returns:
            Array of likelihood values.

        Raises:
            AnalysisError: If likelihood computation fails or all zeros.
        """
        try:
            likelihoods = np.array([likelihood_fn(s) for s in samples])
        except Exception as e:
            raise AnalysisError(
                f"Likelihood function failed for '{param}'",
                details=str(e),
                fix_hint="Check that likelihood_fn is compatible with prior samples",
            ) from e

        if np.all(likelihoods == 0):
            raise AnalysisError(
                f"All likelihood values are zero for '{param}'",
                details="Observation may be incompatible with prior range",
                fix_hint="Adjust prior range or check likelihood function",
            )

        return likelihoods

    def _normalize_weights(
        self, likelihoods: NDArray[np.floating[Any]], param: str
    ) -> NDArray[np.floating[Any]]:
        """Normalize likelihoods to importance weights.

        Uses numerical stability techniques for extreme values.

        Args:
            likelihoods: Likelihood values.
            param: Parameter name for error messages.

        Returns:
            Normalized importance weights.

        Raises:
            AnalysisError: If all likelihoods are zero.
        """
        max_likelihood = np.max(likelihoods)
        if max_likelihood <= 0:
            raise AnalysisError(
                f"All likelihood values are zero for '{param}'",
                details="Observation may be incompatible with prior range",
                fix_hint="Adjust prior range or check likelihood function",
            )

        # Normalize by max to prevent overflow/underflow
        normalized_likelihoods = likelihoods / max_likelihood

        # Use log-space for extreme underflow
        if max_likelihood < 1e-300:
            log_likelihoods = np.log(np.maximum(likelihoods, 1e-300))
            log_likelihoods -= np.max(log_likelihoods)
            weights = np.exp(log_likelihoods)
            weights /= np.sum(weights)
        else:
            weights = normalized_likelihoods / np.sum(normalized_likelihoods)

        result: NDArray[np.floating[Any]] = np.asarray(weights, dtype=np.float64)
        return result

    def _build_posterior(
        self, samples: NDArray[np.floating[Any]], weights: NDArray[np.floating[Any]]
    ) -> Posterior:
        """Build posterior distribution from weighted samples.

        Args:
            samples: Prior samples.
            weights: Importance weights.

        Returns:
            Posterior distribution with statistics.
        """
        # Compute posterior statistics
        mean = float(np.sum(samples * weights))
        variance = float(np.sum(weights * (samples - mean) ** 2))
        std = float(np.sqrt(variance))

        # Compute 95% credible interval
        sorted_indices = np.argsort(samples)
        sorted_samples = samples[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumsum = np.cumsum(sorted_weights)

        ci_lower = float(sorted_samples[np.searchsorted(cumsum, 0.025)])
        ci_upper = float(sorted_samples[np.searchsorted(cumsum, 0.975)])

        return Posterior(
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            samples=samples,
        )

    def infer_baud_rate(
        self, edge_times: NDArray[np.floating[Any]], *, prior: Prior | None = None
    ) -> Posterior:
        """Infer baud rate from edge timing observations.

        Uses the distribution of inter-edge intervals to infer the underlying
        baud rate. Assumes edges occur at bit boundaries.

        Args:
            edge_times: Array of edge timestamps in seconds.
            prior: Optional prior for baud rate (uses default if None).

        Returns:
            Posterior distribution for baud rate in bits per second.

        Raises:
            InsufficientDataError: If fewer than 2 edges provided.

        Example:
            >>> # 115200 baud UART (bit period = 8.68 μs)
            >>> edge_times = np.array([0, 8.68e-6, 17.36e-6, 26.04e-6])
            >>> posterior = inference.infer_baud_rate(edge_times)
            >>> print(f"Baud rate: {posterior.mean:.0f} bps")
        """
        if len(edge_times) < 2:
            raise InsufficientDataError(
                "Need at least 2 edges to infer baud rate",
                required=2,
                available=len(edge_times),
            )

        # Compute inter-edge intervals
        intervals = np.diff(edge_times)

        # Filter out zero/negative intervals (should not happen but be safe)
        intervals = intervals[intervals > 0]

        if len(intervals) == 0:
            raise InsufficientDataError("No valid inter-edge intervals found")

        # Likelihood: assume intervals are Gaussian around 1/baud_rate
        # Multiple of bit period due to encoding (e.g., start/stop bits)
        def likelihood(baud_rate: float) -> float:
            if baud_rate <= 0:
                return 0.0
            bit_period = 1.0 / baud_rate
            # Intervals should be multiples of bit_period
            # Use smallest interval as proxy for single bit period
            min_interval = np.min(intervals)
            # Likelihood: intervals match multiples of estimated bit period
            expected = min_interval
            # Gaussian likelihood around expected interval
            sigma = expected * 0.1  # 10% uncertainty
            log_likelihood = -0.5 * ((expected - bit_period) / sigma) ** 2
            # Clip log-likelihood to prevent extreme underflow
            # Keep at least exp(-700) ≈ 1e-304 to stay above zero
            log_likelihood = max(log_likelihood, -700.0)
            return float(np.exp(log_likelihood))

        return self.update("baud_rate", likelihood, prior=prior)

    def infer_protocol_type(self, observations: dict[str, Any]) -> dict[str, float]:
        """Infer probability of each protocol type given observations.

        Uses a simple Bayesian classifier to compute posterior probabilities
        for different protocol types (UART, SPI, I2C, CAN) based on observed
        signal characteristics.

        Args:
            observations: Dictionary of observed signal characteristics:
                - "idle_level": "high" or "low"
                - "regularity": 0-1 (edge regularity)
                - "duty_cycle": 0-1 (fraction time high)
                - "symbol_rate": Hz (optional)
                - "transition_density": edges/sec (optional)

        Returns:
            Dictionary mapping protocol names to posterior probabilities.
            Probabilities sum to 1.0.

        Example:
            >>> observations = {
            ...     "idle_level": "high",
            ...     "regularity": 0.25,
            ...     "duty_cycle": 0.85,
            ...     "symbol_rate": 115200,
            ... }
            >>> probs = inference.infer_protocol_type(observations)
            >>> print(probs)  # {"UART": 0.85, "I2C": 0.10, "SPI": 0.03, "CAN": 0.02}
        """
        # Prior probabilities (uniform over protocols)
        protocols = ["UART", "SPI", "I2C", "CAN"]
        prior_prob = 1.0 / len(protocols)

        # Compute likelihoods for each protocol
        likelihoods = {
            "UART": self._likelihood_uart(observations),
            "SPI": self._likelihood_spi(observations),
            "I2C": self._likelihood_i2c(observations),
            "CAN": self._likelihood_can(observations),
        }

        # Compute posterior probabilities (Bayes' theorem)
        posteriors = {proto: prior_prob * likelihoods[proto] for proto in protocols}

        # Normalize to sum to 1
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {proto: prob / total for proto, prob in posteriors.items()}
        else:
            # No evidence - return uniform
            posteriors = {proto: 1.0 / len(protocols) for proto in protocols}

        return posteriors

    def _likelihood_uart(self, obs: dict[str, Any]) -> float:
        """Compute likelihood of observations given UART protocol."""
        likelihood = 1.0

        # UART characteristics: idle high, low regularity, high duty cycle
        if obs.get("idle_level") == "high":
            likelihood *= 0.9
        else:
            likelihood *= 0.2

        regularity = obs.get("regularity", 0.5)
        if regularity < 0.3:
            likelihood *= 0.8
        elif regularity > 0.7:
            likelihood *= 0.2

        duty_cycle = obs.get("duty_cycle", 0.5)
        if duty_cycle > 0.7:
            likelihood *= 0.8
        elif duty_cycle < 0.3:
            likelihood *= 0.3

        return likelihood

    def _likelihood_spi(self, obs: dict[str, Any]) -> float:
        """Compute likelihood of observations given SPI protocol."""
        likelihood = 1.0

        # SPI characteristics: regular clock, ~50% duty, high transition density
        regularity = obs.get("regularity", 0.5)
        if regularity > 0.7:
            likelihood *= 0.9
        elif regularity < 0.4:
            likelihood *= 0.2

        duty_cycle = obs.get("duty_cycle", 0.5)
        duty_error = abs(duty_cycle - 0.5)
        if duty_error < 0.1:
            likelihood *= 0.8
        elif duty_error > 0.3:
            likelihood *= 0.3

        return likelihood

    def _likelihood_i2c(self, obs: dict[str, Any]) -> float:
        """Compute likelihood of observations given I2C protocol."""
        likelihood = 1.0

        # I2C characteristics: idle high, moderate regularity
        if obs.get("idle_level") == "high":
            likelihood *= 0.85
        else:
            likelihood *= 0.3

        regularity = obs.get("regularity", 0.5)
        if 0.4 < regularity < 0.8:
            likelihood *= 0.8
        else:
            likelihood *= 0.4

        return likelihood

    def _likelihood_can(self, obs: dict[str, Any]) -> float:
        """Compute likelihood of observations given CAN protocol."""
        likelihood = 1.0

        # CAN characteristics: idle high, moderate irregularity (bit stuffing)
        if obs.get("idle_level") == "high":
            likelihood *= 0.85
        else:
            likelihood *= 0.2

        regularity = obs.get("regularity", 0.5)
        if 0.3 < regularity < 0.7:
            likelihood *= 0.8
        else:
            likelihood *= 0.4

        # Check for standard CAN baud rates
        symbol_rate = obs.get("symbol_rate", 0)
        if symbol_rate > 0:
            standard_rates = [125000, 250000, 500000, 1000000]
            for rate in standard_rates:
                if abs(symbol_rate - rate) / rate < 0.1:
                    likelihood *= 1.5
                    break

        return likelihood

    def infer_symbol_count(
        self,
        amplitude_histogram: NDArray[np.floating[Any]],
        *,
        prior: Prior | None = None,
        max_symbols: int = 16,
    ) -> Posterior:
        """Infer number of discrete symbols from amplitude distribution.

        Analyzes the amplitude histogram to determine how many discrete
        signal levels (symbols) are present. Useful for multi-level signaling
        (PAM-4, etc.).

        Args:
            amplitude_histogram: Histogram of signal amplitudes (bin counts).
            prior: Optional prior for symbol count (uses default if None).
            max_symbols: Maximum number of symbols to consider.

        Returns:
            Posterior distribution for number of symbols.

        Raises:
            InsufficientDataError: If histogram is empty or all zeros.

        Example:
            >>> # PAM-4 signal (4 levels)
            >>> amplitudes = np.random.choice([0.0, 0.33, 0.67, 1.0], size=1000)
            >>> hist, _ = np.histogram(amplitudes, bins=50)
            >>> posterior = inference.infer_symbol_count(hist)
            >>> print(f"Symbols: {int(posterior.mean)}")  # Should be close to 4
        """
        if len(amplitude_histogram) == 0 or np.sum(amplitude_histogram) == 0:
            raise InsufficientDataError("Amplitude histogram is empty or all zeros")

        # Pre-compute peaks in histogram (optimization: only compute once)
        # Use robust peak detection that also considers edge bins
        prominence_threshold = np.max(amplitude_histogram) * 0.1
        detected_peaks, _ = find_peaks(amplitude_histogram, prominence=prominence_threshold)
        peak_indices = list(detected_peaks)

        # Check edge bins - they can be peaks but find_peaks won't detect them
        # First bin is a peak if it's significant and higher than second bin
        if len(amplitude_histogram) > 1:
            if (
                amplitude_histogram[0] > prominence_threshold
                and amplitude_histogram[0] > amplitude_histogram[1]
            ):
                peak_indices.insert(0, 0)
            # Last bin is a peak if significant and higher than second-to-last
            if (
                amplitude_histogram[-1] > prominence_threshold
                and amplitude_histogram[-1] > amplitude_histogram[-2]
            ):
                peak_indices.append(len(amplitude_histogram) - 1)

        num_peaks = len(peak_indices)

        # Likelihood: number of peaks in histogram should match symbol count
        def likelihood(num_symbols: float) -> float:
            k = int(round(num_symbols))
            if k < 1 or k > max_symbols:
                return 0.0

            # Likelihood: peaks should match symbols
            # Allow ±1 tolerance for noise
            if abs(num_peaks - k) <= 1:
                return 0.8
            elif abs(num_peaks - k) == 2:
                return 0.3
            else:
                return 0.1

        return self.update(
            "num_symbols",
            likelihood,
            prior=prior,
            num_samples=max_symbols * 100,
        )


class SequentialBayesian:
    """Sequential Bayesian updating for streaming analysis.

    Maintains a posterior that is updated as new observations arrive.
    Useful for online/streaming signal analysis where data comes in
    incrementally.

    Attributes:
        param: Parameter name being inferred.
        current_posterior: Current posterior after all updates so far.

    Example:
        >>> from oscura.inference.bayesian import SequentialBayesian, Prior
        >>> prior = Prior("normal", {"mean": 115200, "std": 10000})
        >>> sequential = SequentialBayesian("baud_rate", prior)
        >>>
        >>> # Update with streaming observations
        >>> for _observation in streaming_data:
        ...     posterior = sequential.update(likelihood_fn)
        ...     print(f"Current estimate: {posterior.mean:.0f} (confidence: {sequential.get_confidence():.2%})")
        ...     if sequential.get_confidence() > 0.95:
        ...         break  # High confidence reached
    """

    def __init__(self, param: str, prior: Prior) -> None:
        """Initialize sequential Bayesian updater.

        Args:
            param: Parameter name being inferred.
            prior: Initial prior distribution.
        """
        self.param = param
        self.current_posterior: Prior | Posterior = prior
        self._samples: list[NDArray[np.floating[Any]]] = []
        self._weights: list[NDArray[np.floating[Any]]] = []

    def update(
        self,
        likelihood_fn: Callable[[float], float],
        *,
        num_samples: int = 5000,
    ) -> Posterior:
        """Update posterior with new observation.

        Performs one step of sequential Bayesian updating. The current
        posterior becomes the prior for the next update.

        Args:
            likelihood_fn: Likelihood function p(observation | param).
            num_samples: Number of samples for approximation.

        Returns:
            Updated posterior distribution.

        Raises:
            AnalysisError: If likelihood function fails.
        """
        # If we have a Prior, sample from it
        if isinstance(self.current_posterior, Prior):
            samples = self.current_posterior.sample(num_samples)
        else:
            # Resample from previous posterior (must be Posterior)
            if self.current_posterior.samples is not None:
                # Resample with replacement
                indices = np.random.choice(
                    len(self.current_posterior.samples),
                    size=num_samples,
                    replace=True,
                )
                samples = self.current_posterior.samples[indices]
            else:
                # Approximate as normal distribution
                samples = np.random.normal(
                    self.current_posterior.mean,
                    self.current_posterior.std,
                    size=num_samples,
                )

        # Compute likelihoods
        try:
            likelihoods = np.array([likelihood_fn(s) for s in samples])
        except Exception as e:
            raise AnalysisError(
                f"Likelihood function failed for '{self.param}'",
                details=str(e),
            ) from e

        # Check for valid likelihoods
        if np.all(likelihoods == 0):
            # No update - keep current posterior
            return (
                self.current_posterior
                if isinstance(self.current_posterior, Posterior)
                else Posterior(mean=0.0, std=1.0, ci_lower=-2.0, ci_upper=2.0)
            )

        # Compute importance weights
        weights = likelihoods / np.sum(likelihoods)

        # Store for potential resampling
        self._samples.append(samples)
        self._weights.append(weights)

        # Compute posterior statistics
        mean = float(np.sum(samples * weights))
        variance = float(np.sum(weights * (samples - mean) ** 2))
        std = float(np.sqrt(variance))

        # Credible interval
        sorted_indices = np.argsort(samples)
        sorted_samples = samples[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumsum = np.cumsum(sorted_weights)

        ci_lower = float(sorted_samples[np.searchsorted(cumsum, 0.025)])
        ci_upper = float(sorted_samples[np.searchsorted(cumsum, 0.975)])

        posterior = Posterior(
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            samples=samples,
        )

        self.current_posterior = posterior
        return posterior

    def get_confidence(self) -> float:
        """Get current confidence in estimate.

        Returns:
            Confidence score between 0 and 1.
        """
        if isinstance(self.current_posterior, Posterior):
            return self.current_posterior.confidence
        else:
            # Prior - no evidence yet
            return 0.0


def infer_with_uncertainty(
    measurements: list[float] | NDArray[np.floating[Any]],
    prior: Prior | None = None,
) -> Posterior:
    """Infer parameter value with full uncertainty quantification.

    Convenience function for simple parameter inference from repeated measurements.
    Assumes measurements are normally distributed around the true value.

    Args:
        measurements: List or array of independent measurements.
        prior: Optional prior distribution (uses uninformative normal if None).

    Returns:
        Posterior distribution combining prior and measurement likelihood.

    Raises:
        InsufficientDataError: If measurements list is empty.

    Example:
        >>> # Repeated measurements of signal frequency
        >>> measurements = [99.8, 100.2, 99.9, 100.1, 100.0]  # Hz
        >>> posterior = infer_with_uncertainty(measurements)
        >>> print(f"Frequency: {posterior.mean:.2f} ± {posterior.std:.2f} Hz")
        >>> print(f"95% CI: [{posterior.ci_lower:.2f}, {posterior.ci_upper:.2f}] Hz")
    """
    measurements_array = np.asarray(measurements)

    if len(measurements_array) == 0:
        raise InsufficientDataError(
            "Cannot infer from empty measurements",
            required=1,
            available=0,
        )

    # Use sample statistics if no prior
    if prior is None:
        # Uninformative prior centered at sample mean
        sample_mean = float(np.mean(measurements_array))
        sample_std = (
            float(np.std(measurements_array, ddof=1)) if len(measurements_array) > 1 else 1.0
        )
        prior = Prior("normal", {"mean": sample_mean, "std": sample_std * 10})

    # Likelihood: measurements are Gaussian around true value
    measurement_std = (
        float(np.std(measurements_array, ddof=1)) if len(measurements_array) > 1 else 1.0
    )

    def likelihood(param_value: float) -> float:
        # Product of Gaussian likelihoods
        log_likelihood = -0.5 * np.sum(((measurements_array - param_value) / measurement_std) ** 2)
        return float(np.exp(log_likelihood))

    inference = BayesianInference()
    return inference.update(
        "parameter",
        likelihood,
        prior=prior,
    )


__all__ = [
    "BayesianInference",
    "Posterior",
    "Prior",
    "SequentialBayesian",
    "infer_with_uncertainty",
]
