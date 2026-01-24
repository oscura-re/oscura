"""Common constants used across demonstrations for validation and calculations.

This module defines standard tolerances, thresholds, and mathematical constants
used consistently across all 112 demonstrations to ensure SSOT (Single Source of Truth).
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# Validation Tolerances
# =============================================================================

# Standard tolerance levels for measurement validation
TOLERANCE_STRICT = 0.01  # 1% - for precise measurements (clocks, references)
TOLERANCE_NORMAL = 0.05  # 5% - for typical measurements (amplitude, frequency)
TOLERANCE_RELAXED = 0.10  # 10% - for noisy/derived measurements (jitter, harmonics)

# =============================================================================
# Numerical Precision
# =============================================================================

# Float comparison thresholds
FLOAT_EPSILON = 1e-14  # Floating-point comparison threshold (near-zero values)
FLOAT_TOLERANCE = 1e-6  # Relative tolerance for float comparisons

# =============================================================================
# Mathematical Constants
# =============================================================================

# Signal processing constants
SINE_RMS_FACTOR = 1 / np.sqrt(2)  # 0.707... (sine wave peak to RMS conversion)
SQRT2 = np.sqrt(2)  # 1.414... (RMS to peak conversion)

# =============================================================================
# Random Seed
# =============================================================================

# Standard seed for deterministic test data generation
RANDOM_SEED = 42  # Used across all demonstrations for reproducibility
