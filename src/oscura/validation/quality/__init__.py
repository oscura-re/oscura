"""Signal quality analysis module for Oscura."""

from oscura.validation.quality.ensemble import (
    AMPLITUDE_ENSEMBLE,
    EDGE_DETECTION_ENSEMBLE,
    FREQUENCY_ENSEMBLE,
    AggregationMethod,
    EnsembleAggregator,
    EnsembleResult,
    create_edge_ensemble,
    create_frequency_ensemble,
)
from oscura.validation.quality.explainer import (
    ResultExplainer,
    ResultExplanation,
    explain_result,
)
from oscura.validation.quality.scoring import (
    AnalysisQualityScore,
    DataQualityMetrics,
    ReliabilityCategory,
    assess_data_quality,
    calculate_quality_score,
    combine_quality_scores,
    score_analysis_result,
)
from oscura.validation.quality.warnings import (
    QualityWarning,
    SignalQualityAnalyzer,
    check_clipping,
    check_noise,
    check_saturation,
    check_undersampling,
)

__all__ = [
    # Ensemble methods
    "AMPLITUDE_ENSEMBLE",
    "EDGE_DETECTION_ENSEMBLE",
    "FREQUENCY_ENSEMBLE",
    "AggregationMethod",
    # Scoring
    "AnalysisQualityScore",
    "DataQualityMetrics",
    "EnsembleAggregator",
    "EnsembleResult",
    # Warnings
    "QualityWarning",
    "ReliabilityCategory",
    "ResultExplainer",
    # Explainability
    "ResultExplanation",
    "SignalQualityAnalyzer",
    "assess_data_quality",
    "calculate_quality_score",
    "check_clipping",
    "check_noise",
    "check_saturation",
    "check_undersampling",
    "combine_quality_scores",
    "create_edge_ensemble",
    "create_frequency_ensemble",
    "explain_result",
    "score_analysis_result",
]
