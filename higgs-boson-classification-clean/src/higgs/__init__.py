"""
Higgs Boson Classification package.

A modular, reproducible implementation of dataset validation, physics-based
feature engineering, and classification modeling for the UCI HIGGS dataset.
"""

from higgs.data import (
    COLUMN_NAMES,
    LOW_LEVEL_FEATURES,
    HIGH_LEVEL_FEATURES,
    ALL_FEATURES,
    TARGET_COLUMN,
    load_higgs_sample,
    validate_schema,
    validate_target_labels,
    check_class_balance,
    check_missing_and_infinite_values,
    check_constant_features,
    check_duplicates,
    split_dataset,
)
from higgs.physics import (
    EXPECTED_DIMENSIONS,
    GEOMETRIC_FEATURE_NAMES,
    PHI_UNSCALING_FACTOR,
    STANDARDIZED_PHI_MAX,
    compute_delta_phi,
    compute_delta_R,
    compute_geometric_features,
    get_feature_groups,
)
from higgs.models import (
    HiggsMLPModule,
    LogisticRegressionModel,
    MultilayerPerceptronModel,
    XGBoostModel,
)
from higgs.evaluation import (
    calculate_metrics,
    compile_comparison_table,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
)

__all__ = [
    "COLUMN_NAMES",
    "LOW_LEVEL_FEATURES",
    "HIGH_LEVEL_FEATURES",
    "ALL_FEATURES",
    "TARGET_COLUMN",
    "load_higgs_sample",
    "validate_schema",
    "validate_target_labels",
    "check_class_balance",
    "check_missing_and_infinite_values",
    "check_constant_features",
    "check_duplicates",
    "split_dataset",
    "STANDARDIZED_PHI_MAX",
    "PHI_UNSCALING_FACTOR",
    "GEOMETRIC_FEATURE_NAMES",
    "EXPECTED_DIMENSIONS",
    "compute_delta_phi",
    "compute_delta_R",
    "compute_geometric_features",
    "get_feature_groups",
    "LogisticRegressionModel",
    "XGBoostModel",
    "HiggsMLPModule",
    "MultilayerPerceptronModel",
    "calculate_metrics",
    "evaluate_model",
    "compile_comparison_table",
    "plot_roc_curves",
    "plot_confusion_matrix",
]
