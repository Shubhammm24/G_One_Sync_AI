"""
G_One_Sync AI — Multi-Outcome Prediction
==========================================
Shared encoder + per-outcome classification heads for
simultaneous prediction of deterioration, sepsis,
cardiac arrest, and respiratory failure.
"""

from src.modeling.multi_outcome.label_definitions import (
    ALL_OUTCOMES,
    CARDIAC_ARREST_DEFINITION,
    DETERIORATION_DEFINITION,
    NUM_OUTCOMES,
    OUTCOME_COLUMNS,
    OUTCOME_REGISTRY,
    RESPIRATORY_FAILURE_DEFINITION,
    SEPSIS_DEFINITION,
    OutcomeDefinition,
    OutcomeType,
)
from src.modeling.multi_outcome.multi_head_model import (
    ClassificationHead,
    MultiHeadPredictionModel,
)
from src.modeling.multi_outcome.multi_label_trainer import MultiLabelTrainer
from src.modeling.multi_outcome.outcome_schemas import (
    MultiOutcomeExplainResponse,
    MultiOutcomeResponse,
    OutcomePrediction,
    RiskLevel,
)

__all__ = [
    "ALL_OUTCOMES",
    "CARDIAC_ARREST_DEFINITION",
    "ClassificationHead",
    "DETERIORATION_DEFINITION",
    "MultiHeadPredictionModel",
    "MultiLabelTrainer",
    "MultiOutcomeExplainResponse",
    "MultiOutcomeResponse",
    "NUM_OUTCOMES",
    "OUTCOME_COLUMNS",
    "OUTCOME_REGISTRY",
    "OutcomeDefinition",
    "OutcomePrediction",
    "OutcomeType",
    "RESPIRATORY_FAILURE_DEFINITION",
    "RiskLevel",
    "SEPSIS_DEFINITION",
]
