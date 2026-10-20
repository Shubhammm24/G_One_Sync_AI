"""
G_One_Sync AI — Clinical Outcome Label Definitions
=====================================================
Clinical criteria for each predicted outcome.
Used for label generation, model card documentation,
and clinician-facing explanations.

IMPORTANT: These definitions are for the MIMIC-IV demo dataset.
Actual clinical deployment definitions must be validated
by board-certified intensivists before use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OutcomeType(str, Enum):
    """Enumeration of all predicted clinical outcomes."""

    DETERIORATION = "deterioration_next_12h"
    SEPSIS = "sepsis_next_6h"
    CARDIAC_ARREST = "cardiac_arrest_next_12h"
    RESPIRATORY_FAILURE = "respiratory_failure_next_12h"


@dataclass(frozen=True)
class OutcomeDefinition:
    """Clinical definition for a single predicted outcome."""

    outcome_type: OutcomeType
    label_column: str
    display_name: str
    prediction_horizon_hours: int
    description: str
    prevalence_estimate: float  # Expected positive rate in ICU population
    clinical_criteria: list[str] = field(default_factory=list)
    icd10_codes: list[str] = field(default_factory=list)


# ── Outcome Definitions ──────────────────────────────────────────────────

DETERIORATION_DEFINITION = OutcomeDefinition(
    outcome_type=OutcomeType.DETERIORATION,
    label_column="deterioration_next_12h",
    display_name="General Deterioration",
    prediction_horizon_hours=12,
    description=(
        "Any clinically significant deterioration event including ICU transfer, "
        "rapid response activation, unplanned intubation, vasopressor initiation, "
        "cardiac arrest, or death within the prediction horizon."
    ),
    prevalence_estimate=0.10,
    clinical_criteria=[
        "ICU transfer to higher level of care",
        "Rapid Response Team (RRT) activation",
        "Unplanned endotracheal intubation",
        "New vasopressor initiation",
        "Cardiac arrest (Code Blue)",
        "In-ICU death",
        "Sustained HR <40 or >150 BPM for >= 30 min",
        "Sustained SBP <80 mmHg for >= 30 min",
        "Sustained SpO2 <88% for >= 30 min",
    ],
)

SEPSIS_DEFINITION = OutcomeDefinition(
    outcome_type=OutcomeType.SEPSIS,
    label_column="sepsis_next_6h",
    display_name="Sepsis Onset",
    prediction_horizon_hours=6,
    description=(
        "New onset of sepsis defined by Sepsis-3 criteria: suspected or documented "
        "infection with an acute increase in SOFA score >= 2 points."
    ),
    prevalence_estimate=0.07,
    clinical_criteria=[
        "Suspected or documented infection AND",
        "Acute SOFA score increase >= 2 points",
        "OR: qSOFA >= 2 (RR >= 22, SBP <= 100, altered mentation)",
        "Surrogate: sepsis_risk_score > 0.6 with rising lactate > 2.0",
        "Blood culture ordered + antibiotic started within 6h window",
    ],
    icd10_codes=["A41.9", "R65.20", "R65.21"],
)

CARDIAC_ARREST_DEFINITION = OutcomeDefinition(
    outcome_type=OutcomeType.CARDIAC_ARREST,
    label_column="cardiac_arrest_next_12h",
    display_name="Cardiac Arrest",
    prediction_horizon_hours=12,
    description=(
        "Cardiac arrest event requiring cardiopulmonary resuscitation (CPR), "
        "defibrillation, or pulseless electrical activity (PEA) within the "
        "prediction horizon."
    ),
    prevalence_estimate=0.02,
    clinical_criteria=[
        "Code Blue activation for cardiac arrest",
        "CPR initiated (chest compressions)",
        "Defibrillation/cardioversion for pulseless rhythm",
        "Pulseless electrical activity (PEA)",
        "Asystole",
        "Surrogate: HR <30 or >200 sustained > 5 min",
    ],
    icd10_codes=["I46.9", "I46.2", "I46.8"],
)

RESPIRATORY_FAILURE_DEFINITION = OutcomeDefinition(
    outcome_type=OutcomeType.RESPIRATORY_FAILURE,
    label_column="respiratory_failure_next_12h",
    display_name="Respiratory Failure",
    prediction_horizon_hours=12,
    description=(
        "Acute respiratory failure requiring new mechanical ventilation, "
        "non-invasive positive pressure ventilation (NIPPV), or significant "
        "escalation of respiratory support."
    ),
    prevalence_estimate=0.05,
    clinical_criteria=[
        "New endotracheal intubation for respiratory failure",
        "Initiation of non-invasive ventilation (BiPAP/CPAP)",
        "FiO2 escalation to >= 0.6",
        "Sustained SpO2 < 90% on current O2 therapy for >= 15 min",
        "PaO2/FiO2 ratio < 200 (ARDS criteria)",
        "RR > 35 sustained > 30 min",
        "Surrogate: SpO2 < 92% with RR > 28 and rising O2 flow",
    ],
    icd10_codes=["J96.00", "J96.01", "J96.20", "J80"],
)


# ── Registry ──────────────────────────────────────────────────────────────

ALL_OUTCOMES: list[OutcomeDefinition] = [
    DETERIORATION_DEFINITION,
    SEPSIS_DEFINITION,
    CARDIAC_ARREST_DEFINITION,
    RESPIRATORY_FAILURE_DEFINITION,
]

OUTCOME_REGISTRY: dict[str, OutcomeDefinition] = {
    od.label_column: od for od in ALL_OUTCOMES
}

# Ordered list of label column names (canonical order)
OUTCOME_COLUMNS: list[str] = [od.label_column for od in ALL_OUTCOMES]
NUM_OUTCOMES: int = len(ALL_OUTCOMES)
