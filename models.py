from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# -----------------------------------------------------------------------------
# 1. Patient and Population
# -----------------------------------------------------------------------------
class PatientPopulation(BaseModel):
    cancer_type: str = Field(
        ..., description="Specific cancer type (e.g., 'Triple-Negative Breast Cancer', 'NSCLC')"
    )
    disease_stage: Optional[str] = Field(
        None, description="Stage of disease (e.g., 'Stage II-III', 'Metastatic', 'Recurrent')"
    )
    patient_age_range: Optional[str] = Field(
        None, description="Age range or median age of participants (e.g., 'Median 54 years', '18-75')"
    )
    sex_distribution: Optional[str] = Field(
        None, description="Breakdown of sex (e.g., '100% Female', '55% Male / 45% Female')"
    )
    comorbidities_summary: Optional[str] = Field(
        None, description="Summary of patient comorbidities or performance status (ECOG)"
    )
    prior_treatments_summary: Optional[str] = Field(
        None, description="Details on prior lines of therapy or treatment-naive status"
    )

# -----------------------------------------------------------------------------
# 2. Study Design
# -----------------------------------------------------------------------------
class StudyDesign(BaseModel):
    study_type: Literal['RCT', 'Prospective Cohort', 'Retrospective', 'Case Series', 'Meta-Analysis', 'Other'] = Field(
        ..., description="The methodological design of the study. 'RCT' = Randomized Controlled Trial."
    )
    sample_size: int = Field(
        ..., description="Total number of patients enrolled or analyzed in the study"
    )
    intervention: str = Field(
        ..., description="The main treatment, drug, or intervention being studied"
    )
    comparator: Optional[str] = Field(
        None, description="Control group or comparator regime (e.g., 'Placebo', 'Standard of Care'). None if single-arm."
    )
    primary_endpoint: str = Field(
        ..., description="The main outcome measure defined by the authors (e.g., 'Overall Survival', 'pCR')"
    )
    secondary_endpoints: Optional[List[str]] = Field(
        default=[], description="List of secondary outcomes measured"
    )
    followup_duration: Optional[str] = Field(
        None, description="Duration of patient follow-up (e.g., 'Median 24 months', '5 years')"
    )

# -----------------------------------------------------------------------------
# 3. Results and Reliability
# -----------------------------------------------------------------------------
class StudyResults(BaseModel):
    primary_result_text: str = Field(
        ..., description="Short textual summary of the primary outcome conclusion"
    )
    primary_result_numeric: Optional[str] = Field(
        None, description="Key numeric result if available (e.g., 'HR 0.65 (95% CI 0.5-0.8)', 'p<0.001')"
    )
    safety_profile_summary: Optional[str] = Field(
        None, description="General description of safety/tolerability and major adverse events"
    )
    limitations_summary: Optional[str] = Field(
        None, description="Author-stated limitations of the study"
    )

class OncologyStudy(BaseModel):
    filename: str = Field(..., description="Name of the source PDF file")
    title: str = Field(..., description="Extracted title of the paper")
    publication_year: Optional[int] = Field(None, description="Year of publication")

    # Nested Models
    population: PatientPopulation
    design: StudyDesign
    results: StudyResults

    # Derived Reliability Score (Calculated post-extraction, but field exists in model)
    predicted_reliability: Optional[Literal['Low', 'Medium', 'High']] = Field(
        None, description="Heuristic reliability score based on study design and sample size"
    )