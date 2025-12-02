# User Stories & Requirements

## Process Overview
**Rule:** Feature implementation requires a corresponding User Story or Technical Task. This ensures all code changes are tracked and justified.

## Epics & Stories Overview

### Epic 1: Data Preparation
Focus on robustness and usability (Cleaning).
- **US-1.1**: Basic Data Cleaning

### Epic 2: Data Transformation
Focus on adapting external datasets to internal standards (Domain Adaptation).
- **US-2.1**: Feature Renaming & Translation
- **US-2.2**: Target Range Adaptation (Scaling)

### Epic 3: Data Control & Quality
Focus on validity and domain rules.
- **US-3.1**: Basic Column Constraints
- **US-3.2**: Metadata Override

### Epic 4: Model Management & Automation
Focus on reusability and CI/CD integration.
- **US-4.1**: Save Trained Model
- **US-4.2**: Load Pre-trained Model
- **US-4.3**: CLI Model Saving

### Epic 5: Advanced Visualization & Evaluation
Focus on building trust in the synthetic data.
- **US-5.1**: Visual Distribution Comparison
- **US-5.2**: Correlation Matrix

## Detailed Specifications

### Epic 1: Data Preparation

#### US-1.1 Basic Data Cleaning
- **Description**: As a user, I want to drop columns or handle simple NaNs so that I don't need to prep data in Excel or Python scripts beforehand.
- **Status**: ✅ Completed
- **Acceptance Criteria**:
  - Option to select columns to drop (exclude from training).
  - Option to fill NaNs with a simple strategy (e.g., mean/mode) or drop rows with missing values.
- **Effort**: Low

### Epic 2: Data Transformation

#### US-2.1 Feature Renaming & Translation
- **Description**: As a user, I want to rename or translate columns (e.g., "Alter" -> "Age", "Geschlecht" -> "Gender") so that the external example data matches our internal naming conventions.
- **Status**: ✅ Completed
- **Acceptance Criteria**:
  - UI provides a mapping interface (Old Name -> New Name).
  - Renaming is applied before training so the synthetic output has the correct schema.
  - Original source file is preserved (transformation happens in memory/temp).
- **Effort**: Low

#### US-2.2 Target Range Adaptation (Scaling)
- **Description**: As a user, I want to transform numeric columns (e.g., scale an "Age" column from 0-100 to 25-45) so that generic proxy data fits my specific business domain constraints before synthesis.
- **Status**: ✅ Completed
- **Acceptance Criteria**:
  - UI allows selecting a numeric column.
  - User can define a target Min/Max range or Shift/Scale factor.
  - Data is transformed (e.g., MinMaxScalar) to fit the new range.
  - **Critical:** Correlations to other features must be preserved (linear scaling).
  - **Critical:** Variance/StdDev must scale proportionally; distribution shape must be identical.
  - The synthesizer trains on this transformed data.
- **Effort**: Medium

### Epic 3: Data Control & Quality

#### US-3.1 Basic Column Constraints
- **Description**: As a domain expert, I want to set min/max bounds for numeric columns so that the synthetic data makes sense (e.g., no negative ages or invalid dates).
- **Acceptance Criteria**:
  - UI provides inputs for Min/Max for detected numeric columns.
  - Training respects these bounds (passed to SDV constraints).
- **Effort**: High

#### US-3.2 Metadata Override
- **Description**: As a user, I want to manually specify if a column is categorical or numerical so that the model treats it correctly (e.g., treating Zip Codes as categories instead of numbers).
- **Acceptance Criteria**:
  - UI displays detected metadata (types) before training.
  - User can toggle column types or select the correct type.
  - Synthesizer uses the corrected metadata.
- **Effort**: High

### Epic 4: Model Management & Automation

#### US-4.1 Save Trained Model
- **Description**: As a user, I want to save a trained synthesizer to disk so that I can reuse it later without retraining.
- **Acceptance Criteria**:
  - UI provides a "Download Model" button after training.
  - Model is saved in a standard format (e.g., `.pkl`).
  - Filename includes timestamp and/or versioning.
- **Effort**: Medium

#### US-4.2 Load Pre-trained Model
- **Description**: As a user, I want to load a pre-trained model file so that I can generate synthetic data immediately without waiting for training.
- **Acceptance Criteria**:
  - UI allows uploading a model file (`.pkl`).
  - App validates the model file structure.
  - User can specify the number of rows to generate from the loaded model.
- **Effort**: Medium

#### US-4.3 CLI Model Saving
- **Description**: As an automation engineer, I want CLI flags to save and load models so that I can integrate this into CI/CD pipelines.
- **Acceptance Criteria**:
  - New flag `--save-model <path>` to save the artifact after training.
  - New flag `--load-model <path>` to skip training and just generate from the artifact.
- **Effort**: Medium

### Epic 5: Advanced Visualization & Evaluation

#### US-5.1 Visual Distribution Comparison
- **Description**: As a data analyst, I want to see histograms comparing real vs. synthetic distributions so that I can visually verify the quality beyond simple mean/std stats.
- **Status**: ✅ Completed
- **Acceptance Criteria**:
  - UI shows side-by-side or overlaid histograms for numeric columns.
  - UI shows bar charts for categorical columns.
  - Charts are interactive (e.g., zoom, hover) if possible.
- **Effort**: Medium

#### US-5.2 Correlation Matrix
- **Description**: As a data analyst, I want to see a correlation heatmap so that I can ensure column relationships are preserved in the synthetic data.
- **Status**: ✅ Completed
- **Acceptance Criteria**:
  - Heatmap visualization for real data.
  - Heatmap visualization for synthetic data.
  - (Optional) Difference heatmap highlighting discrepancies.
- **Effort**: Medium
