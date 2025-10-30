# Synthetic Data Workbench

A lightweight toolbox for SMEs to convert tabular sample data into synthetic datasets. It combines a Python command-line script with an interactive Streamlit interface.

## Contents
- `data/` – sample CSV files (add your own files here)
- `evaluation_utils.py` – shared helper functions for quality metrics
- `sdv_tabular_example.py` – CLI script for generating synthetic data
- `streamlit_app.py` – Streamlit web app for exploration, synthesis, and evaluation
- `synthetic_tabular_workflow.md` – detailed guide covering tools, evaluation, and rollout

## Prerequisites
- Windows 10/11 (tested) or a compatible OS
- Python 3.13 (tested with 3.13.5); compatible with 3.9–3.13
- Internet access for the initial package installation

## Setup
1. Place the repository in your preferred folder (for example `C:\Cursor_Git\hhz-zdbb-dataincubator`).
2. Open PowerShell and switch to the project directory:
   ```powershell
   cd C:\Cursor_Git\hhz-zdbb-dataincubator
   ```
3. Create and activate a virtual environment:
   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   > If the `py` launcher is missing, use `python -m venv .venv` as an alternative.
4. Install the dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Optional: refresh `requirements.txt` after updating packages:
   ```powershell
   pip freeze > requirements.txt
   ```

## Prepare your data
- Place your CSV files in the `data/` folder. All files ending in `.csv` are detected automatically.
- Each file should include a header row with column names.

## SDV version and model support
- Tested with Python 3.13.5 and the latest SDV version (see `pip show sdv`).
- `CTGANSynthesizer` and `TVAESynthesizer` accept `random_state`; `GaussianCopulaSynthesizer` uses global random sources (the seed is applied through `seed_everything`).
- For different SDV versions, consult the release notes. API changes (for example removed methods) are captured in the improvement notes.

## Reproducibility and tests
- `requirements.txt` lists the validated dependencies (see Setup).
- After installation, run the tests:
  ```powershell
  python -m pytest
  ```
- Generated artifacts (for example `data/*_synthetic.csv`, `__pycache__/`, virtual environments) are excluded via `.gitignore` and not versioned.

## Command-line usage
The script `sdv_tabular_example.py` trains the SDV CTGAN synthesizer (or alternatives) on a selected CSV file and produces a synthetic variant.

### Example invocation
```powershell
python sdv_tabular_example.py `
  --csv data\patients.csv `
  --rows 1500 `
  --output data\patients_synth.csv `
  --report artifacts\patients_report.json
```

- `--csv`: path to the source file.
- `--rows`: number of rows in the synthetic dataset (default = original size).
- `--output`: location for the new CSV.
- `--report`: optional JSON report with metadata, utility score, and detailed metrics (SDMetrics QualityReport).
- `--model`: optional, choose from `ctgan`, `gaussiancopula`, `tvae`.
- `--random-seed`: optional random seed for reproducible results. Models with a `random_state` parameter (CTGAN, TVAE) receive it directly; `GaussianCopula` falls back to global RNG seeds internally.
- `--epochs`, `--batch-size`: optional overrides for the defaults in models that support these parameters (for example CTGAN/TVAE).
- Before training you receive a rough time estimate; during training the CLI continuously updates the progress (heuristic based on dataset size and model type).

## Streamlit usage
The web application provides a guided workflow for non-technical users.

### Launch step by step
1. Open PowerShell and switch to the project directory:
   ```powershell
   cd C:\Cursor_Git\hhz-zdbb-dataincubator
   ```
2. Activate the virtual environment (if it is not already active):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Start the Streamlit app:
   ```powershell
   streamlit run streamlit_app.py
   ```
4. The browser opens automatically. If not, copy the local link displayed (for example `http://localhost:8501`) into the address bar.

### Feature overview
1. **Select file** – dropdown lists every CSV file in `data/`.
2. **Descriptive statistics** – preview of the first rows, numeric KPIs, top categories.
3. **Configure synthesis** – choose the SDV model, seed (for reproducible runs), number of additional rows, and output suffix.
4. **Generate and save** – creates the synthetic dataset and writes it as a CSV next to the original using the suffix (default `_synthetic`). A download button is provided.
5. **Evaluation** – SDMetrics *QualityReport* supplies utility scores (0–1) plus detailed metrics; numeric KPIs are compared side by side.
6. **Info** – tab with explanatory content sourced from `docs/explainers.md` (models, seeds, metrics, workflow tips).

### Typical flow inside the app
1. Pick a CSV in the sidebar dropdown.
2. Review the statistics (tabs "Preview", "Numeric Statistics", "Categorical Values").
3. Fill in the form (model, seed, extra rows, suffix) and click "Generate".
4. Optional: use the "Training settings" expander to adjust epochs/batch size (for CTGAN/TVAE).
5. Wait for the synthesis to complete. The progress bar continuously estimates the remaining time. Once finished, metrics, comparison tables, and the download link appear.

## Interpreting the results
- **Utility score** (0–1): indicates how similar the original and synthetic data behave. Values above 0.7 are often considered good, but thresholds depend on the use case. The score comes from the SDMetrics *QualityReport*.
- **Comparison tables**: confirm whether distributions remain plausible (for example means, standard deviations).
- **Detailed metrics**: view single metrics and per-column results inside the Streamlit "Detailed quality metrics" expander and in the JSON report (`quality_report`).
- **Reproducibility**: both the CLI and Streamlit apply the selected seed. For models without an explicit `random_state` (for example `GaussianCopula`) Python/Numpy (and optionally Torch) are seeded globally.
- **Output files**: the synthetic CSV is stored in the same directory; the filename automatically receives the chosen suffix.

## Troubleshooting
- *No CSV detected*: ensure files ending in `.csv` are stored in `data/`.
- *ImportError / missing module*: check that the virtual environment is active and `pip install ...` completed successfully.
- *SDV freezes or runs very slowly*: for very small datasets switch to `--model gaussiancopula` or lower the number of additional rows.
- *Poor utility score*: check metadata, perform additional preprocessing (for example encoding, outlier handling), or try an alternative model.

## Further reading
- Detailed guide: `synthetic_tabular_workflow.md`
- Custom automation: integrate the CLI script into batch jobs or CI/CD.
- Extensions: multi-table (relational) setups, differential privacy, dashboarding quality metrics.
- Models and concepts: fundamentals on models, seeds, and metrics in `synthetic_tabular_workflow.md` (sections 2–3) plus deeper explanations in `docs/explainers.md`.
- Implementation details: `synthesis_runner.py` encapsulates metadata detection, duration heuristics, and progress callbacks shared by the Streamlit UI and CLI.

## Improvement notes

### Completed
- **T1** Cache the synthesizer registry lookup; `build_synthesizer` now keeps a cache for normalized keys.
- **T2** Make the `seed_everything` fallback visible (logging or UI hint) so it is clear when global seeds are used.
- **T3** Surface a signal when QualityReport properties are missing (log or UI hint) instead of silently ignoring them.
- **T4** Combine metadata detection with DataFrame caching (for example via `st.cache_data`) to speed up large CSVs.
- **T5** Document `SDV` version and model support in the README; removed the separate note document.
- **T6** Removed the legacy planning document `synt.plan.md` (cleaned the repo).
- **F1** Group quality metrics by category in the UI (tabs for column shapes vs. column pair trends) and explain them.
- **F2** Offer optional hyperparameter overrides (epochs, batch size) with sensible defaults.
- **F3** Add a summary card showing seed, model, and row count with a copyable text snippet.
- **F4** Provide actionable tips with error messages (UI and CLI display context-aware guidance).
- **F5a** Allow drag-and-drop CSV uploads (uploader stores files in `data/`).
- **F5b** Detect when CSVs change outside the app and show a notice with automatic reload.
- **F7a** Hover tooltips explain metrics for users without prior knowledge.
- **F7b** Short interpretations for key quality metrics (for example utility score) appear alongside the values.
- **F7c** Dedicated knowledge section with explanations for models, seeds, metrics, and recommended workflows (`docs/explainers.md`).
- **F8a** Provide a rough duration estimate before training (heuristic based on dataset size and model, UI/CLI).
- **F8b** Display a dynamic progress indicator with remaining time heuristic during training (UI & CLI).

### Open
- **F6** Notification system that alerts users when synthesis finishes or aborts (optional future extension).
- **O1** (reserved for additional optimization ideas)

Good luck creating synthetic datasets! Reach out with questions or suggestions.

