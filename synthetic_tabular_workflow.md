# Cursor Vibe: Synthetic Tabular Workflow

## 1. Open-Source Landscape for Tabular Synthesis

| Library | License | Strengths | Limitations | Notes |
| --- | --- | --- | --- | --- |
| SDV (Synthetic Data Vault) | MIT | Mature library with single-/multi-table and time-series modules; SDMetrics evaluation built in; active support | Training on large tables may require GPU/TPU; documentation is spread across sources | https://sdv.dev
| YData Synthetic | Apache-2.0 | GAN/VAE models focused on data quality; simple API, tutorials; CLI and notebook templates | Relatively young OSS project; evaluations less extensive than SDV | https://github.com/ydataai/ydata-synthetic
| Gretel Synthetics | Apache-2.0 | CLI/SDK, differential privacy options, optional SaaS integration; example notebooks | Open-source core has fewer convenience features than SaaS; depends on TensorFlow | https://github.com/gretelai/gretel-synthetics
| DataSynthesizer | MIT | Minimalist, fast, simple "mode/independent/correlated" strategies; great for prototyping | Less precise for complex relationships; barely any automatic evaluation | https://github.com/DataResponsibly/DataSynthesizer
| MOSTLY AI (Community) | Proprietary (free tier) | UI-guided pipeline, built-in RLS; free community edition for smaller datasets | Not fully open source; volume limits; account required | https://mostly.ai

SMBs adopting Cursor Vibe benefit from a Python-centric stack (SDV or YData Synthetic). Both can be run locally, versioned, and automated.

## 2. Recommended stack

- **Primary: SDV (single table → `CTGANSynthesizer`)**
  - Runs fully locally on Python ≥3.9.
  - Includes automatic metadata detection (`SingleTableMetadata.detect_from_dataframe`).
  - SDMetrics provides ready-to-use utility and privacy metrics.
  - Large community with long-term maintenance.

- **Optional: YData Synthetic**
  - Consider when planning GAN/TimeGAN models with GPU support.
  - Notebook-first experience, useful for exploratory experiments.

## 3. Implementation guide (Cursor Vibe)

### 3.1 Prepare the environment

1. Create a virtual environment (PowerShell):
   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. Optional: after updating packages, run `pip freeze > requirements.txt` to capture the state.
3. Run the tests:
   ```powershell
   python -m pytest
   ```

### 3.2 Sample script (`sdv_tabular_example.py`)

The script reads an example file, trains `CTGANSynthesizer`, and writes a synthetic dataset. Evaluation happens through `evaluation_utils.run_quality_report`, which wraps the SDMetrics *QualityReport*.

- `synthesis_runner.py` provides a shared runtime layer (metadata detection, duration heuristic, progress callbacks) consumed by both the CLI and Streamlit.
- The CLI offers a rough runtime estimate before starting and updates progress during training (heuristic).

```text
python sdv_tabular_example.py \
  --csv data/input.csv \
  --rows 1000 \
  --output data/synthetic.csv
```

Key customization points:

- Data schema: optionally provide your own `metadata.json` (see SDV documentation).
- Sampling: `--rows` controls how many synthetic rows are generated (default = original size).
- Model choice: for very small datasets `GaussianCopulaSynthesizer` can be more stable; the script exposes the selection.
- Reproducibility: `--random-seed` sets the model-level `random_state` (CTGAN/TVAE). If a model does not expose that parameter (for example GaussianCopula), global RNGs (Python/Numpy, optionally Torch) are seeded instead.
- The Streamlit UI caches dataset and metadata together (`load_dataset`) so repeated interactions run faster.

### 3.3 Notebooks and tests

- Cursor Vibe supports Jupyter: run `python -m ipykernel install --user --name sdv-env` and create an `.ipynb` notebook.
- For unit tests: use `pytest` with fixtures based on small CSV files; SDV supports deterministic seeds (`synthesizer.set_random_state`).

### 3.4 Streamlit UI (optional)

- Start the app:
  ```powershell
  streamlit run streamlit_app.py
  ```
- Features: pick a CSV from `data/`, inspect preview/statistics, set additional rows, generate synthetic data, review the evaluation, download the result.
- Output files are saved alongside the original with a suffix (default `_synthetic`); for example `patients_synthetic.csv`.
- Evaluation uses the SDMetrics *QualityReport* (via `evaluation_utils.run_quality_report`); scores, detailed metrics, and numeric comparison tables appear under "Results & Validation".
- During synthesis a progress bar shows a heuristic estimate of the remaining time; after completion the summary notes estimated vs. actual duration. The "Info" tab sources its content from `docs/explainers.md` and explains models, seeds, and metrics.
- The "Training settings" expander lets you optionally adjust `epochs` and `batch_size` for CTGAN/TVAE (defaults: 300/500 and 300/256 respectively).
- Upload/changes: CSVs can be uploaded via drag & drop (saved in `data/`). External changes to the selected file are detected via timestamps and trigger a notice plus reload.
- Seeding: the Streamlit UI applies the chosen seed when instantiating the synthesizer; models without `random_state` rely on global seeding.

## 4. Quality and compliance

- **Utility metrics**
  - `evaluation_utils.run_quality_report(real, synthetic, metadata)` returns a utility score [0,1] plus detailed metrics.
  - Visualizations: use `sdmetrics.visualization.get_column_plot` for distributions.

- **Privacy checks**
  - Attribute disclosure: `sdmetrics.single_table.PrivacyReport` highlights re-identification risks.
  - Track Distance-to-Closest-Record (DCR) and document thresholds (for example >0.1).
  - Differential privacy: if required, configure `clamp` and `noise_multiplier` via DP-capable variants (for example Gretel DP, OpenDP).

- **Governance**
  - Logging: record model parameters, training timestamps, and seeds in YAML.
  - Data minimization: pseudonymize/anonymize source samples before training.

## 5. Next steps for an SMB rollout

- **Productization**
  - Wrap the script in a CLI (`typer`/`click`) and store artifacts in `artifacts/`.
  - Automate tests plus quality reports in CI (GitHub Actions/Azure DevOps).

- **Deployment**
  - For batch jobs: scheduled task (Windows Scheduler) or Docker container.
  - For self-service: provide a FastAPI or Flask endpoint that processes incoming CSVs (only in secure environments).

- **Monitoring**
  - Track metrics (utility/privacy scores) in a central dashboard.
  - Define retraining cycles (for example monthly, or ad hoc for schema changes).

- **Compliance**
  - Document the data protection impact assessment (GDPR Art. 35 if applicable).
  - Contracts: clarify whether synthetic data counts as trade secrets and how it is distributed (SaaS vs. on-premises).

This provides a Cursor-compatible roadmap: local Python development, reproducible scripts, validated quality, and clear next steps toward production.

