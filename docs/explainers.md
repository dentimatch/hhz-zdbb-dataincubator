# Synthetic Data – Explanations

## Models
- **CTGAN**: GAN-based approach, strong for heterogeneous tabular data. Typically needs more training time but yields richer distributions.
- **GaussianCopula**: Models dependencies via copulas; fast and stable for smaller datasets or when you have few outliers.
- **TVAE**: Variational Autoencoder suited for continuous data; produces smooth distributions but may require preprocessing when many categorical features are present.

### Choosing a model
1. *Small datasets (<1k rows)* → start with GaussianCopula, then try CTGAN.
2. *Many categorical columns* → favor CTGAN.
3. *Purely numeric data / time series* → use TVAE or SDV's dedicated time-series modules.

## Seeds and reproducibility
- Models with a `random_state` parameter (CTGAN, TVAE) take the seed directly.
- Models without an explicit seed parameter (GaussianCopula) rely on global seeds (`seed_everything`).
- Same seed plus identical data → deterministic, reproducible results.

## Quality metrics
- **Utility score**: Aggregated similarity (0–1). 0.85+ = excellent, 0.7–0.85 = good, 0.55–0.7 = acceptable, below that: reconfigure.
- **Column shapes**: Compare distributions of individual columns; large deviations point to missing modeling signal.
- **Column pair trends**: Check relationships between column pairs—important for feature interactions.
Tips:
- If the utility score is low, adjust parameters (epochs, reduce additional rows) or tweak preprocessing (remove outliers, group categories).
- Combine A/B comparisons with domain knowledge: charts, pivot tables, test models.

## Workflow and troubleshooting
1. Inspect data → NaNs, types, categories.
2. Start synthesis with default parameters.
3. Evaluate results → utility score plus visual checks.
4. Iterate: vary seeds, swap synthesizers, normalize features.
5. Maintain documentation: seed, parameters, evaluation in the report (`quality_report`).

## Runtime and progress
- The runtime estimate is based on dataset size and model (heuristic) and is shown before training starts.
- During training the CLI and Streamlit update progress with a rough remaining-time estimate (prioritizing pragmatism over accuracy).
- After completion the estimated and actual durations are recorded (CLI report, Streamlit summary).

## Further resources
- SDV docs: https://docs.sdv.dev/
- SDMetrics: https://docs.sdv.dev/sdmetrics/
- Best practices for data anonymization: https://opendp.org/

