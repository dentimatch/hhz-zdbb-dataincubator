from __future__ import annotations

import pandas as pd

from synthesis_runner import (
    estimate_duration_minutes,
    format_duration_label,
    format_elapsed_time,
    format_remaining_seconds,
    load_metadata,
    run_training,
)
from sdv.single_table import GaussianCopulaSynthesizer


def test_duration_estimations():
    minutes = estimate_duration_minutes(100, 5, "ctgan", 0)
    assert minutes > 0
    assert format_duration_label(0.4) == "< 1 minute"
    assert format_duration_label(20) == "approx. 15–30 minutes"
    assert format_remaining_seconds(45) == "45s"
    assert format_remaining_seconds(125) == "2m 5s"
    assert format_elapsed_time(0.3) == "<1s"
    assert format_elapsed_time(75) == "1m 15s"


def test_run_training_gaussian_copula():
    data = pd.DataFrame(
        {
            "age": [25, 30, 45, 35, 40],
            "income": [50000, 52000, 61000, 58000, 59000],
            "segment": ["A", "B", "A", "C", "B"],
        }
    )
    metadata = load_metadata(None, data)

    events = []

    def progress_callback(fraction: float, status: str) -> None:
        events.append((fraction, status))

    result = run_training(
        df_real=data,
        metadata=metadata,
        model_name="gaussiancopula",
        random_seed=42,
        registry={"gaussiancopula": GaussianCopulaSynthesizer},
        total_rows=len(data),
        estimated_minutes=estimate_duration_minutes(len(data), len(data.columns), "gaussiancopula", 0),
        progress_callback=progress_callback,
        poll_interval=0.1,
    )

    assert len(result.df_synth) == len(data)
    assert result.actual_seconds >= 0
    assert result.used_global_seed is True
    assert events, "Progress callback should be invoked at least once"

