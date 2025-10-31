"""Shared helpers for evaluating synthetic data quality with SDMetrics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Tuple

import pandas as pd

if TYPE_CHECKING:
    from sdv.metadata import SingleTableMetadata


LOGGER = logging.getLogger(__name__)

ERROR_HINTS = [
    (("nan", "missing", "null", "na"), "Check whether missing values are cleaned or allowed by the model."),
    (("shape", "dimension", "columns"), "Ensure the schema stays consistent (same column names and dtypes)."),
    (("timeout", "converge", "diverge"), "Reduce additional rows or try `GaussianCopula` for smaller datasets."),
    (("datetime", "date", "timestamp"), "Validate date/time formats; convert to a standard format (YYYY-MM-DD) if needed."),
]

def run_quality_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: "SingleTableMetadata",
) -> Tuple[float, pd.DataFrame, List[str]]:
    """Generate a SDMetrics QualityReport and return score, detail table, warnings."""
    # Lazy import to avoid loading heavy SDMetrics dependencies at startup
    from sdmetrics.reports.single_table import QualityReport

    report = QualityReport()
    report.generate(real_df, synthetic_df, metadata.to_dict())
    score = report.get_score()
    detail_frames = []
    warnings: List[str] = []
    for property_name in ("Column Shapes", "Column Pair Trends"):
        try:
            detail_frame = report.get_details(property_name)
        except ValueError:
            warnings.append(
                f"QualityReport property '{property_name}' not available in this SDMetrics version."
            )
            LOGGER.warning("QualityReport missing property '%s'", property_name)
            continue
        if detail_frame is not None and not detail_frame.empty:
            detail_frame = detail_frame.copy()
            detail_frame.insert(0, "property", property_name)
            detail_frames.append(detail_frame)

    if detail_frames:
        details = pd.concat(detail_frames, ignore_index=True)
    else:
        details = pd.DataFrame()

    return score, details, warnings


def error_suggestions(message: str) -> List[str]:
    """Return human-friendly suggestions based on an error message."""

    lowered = message.lower()
    tips: List[str] = []
    for keywords, suggestion in ERROR_HINTS:
        if any(keyword in lowered for keyword in keywords):
            tips.append(suggestion)

    if not tips:
        tips.append(
            "Try an alternative model (for example GaussianCopula) or reduce additional rows if training is unstable."
        )
        tips.append("Confirm that the input data has clean dtypes and enough observations.")

    return tips


def interpret_utility_score(score: float) -> str:
    """Return a simple textual interpretation for the utility score."""

    if score >= 0.85:
        return "Excellent alignment — synthetic data is very close to the original."
    if score >= 0.7:
        return "Good quality — suitable for many analyses, but spot-check samples."
    if score >= 0.55:
        return "Acceptable — review results more carefully and adjust parameters if needed."
    return "Low quality — tweak the training configuration or improve data preprocessing."

