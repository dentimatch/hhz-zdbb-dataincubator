"""Shared helpers for evaluating synthetic data quality with SDMetrics."""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport


LOGGER = logging.getLogger(__name__)

ERROR_HINTS = [
    (("nan", "missing", "null", "na"), "Prüfe, ob fehlende Werte bereinigt oder im Modell erlaubt sind."),
    (("shape", "dimension", "columns"), "Vergewissere dich, dass das Schema stabil bleibt (gleiche Spaltennamen/-typen)."),
    (("timeout", "converge", "diverge"), "Reduziere zusätzliche Zeilen oder versuche `GaussianCopula` für kleinere Datensätze."),
    (("datetime", "date", "timestamp"), "Überprüfe Datums-/Zeitformate; ggf. in standardisiertes Format (YYYY-MM-DD) bringen."),
]

def run_quality_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: SingleTableMetadata,
) -> Tuple[float, pd.DataFrame, List[str]]:
    """Generate a SDMetrics QualityReport and return score, detail table, warnings."""

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
            "Versuche ein alternatives Modell (z. B. GaussianCopula) oder reduziere die Zusatzzeilen, falls das Training instabil ist."
        )
        tips.append("Prüfe, ob die Eingabedaten saubere Typen und ausreichend Beobachtungen besitzen.")

    return tips


def interpret_utility_score(score: float) -> str:
    """Return a simple textual interpretation for the utility score."""

    if score >= 0.85:
        return "Ausgezeichnete Übereinstimmung – synthetische Daten sehr nah am Original."
    if score >= 0.7:
        return "Gute Qualität – für viele Analysen ausreichend, dennoch Stichproben prüfen."
    if score >= 0.55:
        return "Akzeptabel – Ergebnisse genauer prüfen und ggf. Parameter anpassen."
    return "Niedrige Qualität – Trainingskonfiguration anpassen oder Datenvorverarbeitung verbessern."

