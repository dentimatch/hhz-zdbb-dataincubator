"""Streamlit UI for generating synthetic tabular data with SDV.

Run locally (PowerShell):

    streamlit run streamlit_app.py

The app scans the `data/` directory for CSV files, displays descriptive
statistics, lets the user generate synthetic rows, saves them with a suffix,
and evaluates the result using SDV's single table metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import time

import pandas as pd
import streamlit as st
from sdv.single_table import (
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

from evaluation_utils import (
    error_suggestions,
    interpret_utility_score,
    run_quality_report,
)
from synthesis_runner import (
    estimate_duration_minutes,
    format_duration_label,
    format_elapsed_time,
    format_remaining_seconds,
    load_dataset,
    run_training,
)


APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
EXPLAINER_PATH = APP_ROOT / "docs" / "explainers.md"
DEFAULT_SUFFIX = "_synthetic"

DATA_DIR.mkdir(parents=True, exist_ok=True)

SYNTHESIZER_REGISTRY = {
    "CTGAN": CTGANSynthesizer,
    "GaussianCopula": GaussianCopulaSynthesizer,
    "TVAE": TVAESynthesizer,
}

MODEL_ADVANCED_CONFIGS = {
    "CTGAN": {"epochs": 300, "batch_size": 500},
    "TVAE": {"epochs": 300, "batch_size": 256},
}


METRIC_HELP = {
    "utility": "Bewertet die Ähnlichkeit zwischen Original- und synthetischen Daten (0-1, höher ist besser).",
    "model": "Gewähltes SDV-Modell zur Generierung der synthetischen Daten.",
    "rows": "Anzahl der erzeugten Zeilen (Original + zusätzliche Zeilen).",
    "seed": "Zufallsseed für reproduzierbare Ergebnisse.",
}


@dataclass
class SyntheticResult:
    dataframe: pd.DataFrame
    output_path: Path
    utility_score: float
    model_name: str
    total_rows: int
    additional_rows: int
    random_seed: int
    hints: List[str]
    report_details: pd.DataFrame
    duration_label: str
    actual_duration_label: str


@st.cache_data(show_spinner=False)
def list_csv_files() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(p for p in DATA_DIR.glob("*.csv") if p.is_file())


@st.cache_data(show_spinner=False)
def load_dataset_cached(path: str, mtime: float) -> Tuple[pd.DataFrame, object]:
    return load_dataset(path)


@st.cache_data(show_spinner=False)
def load_explainers() -> Optional[str]:
    if not EXPLAINER_PATH.exists():
        return None
    return EXPLAINER_PATH.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return pd.DataFrame()
    stats = numeric_df.describe().T
    stats.index.name = "column"
    return stats


@st.cache_data(show_spinner=False)
def top_value_counts(df: pd.DataFrame, limit: int = 10) -> Dict[str, pd.DataFrame]:
    categorical_df = df.select_dtypes(include=["object", "category"])
    counts: Dict[str, pd.DataFrame] = {}
    for column in categorical_df.columns:
        series = categorical_df[column].astype("string")
        counts[column] = (
            series.value_counts(dropna=False)
            .rename_axis("value")
            .reset_index(name="count")
            .head(limit)
        )
    return counts
def ensure_suffix(suffix: str) -> str:
    suffix = suffix.strip()
    if not suffix:
        return DEFAULT_SUFFIX
    if not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return suffix


def app() -> None:
    st.set_page_config(page_title="Synthetic Data Workbench", layout="wide")
    st.title("Synthetic Data Workbench")

    csv_files = list_csv_files()
    if not csv_files:
        st.warning("Kein CSV in 'data/' gefunden. Bitte Dateien hinzufügen und neu laden.")
        return

    st.sidebar.subheader("Datenauswahl")
    uploaded_file = st.sidebar.file_uploader(
        "CSV hochladen oder ziehen",
        type=["csv"],
        help="Neue Datei wird im Ordner `data/` gespeichert.",
    )
    if uploaded_file is not None:
        target_name = Path(uploaded_file.name).name or "upload.csv"
        target_path = DATA_DIR / target_name
        if target_path.exists():
            timestamp = int(time.time())
            target_path = DATA_DIR / f"{target_path.stem}_{timestamp}{target_path.suffix}"
        target_path.write_bytes(uploaded_file.getbuffer())
        st.sidebar.success(f"Datei unter `{target_path.name}` gespeichert.")
        st.session_state["selected_path"] = target_path
        st.experimental_rerun()

    file_display = {file.name: file for file in csv_files}
    selected_name = st.sidebar.selectbox("CSV-Datei auswählen", options=list(file_display.keys()))
    selected_path = file_display[selected_name]

    selected_mtime = selected_path.stat().st_mtime
    tracking = st.session_state.get("file_tracking")
    if tracking and tracking.get("path") == str(selected_path) and tracking.get("mtime") != selected_mtime:
        st.sidebar.warning("Datei außerhalb der App geändert – Daten werden neu geladen.")

    if st.session_state.get("selected_path") != selected_path:
        st.session_state.pop("synthetic_result", None)
        st.session_state["selected_path"] = selected_path

    st.session_state["file_tracking"] = {"path": str(selected_path), "mtime": selected_mtime}

    df_real, metadata = load_dataset_cached(str(selected_path), selected_mtime)

    st.subheader("1. Datensatz erkunden")
    st.write(f"**Datei:** `{selected_name}` | **Zeilen:** {len(df_real):,} | **Spalten:** {len(df_real.columns)}")

    tab_overview, tab_numeric, tab_categorical, tab_infos = st.tabs([
        "Vorschau",
        "Numerische Statistik",
        "Kategoriale Werte",
        "Infos",
    ])

    with tab_overview:
        st.dataframe(df_real.head(), use_container_width=True)
        st.caption("Erste 5 Zeilen des Originaldatensatzes")

    with tab_numeric:
        numeric_stats = describe_numeric(df_real)
        if numeric_stats.empty:
            st.info("Keine numerischen Spalten gefunden.")
        else:
            st.dataframe(numeric_stats, use_container_width=True)

    with tab_categorical:
        value_counts = top_value_counts(df_real)
        if not value_counts:
            st.info("Keine kategorialen Spalten gefunden.")
        else:
            for column, counts in value_counts.items():
                with st.expander(f"Top-Werte für {column}"):
                    st.table(counts)

    with tab_infos:
        explainer_md = load_explainers()
        if explainer_md:
            st.markdown(explainer_md)
        else:
            st.info("Keine Zusatzinformationen gefunden. Bitte `docs/explainers.md` prüfen.")

    st.subheader("2. Synthetische Daten generieren")

    col_left, col_right = st.columns(2)
    with col_left:
        model_name = st.selectbox("Synthesizer", options=list(SYNTHESIZER_REGISTRY.keys()), index=0)
        random_seed = int(
            st.number_input("Random Seed", value=42, step=1, help="Für reproduzierbare Ergebnisse.")
        )
    with col_right:
        additional_rows = st.number_input(
            "Zusätzliche Zeilen",
            min_value=0,
            max_value=int(len(df_real) * 50) if len(df_real) else 10000,
            value=min(len(df_real), 1000),
            help="Anzahl zusätzlicher synthetischer Zeilen im Vergleich zum Original.",
        )
        suffix = st.text_input("Dateisuffix", value=DEFAULT_SUFFIX, help="Suffix für die Ausgabe (z. B. _synthetic)")

    init_kwargs: Dict[str, int] = {}
    model_defaults = MODEL_ADVANCED_CONFIGS.get(model_name)
    if model_defaults:
        with st.expander("Trainingseinstellungen (optional)", expanded=False):
            epochs_val = st.number_input(
                "Epochs",
                min_value=1,
                max_value=5000,
                value=model_defaults["epochs"],
                step=10,
                help="Anzahl der Trainingsdurchläufe (höher = bessere Qualität, längere Laufzeit).",
            )
            batch_size_val = st.number_input(
                "Batch Size",
                min_value=16,
                max_value=4096,
                value=model_defaults["batch_size"],
                step=16,
                help="Größe der Trainings-Batches. Kleinere Werte = stabiler, längere Laufzeit.",
            )
            init_kwargs["epochs"] = int(epochs_val)
            init_kwargs["batch_size"] = int(batch_size_val)

    additional_rows_int = int(additional_rows)
    total_rows = len(df_real) + additional_rows_int
    estimated_minutes = estimate_duration_minutes(
        rows=len(df_real),
        columns=len(df_real.columns),
        model_name=model_name,
        additional_rows=additional_rows_int,
    )
    duration_label = format_duration_label(estimated_minutes)

    st.markdown(
        f"*Gesamte synthetische Zeilen*: **{total_rows:,}** (Original: {len(df_real):,} + Zusatz: {additional_rows_int:,})"
    )
    st.caption(
        "Geschätzte Dauer: "
        f"{duration_label} (Heuristik basierend auf Datensatzgröße und Modell)"
    )

    if st.button("Generieren", type="primary"):
        progress_placeholder = st.empty()

        progress_bar = progress_placeholder.progress(
            0,
            text=f"Training gestartet (≈ {duration_label})",
        )

        def progress_callback(fraction: float, status_text: str) -> None:
            progress_bar.progress(min(fraction, 0.999), text=status_text)

        try:
            runner_result = run_training(
                df_real=df_real,
                metadata=metadata,
                model_name=model_name,
                random_seed=random_seed,
                registry=SYNTHESIZER_REGISTRY,
                total_rows=total_rows,
                estimated_minutes=estimated_minutes,
                progress_callback=progress_callback,
                init_kwargs=init_kwargs,
            )
            progress_placeholder.empty()

            df_synth = runner_result.df_synth
            actual_duration_label = format_elapsed_time(runner_result.actual_seconds)

            utility_score, report_details, report_warnings = run_quality_report(
                df_real,
                df_synth,
                metadata,
            )
            suffix_normalized = ensure_suffix(suffix)
            output_path = selected_path.with_name(f"{selected_path.stem}{suffix_normalized}.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_synth.to_csv(output_path, index=False)

            base_hints: List[str] = [f"Training abgeschlossen in {actual_duration_label}."]
            if init_kwargs:
                formatted = ", ".join(f"{key}={value}" for key, value in init_kwargs.items())
                base_hints.append(f"Verwendete Trainingsparameter: {formatted}")
            st.session_state["synthetic_result"] = SyntheticResult(
                dataframe=df_synth,
                output_path=output_path,
                utility_score=utility_score,
                model_name=model_name,
                total_rows=total_rows,
                additional_rows=int(additional_rows),
                report_details=report_details,
                random_seed=int(random_seed),
                hints=base_hints,
                duration_label=duration_label,
                actual_duration_label=actual_duration_label,
            )
            if runner_result.used_global_seed:
                st.info(
                    "Hinweis: Das gewählte Modell unterstützt kein `random_state`. "
                    "Globale Zufallsquellen wurden mit dem angegebenen Seed gesetzt."
                )
                st.session_state["synthetic_result"].hints.append(
                    "Globale Zufallsquellen mit angegebenem Seed gesetzt (Modell ohne random_state)."
                )
            if report_warnings:
                st.warning("\n".join(report_warnings))
            st.success(f"Synthetischer Datensatz gespeichert unter `{output_path.name}`")
        except Exception as exc:  # pragma: no cover - Streamlit handles display
            progress_placeholder.empty()
            st.error(f"Fehler beim Generieren: {exc}")
            hints = error_suggestions(str(exc))
            if hints:
                st.info("Tipps:\n- " + "\n- ".join(hints))

    result: Optional[SyntheticResult] = st.session_state.get("synthetic_result")
    if result:
        st.subheader("3. Ergebnisse & Validierung")

        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Utility Score (0-1)",
            f"{result.utility_score:.3f}",
            help=METRIC_HELP["utility"],
        )
        metric_cols[1].metric(
            "Modell",
            result.model_name,
            help=METRIC_HELP["model"],
        )
        metric_cols[2].metric(
            "Zeilen (synthetisch)",
            f"{result.total_rows:,}",
            help=METRIC_HELP["rows"],
        )
        metric_cols[3].metric(
            "Seed",
            str(result.random_seed),
            help=METRIC_HELP["seed"],
        )

        status_text = (
            f"Modell: {result.model_name}\n"
            f"Seed: {result.random_seed}\n"
            f"Zeilen gesamt: {result.total_rows:,} (Zusätzlich: {result.additional_rows:,})\n"
            f"Utility Score: {result.utility_score:.3f}\n"
            f"Geschätzte Dauer: {result.duration_label}\n"
            f"Tatsächliche Dauer: {result.actual_duration_label}"
        )
        st.write("**Zusammenfassung**")
        st.code(status_text, language="text")
        st.caption(interpret_utility_score(result.utility_score))
        if st.button("Zusammenfassung kopieren"):
            st.query_params.update({"summary": status_text})
            st.success("Zusammenfassung via Query-Parameter bereitgestellt.")

        hints = result.hints
        st.write("**Hinweise**")
        if hints:
            hints_text = "\n".join(f"- {hint}" for hint in hints)
            st.write(hints_text)
            if st.button("Hinweise kopieren"):
                st.query_params.update({"hints": hints_text})
                st.success("Hinweise via Query-Parameter bereitgestellt.")
        else:
            st.caption("Keine Hinweise verfügbar.")

        preview_tab, compare_tab = st.tabs(["Synthetische Vorschau", "Vergleich" ])
        with preview_tab:
            st.dataframe(result.dataframe.head(), use_container_width=True)
            st.caption("Erste 5 Zeilen des synthetischen Datensatzes")

        with compare_tab:
            synth_numeric = describe_numeric(result.dataframe)
            if not synth_numeric.empty and not describe_numeric(df_real).empty:
                combined = (
                    describe_numeric(df_real)
                    .add_suffix("_original")
                    .join(synth_numeric.add_suffix("_synthetic"), how="outer")
                )
                st.dataframe(combined, use_container_width=True)
            else:
                st.info("Numerische Kennzahlen konnten nicht verglichen werden.")

        with st.expander("Detailierte Qualitätsmetriken"):
            details = result.report_details
            if details.empty:
                st.info("Keine Detailmetriken verfügbar.")
            elif "property" not in details.columns:
                st.dataframe(details, use_container_width=True)
            else:
                property_labels = {
                    "Column Shapes": "Column Shapes",
                    "Column Pair Trends": "Column Pair Trends",
                }
                categories = list(details["property"].unique())
                tab_labels = [property_labels.get(cat, cat) for cat in categories]
                tabs = st.tabs(tab_labels)
                for tab, category in zip(tabs, categories):
                    with tab:
                        subset = details[details["property"] == category].drop(
                            columns=["property"],
                            errors="ignore",
                        )
                        st.dataframe(subset, use_container_width=True)
                        st.caption(
                            "Metrics für "
                            f"{property_labels.get(category, category)}"
                        )

        csv_bytes = result.dataframe.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Synthetischen Datensatz herunterladen",
            data=csv_bytes,
            file_name=result.output_path.name,
            mime="text/csv",
        )

        st.caption(f"Datei gespeichert in `{result.output_path}`")


if __name__ == "__main__":
    app()

