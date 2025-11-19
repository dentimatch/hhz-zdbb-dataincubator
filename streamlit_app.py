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
DEFAULT_SUFFIX = "_synthetic"

DATA_DIR.mkdir(parents=True, exist_ok=True)

SYNTHESIZER_REGISTRY = {
    "CTGAN": "CTGANSynthesizer",
    "GaussianCopula": "GaussianCopulaSynthesizer",
    "TVAE": "TVAESynthesizer",
}

MODEL_ADVANCED_CONFIGS = {
    "CTGAN": {"epochs": 300, "batch_size": 500},
    "TVAE": {"epochs": 300, "batch_size": 256},
}


METRIC_HELP = {
    "utility": "Rates how similar the original and synthetic data are (0-1, higher is better).",
    "model": "Selected SDV model used to generate the synthetic data.",
    "rows": "Number of rows produced (original + additional rows).",
    "seed": "Random seed for reproducible results.",
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
    st.set_page_config(
        page_title="Synthetic Data Workbench", 
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🏠"
    )
    
    # Hide Streamlit's automatic page navigation using CSS and JavaScript
    # Also add styling to highlight the current page
    nav_css = """
    <style>
    /* Hide Streamlit's automatic navigation completely - multiple selectors for reliability */
    [data-testid="stSidebarNav"],
    [data-testid="stSidebarNav"] *,
    nav[data-testid="stSidebarNav"],
    nav[data-testid="stSidebarNav"] ul,
    nav[data-testid="stSidebarNav"] li,
    section[data-testid="stSidebarNav"],
    section[data-testid="stSidebarNav"] > * {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
        
    }
    /* Highlight the Home page link */
    div[data-testid="stSidebar"] a[href*="streamlit_app.py"] {
        background-color: rgba(38, 39, 48, 0.6);
        border-left: 3px solid #ff6b6b;
        padding-left: 1rem;
        border-radius: 0.25rem;
    }
    </style>
    <script>
    // Ensure navigation is hidden even if CSS doesn't catch it initially
    window.addEventListener('load', function() {
        const nav = document.querySelector('[data-testid="stSidebarNav"]');
        if (nav) {
            nav.style.display = 'none';
            nav.style.visibility = 'hidden';
            nav.style.height = '0';
            nav.style.width = '0';
        }
    });
    </script>
    """
    st.markdown(nav_css, unsafe_allow_html=True)
    
    # Add custom navigation links in sidebar
    st.sidebar.markdown("### Navigation")
    st.sidebar.page_link("streamlit_app.py", label="🏠 Home")
    st.sidebar.page_link("pages/Info.py", label="📚 Info & Documentation")
    st.sidebar.markdown("---")
    
    st.title("Synthetic Data Workbench")

    csv_files = list_csv_files()
    if not csv_files:
        st.warning("No CSV files found in 'data/'. Please add files and reload.")
        return

    st.sidebar.subheader("Data selection")
    uploaded_file = st.sidebar.file_uploader(
        "Upload or drop a CSV",
        type=["csv"],
        help="The uploaded file is stored in the `data/` folder.",
    )
    if uploaded_file is not None:
        target_name = Path(uploaded_file.name).name or "upload.csv"
        target_path = DATA_DIR / target_name
        if target_path.exists():
            timestamp = int(time.time())
            target_path = DATA_DIR / f"{target_path.stem}_{timestamp}{target_path.suffix}"
        target_path.write_bytes(uploaded_file.getbuffer())
        st.sidebar.success(f"File saved as `{target_path.name}`.")
        st.session_state["selected_path"] = target_path
        st.rerun()

    file_display = {file.name: file for file in csv_files}
    # Add placeholder option and check session state
    options = ["-- Select a CSV file --"] + list(file_display.keys())
    
    # Get current selection from session state or default to placeholder
    if "selected_path" in st.session_state:
        current_selection = st.session_state["selected_path"]
        # Find the index of the currently selected file
        try:
            selected_name = Path(current_selection).name
            if selected_name in file_display:
                default_index = list(file_display.keys()).index(selected_name) + 1
            else:
                default_index = 0
        except (ValueError, AttributeError):
            default_index = 0
    else:
        default_index = 0

    selected_name = st.sidebar.selectbox("Select CSV file", options=options, index=default_index)

    # If placeholder selected, show instructions and return early
    if selected_name == "-- Select a CSV file --":
        st.session_state.pop("selected_path", None)
        st.session_state.pop("synthetic_result", None)
        st.info("👈 Please select a CSV file from the sidebar to begin.")
        st.markdown("""
        ### Getting Started
        
        1. **Select a CSV file** from the dropdown in the sidebar
        2. **Explore** the dataset statistics and preview
        3. **Generate** synthetic data using one of the available models
        4. **Review** the quality metrics and download results
        
        Upload new files using the file uploader in the sidebar.
        """)
        return

    selected_path = file_display[selected_name]

    selected_mtime = selected_path.stat().st_mtime
    tracking = st.session_state.get("file_tracking")
    if tracking and tracking.get("path") == str(selected_path) and tracking.get("mtime") != selected_mtime:
        st.sidebar.warning("File modified outside the app—reloading data.")

    if st.session_state.get("selected_path") != selected_path:
        st.session_state.pop("synthetic_result", None)
        st.session_state["selected_path"] = selected_path

    st.session_state["file_tracking"] = {"path": str(selected_path), "mtime": selected_mtime}

    # Define containers for the new 4-step flow
    st.subheader("1. Explore the dataset")
    section1_content = st.empty()
    
    st.subheader("2. Prepare & Transform")
    section2_content = st.empty()

    st.subheader("3. Generate synthetic data")
    section3_content = st.empty()

    # Show loading states
    with section1_content.container():
        with st.spinner("Loading dataset and detecting metadata..."):
            st.info("🔄 Analyzing dataset...")

    # Load data
    df_original, metadata_original = load_dataset_cached(str(selected_path), selected_mtime)
    df_train = df_original.copy()
    metadata_train = metadata_original

    # ------------------------------------------------------------------
    # Step 1: Explore
    # ------------------------------------------------------------------
    with section1_content.container():
        st.write(f"**File:** `{selected_name}` | **Rows:** {len(df_original):,} | **Columns:** {len(df_original.columns)}")

        tab_overview, tab_numeric, tab_categorical = st.tabs([
            "Preview",
            "Numeric statistics",
            "Categorical values",
        ])

        with tab_overview:
            st.dataframe(df_original.head(), width="stretch")
            st.caption("First 5 rows of the original dataset")

        with tab_numeric:
            numeric_stats = describe_numeric(df_original)
            if numeric_stats.empty:
                st.info("No numeric columns found.")
            else:
                st.dataframe(numeric_stats, width="stretch")

        with tab_categorical:
            value_counts = top_value_counts(df_original)
            if not value_counts:
                st.info("No categorical columns found.")
            else:
                for column, counts in value_counts.items():
                    with st.expander(f"Top values for {column}"):
                        st.table(counts)

    # ------------------------------------------------------------------
    # Step 2: Prepare & Transform (US-1.1, US-2.x)
    # ------------------------------------------------------------------
    with section2_content.container():
        st.caption("Select columns to include in the synthetic data generation.")
        
        schema_data_key = f"schema_data_{selected_name}_{selected_mtime}"
        schema_widget_key = f"schema_widget_{selected_name}_{selected_mtime}"
        active_cols_key = f"active_columns_{selected_name}_{selected_mtime}"
        renaming_map_key = f"renaming_map_{selected_name}_{selected_mtime}"  # New key for renaming
        cleaned_df_key = f"cleaned_df_{selected_name}_{selected_mtime}"
        cleaned_meta_key = f"cleaned_meta_{selected_name}_{selected_mtime}"

        if schema_data_key not in st.session_state:
            st.session_state[schema_data_key] = pd.DataFrame({
                "Include": [True] * len(df_original.columns),
                "Column": df_original.columns,
                "Rename To": df_original.columns,  # Initialize with same names
                "Type": df_original.dtypes.astype(str).values
            })

        # Unified Form for Selection, Renaming, and Cleaning
        with st.form(key=f"config_form_{selected_name}_{selected_mtime}"):
            st.subheader("Feature Selection & Renaming")
            edited_schema = st.data_editor(
                st.session_state[schema_data_key],
                column_config={
                    "Include": st.column_config.CheckboxColumn(
                        "Train?",
                        help="Uncheck to exclude this column from synthesis",
                        default=True,
                    ),
                    "Column": st.column_config.TextColumn("Original Name", disabled=True),
                    "Rename To": st.column_config.TextColumn("Rename To (Optional)", help="Edit this to rename the feature for training"),
                    "Type": st.column_config.TextColumn("Detected Type", disabled=True),
                },
                disabled=["Column", "Type"],
                hide_index=True,
                use_container_width=True,
                key=schema_widget_key,
            )
            
            st.subheader("Data Cleaning")
            # Cleaning options (always visible but conditional on execution)
            # Restore previous strategy
            cleaning_strategy_key = f"cleaning_strategy_{selected_name}_{selected_mtime}"
            if cleaning_strategy_key not in st.session_state:
                st.session_state[cleaning_strategy_key] = "Keep (let SDV handle it)"
            
            options = [
                "Keep (let SDV handle it)",
                "Drop rows with missing values",
                "Fill with Mean/Mode",
            ]
            try:
                current_index = options.index(st.session_state[cleaning_strategy_key])
            except ValueError:
                current_index = 0

            nan_strategy_selection = st.radio(
                "Handle missing values (if detected):",
                options=options,
                index=current_index,
                horizontal=True,
                help="These rules are applied only if missing values (NaNs) are found in the selected columns."
            )
            
            confirm_config = st.form_submit_button("Confirm Configuration", type="primary")

        if confirm_config:
            # 1. Save Schema State
            st.session_state[schema_data_key] = edited_schema.copy()
            st.session_state[cleaning_strategy_key] = nan_strategy_selection
            
            # 2. Extract Selection & Renaming
            included_df = edited_schema[edited_schema["Include"]]
            candidate_cols = included_df["Column"].tolist()
            
            new_renaming_map = {}
            for _, row in included_df.iterrows():
                orig = row["Column"]
                new_name = row["Rename To"]
                if new_name and new_name.strip() and new_name != orig:
                    new_renaming_map[orig] = new_name.strip()

            if not candidate_cols:
                st.warning("⚠️ At least one column must be included. Configuration not updated.")
            else:
                # 3. Update Active Selection
                st.session_state[active_cols_key] = candidate_cols
                st.session_state[renaming_map_key] = new_renaming_map
                
                # Clear old cleaning results to ensure fresh processing
                st.session_state.pop(cleaned_df_key, None)
                st.session_state.pop(cleaned_meta_key, None)
                
                # 4. Create Draft & Rename
                temp_draft = df_original[candidate_cols].copy()
                if new_renaming_map:
                    temp_draft.rename(columns=new_renaming_map, inplace=True)
                
                # 5. Check & Apply Cleaning
                has_nans = temp_draft.isnull().values.any()
                final_df = temp_draft
                msg = f"✅ Configured: {len(candidate_cols)} columns selected"
                
                if new_renaming_map:
                    msg += f", {len(new_renaming_map)} renamed"
                
                if has_nans and nan_strategy_selection != "Keep (let SDV handle it)":
                    if nan_strategy_selection == "Drop rows with missing values":
                        before = len(final_df)
                        final_df = final_df.dropna()
                        dropped = before - len(final_df)
                        msg += f", {dropped} rows dropped (cleaning)"
                    elif nan_strategy_selection == "Fill with Mean/Mode":
                        filled_count = 0
                        for col in final_df.columns:
                            if final_df[col].isnull().any():
                                if final_df[col].dtype.kind in "biufc":
                                    final_df[col] = final_df[col].fillna(final_df[col].mean())
                                else:
                                    mode_val = final_df[col].mode()
                                    fill_val = mode_val[0] if not mode_val.empty else "Missing"
                                    final_df[col] = final_df[col].fillna(fill_val)
                                filled_count += 1
                        msg += f", {filled_count} cols filled (cleaning)"
                    
                    # Save Cleaned Data
                    from sdv.metadata import SingleTableMetadata
                    new_meta = SingleTableMetadata()
                    new_meta.detect_from_dataframe(final_df)
                    st.session_state[cleaned_df_key] = final_df
                    st.session_state[cleaned_meta_key] = new_meta
                    msg += "."
                elif has_nans:
                    msg += ". (NaNs preserved)."
                else:
                    msg += ". (No NaNs found)."

                st.success(msg)
                st.rerun()

        # --- Logic to Prepare Downstream Data (Read-Only View) ---
        if active_cols_key not in st.session_state:
            st.session_state[active_cols_key] = list(df_original.columns)
        if renaming_map_key not in st.session_state:
            st.session_state[renaming_map_key] = {}
            
        selected_cols = st.session_state[active_cols_key]
        renaming_map = st.session_state[renaming_map_key]
        
        # Reconstruct the effective training data from state
        # (Use cached cleaned data if available, else re-derive raw subset)
        if cleaned_df_key in st.session_state:
            df_train = st.session_state[cleaned_df_key]
            metadata_train = st.session_state[cleaned_meta_key]
            status_msg = "✅ **Ready for Training** (Cleaned)"
        else:
            # Re-derive raw
            df_train = df_original[selected_cols].copy()
            if renaming_map:
                df_train.rename(columns=renaming_map, inplace=True)
                
            if set(df_train.columns) == set(df_original.columns) and not renaming_map:
                metadata_train = metadata_original
            else:
                from sdv.metadata import SingleTableMetadata
                metadata_train = SingleTableMetadata()
                metadata_train.detect_from_dataframe(df_train)
            status_msg = "✅ **Ready for Training** (Raw)"

        st.markdown(f"{status_msg} | Rows: {len(df_train):,} | Columns: {len(df_train.columns)}")

    # ------------------------------------------------------------------
    # Step 3: Generate
    # ------------------------------------------------------------------
    with section3_content.container():
        col_left, col_right = st.columns(2)
        with col_left:
            model_name = st.selectbox("Synthesizer", options=list(SYNTHESIZER_REGISTRY.keys()), index=0)
            random_seed = int(
                st.number_input("Random Seed", value=42, step=1, help="For reproducible results.")
            )
        with col_right:
            additional_rows = st.number_input(
                "Additional rows",
                min_value=0,
                max_value=int(len(df_train) * 50) if len(df_train) else 10000,
                value=min(len(df_train), 1000),
                help="Number of additional synthetic rows compared to the original.",
            )
            suffix = st.text_input("File suffix", value=DEFAULT_SUFFIX, help="Suffix for the output (e.g. _synthetic)")

        init_kwargs: Dict[str, int] = {}
        model_defaults = MODEL_ADVANCED_CONFIGS.get(model_name)
        if model_defaults:
            with st.expander("Training settings (optional)", expanded=False):
                epochs_val = st.number_input(
                    "Epochs",
                    min_value=1,
                    max_value=5000,
                    value=model_defaults["epochs"],
                    step=10,
                    help="Number of training passes (higher = better quality, longer runtime).",
                )
                batch_size_val = st.number_input(
                    "Batch Size",
                    min_value=16,
                    max_value=4096,
                    value=model_defaults["batch_size"],
                    step=16,
                    help="Training batch size. Smaller values = more stable, longer runtime.",
                )
                init_kwargs["epochs"] = int(epochs_val)
                init_kwargs["batch_size"] = int(batch_size_val)

        additional_rows_int = int(additional_rows)
        total_rows = len(df_train) + additional_rows_int
        estimated_minutes = estimate_duration_minutes(
            rows=len(df_train),
            columns=len(df_train.columns),
            model_name=model_name,
            additional_rows=additional_rows_int,
        )
        duration_label = format_duration_label(estimated_minutes)

        st.markdown(
            f"*Total synthetic rows*: **{total_rows:,}** (Original: {len(df_train):,} + Extra: {additional_rows_int:,})"
        )
        st.caption(
            "Estimated duration: "
            f"{duration_label} (heuristic based on dataset size and model)"
        )

        if st.button("Generate", type="primary"):
            progress_placeholder = st.empty()

            progress_bar = progress_placeholder.progress(
                0,
                text=f"Training started (≈ {duration_label})",
            )

            def progress_callback(fraction: float, status_text: str) -> None:
                progress_bar.progress(min(fraction, 0.999), text=status_text)

            try:
                runner_result = run_training(
                    df_real=df_train,
                    metadata=metadata_train,
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
                    df_train,
                    df_synth,
                    metadata,
                )
                suffix_normalized = ensure_suffix(suffix)
                output_path = selected_path.with_name(f"{selected_path.stem}{suffix_normalized}.csv")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df_synth.to_csv(output_path, index=False)

                base_hints: List[str] = [f"Training finished in {actual_duration_label}."]
                if init_kwargs:
                    formatted = ", ".join(f"{key}={value}" for key, value in init_kwargs.items())
                    base_hints.append(f"Training parameters: {formatted}")
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
                        "Note: the selected model does not support `random_state`. "
                        "Global random generators were seeded with the provided value."
                    )
                    st.session_state["synthetic_result"].hints.append(
                        "Global random generators seeded with the provided value (model without random_state)."
                    )
                if report_warnings:
                    st.warning("\n".join(report_warnings))
                st.success(f"Synthetic dataset saved as `{output_path.name}`")
                st.rerun()
            except Exception as exc:  # pragma: no cover - Streamlit handles display
                progress_placeholder.empty()
                st.error(f"Error during generation: {exc}")
                hints = error_suggestions(str(exc))
                if hints:
                    st.info("Tips:\n- " + "\n- ".join(hints))

    # ------------------------------------------------------------------
    # Step 4: Evaluate (Existing Logic)
    # ------------------------------------------------------------------
    result: Optional[SyntheticResult] = st.session_state.get("synthetic_result")
    if result:
        st.subheader("4. Results & validation")

        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Utility Score (0-1)",
            f"{result.utility_score:.3f}",
            help=METRIC_HELP["utility"],
        )
        metric_cols[1].metric(
            "Model",
            result.model_name,
            help=METRIC_HELP["model"],
        )
        metric_cols[2].metric(
            "Rows (synthetic)",
            f"{result.total_rows:,}",
            help=METRIC_HELP["rows"],
        )
        metric_cols[3].metric(
            "Seed",
            str(result.random_seed),
            help=METRIC_HELP["seed"],
        )

        status_text = (
            f"Model: {result.model_name}\n"
            f"Seed: {result.random_seed}\n"
            f"Total rows: {result.total_rows:,} (Additional: {result.additional_rows:,})\n"
            f"Utility score: {result.utility_score:.3f}\n"
            f"Estimated duration: {result.duration_label}\n"
            f"Actual duration: {result.actual_duration_label}"
        )
        st.write("**Summary**")
        st.code(status_text, language="text")
        st.caption(interpret_utility_score(result.utility_score))
        if st.button("Copy summary"):
            st.query_params.update({"summary": status_text})
            st.success("Summary provided via query parameter.")

        hints = result.hints
        st.write("**Notes**")
        if hints:
            hints_text = "\n".join(f"- {hint}" for hint in hints)
            st.write(hints_text)
            if st.button("Copy notes"):
                st.query_params.update({"hints": hints_text})
                st.success("Notes provided via query parameter.")
        else:
            st.caption("No notes available.")

        preview_tab, compare_tab = st.tabs(["Synthetic preview", "Comparison"])
        with preview_tab:
            st.dataframe(result.dataframe.head(), width="stretch")
            st.caption("First 5 rows of the synthetic dataset")

        with compare_tab:
            synth_numeric = describe_numeric(result.dataframe)
            train_numeric = describe_numeric(df_train)
            if not synth_numeric.empty and not train_numeric.empty:
                combined = (
                    train_numeric
                    .add_suffix("_original")
                    .join(synth_numeric.add_suffix("_synthetic"), how="outer")
                )
                st.dataframe(combined, width="stretch")
            else:
                st.info("Numeric metrics could not be compared.")

        with st.expander("Detailed quality metrics"):
            details = result.report_details
            if details.empty:
                st.info("No detailed metrics available.")
            elif "property" not in details.columns:
                st.dataframe(details, width="stretch")
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
                        st.dataframe(subset, width="stretch")
                        st.caption(
                            "Metrics for "
                            f"{property_labels.get(category, category)}"
                        )

        csv_bytes = result.dataframe.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download synthetic dataset",
            data=csv_bytes,
            file_name=result.output_path.name,
            mime="text/csv",
        )

        st.caption(f"File saved to `{result.output_path}`")


if __name__ == "__main__":
    app()
