from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np


@dataclass
class ProcessingResult:
    """Result of the data processing pipeline."""
    dataframe: pd.DataFrame
    metadata: object  # SingleTableMetadata
    summary: List[str]
    renaming_map: Dict[str, str]


@dataclass
class ScalingRule:
    """Rule for scaling a numeric column."""
    column: str
    target_min: float
    target_max: float


def process_data(
    df: pd.DataFrame,
    selected_columns: List[str],
    renaming_map: Optional[Dict[str, str]] = None,
    cleaning_strategy: str = "Keep (let SDV handle it)",
    scaling_rules: Optional[List[ScalingRule]] = None,
) -> ProcessingResult:
    """
    Process the dataframe by selecting columns, renaming, cleaning, and scaling.

    Args:
        df: Input dataframe.
        selected_columns: List of columns to include.
        renaming_map: Dictionary mapping original column names to new names.
        cleaning_strategy: Strategy for handling missing values.
                           Options: "Keep (let SDV handle it)", "Drop rows with missing values", "Fill with Mean/Mode".
        scaling_rules: List of scaling rules to apply.

    Returns:
        ProcessingResult containing the processed dataframe, metadata, and summary of changes.
    """
    # Lazy import to avoid heavy dependencies at module level if not needed immediately
    from sdv.metadata import SingleTableMetadata

    if renaming_map is None:
        renaming_map = {}
    if scaling_rules is None:
        scaling_rules = []

    # 1. Select Columns
    # Create a copy to avoid modifying the original
    final_df = df[selected_columns].copy()
    
    msg_parts = [f"✅ Configured: {len(selected_columns)} columns selected"]

    # 2. Rename Columns
    if renaming_map:
        final_df.rename(columns=renaming_map, inplace=True)
        msg_parts.append(f"{len(renaming_map)} renamed")

    # 3. Data Cleaning
    has_nans = final_df.isnull().values.any()
    
    if has_nans and cleaning_strategy != "Keep (let SDV handle it)":
        if cleaning_strategy == "Drop rows with missing values":
            before = len(final_df)
            final_df = final_df.dropna()
            dropped = before - len(final_df)
            msg_parts.append(f"{dropped} rows dropped (cleaning)")
        elif cleaning_strategy == "Fill with Mean/Mode":
            filled_count = 0
            for col in final_df.columns:
                if final_df[col].isnull().any():
                    if final_df[col].dtype.kind in "biufc":  # Numeric
                        final_df[col] = final_df[col].fillna(final_df[col].mean())
                    else:
                        mode_val = final_df[col].mode()
                        fill_val = mode_val[0] if not mode_val.empty else "Missing"
                        final_df[col] = final_df[col].fillna(fill_val)
                    filled_count += 1
            msg_parts.append(f"{filled_count} cols filled (cleaning)")

    # 4. Value Scaling
    scaled_count = 0
    if scaling_rules:
        for rule in scaling_rules:
            # Map back to the *renamed* column name if applicable
            # The rule.column comes from the UI which might be the original name or tracked name.
            # Based on previous logic, the UI passed original names in the rule, but we need to check.
            # The UI logic was: target_col_name = new_renaming_map.get(orig_col_name, orig_col_name)
            # We will assume the rule.column is the ORIGINAL column name, as that's how the UI seemed to structure it.
            
            target_col_name = renaming_map.get(rule.column, rule.column)
            
            if target_col_name in final_df.columns:
                t_min = rule.target_min
                t_max = rule.target_max
                
                if t_min >= t_max:
                    # Skip invalid rules
                    continue

                col_data = final_df[target_col_name]
                
                # Skip if non-numeric
                if not pd.api.types.is_numeric_dtype(col_data):
                    continue
                    
                c_min = col_data.min()
                c_max = col_data.max()
                
                if pd.isna(c_min) or pd.isna(c_max):
                    # Skip if column is all NaNs (and wasn't filled)
                    continue

                if c_min == c_max:
                    # Constant column
                    final_df[target_col_name] = t_min
                else:
                    # Apply Formula: (X - min) / (max - min) * (t_max - t_min) + t_min
                    scale_factor = (t_max - t_min) / (c_max - c_min)
                    final_df[target_col_name] = (col_data - c_min) * scale_factor + t_min
                
                scaled_count += 1
        
        if scaled_count > 0:
            msg_parts.append(f"{scaled_count} cols scaled")

    # 5. Detect Metadata
    new_meta = SingleTableMetadata()
    new_meta.detect_from_dataframe(final_df)

    return ProcessingResult(
        dataframe=final_df,
        metadata=new_meta,
        summary=msg_parts,
        renaming_map=renaming_map
    )
