import pytest
import pandas as pd
import numpy as np
from data_processor import process_data, ScalingRule

def test_process_data_selection_and_renaming():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    })
    
    selected = ["A", "B"]
    renaming = {"A": "Alpha"}
    
    result = process_data(df, selected, renaming_map=renaming)
    
    assert list(result.dataframe.columns) == ["Alpha", "B"]
    assert len(result.dataframe) == 3
    assert "C" not in result.dataframe.columns

def test_process_data_cleaning_drop():
    df = pd.DataFrame({
        "A": [1, np.nan, 3],
        "B": [4, 5, 6]
    })
    
    result = process_data(
        df, 
        ["A", "B"], 
        cleaning_strategy="Drop rows with missing values"
    )
    
    assert len(result.dataframe) == 2
    assert result.dataframe["A"].isnull().sum() == 0

def test_process_data_cleaning_fill():
    df = pd.DataFrame({
        "A": [1, np.nan, 3],
        "B": ["x", np.nan, "x"]
    })
    
    result = process_data(
        df, 
        ["A", "B"], 
        cleaning_strategy="Fill with Mean/Mode"
    )
    
    assert len(result.dataframe) == 3
    assert result.dataframe["A"].isnull().sum() == 0
    assert result.dataframe["A"].iloc[1] == 2.0  # Mean of 1 and 3
    assert result.dataframe["B"].iloc[1] == "x"  # Mode

def test_process_data_scaling():
    df = pd.DataFrame({
        "A": [0, 50, 100],
        "B": [10, 20, 30]
    })
    
    rules = [
        ScalingRule(column="A", target_min=0, target_max=1),
        ScalingRule(column="B", target_min=0, target_max=100)
    ]
    
    result = process_data(
        df, 
        ["A", "B"], 
        scaling_rules=rules
    )
    
    # Check A: 0->0, 50->0.5, 100->1
    assert result.dataframe["A"].iloc[0] == 0.0
    assert result.dataframe["A"].iloc[1] == 0.5
    assert result.dataframe["A"].iloc[2] == 1.0
    
    # Check B: 10->0, 20->50, 30->100
    assert result.dataframe["B"].iloc[0] == 0.0
    assert result.dataframe["B"].iloc[1] == 50.0
    assert result.dataframe["B"].iloc[2] == 100.0

def test_process_data_scaling_with_renaming():
    df = pd.DataFrame({
        "A": [0, 100]
    })
    
    # Rename A -> Alpha
    renaming = {"A": "Alpha"}
    # Rule applies to original name "A"
    rules = [ScalingRule(column="A", target_min=0, target_max=1)]
    
    result = process_data(
        df, 
        ["A"], 
        renaming_map=renaming,
        scaling_rules=rules
    )
    
    assert "Alpha" in result.dataframe.columns
    assert result.dataframe["Alpha"].max() == 1.0

def test_process_data_scaling_constant_column():
    df = pd.DataFrame({
        "A": [5, 5, 5]
    })
    
    rules = [ScalingRule(column="A", target_min=0, target_max=1)]
    
    result = process_data(df, ["A"], scaling_rules=rules)
    
    # Should be set to target_min
    assert (result.dataframe["A"] == 0).all()

def test_process_data_scaling_invalid_range():
    df = pd.DataFrame({"A": [1, 2, 3]})
    # Min > Max
    rules = [ScalingRule(column="A", target_min=10, target_max=0)]
    
    result = process_data(df, ["A"], scaling_rules=rules)
    
    # Should be unchanged
    assert result.dataframe["A"].iloc[0] == 1
