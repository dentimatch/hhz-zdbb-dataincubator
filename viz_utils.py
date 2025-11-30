"""
Shared helper functions for plotting and visualization.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

def plot_numeric_distribution(df_real: pd.DataFrame, df_synth: pd.DataFrame, column: str):
    """Plots overlaid histograms for a numeric column."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    sns.histplot(
        df_real[column], 
        label="Real", 
        color="blue", 
        kde=True, 
        stat="density", 
        element="step", 
        alpha=0.3, 
        ax=ax
    )
    sns.histplot(
        df_synth[column], 
        label="Synthetic", 
        color="red", 
        kde=True, 
        stat="density", 
        element="step", 
        alpha=0.3, 
        ax=ax
    )
    
    ax.set_title(f"Distribution: {column}")
    ax.legend()
    return fig

def plot_categorical_distribution(df_real: pd.DataFrame, df_synth: pd.DataFrame, column: str):
    """Plots side-by-side bar charts for a categorical column."""
    # Compute value counts as percentages
    real_counts = df_real[column].value_counts(normalize=True).reset_index()
    real_counts.columns = ["category", "proportion"]
    real_counts["dataset"] = "Real"
    
    synth_counts = df_synth[column].value_counts(normalize=True).reset_index()
    synth_counts.columns = ["category", "proportion"]
    synth_counts["dataset"] = "Synthetic"
    
    combined = pd.concat([real_counts, synth_counts])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=combined,
        x="category",
        y="proportion",
        hue="dataset",
        palette={"Real": "blue", "Synthetic": "red"},
        alpha=0.7,
        ax=ax
    )
    
    ax.set_title(f"Distribution: {column}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, title: str):
    """Plots a correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return None
        
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title(title)
    return fig



