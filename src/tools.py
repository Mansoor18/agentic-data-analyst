import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import os
from langchain_core.tools import tool

_dataframe = None

def set_dataframe(df):
    global _dataframe
    _dataframe = df

def get_dataframe():
    return _dataframe

@tool
def get_dataset_info() -> str:
    """Get basic information about the loaded dataset including shape, columns, dtypes and missing values. Call this first."""
    df = get_dataframe()
    if df is None:
        return "No dataset loaded."
    info = []
    info.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    info.append(f"\nColumns and dtypes:")
    for col, dtype in df.dtypes.items():
        nulls = df[col].isnull().sum()
        info.append(f"  - {col}: {dtype} ({nulls} nulls)")
    info.append(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    return "\n".join(info)

@tool
def run_python_code(code: str) -> str:
    """Execute Python code for data analysis. The dataframe is available as 'df'. Always print results."""
    df = get_dataframe()
    if df is None:
        return "No dataset loaded."
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        exec_globals = {
            "df": df, "pd": pd, "np": np,
            "plt": plt, "sns": sns, "os": os
        }
        exec(code, exec_globals)
        output = buffer.getvalue()
        return output if output else "Code executed successfully with no output."
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        sys.stdout = old_stdout

@tool
def get_summary_statistics() -> str:
    """Get summary statistics for all numeric columns in the dataset."""
    df = get_dataframe()
    if df is None:
        return "No dataset loaded."
    return df.describe(include='all').to_string()

@tool
def get_correlation_matrix() -> str:
    """Calculate and return the correlation matrix for numeric columns."""
    df = get_dataframe()
    if df is None:
        return "No dataset loaded."
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return "No numeric columns found."
    return numeric_df.corr().to_string()

@tool
def get_value_counts(column: str) -> str:
    """Get value counts for a specific column. Pass the exact column name as a string."""
    df = get_dataframe()
    if df is None:
        return "No dataset loaded."
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {list(df.columns)}"
    counts = df[column].value_counts()
    pct = df[column].value_counts(normalize=True) * 100
    result = pd.DataFrame({'count': counts, 'percentage': pct.round(2)})
    return result.to_string()