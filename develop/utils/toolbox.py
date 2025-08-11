import os

def reorder_columns(df, cols_first):
    cols_existing = [c for c in cols_first if c in df.columns]
    cols_rest = [c for c in df.columns if c not in cols_existing]
    return df[cols_existing + cols_rest]