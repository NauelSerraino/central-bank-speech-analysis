import logging
import numpy as np
import pandas as pd


class EmbeddingNormComputer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Computing norm...")
        df = df.copy()
        df["norm"] = df["embedding"].apply(
            lambda v:  round(np.linalg.norm(v), 4) if isinstance(v, (list, np.ndarray)) else np.nan
        )
        df["norm"] = df["norm"].round(4)
        return df