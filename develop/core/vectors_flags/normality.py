import logging
import numpy as np
import pandas as pd
from scipy.stats import shapiro


class EmbeddingNormalityChecker:
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    def check(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'p_value' and 'is_normal' columns to the original DataFrame.

        Assumes:
        - 'embedding' column exists and contains list or np.array
        - Index includes 'word' and 'year' or they are columns
        """
        logging.info("Checking for normality...")
        df = df.copy()

        p_values = []
        is_normals = []

        for vec in df["embedding"]:
            if isinstance(vec, float) and np.isnan(vec):
                p = np.nan
                is_normal = False
            else:
                try:
                    p = shapiro(vec).pvalue
                    p = round(p, 4) 
                    is_normal = p >= self.alpha
                except Exception:
                    p = np.nan
                    is_normal = False

            p_values.append(p)
            is_normals.append(is_normal)

        df["p_value"] = p_values
        df["is_normal"] = is_normals

        return df

