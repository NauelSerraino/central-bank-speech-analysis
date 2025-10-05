import numpy as np
import pandas as pd
from sklearn.metrics import euclidean_distances

from develop.core.fixed_chain.base import TemporalDriftMetric
from develop.core.metrics.base import BaseMetric

class EmbeddingDrift(TemporalDriftMetric):
    
    def compute(self, df: pd.DataFrame, mode: str = "chain") -> pd.DataFrame:
        """
        Adds columns:
        - f"score_{metric_type}" : float
        - "year_t-1" : int or None
        to the original DataFrame, aligned by word and year.
        """
        df = df.copy()
        df = df.reset_index().rename(columns={'index':'word'})
        score_col = f"score_{self.metric.metric_type.value}_{mode}"
        df[score_col] = None
        df["year_t-1"] = None

        for word, group in df.groupby("word"):
            group_sorted = group.sort_values("year")
            valid_mask = group_sorted["embedding"].apply(
                lambda x: isinstance(x, np.ndarray) and not np.isnan(x).any()
            )

            if valid_mask.sum() < 2:
                continue

            group_valid = group_sorted[valid_mask]
            emb = np.stack(group_valid["embedding"].values)
            years = group_valid["year"].values

            base = emb[0]
            scores = [None]  # first has no comparison
            prev_years = [None]

            for i in range(1, len(emb)):
                prev = emb[i - 1] if mode == "chain" else base
                curr = emb[i]
                score = self.metric.compute_distance(prev, curr)

                scores.append(score)
                prev_years.append(years[i - 1] if mode == "chain" else years[0])

            df.loc[group_valid.index, score_col] = scores
            df.loc[group_valid.index, "year_t-1"] = prev_years

        return df
