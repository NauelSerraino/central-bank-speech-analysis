import pandas as pd
from develop.core.fixed_chain.base import TemporalDriftMetric
from develop.core.metrics.base import BaseMetric

class JaccardDrift(TemporalDriftMetric):
    def __init__(self, metric: BaseMetric, top_k: int = 10):
        self.top_k = top_k
        self.metric = metric  

    def compute(self, df: pd.DataFrame, mode: str = "chain") -> pd.DataFrame:
        """
        Adds 'jaccard_score' and 'year_t-1' to the original DataFrame for each word/year.
        Assumes column f"most_similar_{self.metric.metric_type.value}" exists and contains list of tuples.
        """
        df = df.copy()
        jaccard_scores = []
        prev_years = []

        for word, group in df.groupby("word"):
            group_sorted = group.sort_values("year")
            neighbors = (
                group_sorted[f"most_similar_{self.metric.metric_type.value}"]
                .apply(lambda lst: lst if isinstance(lst, list) else [])  # fill NA
                .apply(lambda lst: [w for w, _ in lst[:self.top_k]])
                .tolist()
            )
            years = group_sorted["year"].tolist()
            base = set(neighbors[0])

            scores = [None]  # first year has no comparison
            prevs = [None]

            for i in range(1, len(neighbors)):
                current = set(neighbors[i])
                previous = set(neighbors[i - 1]) if mode == "chain" else base

                intersection = len(previous & current)
                union = len(previous | current)
                jaccard = intersection / union if union > 0 else 0.0

                scores.append(jaccard)
                prevs.append(years[i - 1] if mode == "chain" else years[0])

            df.loc[group_sorted.index, f"jaccard_score_{mode}"] = scores
            df.loc[group_sorted.index, "year_t-1"] = prevs

        return df
