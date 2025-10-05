import numpy as np

from develop.core.metrics.base import BaseMetric


class DotProductMetric(BaseMetric):
    def pairwise(self, batch_emb: np.ndarray, full_emb: np.ndarray) -> np.ndarray:
        """
        Compute dot product similarities between batch_emb and full_emb.
        """
        return batch_emb @ full_emb.T