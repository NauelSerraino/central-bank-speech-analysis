import numpy as np
from scipy.stats import kendalltau
from develop.core.metrics.base import BaseMetric
from develop.core.metrics.metric_type import MetricType


class KendallTauMetric(BaseMetric):
    @property
    def metric_type(self) -> MetricType:
        return MetricType.KENDALLTAU  # you need to add this to MetricType enum

    def pairwise(self, batch_emb: np.ndarray, full_emb: np.ndarray) -> np.ndarray:
        # Number of vectors
        n_batch = batch_emb.shape[0]
        n_full = full_emb.shape[0]
        dist_mat = np.zeros((n_batch, n_full))
        
        # Precompute ranks for full_emb
        full_ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 1, full_emb)

        # Compute pairwise Kendall Tau distance = 1 - correlation
        for i in range(n_batch):
            batch_rank = np.argsort(np.argsort(batch_emb[i]))
            for j in range(n_full):
                tau, _ = kendalltau(batch_rank, full_ranks[j])
                dist_mat[i, j] = 1 - tau if not np.isnan(tau) else 1.0  # treat NaN as max distance

        return dist_mat
