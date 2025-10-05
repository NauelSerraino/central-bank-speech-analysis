import numpy as np
from scipy.stats import spearmanr
from develop.core.metrics.base import BaseMetric
from develop.core.metrics.metric_type import MetricType

class SpearmanMetric(BaseMetric):
    @property
    def metric_type(self) -> MetricType:
        return MetricType.SPEARMAN  # add this to your enum

    def pairwise(self, batch_emb: np.ndarray, full_emb: np.ndarray) -> np.ndarray:
        n_batch = batch_emb.shape[0]
        n_full = full_emb.shape[0]
        dist_mat = np.zeros((n_batch, n_full))
        
        # Precompute ranks for full_emb
        full_ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 1, full_emb)
        
        for i in range(n_batch):
            batch_rank = np.argsort(np.argsort(batch_emb[i]))
            for j in range(n_full):
                corr, _ = spearmanr(batch_rank, full_ranks[j])
                dist_mat[i, j] = 1 - corr if not np.isnan(corr) else 1.0
        
        return dist_mat
