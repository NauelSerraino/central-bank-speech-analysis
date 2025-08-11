import numpy as np
from sklearn.metrics import euclidean_distances

from develop.core.metrics.base import BaseMetric
from develop.core.metrics.metric_type import MetricType


class EuclideanMetric(BaseMetric):
    @property
    def metric_type(self) -> MetricType:
        return MetricType.EUCLIDEAN
    
    def pairwise(self, batch_emb: np.ndarray, full_emb: np.ndarray) -> np.ndarray:
        dists = euclidean_distances(batch_emb, full_emb)
        return dists