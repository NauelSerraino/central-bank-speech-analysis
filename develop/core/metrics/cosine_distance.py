import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from develop.core.metrics.base import BaseMetric
from develop.core.metrics.metric_type import MetricType


class CosineMetric(BaseMetric):
    @property
    def metric_type(self) -> MetricType:
        return MetricType.COSINE
    
    def pairwise(self, batch_emb: np.ndarray, full_emb: np.ndarray) -> np.ndarray:
        cosine_dist = 1 - cosine_similarity(batch_emb, full_emb)
        return cosine_dist
