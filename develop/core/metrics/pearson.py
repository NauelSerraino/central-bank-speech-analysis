import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from develop.core.metrics.base import BaseMetric
from develop.core.metrics.metric_type import MetricType


class PearsonMetric(BaseMetric):
    @property
    def metric_type(self) -> MetricType:
        return MetricType.PEARSON
    
    def pairwise(self, batch_emb: np.ndarray, full_emb: np.ndarray) -> np.ndarray:
        batch_centered = batch_emb - batch_emb.mean(axis=1, keepdims=True)
        full_centered = full_emb - full_emb.mean(axis=1, keepdims=True)
        pearson_sim = cosine_similarity(batch_centered, full_centered)
        pearson_dist = 1 - pearson_sim
        return pearson_dist