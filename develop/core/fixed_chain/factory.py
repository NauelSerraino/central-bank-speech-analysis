from develop.core.fixed_chain.distance_measures import EmbeddingDrift
from develop.core.fixed_chain.neighboor_measures import JaccardDrift
from develop.core.metrics.pearson import PearsonMetric
from develop.core.metrics.cosine_distance import CosineMetric

METRIC_MAP = {
    "cosine": CosineMetric,
    "pearson": PearsonMetric,
    # add others here
}

DRIFT_MAP = {
    "embedding": EmbeddingDrift,
    "jaccard": JaccardDrift,
}

def drift_factory(drift_name: str, metric_name: str, **kwargs):
    if drift_name not in DRIFT_MAP:
        raise ValueError(f"Unknown drift type {drift_name}")
    if metric_name not in METRIC_MAP:
        raise ValueError(f"Unknown metric type {metric_name}")

    metric_instance = METRIC_MAP[metric_name]()
    drift_class = DRIFT_MAP[drift_name]

    return drift_class(metric=metric_instance, **kwargs)
