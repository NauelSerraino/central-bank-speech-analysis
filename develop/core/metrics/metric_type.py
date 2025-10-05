from enum import Enum

class MetricType(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    PEARSON = "pearson"
    KENDALLTAU = "kendal"
    SPEARMAN = "spearman"