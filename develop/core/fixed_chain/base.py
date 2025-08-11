from abc import ABC, abstractmethod

import pandas as pd

from develop.core.metrics.base import BaseMetric

class TemporalDriftMetric(ABC):
    def __init__(self, metric: BaseMetric):
        self.metric = metric
        
    @abstractmethod
    def compute(self, df: pd.DataFrame, mode: str = "chain",) -> pd.DataFrame:
        pass
