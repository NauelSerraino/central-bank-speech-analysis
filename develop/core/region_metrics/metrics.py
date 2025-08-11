import logging
import numpy as np
import pandas as pd

class SingletonMetricWrapper:
    _instance = None

    def __new__(cls, metric=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.eu_embeddings = {}
            cls._instance.usa_embeddings = {}
            cls._instance.metric = metric
        return cls._instance

    def load_embeddings(self, df: pd.DataFrame):
        self.eu_embeddings = {
            (row.word, row.year): row.embedding 
            for _, row in df[df.region == 'EU'].iterrows()
        }
        self.usa_embeddings = {
            (row.word, row.year): row.embedding 
            for _, row in df[df.region == 'USA'].iterrows()
        }


    def compute_distances_for_df(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Computing USA-EU semantic differences...")
        if self.metric is None:
            raise ValueError("Metric not set during initialization.")

        def compute_distance(row):
            key = (row['word'], row['year'])
            emb_eu = self.eu_embeddings.get(key)
            emb_usa = self.usa_embeddings.get(key)
            if emb_eu is not None and emb_usa is not None:
                emb_eu_2d = emb_eu.reshape(1, -1)
                emb_usa_2d = emb_usa.reshape(1, -1)
                dist = self.metric.pairwise(emb_eu_2d, emb_usa_2d)[0, 0]
                return round(dist, 4) 
            else:
                return np.nan

        col_name = f'distance_{self.metric.metric_type.value}'
        df[col_name] = df.apply(compute_distance, axis=1).round(4)
        return df
