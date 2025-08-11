from asyncio import sleep
import os
from sklearn.preprocessing import MinMaxScaler
import spacy
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from scipy import stats
from tqdm import tqdm
from scipy.stats import bernoulli
from scipy.stats import entropy
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import warnings

import sys

from develop.core.fixed_chain.distance_measures import EmbeddingDrift
from develop.core.fixed_chain.factory import drift_factory
from develop.core.fixed_chain.neighboor_measures import JaccardDrift
from develop.core.metrics.cosine_distance import CosineMetric
from develop.core.metrics.euclidean import EuclideanMetric
from develop.core.metrics.metric_type import MetricType
from develop.core.metrics.wrd import WRDMetric
from develop.core.region_metrics.metrics import SingletonMetricWrapper
from develop.core.vectors_flags.norm import EmbeddingNormComputer
from develop.core.vectors_flags.normality import EmbeddingNormalityChecker
from develop.utils.toolbox import reorder_columns
from develop.utils.logger import LoggerManager
warnings.filterwarnings("ignore")

import sys
from develop.utils.paths import DATA_ALT, MODEL, MODEL_USA, MODEL_EU

nlp = spacy.load("en_core_web_sm")

log_mgr = LoggerManager(
    name = "create_metrics", 
    log_file = "create_metrics.log",
    clear_log = True
    )
logger = log_mgr.get_logger()

def load_model(
    MODEL: str,
    region: Optional[str] = None, 
    year: Optional[int] = None,
    compass=False,
    ):
    if compass:
        logger.info("Loading compass model")
        return Word2Vec.load(os.path.join(MODEL, "compass.model"))
    else:
        return Word2Vec.load(os.path.join(MODEL, f"{year}_{region}.model"))
    

def concat_dfs(dict_dfs):
    df_out = pd.concat(dict_dfs.values())
    df_out = reorder_columns(df_out, ['year', 'count'])
    df_out = df_out.reset_index().rename(columns={'index':'word'})
    return df_out

    
def process_year_range(start_year, end_year, output_dir, input_dir):
    
    years = list(range(start_year, end_year + 1))
    dfs_dict = {}
    compass = load_model(input_dir, compass=True)
    compass_vocabulary = compass.wv.index_to_key
    
    for region in ['EU', 'USA']:
        logger.info(f"Processing {region} years {start_year}-{end_year}...")
        for year in tqdm(years, desc=f"Loading and word count {start_year}-{end_year} {region}"):
            model = load_model(input_dir, region=region, year=year)
            cosine_metric = CosineMetric()
            dfs_dict[(region, year)] = cosine_metric.create_df_words_embeddings(
            model, year, region=region, compass_vocabulary=compass_vocabulary)

    df = concat_dfs(dfs_dict)
    
    normality = EmbeddingNormalityChecker()
    df = normality.check(df)
    
    norm = EmbeddingNormComputer()
    df = norm.compute(df)
    
    metric_wrapper = SingletonMetricWrapper(metric=CosineMetric())
    metric_wrapper.load_embeddings(df)
    df = metric_wrapper.compute_distances_for_df(df)

    # metric = drift_factory("embedding", MetricType.COSINE.value)
    # df_chain = metric.compute(df, mode="chain")
    # df_fixed = metric.compute(df, mode="fixed")

    # jaccard = drift_factory("jaccard", MetricType.COSINE.value, top_k=10)
    # df_chain = jaccard.compute(df_chain, mode="chain")
    # df_fixed = jaccard.compute(df_fixed, mode="fixed")
    
    os.makedirs(output_dir, exist_ok=True)
    df_path = os.path.join(output_dir, f"df_{start_year}-{end_year}.csv")
    df.to_csv(df_path, index=False, sep='|')
    logger.info(f"Saved: {df_path}")


if __name__ == "__main__":
    output_dir = os.path.join(DATA_ALT, "04_create_metrics")
    start_year = 1998
    end_year = 2024

    process_year_range(start_year, end_year, output_dir, MODEL)