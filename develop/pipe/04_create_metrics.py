import ast
import os
import spacy
from typing import Optional
import pandas as pd
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import warnings

from develop.core.metrics.cosine_distance import CosineMetric
from develop.core.vectors_flags.norm import EmbeddingNormComputer
from develop.utils.toolbox import reorder_columns
from develop.utils.logger import LoggerManager
warnings.filterwarnings("ignore")

from develop.utils.paths import DATA, MODEL

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
      
    norm = EmbeddingNormComputer()
    df = norm.compute(df[:])
    
    df_path = os.path.join(output_dir, f"df_{start_year}-{end_year}.parquet")
    df['most_similar_cosine'] = df['most_similar_cosine'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else None
    ) # sanitize most_similar_cosine column

    df.to_parquet(df_path, index=False)
    logger.info(f"Saved: {df_path}")


if __name__ == "__main__":
    output_dir = os.path.join(DATA, "04_create_metrics")
    start_year = 2000
    end_year = 2024

    process_year_range(start_year, end_year, output_dir, MODEL)