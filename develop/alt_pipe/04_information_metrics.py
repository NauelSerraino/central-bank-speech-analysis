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
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, "../.."))))
from develop.utils.paths import DATA_ALT, MODEL

nlp = spacy.load("en_core_web_sm")

def load_model(
    year: Optional[int] = None, 
    compass=False
    ):
    if compass:
        print("Loading compass model")
        return Word2Vec.load(os.path.join(MODEL, "compass.model"))
    else:
        return Word2Vec.load(os.path.join(MODEL, f"{year}.model"))
    
def create_df_words_embeddings(m):
    emb = m.wv.vectors
    w = m.wv.index2word
    df = pd.DataFrame(emb, index=w)
    return df

def merge_dfs(dict_dfs):
    
    for k, v in dict_dfs.items():
        v['year'] = k
        cols = ['year'] + [col for col in v.columns if col != 'year']  
        dict_dfs[k] = v[cols]  
    
    df_out = pd.concat(dict_dfs.values())
    df_out = df_out[df_out.index.notnull()]
    return df_out

def assign_count_per_each_word(df, model):
    df['count'] = df.index.map(lambda x: model.wv.vocab[x].count)
    return df

def run_divergences_pipe(df):
    
    def custom_merge_dfs(group_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
        """Generate the merged df of group and reference.
        It does also create an embedding column and a normalized version of it.

        Args:
            group_df (pd.DataFrame): Group DataFrame.
            ref_df (pd.DataFrame): Reference DataFrame.

        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        breakpoint()
        year = group_df.year.unique()[0]
        ref_year = ref_df.year.unique()[0]
        scaler = MinMaxScaler()
        group_cols = [
            'embedding', 
            'embedding_min_max_norm', 
            'count'
            ]
        ref_cols = [
            'embedding_t-1',
            'embedding_t-1_min_max_norm',
            'count'
            ]
        
        group_df['embedding'] = group_df.iloc[:, 1:-1].values.tolist()
        group_df['embedding_min_max_norm'] = scaler.fit_transform(group_df.iloc[:, 1:-1]).tolist()
        ref_df['embedding_t-1'] = ref_df.iloc[:, 1:-1].values.tolist()
        ref_df['embedding_t-1_min_max_norm'] = scaler.fit_transform(ref_df.iloc[:, 1:-1]).tolist()

        group_df = group_df[group_cols].reset_index()
        ref_df = ref_df[ref_cols].reset_index()
        
        merged = pd.merge(group_df, ref_df, on='index', suffixes=('', '_t-1'), how='left')
        merged['year'] = year
        merged['ref_year'] = ref_year
        
        return merged
    
    def _compute_metrics(df: pd.DataFrame, eps: int = 1e-10) -> pd.DataFrame:
        """Return the Euclidean distance, cosine similarity, cross-entropy and KL-divergence between two embeddings.
        If the embeddings are not in both distributions, the distances are set to NaN.

        Args:
            df (pd.DataFrame): _description_
            eps (int, optional): _description_. Defaults to 1e-10.

        Returns:
            pd.DataFrame: _description_
        """
        def compute_distance(row):
            emb1 = np.array(row['embedding'])
            emb2 = np.array(row['embedding_t-1'])
            
            if np.isnan(emb1).any() or np.isnan(emb2).any():
                return np.nan
            
            return euclidean_distances([emb1], [emb2])[0][0]

        def compute_cosine_similarity(row):
            emb1 = np.array(row['embedding'])
            emb2 = np.array(row['embedding_t-1'])
            
            if np.isnan(emb1).any() or np.isnan(emb2).any():
                return np.nan
            
            return cosine_similarity([emb1], [emb2])[0][0]
        
        def compute_cross_entropy(row):
            prob_i = np.array(row['embedding_min_max_norm'])
            prob_j = np.array(row['embedding_t-1_min_max_norm'])

            if np.isnan(prob_i).any() or np.isnan(prob_j).any():
                return np.nan

            prob_i = np.clip(prob_i, eps, 1 - eps)  # Avoid log(0)
            prob_j = np.clip(prob_j, eps, 1 - eps)

            return -(
                np.sum(prob_i * np.log(prob_j)) +
                np.sum((1 - prob_i) * np.log(1 - prob_j))
            )

        def compute_kl_div(row):
            prob_i = np.array(row['embedding_min_max_norm'])
            prob_j = np.array(row['embedding_t-1_min_max_norm'])

            if np.isnan(prob_i).any() or np.isnan(prob_j).any():
                return np.nan

            prob_i = np.clip(prob_i, eps, 1 - eps)  # Avoid log(0)
            prob_j = np.clip(prob_j, eps, 1 - eps)

            return (
                np.sum(prob_i * (np.log(prob_i) - np.log(prob_j))) +
                np.sum((1 - prob_i) * (np.log(1 - prob_i) - np.log(1 - prob_j)))
            )
            
        def compute_entropy(row, column_name="embedding_min_max_norm", eps=1e-10):
            prob_i = np.array(row[column_name])
            
            if np.isnan(prob_i).any():
                return np.nan
            
            prob_i = np.clip(prob_i, eps, 1 - eps)            
            entropy_values = stats.bernoulli(prob_i).entropy()
            
            return np.sum(entropy_values)
        
        df['euclidean_distance'] = df.apply(compute_distance, axis=1)
        df['cosine_similarity'] = df.apply(compute_cosine_similarity, axis=1)
        df['cross_entropy'] = df.apply(compute_cross_entropy, axis=1)
        df['kl_divergence'] = df.apply(compute_kl_div, axis=1)
        df['entropy'] = df.apply(compute_entropy, axis=1)
        df['entropy_t-1'] = df.apply(
            compute_entropy, 
            axis=1, 
            column_name='embedding_t-1_min_max_norm'
            )
 
        return df

    def _compute_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute vectorized descriptive stats on embedding_min_max_norm.

        Args:
            df (pd.DataFrame): Must contain 'embedding_min_max_norm' as arrays/lists of equal length.

        Returns:
            pd.DataFrame: With new columns: 'mean', 'median', 'std', 'iqr', 'range'
        """
        arr = np.vstack(df['embedding_min_max_norm'].values)

        df['mean'] = arr.mean(axis=1)
        df['median'] = np.median(arr, axis=1)
        df['std'] = arr.std(axis=1)
        df['iqr'] = stats.iqr(arr, axis=1)
        df['range'] = arr.max(axis=1) - arr.min(axis=1)

        return df
    
    def _assing_reference_flag(df):
        breakpoint()
        df['word_present_both'] = np.where(
            df['embedding_t-1'].isnull() & df['embedding_t-1_min_max_norm'].isnull(),
            False,
            True
            )
        return df   
    
    def _pop_embeddings(df):
        df = df.drop(columns=['embedding', 'embedding_t-1', 'embedding_min_max_norm', 'embedding_t-1_min_max_norm'])
        return df
    
    def _assign_year(df, year, ref_year):
        df['year'] = year
        df['ref_year'] = ref_year
        return df

    def _check_consecutive_years(list_years):
        return all(
            list_years[i] + 1 == list_years[i + 1] for i in range(len(list_years) - 1)
            )

    references = ['t-1', 't_0']
    groups = df.groupby('year')
    
    for ref_name in references:
        if ref_name == 't-1':
            results_chain = []
            years = list(df['year'].unique())
            prev_year = None 
            for year, group in tqdm(groups, desc="Computing Cosine Similarities: t-1"):    
                if (year == 1920):
                    continue   
                
                if prev_year is not None:
                    if ((prev_year is 1920) and (year - prev_year > 1)):
                        continue
                     
                ref_year = year - 1
                ref_df = df[df['year'] == ref_year]
                
                breakpoint()
                merged = custom_merge_dfs(group, ref_df) 
                single_df_chain = _compute_metrics(merged)
                single_df_chain = _compute_descriptive_statistics(merged)
                single_df_chain = _assing_reference_flag(merged)
                single_df_chain = _pop_embeddings(single_df_chain)
                results_chain.append(single_df_chain)
                
                prev_year = year
        
        elif ref_name == 't_0':
            results_fixed = []
            ref_df = df[df['year'] == 1920].copy()
            
            for year, group in tqdm(groups, desc="Computing Cosine Similarities: t_0"):
                if year == 1920:
                    continue
            
                merged = custom_merge_dfs(group, ref_df)
                single_df_fixed = _compute_metrics(merged)
                single_df_fixed = _compute_descriptive_statistics(merged)
                single_df_fixed = _assing_reference_flag(merged)
                single_df_fixed = _pop_embeddings(single_df_fixed)
                results_fixed.append(single_df_fixed)
                
    df_chain = pd.concat(results_chain)
    df_fixed = pd.concat(results_fixed)
    
    return df_chain, df_fixed

def apply_pos_ner(text):
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    ner_tags = [(ent.text, ent.label_) for ent in doc.ents]
    return pos_tags, ner_tags

cache = {}
def cached_apply_pos_ner(x):
    if x not in cache:
        cache[x] = apply_pos_ner(x)
    return cache[x]
    
def process_year_range(start_year, end_year, output_dir):
    """Processes embeddings for a given time range and saves the results."""
    
    print(f"Processing years {start_year}-{end_year}...")
    years = list(range(start_year, end_year + 1))
    print(f"Years: {years}")
    dfs_dict = {}

    for year in tqdm(years, desc=f"Loading and word count {start_year}-{end_year}"):
        model = load_model(year)            
        df = create_df_words_embeddings(model)
        dfs_dict[year] = assign_count_per_each_word(df, model)

    df = merge_dfs(dfs_dict)
    breakpoint()
    df_chain, df_fixed = run_divergences_pipe(df)
    # TODO: Uncomment this line to add POS and NER tags
    # df_chain[['pos_tags', 'ner_tags']] = df_chain['index'].map(cached_apply_pos_ner).apply(pd.Series)
    # df_fixed[['pos_tags', 'ner_tags']] = df_fixed['index'].map(cached_apply_pos_ner).apply(pd.Series)

    os.makedirs(output_dir, exist_ok=True)
    df_chain_path = os.path.join(output_dir, f"df_chain_{start_year}-{end_year}.csv")
    df_fixed_path = os.path.join(output_dir, f"df_fixed_{start_year}-{end_year}.csv")
    
    df_chain.to_csv(df_chain_path, index=False, sep='|')
    df_fixed.to_csv(df_fixed_path, index=False, sep='|')

    print(f"Saved: {df_chain_path}")
    print(f"Saved: {df_fixed_path}")


if __name__ == "__main__":
    output_dir = os.path.join(DATA_ALT, "05_information_metrics")

    start_year = 1920
    end_year = 2020

    process_year_range(start_year, end_year, output_dir)
