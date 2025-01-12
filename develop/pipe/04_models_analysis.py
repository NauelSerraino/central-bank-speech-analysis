import os
import pandas as pd
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, "../.."))))
from develop.utils.paths import DATA, MODEL

m = Word2Vec.load("model/compass.model")

def load_model(year, compass=False):
    if compass:
        print("Loading compass model")
        return Word2Vec.load(os.path.join(MODEL, "compass.model"))
    else:
        return Word2Vec.load(os.path.join(MODEL, f"combined_{year}.model"))

def create_df_words_embeddings(m):
    emb = m.wv.vectors
    w = m.wv.index2word
    df = pd.DataFrame(emb, index=w)
    return df

def normalize_column_wise(df):
    return (df - df.min()) / (df.max() - df.min())

def assign_count_per_each_word(df):
    df['count'] = df.index.map(lambda x: m.wv.vocab[x].count)
    return df

def filter_words(df):
    tot_obs = df['count'].sum()
    df = df[df['count'] > tot_obs * 1e-6] # 0.0001%
    return df

def save_df(df, year=None):
    os.makedirs(os.path.join(DATA, "04_embeddings"), exist_ok=True)
    if year is None:
        df.to_csv(os.path.join(DATA, "04_embeddings", "compass.csv"))
    else:
        df.to_csv(os.path.join(DATA, "04_embeddings", f"{year}.csv"))

if __name__ == '__main__':
    years = [i for i in range(1991, 2024)]
    for year in tqdm(years):
        m = load_model(year)
        df = create_df_words_embeddings(m)
        df = normalize_column_wise(df)
        df = assign_count_per_each_word(df)
        df = filter_words(df)
        save_df(df, year)
        
    m = load_model(None, compass=True)
    df = create_df_words_embeddings(m)
    df = normalize_column_wise(df)
    df = assign_count_per_each_word(df)
    df = filter_words(df)
    save_df(df)
    
    
