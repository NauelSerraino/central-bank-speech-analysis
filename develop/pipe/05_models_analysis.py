import pandas as pd
from gensim.models.word2vec import Word2Vec

m = Word2Vec.load("model/compass.model")

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

if __name__ == '__main__':
    df = create_df_words_embeddings(m)
    df = normalize_column_wise(df)
    df = assign_count_per_each_word(df)
    breakpoint()
    print(df)
