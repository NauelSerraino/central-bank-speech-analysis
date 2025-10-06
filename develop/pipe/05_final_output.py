import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances
import warnings

from develop.utils.paths import DATA
from develop.core.vectors_flags.label_assinger import LabelAssigner

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# ----------------------
# CONFIG
# ----------------------
FIG_DIR = os.path.join(DATA, "05_graphs")
os.makedirs(FIG_DIR, exist_ok=True)

TOPIC_COLORS = {
    "Interest Rates": "#1f77b4",
    "Inflation": "#d62728",
    "Employment": "#ff7f0e",
    "Financial Markets": "#9467bd",
    "Fiscal Policy": "#8c564b",
    "International Trade": "#17becf",
    "Inequality": "#e377c2",
    "Climate Change": "#2ca02c",
    "Industries": "#bcbd22",
    "Behavioral Economics": "#7f7f7f",
    "Forward Guidance": "#aec7e8"
}

synthetic_mapping = {
    "monetary policy and interest rates": "Interest Rates",
    "inflation and consumer prices": "Inflation",
    "labor market and employment": "Employment",
    "financial markets and banking": "Financial Markets",
    "fiscal policy and government spending": "Fiscal Policy",
    "international trade and global economy": "International Trade",
    "economic development and inequality": "Inequality",
    "climate change and sustainable finance": "Climate Change",
    "industrial and sectoral economics": "Industries",
    "behavioral economics and expectations": "Behavioral Economics",
    "policy communication and forward guidance": "Forward Guidance"
}

region_to_bank = {"USA": "Fed", "EU": "ECB"}

# ----------------------
# LOAD AND PREPARE DATA
# ----------------------
BERTOPIC = os.path.join(DATA, "01_bertopic")
df = pd.read_parquet(os.path.join(BERTOPIC, "bertopic_topic_counts_year_region.parquet"))
raw_mapping = pd.read_parquet(os.path.join(BERTOPIC, "bertopic_topic_words.parquet"))
mapping_words = raw_mapping.groupby("topic")['word'].apply(list).reset_index()

topic_to_words = dict(zip(mapping_words['topic'], mapping_words['word']))
df['word'] = df['topic'].map(topic_to_words)

assigner = LabelAssigner.get_instance()
df = assigner.assign_labels(df)
df['region'] = df['region'].map(region_to_bank)
df['short_topic'] = df['label'].map(synthetic_mapping)

def aggregate_words(df, group_col, word_col):
    """
    Group by `group_col`, aggregate unique words in `word_col`,
    and count the number of unique words per group.
    
    Returns a DataFrame with columns: group_col, words, word_count
    """
    result = df.groupby(group_col, as_index=False).agg(
        words=(word_col, lambda x: list(set(sum(x, []))))
    )
    result['word_count'] = result['words'].apply(len)
    return result

def prepare_groupby_df(df):
    df = (
        df.groupby(['year', 'region', 'short_topic'])['count']
          .sum()
          .reset_index()
    )
    df = df[(df['year'] >= 2000) & (df['year'] <= 2024)]
    df = df[~df['short_topic'].isna()]
    return df

topics_df = prepare_groupby_df(df)
topic_counter = aggregate_words(df, "short_topic", "word")
topic_counter.to_csv(os.path.join(FIG_DIR, "topic_counter.csv"), index=False)

# ----------------------
# PLOTTING UTILITIES
# ----------------------
def save_fig(name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_stacked_topics_normalized(df, region, topic_colors=TOPIC_COLORS):
    df_region = df[df['region'] == region]
    pivot = df_region.pivot_table(
        index='year',
        columns='short_topic',
        values='count',
        aggfunc='sum',
        fill_value=0
    )

    pivot = pivot.div(pivot.sum(axis=1), axis=0).astype(float)
    colors = [topic_colors.get(col, "#cccccc") for col in pivot.columns]
    fontsize = 14

    pivot.plot(kind='bar', stacked=True, figsize=(12, 6), color=colors, fontsize=fontsize)
    plt.title(f"Normalized Topic Distribution per Year ({region})", fontsize=fontsize)
    plt.ylabel("Proportion", fontsize=fontsize)
    plt.xlabel("Year", fontsize=fontsize)
    plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), fontsize=fontsize)
    save_fig(f"stacked_topics_{region}")

plot_stacked_topics_normalized(topics_df, "Fed")
plot_stacked_topics_normalized(topics_df, "ECB")

# ----------------------
# LOAD EMBEDDINGS
# ----------------------
METRICS = os.path.join(DATA, "04_create_metrics")
emb = pd.read_parquet(os.path.join(METRICS, "df_2000-2024.parquet"))
emb = emb[['word', 'embedding', 'region', 'year', 'count', 'norm']]
emb['region'] = emb['region'].map(region_to_bank)

label_to_words = df.explode('word').groupby('short_topic')['word'].apply(lambda x: set(x.dropna())).to_dict()
word_to_topic = {w: t for t, words in label_to_words.items() for w in words}
emb['short_topic'] = emb['word'].map(word_to_topic)
emb = emb[~emb['short_topic'].isna()]
emb['importance_raw'] = emb['count'] * emb['norm']
emb['importance'] = emb['importance_raw'] / emb['importance_raw'].sum()

# ----------------------
# UNIQUE WORDS
# ----------------------
counts = (
    emb.groupby(["year", "region"])['word']
      .nunique()
      .reset_index()
      .query("year >= 2000")
)
pivot = counts.pivot(index="year", columns="region", values="word").fillna(0)
years = pivot.index.values
regions = pivot.columns
x = np.arange(len(years))
width = 0.6 / len(regions)

plt.figure(figsize=(12,6))
for i, region in enumerate(regions):
    plt.bar(x + i*width, pivot[region], width, label=region)
plt.xticks(x + width*(len(regions)-1)/2, years, rotation=90)
plt.xlabel("Year")
plt.ylabel("Unique words")
plt.title("Unique Words by Year and Institution")
plt.legend(title="Institution")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
save_fig("unique_words")

# ----------------------
# CENTROIDS
# ----------------------
def compute_weighted_centroids(df, topics, region):
    df_region = df[df['region'] == region].copy()
    result = {}
    for topic, words in topics.items():
        if not words:
            continue
        filtered = df_region[df_region['word'].isin(words)]
        if filtered.empty:
            continue
        filtered['weighted_embedding'] = filtered.apply(
            lambda row: np.array(row['embedding']) * row['importance'], axis=1
        )
        grouped_sum = filtered.groupby('year')['weighted_embedding'].apply(lambda x: np.sum(x.tolist(), axis=0))
        weight_sum = filtered.groupby('year')['importance'].sum()
        centroids = grouped_sum.div(weight_sum)
        result[topic] = pd.DataFrame(list(centroids), index=centroids.index)
    return result

centroids_ECB = compute_weighted_centroids(emb, label_to_words, region='ECB')
centroids_Fed = compute_weighted_centroids(emb, label_to_words, region='Fed')

# ----------------------
# INTER-REGIONAL DISTANCES
# ----------------------
def compute_region_cosine_distances(centroids_ECB, centroids_Fed):
    records = []
    common_topics = set(centroids_ECB.keys()) & set(centroids_Fed.keys())
    for topic in common_topics:
        df_ECB, df_Fed = centroids_ECB[topic], centroids_Fed[topic]
        common_years = df_ECB.index.intersection(df_Fed.index)
        for year in common_years:
            dist = cosine_distances(df_ECB.loc[year].values.reshape(1,-1),
                                    df_Fed.loc[year].values.reshape(1,-1))[0,0]
            records.append({'year': year, 'topic': topic, 'cosine_distance': dist})
    return pd.DataFrame(records)

region_distances = compute_region_cosine_distances(centroids_ECB, centroids_Fed)

def plot_topic_distances_wide(df, mode='per topic'):
    heatmap_data = df.pivot(index='year', columns='topic', values='cosine_distance').fillna(0)
    
    
    if mode=='per topic':
        heatmap_data = heatmap_data.apply(lambda x: (x-x.min())/(x.max()-x.min()) if x.max()>x.min() else 0, axis=0)
    elif mode=='per year':
        heatmap_data = heatmap_data.T.apply(lambda x: (x-x.min())/(x.max()-x.min()) if x.max()>x.min() else 0, axis=0).T
    elif mode=='global':
        flat = heatmap_data.values.flatten()
        norm_flat = (flat - flat.min())/(flat.max()-flat.min())
        heatmap_data = pd.DataFrame(norm_flat.reshape(heatmap_data.shape),
                                    columns=heatmap_data.columns, index=heatmap_data.index)
           
    # SORT VALUES BY AVERAGE PER TOPIC 
    topic_order = heatmap_data.mean(axis=0).sort_values(ascending=False).index.to_list()
    heatmap_data = heatmap_data[topic_order]
    
    plt.figure(figsize=(12, len(heatmap_data.columns)*0.5+2))
    sns.heatmap(heatmap_data.T, cmap='Reds', linewidths=0.5, linecolor='white',
                cbar_kws={'label': f'Normalized Distance ({mode})'})
    plt.title(f'ECB–Fed Topic Distances Over Time ({mode})')
    plt.xlabel('Year')
    plt.ylabel('Topic')
    save_fig(f"region_distance_{mode}")

for m in ['per topic', 'per year', 'global']:
    plot_topic_distances_wide(region_distances, m)

# ----------------------
# INTRA-REGIONAL SHIFTS
# ----------------------
def compute_topic_shifts(centroids, mode='fixed'):
    records = []
    for topic, df_topic in centroids.items():
        df_topic = df_topic.sort_index()
        prev_vec = None
        base_vec = df_topic.iloc[0].values.reshape(1,-1)
        for year in df_topic.index:
            vec = df_topic.loc[year].values.reshape(1,-1)
            if mode=='fixed': dist = cosine_distances(base_vec, vec)[0,0]
            elif mode=='chain': dist = cosine_distances(prev_vec, vec)[0,0] if prev_vec is not None else 0.0
            records.append({'year': year, 'topic': topic, 'cosine_distance': dist})
            prev_vec = vec
    return pd.DataFrame(records)

def plot_topic_shifts(df, mode='fixed', region=None, color='red'):
    heatmap_data = df.pivot(index='topic', columns='year', values='cosine_distance')
    heatmap_data = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0, axis=1)
    
    # SORT VALUES BY AVERAGE PER TOPIC
    topic_order = heatmap_data.T.mean(axis=0).sort_values(ascending=False).index
    heatmap_data = heatmap_data.T[topic_order].T
    
    color_map_dict = {'red': 'Reds','blue': 'Blues','orange': 'Oranges','green': 'Greens','purple': 'Purples'}
    cmap = color_map_dict.get(color.lower(), 'Reds')
    plt.figure(figsize=(12, len(heatmap_data) * 0.5 + 2))
    sns.heatmap(heatmap_data, cmap=cmap, linewidths=0.5, linecolor='white',
                cbar_kws={'label': f'Normalized Distance ({mode})'})
    title_region = f" - {region}" if region else ""
    plt.title(f'Topic Shifts Over Time ({mode}{title_region})')
    plt.xlabel('Year')
    plt.ylabel('Topic')
    save_fig(f"topic_shifts_{region}_{mode}")

fixed_ECB = compute_topic_shifts(centroids_ECB, 'fixed')
chain_ECB = compute_topic_shifts(centroids_ECB, 'chain')
plot_topic_shifts(fixed_ECB, 'fixed', 'ECB', color='blue')
plot_topic_shifts(chain_ECB, 'chain', 'ECB', color='blue')

fixed_Fed = compute_topic_shifts(centroids_Fed, 'fixed')
chain_Fed = compute_topic_shifts(centroids_Fed, 'chain')
plot_topic_shifts(fixed_Fed, 'fixed', 'Fed', color='purple')
plot_topic_shifts(chain_Fed, 'chain', 'Fed', color='purple')
