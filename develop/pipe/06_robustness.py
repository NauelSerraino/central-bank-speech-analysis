"""
Robustness analysis for the EmbedDynamics pipeline.

Three analyses:
  1. Centroid weighting sensitivity  — compares freq*norm vs freq-only vs uniform
  2. TWEC seed stability             — re-trains TWEC under 5 seeds, compares rankings
  3. TWEC hyperparameter stability   — varies size ∈ {50,100,200} and window ∈ {3,5,10}

All outputs (CSV tables + PNG figures) go to data/06_robustness/.
"""

import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from develop.utils.paths import DATA
from develop.utils.logger import LoggerManager
from develop.core.vectors_flags.label_assinger import LabelAssigner

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

log_mgr = LoggerManager(name="robustness", log_file="06_robustness.log", clear_log=True)
logger  = log_mgr.get_logger()

# ── paths ──────────────────────────────────────────────────────────────────────
OUT_DIR  = os.path.join(DATA, "06_robustness")
BERTOPIC = os.path.join(DATA, "01_bertopic")
METRICS  = os.path.join(DATA, "04_create_metrics")
CORPUS   = os.path.join(DATA, "00_preprocessed_corpus")
os.makedirs(OUT_DIR, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
LABELS = [
    "monetary policy and interest rates",
    "inflation and consumer prices",
    "labor market and employment",
    "financial markets and banking",
    "fiscal policy and government spending",
    "international trade and global economy",
    "economic development and inequality",
    "climate change and sustainable finance",
    "industrial and sectoral economics",
    "behavioral economics and expectations",
    "policy communication and forward guidance",
]
SYNTHETIC = {
    "monetary policy and interest rates":     "Interest Rates",
    "inflation and consumer prices":          "Inflation",
    "labor market and employment":            "Employment",
    "financial markets and banking":          "Financial Markets",
    "fiscal policy and government spending":  "Fiscal Policy",
    "international trade and global economy": "International Trade",
    "economic development and inequality":    "Inequality",
    "climate change and sustainable finance": "Climate Change",
    "industrial and sectoral economics":      "Industries",
    "behavioral economics and expectations":  "Behavioral Economics",
    "policy communication and forward guidance": "Forward Guidance",
}
REGION_MAP = {"USA": "Fed", "EU": "ECB"}

WEIGHT_SCHEMES = {
    "freq_x_norm": lambda df: df["count"] * df["norm"],
    "freq_only":   lambda df: df["count"].astype(float),
    "uniform":     lambda df: pd.Series(1.0, index=df.index),
}

TWEC_HP_SIZES   = [50, 100, 200]   # window fixed at 5
TWEC_HP_WINDOWS = [3, 5, 10]       # size fixed at 100
TWEC_HP_SEED    = 1                # fixed to isolate hp effects from seed noise


# ── helpers ────────────────────────────────────────────────────────────────────

def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close()


def load_base_data():
    raw_words  = pd.read_parquet(os.path.join(BERTOPIC, "bertopic_topic_words.parquet"))
    raw_counts = pd.read_parquet(os.path.join(BERTOPIC, "bertopic_topic_counts_year_region.parquet"))
    emb = pd.read_parquet(os.path.join(METRICS, "df_2000-2024.parquet"))
    emb = emb[["word", "embedding", "region", "year", "count", "norm"]].copy()
    emb["region"]    = emb["region"].map(REGION_MAP)
    emb["embedding"] = emb["embedding"].apply(np.array)
    emb = emb[(emb["year"] >= 2000) & (emb["year"] <= 2024)]
    return raw_words, raw_counts, emb


def build_label_to_words(df_with_short_topic):
    """Explode word lists and return {short_topic: set_of_words}."""
    return (
        df_with_short_topic.explode("word")
        .groupby("short_topic")["word"]
        .apply(lambda x: set(x.dropna()))
        .to_dict()
    )


def compute_centroids(emb_df, label_to_words, weight_col="importance"):
    """Per-region, per-topic, per-year weighted centroid from emb_df."""
    word_to_topic = {w: t for t, ws in label_to_words.items() for w in ws}
    df = emb_df.copy()
    df["_topic"] = df["word"].map(word_to_topic)
    df = df[df["_topic"].notna()]

    result = {"ECB": {}, "Fed": {}}
    for region in ["ECB", "Fed"]:
        df_r = df[df["region"] == region]
        for topic, words in label_to_words.items():
            df_t = df_r[df_r["_topic"] == topic]
            if df_t.empty:
                continue
            year_centroids = {}
            for year, grp in df_t.groupby("year"):
                w    = grp[weight_col].values
                vecs = np.stack(grp["embedding"].values)
                if w.sum() == 0:
                    continue
                year_centroids[year] = (vecs * w[:, None]).sum(0) / w.sum()
            if year_centroids:
                result[region][topic] = year_centroids
    return result


def compute_inter_distances(centroids):
    """Cosine distance between ECB and Fed centroids, per topic per year."""
    records = []
    for topic in set(centroids["ECB"]) & set(centroids["Fed"]):
        ecb, fed = centroids["ECB"][topic], centroids["Fed"][topic]
        for year in set(ecb) & set(fed):
            d = cosine_distances(
                ecb[year].reshape(1, -1),
                fed[year].reshape(1, -1),
            )[0, 0]
            records.append({"topic": topic, "year": year, "distance": d})
    return pd.DataFrame(records)


def mean_ranking(dist_df):
    """Mean distance per topic, sorted descending (most divergent first)."""
    return dist_df.groupby("topic")["distance"].mean().sort_values(ascending=False)


def compute_davies_bouldin(emb_df, label_to_words, centroids):
    """
    Davies-Bouldin index (Davies & Bouldin, 1979) with cosine distance.
    DB = (1/K) * sum_i max_{j≠i} (s_i + s_j) / d(c_i, c_j)
    where s_i = mean cosine distance of topic i's words to centroid c_i,
    and d(c_i, c_j) = cosine distance between centroids i and j.
    Averaged over region-year slices. Lower = better.
    """
    word_to_topic = {w: t for t, ws in label_to_words.items() for w in ws}
    df = emb_df.copy()
    df["_topic"] = df["word"].map(word_to_topic)
    df = df[df["_topic"].notna()]

    scores = []
    for region in ["ECB", "Fed"]:
        df_r = df[df["region"] == region]
        for year, grp in df_r.groupby("year"):
            topic_list = [t for t in centroids[region] if year in centroids[region][t]]
            if len(topic_list) < 2:
                continue

            centroid_mat = np.stack([centroids[region][t][year] for t in topic_list])
            topic_to_idx = {t: i for i, t in enumerate(topic_list)}
            grp_valid    = grp[grp["_topic"].isin(topic_to_idx)]
            if grp_valid.empty:
                continue

            K      = len(topic_list)
            embs   = np.stack(grp_valid["embedding"].values)
            labels = np.array([topic_to_idx[t] for t in grp_valid["_topic"]])

            # within-cluster scatter s_i
            dists_to_c = cosine_distances(embs, centroid_mat)   # (n_words, K)
            a          = dists_to_c[np.arange(len(labels)), labels]
            s = np.array([a[labels == i].mean() if (labels == i).any() else 0.0
                          for i in range(K)])

            # between-centroid distances
            c_dists = cosine_distances(centroid_mat)            # (K, K)
            np.fill_diagonal(c_dists, np.inf)

            # R_i = max_{j≠i} (s_i + s_j) / d(c_i, c_j)
            R = ((s[:, None] + s[None, :]) / c_dists).max(axis=1)
            scores.append(R.mean())

    return float(np.mean(scores)) if scores else 0.0


def spearman_matrix(rankings_dict):
    """Build topic × variant rank matrix and its Spearman correlation matrix."""
    all_topics = sorted({t for r in rankings_dict.values() for t in r.index})
    rank_df = pd.DataFrame(
        {k: v.reindex(all_topics) for k, v in rankings_dict.items()}
    )
    return rank_df, rank_df.corr(method="spearman")


# ── analysis 1: weighting sensitivity ─────────────────────────────────────────

def run_weighting_sensitivity(emb, label_to_words):
    """
    Compute inter-institution centroid distances under three weighting schemes.
    Measures:
      - Spearman ρ of topic divergence rankings across schemes
      - Davies-Bouldin index per scheme (Davies & Bouldin, 1979): ratio of
        within-cluster scatter to between-centroid distance; lower = better
    """
    rankings = {}
    db_scores = {}

    for name, weight_fn in WEIGHT_SCHEMES.items():
        df       = emb.copy()
        raw_w    = weight_fn(df)
        df["importance"] = raw_w / raw_w.sum()
        centroids          = compute_centroids(df, label_to_words)
        dist_df            = compute_inter_distances(centroids)
        rankings[name]     = mean_ranking(dist_df)
        db_scores[name]    = compute_davies_bouldin(df, label_to_words, centroids)
        logger.info(f"Scheme '{name}' done. Davies-Bouldin={db_scores[name]:.4f}")

    rank_df, corr = spearman_matrix(rankings)
    rank_df.to_csv(os.path.join(OUT_DIR, "weighting_rankings.csv"))
    corr.to_csv(os.path.join(OUT_DIR, "weighting_spearman.csv"))

    db_df = pd.Series(db_scores, name="davies_bouldin")
    db_df.to_csv(os.path.join(OUT_DIR, "weighting_davies_bouldin.csv"), header=True)
    logger.info("Davies-Bouldin per scheme (lower = better):\n" + db_df.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    rank_df.plot(kind="bar", ax=axes[0], colormap="tab10", width=0.7)
    axes[0].set_title("Mean ECB–Fed distance per topic\nby weighting scheme")
    axes[0].set_ylabel("Mean cosine distance")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=9)

    sns.heatmap(
        corr.astype(float), annot=True, fmt=".3f",
        cmap="Greens", vmin=0.7, vmax=1.0, ax=axes[1],
    )
    axes[1].set_title("Spearman ρ of topic divergence rankings\nacross weighting schemes")

    axes[2].bar(db_df.index, db_df.values, color=["steelblue", "tomato", "seagreen"])
    axes[2].set_title("Davies-Bouldin index per scheme\n(lower = better)")
    axes[2].set_ylabel("Davies-Bouldin index")
    axes[2].set_ylim(0, db_df.max() * 1.1)

    save_fig("weighting_sensitivity")
    return rank_df, corr


# ── analysis 3: TWEC seed stability ───────────────────────────────────────────

def _train_twec_seed(seed, out_dir):
    from twec.twec import TWEC
    os.makedirs(out_dir, exist_ok=True)
    # Reduced iterations vs. the main pipeline (siter/diter 1→100 instead of 1→100)
    # to keep runtime manageable; sufficient to test embedding stability.
    aligner = TWEC(size=100, sg=0, siter=1, diter=1,
                   window=5, workers=1, seed=seed, opath=out_dir)
    aligner.train_compass(CORPUS, overwrite=True)
    aligner.siter = 100
    aligner.diter = 100
    for fname in os.listdir(CORPUS):
        aligner.train_slice(os.path.join(CORPUS, fname), save=True)
    logger.info(f"TWEC seed={seed} trained → {out_dir}")


def _centroids_from_model_dir(model_dir, label_to_words, count_emb):
    """
    Load gensim Word2Vec slice models from model_dir, extract word vectors,
    and compute weighted centroids (reusing corpus counts from count_emb).
    """
    from gensim.models import Word2Vec

    all_words    = {w for ws in label_to_words.values() for w in ws}
    word_to_topic = {w: t for t, ws in label_to_words.items() for w in ws}

    records = []
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".model") or fname == "compass.model":
            continue
        parts = fname.replace(".model", "").split("_")
        if len(parts) != 2:
            continue
        try:
            year   = int(parts[0])
            region = REGION_MAP.get(parts[1], parts[1])
        except ValueError:
            continue
        if year < 2000 or year > 2024:
            continue

        model = Word2Vec.load(os.path.join(model_dir, fname))
        for word in all_words:
            if word in model.wv:
                records.append({
                    "word": word, "year": year, "region": region,
                    "embedding": model.wv[word],
                    "topic": word_to_topic[word],
                })

    if not records:
        return {"ECB": {}, "Fed": {}}

    df = pd.DataFrame(records)
    df = df.merge(
        count_emb[["word", "year", "region", "count"]],
        on=["word", "year", "region"], how="left",
    )
    df["count"] = df["count"].fillna(1.0)
    df["norm"]  = df["embedding"].apply(np.linalg.norm)
    raw_w       = df["count"] * df["norm"]
    df["importance"] = raw_w / raw_w.sum()

    result = {"ECB": {}, "Fed": {}}
    for region in ["ECB", "Fed"]:
        df_r = df[df["region"] == region]
        for topic in df_r["topic"].dropna().unique():
            df_t = df_r[df_r["topic"] == topic]
            year_centroids = {}
            for year, grp in df_t.groupby("year"):
                w    = grp["importance"].values
                vecs = np.stack(grp["embedding"].values)
                if w.sum() == 0:
                    continue
                year_centroids[year] = (vecs * w[:, None]).sum(0) / w.sum()
            if year_centroids:
                result[region][topic] = year_centroids
    return result


def _ranking_from_centroids(centroids):
    """Mean ECB–Fed cosine distance per topic, sorted descending."""
    dist_df = compute_inter_distances(centroids)
    if dist_df.empty:
        return pd.Series(dtype=float)
    return mean_ranking(dist_df)


def run_twec_stability(label_to_words, count_emb, seeds=None):
    """
    Train TWEC once per seed (workers=1 for determinism), then compare
    the topic divergence rankings across seeds via Spearman ρ.
    Skips training if the model directory already exists.
    """
    if seeds is None:
        seeds = [1, 42, 456, 789, 1000]

    rankings = {}
    for seed in seeds:
        seed_dir = os.path.join(OUT_DIR, f"twec_seed_{seed}")
        if not os.path.exists(os.path.join(seed_dir, "compass.model")):
            _train_twec_seed(seed, seed_dir)
        else:
            logger.info(f"Seed={seed}: models found, skipping training.")
        centroids = _centroids_from_model_dir(seed_dir, label_to_words, count_emb)
        rankings[seed] = _ranking_from_centroids(centroids)
        logger.info(f"Seed={seed} top topic: {rankings[seed].index[0]} "
                    f"(dist={rankings[seed].iloc[0]:.4f})")

    rank_df, corr = spearman_matrix(rankings)
    rank_df.columns = [f"seed={s}" for s in rank_df.columns]
    rank_df.to_csv(os.path.join(OUT_DIR, "twec_seed_rankings.csv"))
    corr.index   = [f"seed={s}" for s in corr.index]
    corr.columns = [f"seed={s}" for s in corr.columns]
    corr.to_csv(os.path.join(OUT_DIR, "twec_seed_stability.csv"))

    logger.info("TWEC seed Spearman matrix:\n" + corr.to_string())

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr.astype(float), annot=True, fmt=".3f",
                cmap="Blues", vmin=0.7, vmax=1.0)
    plt.title("Spearman ρ of topic divergence rankings across TWEC seeds")
    save_fig("twec_seed_stability")

    return corr


# ── analysis 4: TWEC hyperparameter stability ─────────────────────────────────

def _train_twec_hp(size, window, out_dir):
    from twec.twec import TWEC
    os.makedirs(out_dir, exist_ok=True)
    aligner = TWEC(size=size, sg=0, siter=1, diter=1,
                   window=window, workers=1, seed=TWEC_HP_SEED, opath=out_dir)
    aligner.train_compass(CORPUS, overwrite=True)
    aligner.siter = 100
    aligner.diter = 100
    for fname in os.listdir(CORPUS):
        aligner.train_slice(os.path.join(CORPUS, fname), save=True)
    logger.info(f"TWEC size={size} window={window} trained → {out_dir}")


def run_twec_hp_stability(label_to_words, count_emb):
    """
    Vary embedding size (window=5 fixed) and context window (size=100 fixed)
    independently. Compare topic divergence rankings via Spearman ρ.
    Skips training if the model directory already exists.
    """
    def _get_ranking(size, window):
        hp_dir = os.path.join(OUT_DIR, f"twec_hp_s{size}_w{window}")
        if not os.path.exists(os.path.join(hp_dir, "compass.model")):
            _train_twec_hp(size, window, hp_dir)
        else:
            logger.info(f"size={size} window={window}: models found, skipping training.")
        centroids = _centroids_from_model_dir(hp_dir, label_to_words, count_emb)
        ranking   = _ranking_from_centroids(centroids)
        logger.info(f"size={size} window={window} top topic: {ranking.index[0]} "
                    f"(dist={ranking.iloc[0]:.4f})")
        return ranking

    # Vary size, window fixed at 5
    rankings_size = {f"size={s}": _get_ranking(s, 5) for s in TWEC_HP_SIZES}

    # Vary window, size fixed at 100
    rankings_window = {f"window={w}": _get_ranking(100, w) for w in TWEC_HP_WINDOWS}

    _, corr_size = spearman_matrix(rankings_size)
    corr_size.to_csv(os.path.join(OUT_DIR, "twec_hp_size_stability.csv"))
    logger.info("TWEC size Spearman matrix:\n" + corr_size.to_string())

    _, corr_window = spearman_matrix(rankings_window)
    corr_window.to_csv(os.path.join(OUT_DIR, "twec_hp_window_stability.csv"))
    logger.info("TWEC window Spearman matrix:\n" + corr_window.to_string())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(corr_size.astype(float), annot=True, fmt=".3f",
                cmap="Blues", vmin=0.7, vmax=1.0, ax=ax1)
    ax1.set_title("Spearman ρ across embedding sizes\n(window=5 fixed)")

    sns.heatmap(corr_window.astype(float), annot=True, fmt=".3f",
                cmap="Blues", vmin=0.7, vmax=1.0, ax=ax2)
    ax2.set_title("Spearman ρ across context windows\n(size=100 fixed)")

    save_fig("twec_hp_stability")
    return corr_size, corr_window


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Loading data...")
    raw_words, raw_counts, emb = load_base_data()

    # Base label assignment (same as 05_final_output.py)
    assigner          = LabelAssigner.get_instance()
    mapping_words     = raw_words.groupby("topic")["word"].apply(list).reset_index()
    topic_to_words_map = dict(zip(mapping_words["topic"], mapping_words["word"]))

    df_base = raw_counts.copy()
    df_base["word"]   = df_base["topic"].map(topic_to_words_map)
    df_base           = assigner.assign_labels(df_base)
    df_base["region"] = df_base["region"].map(REGION_MAP)
    df_base["short_topic"] = df_base["label"].map(SYNTHETIC)
    df_base = df_base[df_base["short_topic"].notna()]

    base_l2w = build_label_to_words(df_base)

    base_emb = emb.copy()
    raw_w    = base_emb["count"] * base_emb["norm"]
    base_emb["importance"] = raw_w / raw_w.sum()

    # ── 1. weighting sensitivity ───────────────────────────────────────────────
    logger.info("─── Analysis 1: Weighting scheme sensitivity ───")
    _, weight_corr = run_weighting_sensitivity(base_emb, base_l2w)
    logger.info("Weighting Spearman matrix:\n" + weight_corr.to_string())

    # ── 2. TWEC seed stability ─────────────────────────────────────────────────
    logger.info("─── Analysis 2: TWEC seed stability ───")
    logger.info("Training TWEC 5× with workers=1 (may take 10–40 min depending on corpus size).")
    twec_corr = run_twec_stability(base_l2w, emb)

    # ── 3. TWEC hyperparameter stability ──────────────────────────────────────
    logger.info("─── Analysis 3: TWEC hyperparameter stability ───")
    logger.info(f"Sizes to test: {TWEC_HP_SIZES} (window=5); windows to test: {TWEC_HP_WINDOWS} (size=100).")
    hp_corr_size, hp_corr_window = run_twec_hp_stability(base_l2w, emb)

    logger.info(f"All outputs saved to {OUT_DIR}")
