import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.metrics.pairwise import cosine_distances

from develop.utils.paths import DATA
from develop.core.vectors_flags.label_assinger import LabelAssigner
from develop.utils.logger import LoggerManager

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

log_mgr = LoggerManager(name="final_output", log_file="05_final_output.log", clear_log=True)
logger  = log_mgr.get_logger()

# ── paths ──────────────────────────────────────────────────────────────────────
FIG_DIR  = os.path.join(DATA, "05_graphs")
BERTOPIC = os.path.join(DATA, "01_bertopic")
METRICS  = os.path.join(DATA, "04_create_metrics")
os.makedirs(FIG_DIR, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
EPISODES = {
    "GFC":             (2008, "#d73027"),
    "Euro Crisis":     (2010, "#fc8d59"),
    "COVID":           (2020, "#4575b4"),
    "Infl. Surge":     (2021, "#2ca02c"),
}

TOPIC_COLORS = {
    "Interest Rates":      "#1f77b4",
    "Inflation":           "#d62728",
    "Employment":          "#ff7f0e",
    "Financial Markets":   "#9467bd",
    "Fiscal Policy":       "#8c564b",
    "International Trade": "#17becf",
    "Inequality":          "#e377c2",
    "Climate Change":      "#2ca02c",
    "Industries":          "#bcbd22",
    "Behavioral Economics":"#7f7f7f",
    "Forward Guidance":    "#aec7e8",
}

SYNTHETIC_MAPPING = {
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


# ── data helpers ───────────────────────────────────────────────────────────────

def load_bertopic():
    df       = pd.read_parquet(os.path.join(BERTOPIC, "bertopic_topic_counts_year_region.parquet"))
    raw_words = pd.read_parquet(os.path.join(BERTOPIC, "bertopic_topic_words.parquet"))
    topic_to_words = (
        raw_words.groupby("topic")["word"].apply(list)
        .to_dict()
    )
    df["word"] = df["topic"].map(topic_to_words)
    return df


def assign_labels(df):
    assigner = LabelAssigner.get_instance()
    df = assigner.assign_labels(df)
    n_total     = df["topic"].nunique()
    n_unlabeled = df[df["label"].isna()]["topic"].nunique()
    logger.info(f"Selected τ* = {assigner.threshold:.2f}")
    logger.info(f"Unlabeled topics: {n_unlabeled}/{n_total} ({n_unlabeled/n_total:.1%})")
    df["region"]      = df["region"].map(REGION_MAP)
    df["short_topic"] = df["label"].map(SYNTHETIC_MAPPING)
    return df


def prepare_topics_df(df):
    return (
        df.groupby(["year", "region", "short_topic"])["count"]
          .sum()
          .reset_index()
          .query("2000 <= year <= 2025")
          .loc[lambda d: d["short_topic"].notna()]
    )


def aggregate_word_counts(df):
    result = df.groupby("short_topic", as_index=False).agg(
        words=("word", lambda x: list(set(sum(x, []))))
    )
    result["word_count"] = result["words"].apply(len)
    return result


def load_embeddings(df):
    emb = pd.read_parquet(os.path.join(METRICS, "df_2000-2025.parquet"))
    emb = emb[["word", "embedding", "region", "year", "count", "norm"]].copy()
    emb["region"] = emb["region"].map(REGION_MAP)

    label_to_words = (
        df.explode("word")
          .groupby("short_topic")["word"]
          .apply(lambda x: set(x.dropna()))
          .to_dict()
    )
    word_to_topic = {w: t for t, words in label_to_words.items() for w in words}

    emb["short_topic"]    = emb["word"].map(word_to_topic)
    emb                   = emb[emb["short_topic"].notna()]
    raw_w                 = emb["count"] * emb["norm"]
    emb["importance"]     = raw_w / raw_w.sum()
    return emb, label_to_words


# ── computation ────────────────────────────────────────────────────────────────

def compute_weighted_centroids(emb, label_to_words, region):
    df_r   = emb[emb["region"] == region].copy()
    result = {}
    for topic, words in label_to_words.items():
        filtered = df_r[df_r["word"].isin(words)]
        if filtered.empty:
            continue
        filtered["weighted_embedding"] = filtered.apply(
            lambda row: np.array(row["embedding"]) * row["importance"], axis=1
        )
        grouped_sum = filtered.groupby("year")["weighted_embedding"].apply(
            lambda x: np.sum(x.tolist(), axis=0)
        )
        weight_sum = filtered.groupby("year")["importance"].sum()
        centroids  = grouped_sum.div(weight_sum)
        result[topic] = pd.DataFrame(list(centroids), index=centroids.index)
    return result


def compute_inter_distances(centroids_ECB, centroids_Fed):
    records = []
    for topic in set(centroids_ECB) & set(centroids_Fed):
        ecb, fed = centroids_ECB[topic], centroids_Fed[topic]
        for year in ecb.index.intersection(fed.index):
            d = cosine_distances(
                ecb.loc[year].values.reshape(1, -1),
                fed.loc[year].values.reshape(1, -1),
            )[0, 0]
            records.append({"year": year, "topic": topic, "cosine_distance": d})
    return pd.DataFrame(records)


def compute_topic_shifts(centroids, mode="fixed"):
    records = []
    for topic, df_topic in centroids.items():
        df_topic = df_topic.sort_index()
        base_vec = df_topic.iloc[0].values.reshape(1, -1)
        prev_vec = None
        for year in df_topic.index:
            vec = df_topic.loc[year].values.reshape(1, -1)
            if mode == "fixed":
                dist = cosine_distances(base_vec, vec)[0, 0]
            else:
                dist = cosine_distances(prev_vec, vec)[0, 0] if prev_vec is not None else 0.0
            records.append({"year": year, "topic": topic, "cosine_distance": dist})
            prev_vec = vec
    return pd.DataFrame(records)


# ── plotting ───────────────────────────────────────────────────────────────────

def annotate_episodes(ax):
    for label, (year, color) in EPISODES.items():
        ax.axvline(x=year, linestyle="--", color=color, linewidth=1.1, alpha=0.8)
        ax.text(year + 0.15, 0.97, label, rotation=90, va="top", ha="left",
                fontsize=8.5, color=color, transform=ax.get_xaxis_transform())


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_stacked_topics(topics_df, region):
    pivot = (
        topics_df[topics_df["region"] == region]
        .pivot_table(index="year", columns="short_topic", values="count", aggfunc="sum", fill_value=0)
        .pipe(lambda p: p.div(p.sum(axis=1), axis=0))
        .astype(float)
    )
    colors   = [TOPIC_COLORS.get(c, "#cccccc") for c in pivot.columns]
    fontsize = 14
    pivot.plot(kind="bar", stacked=True, figsize=(12, 6), color=colors, fontsize=fontsize)
    plt.title(f"Normalized Topic Distribution per Year ({region})", fontsize=fontsize)
    plt.ylabel("Proportion", fontsize=fontsize)
    plt.xlabel("Year", fontsize=fontsize)
    plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), fontsize=fontsize)
    save_fig(f"stacked_topics_{region}")
    return pivot


def plot_pairwise_frequencies(fed, ecb):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        10,
        "axes.edgecolor":   "#E0E0E0",
        "axes.linewidth":   0.8,
        "axes.labelcolor":  "#333333",
        "axes.titleweight": "semibold",
        "axes.titlesize":   11,
        "xtick.color":      "#555555",
        "ytick.color":      "#555555",
        "grid.color":       "#EAEAEA",
        "grid.linestyle":   "-",
        "grid.linewidth":   0.7,
        "legend.edgecolor": "none",
    })

    cols  = sorted(set(fed.columns) & set(ecb.columns))
    ncols = 3
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(17, nrows * 3.4))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.plot(ecb.index, ecb[col], color="#005EB8", label="ECB", linewidth=2.5, alpha=0.9)
        ax.plot(fed.index, fed[col], color="#E67E22", label="Fed", linewidth=2.5, linestyle="--", alpha=0.9)
        ax.fill_between(fed.index, ecb[col], fed[col], color="#B0B0B0", alpha=0.12)
        ax.set_title(col, fontsize=11, fontweight="semibold", pad=6, color="#222222")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(alpha=0.25)
        ax.set_facecolor("#FAFAFA")
        annotate_episodes(ax)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=2, fontsize=20, frameon=False)
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(bottom=0.08)
    save_fig("pairwise")


def plot_shannon_entropy(fed, ecb):
    def normalized_entropy(row):
        x = row.values.astype(float)
        x = x[x > 0] / x.sum()
        return -np.sum(x * np.log(x)) / np.log(len(x))

    sh_fed = fed.apply(normalized_entropy, axis=1)
    sh_ecb = ecb.apply(normalized_entropy, axis=1)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 3.5))
    plt.plot(sh_ecb.index, sh_ecb, color="#005EB8", marker="o", linewidth=2, label="ECB", alpha=0.9)
    plt.plot(sh_fed.index, sh_fed, color="#E67E22", marker="o", linewidth=2, label="Fed", alpha=0.9, linestyle="--")
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Normalized Shannon Entropy", fontsize=10)
    plt.xticks(rotation=45, fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    annotate_episodes(plt.gca())
    plt.tight_layout()
    save_fig("shannon-graph")


def _unique_words_pivot(emb):
    counts = (
        emb.groupby(["year", "region"])["word"].nunique()
           .reset_index()
           .query("year >= 2000")
    )
    pivot = counts.pivot(index="year", columns="region", values="word").fillna(0)
    pivot.index = pivot.index.astype(int)
    return pivot


def plot_unique_words(emb_bertopic):
    emb_raw = pd.read_parquet(os.path.join(METRICS, "df_2000-2025.parquet"))
    emb_raw["region"] = emb_raw["region"].map(REGION_MAP)

    raw_pivot  = _unique_words_pivot(emb_raw)
    bert_pivot = _unique_words_pivot(emb_bertopic).reindex(raw_pivot.index, fill_value=0)

    regions = raw_pivot.columns.tolist()
    years   = raw_pivot.index.values
    x       = np.arange(len(years))
    width   = 0.6 / len(regions)

    colors = {"ECB": "#2166ac", "Fed": "#d6604d"}

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, region in enumerate(regions):
        offset  = x + i * width
        covered = bert_pivot[region].values
        total   = raw_pivot[region].values
        ax.bar(offset, total,    width, color=colors[region], alpha=0.15,
               edgecolor=colors[region], linewidth=1.2, label=f"{region} (total vocab)")
        ax.bar(offset, covered,  width, color=colors[region], alpha=0.85,
               label=f"{region} (BERTopic)")

    ax.set_xticks(x + width * (len(regions) - 1) / 2)
    ax.set_xticklabels(years, rotation=90)
    ax.set_yscale("log")
    yticks = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Year")
    ax.set_ylabel("Unique words (log scale)")
    ax.set_title("Unique Words by Year and Institution")
    ax.legend(title="Institution", ncol=2)
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.7)
    plt.tight_layout()
    save_fig("unique_words")


def plot_inter_distances(region_distances, mode="per_topic"):
    heatmap = region_distances.pivot(index="year", columns="topic", values="cosine_distance").fillna(0)
    if mode == "per_topic":
        heatmap = heatmap.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0, axis=0)
    elif mode == "per_year":
        heatmap = heatmap.T.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0, axis=0).T
    elif mode == "global":
        flat      = heatmap.values.flatten()
        norm_flat = (flat - flat.min()) / (flat.max() - flat.min())
        heatmap   = pd.DataFrame(norm_flat.reshape(heatmap.shape), columns=heatmap.columns, index=heatmap.index)
    topic_order = heatmap.mean(axis=0).sort_values(ascending=False).index
    heatmap = heatmap[topic_order]
    plt.figure(figsize=(12, len(heatmap.columns) * 0.5 + 2))
    sns.heatmap(heatmap.T, cmap="Reds", linewidths=0.5, linecolor="white",
                cbar_kws={"label": f"Normalized Distance ({mode})"})
    plt.title(f"ECB–Fed Topic Distances Over Time ({mode})")
    plt.xlabel("Year")
    plt.ylabel("Topic")
    save_fig(f"region_distance_{mode}")


def plot_topic_shifts(shift_df, mode, region, color):
    cmap_map = {"red": "Reds", "blue": "Blues", "orange": "Oranges",
                "green": "Greens", "purple": "Purples"}
    heatmap = shift_df.pivot(index="topic", columns="year", values="cosine_distance")
    heatmap = heatmap.apply(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0, axis=1
    )
    topic_order = heatmap.T.mean(axis=0).sort_values(ascending=False).index
    heatmap = heatmap.T[topic_order].T
    plt.figure(figsize=(12, len(heatmap) * 0.5 + 2))
    sns.heatmap(heatmap, cmap=cmap_map.get(color, "Reds"), linewidths=0.5, linecolor="white",
                cbar_kws={"label": f"Normalized Distance ({mode})"})
    plt.title(f"Topic Shifts Over Time ({mode} - {region})")
    plt.xlabel("Year")
    plt.ylabel("Topic")
    save_fig(f"topic_shifts_{region}_{mode}")


# ── export ─────────────────────────────────────────────────────────────────────

def save_topic_counter_latex(topic_counter, out_dir):
    short_to_long = {v: k.title() for k, v in SYNTHETIC_MAPPING.items()}
    rows  = topic_counter[["short_topic", "word_count"]].sort_values("word_count", ascending=False)
    lines = [
        r"\begin{table}[h!]",
        r"    \centering",
        r"    \caption{Number of unique words per topic.}",
        r"    \label{tab:unique_words}",
        r"    \begin{tabular}{l c}",
        r"        \hline",
        r"        \textbf{Topic} & \textbf{Unique Words} \\",
        r"        \hline",
    ]
    for _, row in rows.iterrows():
        topic = short_to_long.get(row["short_topic"], str(row["short_topic"]).title())
        lines.append(f"        {topic} & {int(row['word_count'])} \\\\")
    lines += [r"        \hline", r"    \end{tabular}", r"\end{table}"]
    path = os.path.join(out_dir, "tab_unique_words.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"LaTeX table saved: {path}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- BERTopic data ---
    logger.info("Loading BERTopic outputs...")
    df = load_bertopic()
    logger.info(f"Loaded {df['topic'].nunique()} topics, {df['count'].sum()} total assignments")

    logger.info("Running label assignment...")
    df = assign_labels(df)

    topics_df     = prepare_topics_df(df)
    topic_counter = aggregate_word_counts(df)
    topic_counter.to_csv(os.path.join(FIG_DIR, "topic_counter.csv"), index=False)
    logger.info("Topic word counts:\n" +
                topic_counter[["short_topic", "word_count"]]
                .sort_values("word_count", ascending=False)
                .to_string(index=False))
    save_topic_counter_latex(topic_counter, FIG_DIR)

    # --- Topic frequency plots ---
    logger.info("Plotting stacked topic distributions...")
    pivot_fed = plot_stacked_topics(topics_df, "Fed")
    pivot_ecb = plot_stacked_topics(topics_df, "ECB")
    plot_pairwise_frequencies(pivot_fed, pivot_ecb)
    plot_shannon_entropy(pivot_fed, pivot_ecb)

    # --- Embeddings ---
    logger.info("Loading TWEC embeddings...")
    emb, label_to_words = load_embeddings(df)
    logger.info(f"Embeddings loaded: {len(emb)} rows, {emb['word'].nunique()} unique words, "
                f"years {emb['year'].min():.0f}–{emb['year'].max():.0f}")
    logger.info(f"Words matched to topics: {emb['word'].nunique()} "
                f"({emb['short_topic'].nunique()} topics covered)")

    plot_unique_words(emb_bertopic=emb)

    # --- Centroids ---
    logger.info("Computing weighted centroids...")
    centroids_ECB = compute_weighted_centroids(emb, label_to_words, "ECB")
    centroids_Fed = compute_weighted_centroids(emb, label_to_words, "Fed")
    logger.info(f"Centroids: ECB={len(centroids_ECB)} topics, Fed={len(centroids_Fed)} topics")

    # --- Inter-institution distances ---
    logger.info("Computing ECB–Fed inter-institution distances...")
    region_distances = compute_inter_distances(centroids_ECB, centroids_Fed)
    region_distances.round({"cosine_distance": 4}).to_csv(os.path.join(FIG_DIR, "distances_ECB_Fed.csv"), index=False)
    ranking = region_distances.groupby("topic")["cosine_distance"].mean().sort_values(ascending=False)
    logger.info("Mean ECB–Fed cosine distance per topic:\n" + ranking.to_string())
    for mode in ["per_topic", "per_year", "global"]:
        plot_inter_distances(region_distances, mode)

    # --- Intra-institution shifts ---
    logger.info("Computing intra-institution topic shifts...")
    for region, centroids, color in [("ECB", centroids_ECB, "blue"), ("Fed", centroids_Fed, "purple")]:
        fixed = compute_topic_shifts(centroids, "fixed")
        chain = compute_topic_shifts(centroids, "chain")
        fixed.round({"cosine_distance": 4}).to_csv(os.path.join(FIG_DIR, f"shifts_fixed_{region}.csv"), index=False)
        chain.round({"cosine_distance": 4}).to_csv(os.path.join(FIG_DIR, f"shifts_chain_{region}.csv"), index=False)
        plot_topic_shifts(fixed, "fixed", region, color)
        plot_topic_shifts(chain, "chain", region, color)
        logger.info(f"{region} fixed-base mean shifts:\n" +
                    fixed.groupby("topic")["cosine_distance"].mean()
                    .sort_values(ascending=False).to_string())

    logger.info(f"All outputs saved to {FIG_DIR}")
