"""
Two figures:

  freq_distance_scatter.png
      Left panel  — raw scatter: topic frequency share vs ECB-Fed centroid
                    distance, ECB circles, Fed squares, coloured by topic.
      Right panel — within-topic scatter: both variables demeaned by topic.

  freq_distance_within_by_topic.png
      3x3 faceted grid.  One subplot per topic, showing the within-topic
      relationship (demeaned share vs demeaned distance) with its own OLS
      regression line and r / p-value.  All subplots share the same scale.

Outputs
-------
  data/05_graphs/freq_distance_scatter.png
  data/05_graphs/freq_distance_within_by_topic.png
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

from develop.utils.paths import DATA
from develop.core.vectors_flags.label_assinger import LabelAssigner
from develop.utils.logger import LoggerManager

log_mgr = LoggerManager(name="freq_distance_scatter",
                        log_file="05b_freq_distance_scatter.log", clear_log=True)
logger = log_mgr.get_logger()

BERTOPIC        = os.path.join(DATA, "01_bertopic")
GRAPHS          = os.path.join(DATA, "05_graphs")
OUT_PATH        = os.path.join(GRAPHS, "freq_distance_scatter.png")
OUT_PATH_TOPIC  = os.path.join(GRAPHS, "freq_distance_within_by_topic.png")

REGION_MAP = {"USA": "Fed", "EU": "ECB"}

SYNTHETIC_MAPPING = {
    "monetary policy and interest rates":        "Interest Rates",
    "inflation and consumer prices":             "Inflation",
    "labor market and employment":               "Employment",
    "financial markets and banking":             "Financial Markets",
    "fiscal policy and government spending":     "Fiscal Policy",
    "international trade and global economy":    "International Trade",
    "economic development and inequality":       "Inequality",
    "climate change and sustainable finance":    "Climate Change",
    "industrial and sectoral economics":         "Industries",
    "behavioral economics and expectations":     "Behavioral Economics",
    "policy communication and forward guidance": "Forward Guidance",
}

TOPIC_COLORS = {
    "Interest Rates":       "#1f77b4",
    "Inflation":            "#d62728",
    "Employment":           "#ff7f0e",
    "Financial Markets":    "#9467bd",
    "Fiscal Policy":        "#8c564b",
    "International Trade":  "#17becf",
    "Inequality":           "#e377c2",
    "Climate Change":       "#2ca02c",
    "Industries":           "#bcbd22",
    "Behavioral Economics": "#7f7f7f",
    "Forward Guidance":     "#aec7e8",
}


# ── data loading ───────────────────────────────────────────────────────────────

def load_freq_shares():
    df = pd.read_parquet(os.path.join(BERTOPIC, "bertopic_topic_counts_year_region.parquet"))
    raw_words = pd.read_parquet(os.path.join(BERTOPIC, "bertopic_topic_words.parquet"))
    topic_to_words = raw_words.groupby("topic")["word"].apply(list).to_dict()
    df["word"] = df["topic"].map(topic_to_words)

    assigner = LabelAssigner.get_instance()
    df = assigner.assign_labels(df)
    df["region"]      = df["region"].map(REGION_MAP)
    df["short_topic"] = df["label"].map(SYNTHETIC_MAPPING)

    df = (df.groupby(["year", "region", "short_topic"])["count"]
            .sum()
            .reset_index()
            .query("2000 <= year <= 2025")
            .loc[lambda d: d["short_topic"].notna()])

    totals = df.groupby(["year", "region"])["count"].transform("sum")
    df["share"] = df["count"] / totals
    return df


def load_distances():
    return pd.read_csv(os.path.join(GRAPHS, "distances_ECB_Fed.csv"))


def build_merged(freq, dist):
    ecb = freq[freq["region"] == "ECB"][["year", "short_topic", "share"]].rename(
        columns={"share": "ecb_share"})
    fed = freq[freq["region"] == "Fed"][["year", "short_topic", "share"]].rename(
        columns={"share": "fed_share"})
    merged = (ecb.merge(fed, on=["year", "short_topic"])
                 .merge(dist.rename(columns={"topic": "short_topic",
                                             "cosine_distance": "distance"}),
                        on=["year", "short_topic"]))
    topic_means = merged.groupby("short_topic")[
        ["ecb_share", "fed_share", "distance"]].transform("mean")
    merged["ecb_dm"]  = merged["ecb_share"] - topic_means["ecb_share"]
    merged["fed_dm"]  = merged["fed_share"]  - topic_means["fed_share"]
    merged["dist_dm"] = merged["distance"]   - topic_means["distance"]
    return merged


# ── helpers ────────────────────────────────────────────────────────────────────

def add_regression(ax, x, y, x0, x1, fontsize=9):
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x0, x1, 200)
    ax.plot(x_line, intercept + slope * x_line,
            color="#333333", linewidth=1.5, linestyle="--", zorder=4)
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.97, 0.03, f"r = {r:.2f}\n{p_str}",
            transform=ax.transAxes, fontsize=fontsize, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"))


def shared_legend(fig, topics, anchor_topics, anchor_markers):
    topic_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=TOPIC_COLORS.get(t, "#333"), markersize=7)
        for t in topics
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#555555", markersize=7, label="ECB"),
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor="#555555", markersize=7, label="Fed"),
    ]
    fig.legend(topic_handles, topics,
               loc="lower center", bbox_to_anchor=anchor_topics,
               ncol=5, fontsize=8, frameon=False)
    fig.legend(handles=marker_handles,
               loc="lower center", bbox_to_anchor=anchor_markers,
               ncol=2, fontsize=9, frameon=False)


# ── figure 1: raw + within-topic ──────────────────────────────────────────────

def plot(merged):
    topics = sorted(merged["short_topic"].unique())
    x_max  = max(merged["ecb_share"].max(), merged["fed_share"].max()) * 1.06
    y_max  = merged["distance"].max() * 1.06
    x_abs  = max(merged["ecb_dm"].abs().max(), merged["fed_dm"].abs().max()) * 1.08
    y_abs  = merged["dist_dm"].abs().max() * 1.08

    fig, (ax_raw, ax_dm) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    for topic in topics:
        sub   = merged[merged["short_topic"] == topic]
        color = TOPIC_COLORS.get(topic, "#333333")
        for ax, xcol in [(ax_raw, "ecb_share"), (ax_dm, "ecb_dm")]:
            ax.scatter(sub[xcol], sub["distance"] if ax is ax_raw else sub["dist_dm"],
                       color=color, marker="o", s=55, alpha=0.80,
                       edgecolors="white", linewidths=0.35, zorder=3,
                       label=topic if ax is ax_raw else None)
        for ax, xcol in [(ax_raw, "fed_share"), (ax_dm, "fed_dm")]:
            ax.scatter(sub[xcol], sub["distance"] if ax is ax_raw else sub["dist_dm"],
                       color=color, marker="s", s=50, alpha=0.80,
                       edgecolors="white", linewidths=0.35, zorder=3)

    add_regression(ax_raw,
                   pd.concat([merged["ecb_share"], merged["fed_share"]]),
                   pd.concat([merged["distance"],  merged["distance"]]),
                   0, x_max)
    ax_raw.set_xlim(0, x_max); ax_raw.set_ylim(0, y_max)
    ax_raw.set_xlabel("Frequency share", fontsize=11)
    ax_raw.set_ylabel("Cosine distance (ECB vs Fed centroid)", fontsize=11)
    ax_raw.set_title("Raw", fontsize=13, fontweight="bold")
    ax_raw.tick_params(labelsize=9)

    add_regression(ax_dm,
                   pd.concat([merged["ecb_dm"], merged["fed_dm"]]),
                   pd.concat([merged["dist_dm"], merged["dist_dm"]]),
                   -x_abs, x_abs)
    ax_dm.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
    ax_dm.axvline(0, color="#cccccc", linewidth=0.8, zorder=1)
    ax_dm.set_xlim(-x_abs, x_abs); ax_dm.set_ylim(-y_abs, y_abs)
    ax_dm.set_xlabel("Frequency share (deviation from topic mean)", fontsize=11)
    ax_dm.set_ylabel("Cosine distance (deviation from topic mean)", fontsize=11)
    ax_dm.set_title("Within-topic", fontsize=13, fontweight="bold")
    ax_dm.tick_params(labelsize=9)

    shared_legend(fig, topics, anchor_topics=(0.42, -0.10), anchor_markers=(0.82, -0.10))
    fig.suptitle("Topic frequency share vs ECB-Fed centroid distance", fontsize=13)

    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {OUT_PATH}")
    plt.close()


# ── figure 2: within-topic per topic ──────────────────────────────────────────

def plot_within_by_topic(merged):
    topics = sorted(merged["short_topic"].unique())
    ncols, nrows = 3, int(np.ceil(len(topics) / 3))

    # shared symmetric scale across all subplots
    x_abs = max(merged["ecb_dm"].abs().max(), merged["fed_dm"].abs().max()) * 1.08
    y_abs = merged["dist_dm"].abs().max() * 1.08

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5 * nrows),
                             constrained_layout=True)
    axes = axes.flatten()

    for ax, topic in zip(axes, topics):
        sub   = merged[merged["short_topic"] == topic]
        color = TOPIC_COLORS.get(topic, "#333333")

        ax.scatter(sub["ecb_dm"], sub["dist_dm"],
                   color=color, marker="o", s=60, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3, label="ECB")
        ax.scatter(sub["fed_dm"], sub["dist_dm"],
                   color=color, marker="s", s=55, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3, label="Fed")

        x_all = pd.concat([sub["ecb_dm"], sub["fed_dm"]])
        y_all = pd.concat([sub["dist_dm"], sub["dist_dm"]])
        if len(x_all.dropna()) >= 4:
            add_regression(ax, x_all, y_all, -x_abs, x_abs, fontsize=8)

        ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
        ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=1)
        ax.set_xlim(-x_abs, x_abs)
        ax.set_ylim(-y_abs, y_abs)
        ax.set_title(topic, fontsize=11, fontweight="bold",
                     color=TOPIC_COLORS.get(topic, "#333333"))
        ax.set_xlabel("Share deviation", fontsize=9)
        ax.set_ylabel("Distance deviation", fontsize=9)
        ax.tick_params(labelsize=8)

    for ax in axes[len(topics):]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#555555", markersize=7, label="ECB"),
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor="#555555", markersize=7, label="Fed"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Within-topic: frequency share deviation vs distance deviation",
                 fontsize=13)

    plt.savefig(OUT_PATH_TOPIC, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {OUT_PATH_TOPIC}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Loading topic frequency shares...")
    freq = load_freq_shares()
    logger.info(f"  {len(freq)} year x institution x topic rows")

    logger.info("Loading ECB-Fed distances...")
    dist = load_distances()
    logger.info(f"  {len(dist)} year x topic rows")

    logger.info("Building merged dataset...")
    merged = build_merged(freq, dist)

    logger.info("Plotting...")
    plot(merged)
    plot_within_by_topic(merged)
    logger.info("Done.")
