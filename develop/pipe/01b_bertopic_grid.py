import argparse
import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer

from develop.utils.paths import DATA
from develop.core.vectors_flags.label_assinger import LabelAssigner
from develop.utils.logger import LoggerManager

log_mgr = LoggerManager(name="bertopic_grid", log_file="01b_bertopic_grid.log", clear_log=True)
logger = log_mgr.get_logger()

CORPUS_DIR   = os.path.join(DATA, "00_preprocessed_corpus")
BERTOPIC_DIR = os.path.join(DATA, "01_bertopic")
GRID_DIR     = os.path.join(DATA, "bertopic_grid")

# Full factorial grid — fixed: n_components=5, min_dist=0.0, metric=cosine, random_state=1
N_NEIGHBORS_VALS      = [5, 10, 15]
MIN_CLUSTER_SIZES     = [5, 10, 15]
PARAGRAPHS_PER_DOC_VALS = [2, 3, 5, 7, 10]

CONFIGS = [
    {"n_neighbors": nn, "min_cluster_size": mc, "paragraphs_per_doc": ppd}
    for nn, mc, ppd in itertools.product(N_NEIGHBORS_VALS, MIN_CLUSTER_SIZES, PARAGRAPHS_PER_DOC_VALS)
]


def config_id(cfg):
    return f"n{cfg['n_neighbors']}_mc{cfg['min_cluster_size']}_p{cfg['paragraphs_per_doc']}"


def load_documents(folder_path, paragraphs_per_doc):
    docs, years, regions = [], [], []
    for file in sorted(Path(folder_path).glob("*.txt")):
        parts = file.stem.split("_")
        year, region = int(parts[0]), parts[1].upper()
        with open(file, "r", encoding="utf-8") as f:
            paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]
            for i in range(0, len(paragraphs), paragraphs_per_doc):
                grouped = " ".join(paragraphs[i:i + paragraphs_per_doc])
                if grouped.strip():
                    docs.append(grouped)
                    years.append(year)
                    regions.append(region)
    return pd.DataFrame({"doc": docs, "year": years, "region": regions})


def get_embeddings(docs_df, emb_model, paragraphs_per_doc):
    cache_path = os.path.join(BERTOPIC_DIR, f"doc_embeddings_p{paragraphs_per_doc}.npy")
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        if len(embeddings) == len(docs_df):
            logger.info(f"  Embeddings loaded from cache: {cache_path} ({len(embeddings)} docs)")
            return embeddings
        logger.info(f"  Cache shape mismatch — recomputing.")
    logger.info(f"  Computing embeddings for {len(docs_df)} docs (paragraphs_per_doc={paragraphs_per_doc})...")
    embeddings = emb_model.encode(docs_df["doc"].tolist(), show_progress_bar=False)
    np.save(cache_path, embeddings)
    logger.info(f"  Embeddings saved: {cache_path}")
    return embeddings


def run_bertopic_config(cfg, docs_df, embeddings, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    umap_model = UMAP(
        n_neighbors=cfg["n_neighbors"],
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        low_memory=False,
        random_state=1,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg["min_cluster_size"],
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=10,
        language="english",
        calculate_probabilities=False,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(docs_df["doc"].tolist(), embeddings=embeddings)
    df = docs_df.copy()
    df["topic"] = topics

    topic_info = topic_model.get_topic_info()
    n_topics   = len(topic_info[topic_info["Topic"] != -1])
    noise_rate = (df["topic"] == -1).sum() / len(df)

    topic_words = {t: topic_model.get_topic(t) for t in topic_info["Topic"] if t != -1}
    pd.DataFrame([
        {"topic": tid, "word": word, "prob": prob}
        for tid, wps in topic_words.items()
        for word, prob in wps
    ]).to_parquet(os.path.join(out_dir, "bertopic_topic_words.parquet"), index=False)

    (
        df.groupby(["year", "region", "topic"]).size().reset_index(name="count")
    ).to_parquet(os.path.join(out_dir, "bertopic_topic_counts_year_region.parquet"), index=False)

    return n_topics, noise_rate


def run_label_assignment(out_dir):
    df  = pd.read_parquet(os.path.join(out_dir, "bertopic_topic_counts_year_region.parquet"))
    raw = pd.read_parquet(os.path.join(out_dir, "bertopic_topic_words.parquet"))

    topic_to_words = dict(zip(
        raw.groupby("topic")["word"].apply(list).index,
        raw.groupby("topic")["word"].apply(list).values,
    ))
    df["word"] = df["topic"].map(topic_to_words)

    assigner = LabelAssigner.get_instance()
    with contextlib.redirect_stdout(io.StringIO()):
        df = assigner.assign_labels(df)

    n_total      = df["topic"].nunique()
    n_labeled    = df[~df["label"].isna()]["topic"].nunique()
    n_unlabeled  = n_total - n_labeled
    labeled_rate = n_labeled / n_total

    total_docs     = df["count"].sum()
    labeled_topics = set(df[~df["label"].isna()]["topic"].unique())
    doc_coverage   = df[df["topic"].isin(labeled_topics)]["count"].sum() / total_docs

    return labeled_rate, n_total, n_labeled, n_unlabeled, assigner.threshold, doc_coverage


def load_cached_result(out_dir):
    path = os.path.join(out_dir, "config_result.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_result(out_dir, result):
    path = os.path.join(out_dir, "config_result.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def pareto_front(results):
    valid = [r for r in results if r["labeled_rate"] is not None and r["n_labeled"] is not None]
    pareto = []
    for r in valid:
        dominated = any(
            other["labeled_rate"] >= r["labeled_rate"]
            and other["n_labeled"] >= r["n_labeled"]
            and other is not r
            for other in valid
        )
        if not dominated:
            pareto.append(r)
    return pareto


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="Rerun all configurations from scratch, ignoring cached results")
    args = parser.parse_args()

    os.makedirs(GRID_DIR, exist_ok=True)

    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    results   = []
    total     = len(CONFIGS)

    for ppd in PARAGRAPHS_PER_DOC_VALS:
        logger.info(f"=== paragraphs_per_doc={ppd} ===")
        docs_df    = load_documents(CORPUS_DIR, ppd)
        logger.info(f"Loaded {len(docs_df)} documents")
        embeddings = get_embeddings(docs_df, emb_model, ppd)

        ppd_configs = [c for c in CONFIGS if c["paragraphs_per_doc"] == ppd]
        for cfg in ppd_configs:
            i   = CONFIGS.index(cfg) + 1
            cid = config_id(cfg)
            out_dir = os.path.join(GRID_DIR, cid)

            if not args.overwrite:
                cached = load_cached_result(out_dir)
                if cached is not None:
                    logger.info(f"[{i}/{total}] {cid} — skipped (cached)")
                    results.append(cached)
                    continue

            logger.info(f"[{i}/{total}] {cid} — running BERTopic...")
            n_topics, noise_rate = run_bertopic_config(cfg, docs_df, embeddings, out_dir)
            logger.info(f"  BERTopic: {n_topics} topics, noise={noise_rate:.1%}")

            if n_topics < 10:
                logger.warning(f"  Skipping label assignment: too few topics ({n_topics})")
                labeled_rate, n_total, n_labeled, n_unlabeled, tau, doc_coverage = None, n_topics, None, None, None, None
            else:
                try:
                    labeled_rate, n_total, n_labeled, n_unlabeled, tau, doc_coverage = run_label_assignment(out_dir)
                    logger.info(f"  Labels: {n_labeled}/{n_total} labeled ({labeled_rate:.1%}), "
                                f"doc_coverage={doc_coverage:.1%}, τ*={tau:.2f}")
                except Exception as e:
                    logger.warning(f"  Label assignment failed: {e}")
                    labeled_rate, n_total, n_labeled, n_unlabeled, tau, doc_coverage = None, n_topics, None, None, None, None

            result = {
                "config":             cid,
                "paragraphs_per_doc": ppd,
                "n_neighbors":        cfg["n_neighbors"],
                "min_cluster_size":   cfg["min_cluster_size"],
                "n_topics":           n_topics,
                "noise_rate":         round(noise_rate, 4),
                "n_labeled":          n_labeled,
                "n_unlabeled":        n_unlabeled,
                "labeled_rate":       round(labeled_rate, 4) if labeled_rate is not None else None,
                "doc_coverage":       round(doc_coverage, 4) if doc_coverage is not None else None,
                "tau_star":           round(tau, 2) if tau is not None else None,
            }
            save_result(out_dir, result)
            results.append(result)

    summary    = pd.DataFrame(results)
    pareto_ids = {r["config"] for r in pareto_front(results)}
    summary["is_pareto"] = summary["config"].isin(pareto_ids)

    # Harmonic mean of:
    #   labeled_rate — fraction of topics passing τ* (minimises unlabeled micro-topics)
    #   doc_coverage — fraction of documents in labeled topics (minimises BERTopic noise + unlabeled topics)
    valid_mask = summary["labeled_rate"].notna() & summary["doc_coverage"].notna()
    lr = summary.loc[valid_mask, "labeled_rate"]
    dc = summary.loc[valid_mask, "doc_coverage"]
    summary.loc[valid_mask, "hmean_score"] = 2 * lr * dc / (lr + dc)
    summary = summary.sort_values("hmean_score", ascending=False)

    best = summary[valid_mask].iloc[0]
    logger.info("\n=== Grid Search Summary ===\n" + summary.to_string(index=False))
    logger.info(f"Pareto-optimal: {sorted(pareto_ids)}")
    logger.info(
        f"Best config (hmean): {best['config']} — "
        f"paragraphs_per_doc={int(best['paragraphs_per_doc'])}, "
        f"labeled_rate={best['labeled_rate']:.3f}, "
        f"n_labeled={int(best['n_labeled'])}, "
        f"doc_coverage={best['doc_coverage']:.3f}, "
        f"hmean={best['hmean_score']:.3f}, "
        f"τ*={best['tau_star']}"
    )

    summary_path = os.path.join(GRID_DIR, "grid_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

    best_cfg = {
        "n_neighbors":        int(best["n_neighbors"]),
        "min_cluster_size":   int(best["min_cluster_size"]),
        "paragraphs_per_doc": int(best["paragraphs_per_doc"]),
    }
    best_cfg_path = os.path.join(GRID_DIR, "best_config.json")
    with open(best_cfg_path, "w") as f:
        json.dump(best_cfg, f, indent=2)
    logger.info(f"Best config saved to {best_cfg_path}")
