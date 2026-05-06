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
from develop.utils.logger import LoggerManager

log_mgr = LoggerManager(name="bertopic", log_file="01_bertopic.log", clear_log=True)
logger  = log_mgr.get_logger()

GRID_DIR          = os.path.join(DATA, "bertopic_grid")
BEST_CONFIG_PATH  = os.path.join(GRID_DIR, "best_config.json")
DEFAULT_CONFIG    = {"n_neighbors": 5, "min_cluster_size": 20, "paragraphs_per_doc": 5}


def load_best_config() -> dict:
    if os.path.exists(BEST_CONFIG_PATH):
        with open(BEST_CONFIG_PATH) as f:
            cfg = json.load(f)
        logger.info(f"Grid config loaded: {cfg}")
        return cfg
    logger.warning(f"No grid config found at {BEST_CONFIG_PATH} — using defaults: {DEFAULT_CONFIG}")
    return DEFAULT_CONFIG.copy()


def load_documents(folder_path, paragraphs_per_doc=1):
    """
    Load speeches and group paragraphs into docs.
    Assumes filenames like: YYYY_REGION_something.txt
    Example: 2015_EU_speech1.txt
    """
    docs, years, regions = [], [], []

    files = sorted(Path(folder_path).glob("*.txt"))
    logger.info(f"Found {len(files)} files in {folder_path}")

    for file in files:
        parts = file.stem.split("_")
        year = int(parts[0])
        region = parts[1].upper()

        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

            for i in range(0, len(paragraphs), paragraphs_per_doc):
                grouped = " ".join(paragraphs[i:i+paragraphs_per_doc])
                if grouped.strip():
                    docs.append(grouped)
                    years.append(year)
                    regions.append(region)

    df = pd.DataFrame({"doc": docs, "year": years, "region": regions})
    logger.info(f"Loaded {len(df)} documents | "
                f"years {df['year'].min()}–{df['year'].max()} | "
                f"regions: {df['region'].value_counts().to_dict()} | "
                f"paragraphs_per_doc={paragraphs_per_doc}")
    return df


def load_or_compute_embeddings(docs, embedding_model: SentenceTransformer, cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        if len(embeddings) == len(docs):
            logger.info(f"Embeddings loaded from cache: {cache_path} ({len(embeddings)} docs)")
            return embeddings
        logger.info(f"Cache shape mismatch ({len(embeddings)} vs {len(docs)} docs) — recomputing.")

    logger.info(f"Computing sentence embeddings for {len(docs)} documents...")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    np.save(cache_path, embeddings)
    logger.info(f"Embeddings saved to cache: {cache_path}")
    return embeddings


def train_and_save(df, embedding_model: SentenceTransformer, cfg: dict) -> None:
    saving_folder = os.path.join(DATA, "01_bertopic")
    os.makedirs(saving_folder, exist_ok=True)

    ppd        = cfg["paragraphs_per_doc"]
    cache_path = os.path.join(saving_folder, f"doc_embeddings_p{ppd}.npy")
    embeddings = load_or_compute_embeddings(list(df["doc"]), embedding_model, cache_path)

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

    logger.info(
        f"Initializing BERTopic: "
        f"UMAP(n_neighbors={umap_model.n_neighbors}, n_components={umap_model.n_components}, "
        f"min_dist={umap_model.min_dist}, metric={umap_model.metric}, random_state={umap_model.random_state}), "
        f"HDBSCAN(min_cluster_size={hdbscan_model.min_cluster_size}, metric={hdbscan_model.metric}), "
        f"paragraphs_per_doc={ppd}, top_n_words=10"
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=10,
        language="english",
        calculate_probabilities=False,
        verbose=True,
    )

    logger.info(f"Fitting BERTopic on {len(df)} documents...")
    topics, _ = topic_model.fit_transform(df["doc"], embeddings=embeddings)
    df["topic"] = topics

    topic_info = topic_model.get_topic_info()
    n_topics  = len(topic_info[topic_info["Topic"] != -1])
    n_noise   = (df["topic"] == -1).sum()
    noise_pct = n_noise / len(df)
    logger.info(f"BERTopic done: {n_topics} topics extracted")
    logger.info(f"Noise documents (topic=-1): {n_noise}/{len(df)} ({noise_pct:.1%})")

    # Top words per topic
    topic_words = {
        t: topic_model.get_topic(t)
        for t in topic_info["Topic"] if t != -1
    }
    logger.info("Top 5 words per topic (first 10 topics):")
    for tid, words_probs in list(topic_words.items())[:10]:
        top = ", ".join(w for w, _ in words_probs[:5])
        logger.info(f"  Topic {tid:>3}: {top}")

    # Topic size distribution
    sizes = topic_info[topic_info["Topic"] != -1]["Count"].describe()
    logger.info(f"Topic size distribution: min={sizes['min']:.0f} "
                f"mean={sizes['mean']:.1f} max={sizes['max']:.0f}")

    # Save model
    topic_model.save("bertopic_model")
    logger.info("BERTopic model saved.")

    # Save topic words
    topic_words_df = [
        {"topic": tid, "word": word, "prob": prob}
        for tid, words_probs in topic_words.items()
        for word, prob in words_probs
    ]
    pd.DataFrame(topic_words_df).to_parquet(
        os.path.join(saving_folder, "bertopic_topic_words.parquet"),
        index=False
    )

    # Save topic prevalence
    topic_counts = (
        df.groupby(["year", "region", "topic"])
          .size()
          .reset_index(name="count")
    )
    topic_counts.to_parquet(
        os.path.join(saving_folder, "bertopic_topic_counts_year_region.parquet"),
        index=False
    )
    logger.info(f"Outputs saved to {saving_folder}")


# === Main ===
if __name__ == "__main__":
    cfg    = load_best_config()
    folder = os.path.join(DATA, "00_preprocessed_corpus")

    df              = load_documents(folder, cfg["paragraphs_per_doc"])
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    train_and_save(df.copy(), embedding_model, cfg)

    logger.info("=== BERTopic pipeline complete ===")
