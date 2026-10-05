"""
LDA vs BERTopic topic coherence benchmark.

Trains LDA (static) on the same preprocessed corpus and compares C_v coherence
against BERTopic's extracted topics. Both methods assign fixed, time-invariant
topic identities; the difference is representational — neural sentence embeddings
(BERTopic) vs. bag-of-words co-occurrence statistics (LDA).

BERTopic topic words are derived from saved bertopic_topic_words.parquet
via LabelAssigner aggregation. K = K_TOPICS for both methods.

Outputs
───────
  data/00b_dtm_baseline/coherence_comparison.csv
  data/00b_dtm_baseline/tab_coherence.tex
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import glob
import json
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel, Word2Vec
from itertools import combinations
from sentence_transformers import util as st_util
from sklearn.metrics.pairwise import cosine_distances

from develop.utils.paths import DATA
from develop.utils.logger import LoggerManager
from develop.core.vectors_flags.label_assinger import LabelAssigner

log_mgr = LoggerManager(name="lda_baseline", log_file="00b_lda_baseline.log", clear_log=True)
logger = log_mgr.get_logger()

CORPUS_DIR   = os.path.join(DATA, "00_preprocessed_corpus")
BERTOPIC_DIR = os.path.join(DATA, "01_bertopic")
COMPASS_PATH = os.path.join(DATA, "03_twec", "compass.model")
OUT_DIR      = os.path.join(DATA, "00b_dtm_baseline")

LDA_MODEL_PATH   = os.path.join(OUT_DIR, "lda.model")
BT_CACHE_PATH    = os.path.join(OUT_DIR, "bertopic_labels.json")

K_TOPICS     = 11
TOP_N_WORDS  = 10
LDA_PASSES   = 10
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)


# ── Corpus loading ─────────────────────────────────────────────────────────────

def load_corpus():
    docs, year_list = [], []
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt"))):
        year = int(os.path.basename(path)[:4])
        paras = [p.strip().split()
                 for p in open(path, encoding="utf-8").read().split("\n\n")
                 if p.strip()]
        docs.extend(paras)
        year_list.extend([year] * len(paras))
    return docs, year_list


# ── BERTopic topic words ───────────────────────────────────────────────────────

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


def bertopic_topic_words():
    """Replicate the pipeline's label assignment: encode each raw BERTopic topic
    as its word list string, assign a label, then aggregate words per label."""
    tw = pd.read_parquet(os.path.join(BERTOPIC_DIR, "bertopic_topic_words.parquet"))
    tw = tw[tw["topic"] != -1].copy()

    # Build topic-level word lists (same structure as pipeline's load_bertopic)
    topic_to_words = (tw.sort_values("prob", ascending=False)
                        .groupby("topic")["word"]
                        .apply(list)
                        .to_dict())

    # One row per raw topic with word list as the 'word' column
    topic_df = pd.DataFrame([
        {"topic": t, "word": words, "count": len(words)}
        for t, words in topic_to_words.items()
    ])

    la = LabelAssigner.get_instance()
    topic_df = la.assign_labels(topic_df, auto_threshold=True)
    topic_df["short_topic"] = topic_df["label"].map(SYNTHETIC_MAPPING)
    topic_df = topic_df.dropna(subset=["short_topic"])

    # Aggregate: collect all words from raw topics mapped to each short label
    label_to_words = {}
    for short, grp in topic_df.groupby("short_topic"):
        all_topics = grp["topic"].tolist()
        word_prob = {}
        for t in all_topics:
            for _, row in tw[tw["topic"] == t].iterrows():
                word_prob[row["word"]] = word_prob.get(row["word"], 0) + row["prob"]
        top = sorted(word_prob, key=lambda x: -word_prob[x])[:TOP_N_WORDS]
        label_to_words[short] = top

    logger.info(f"BERTopic labels ({len(label_to_words)}): {list(label_to_words.keys())}")
    return label_to_words


# ── Coherence ─────────────────────────────────────────────────────────────────

def cv_coherence(topic_word_lists, tokenized_docs, dictionary):
    valid = [[w for w in ws if w in dictionary.token2id] for ws in topic_word_lists]
    valid = [ws for ws in valid if len(ws) >= 5]
    if not valid:
        return float("nan")
    cm = CoherenceModel(topics=valid, texts=tokenized_docs,
                        dictionary=dictionary, coherence="c_v")
    return cm.get_coherence()


def embedding_coherence(topic_word_lists, wv):
    """Mean pairwise cosine similarity between topic word vectors (averaged across topics).
    Uses the TWEC compass word vectors — the same embedding space as the paper."""
    topic_scores = []
    for words in topic_word_lists:
        vecs = [wv[w] for w in words if w in wv]
        if len(vecs) < 2:
            continue
        vecs = np.array(vecs)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms == 0, 1, norms)
        pairs = list(combinations(range(len(vecs)), 2))
        sims = [float(vecs[i] @ vecs[j]) for i, j in pairs]
        topic_scores.append(np.mean(sims))
    return float(np.mean(topic_scores)) if topic_scores else float("nan")


def npmi_coherence(topic_word_lists, tokenized_docs, dictionary):
    valid = [[w for w in ws if w in dictionary.token2id] for ws in topic_word_lists]
    valid = [ws for ws in valid if len(ws) >= 2]
    if not valid:
        return float("nan")
    cm = CoherenceModel(topics=valid, texts=tokenized_docs,
                        dictionary=dictionary, coherence="c_npmi")
    return cm.get_coherence()


def topic_diversity(topic_word_lists):
    """Fraction of unique words across all topic top-N lists."""
    all_words = [w for ws in topic_word_lists for w in ws]
    if not all_words:
        return float("nan")
    return len(set(all_words)) / len(all_words)


def topic_uniqueness(topic_word_lists):
    """Mean fraction of each topic's words that appear in no other topic."""
    word_counts = {}
    for ws in topic_word_lists:
        for w in ws:
            word_counts[w] = word_counts.get(w, 0) + 1
    scores = []
    for ws in topic_word_lists:
        if not ws:
            continue
        unique_frac = sum(1 for w in ws if word_counts[w] == 1) / len(ws)
        scores.append(unique_frac)
    return float(np.mean(scores)) if scores else float("nan")


def pairwise_overlap(topic_word_lists):
    """Mean Jaccard similarity between all topic pairs (lower = more distinct)."""
    pairs = list(combinations(range(len(topic_word_lists)), 2))
    if not pairs:
        return float("nan")
    sims = []
    for i, j in pairs:
        a, b = set(topic_word_lists[i]), set(topic_word_lists[j])
        union = a | b
        sims.append(len(a & b) / len(union) if union else 0.0)
    return float(np.mean(sims))


def davies_bouldin_cosine(topic_word_lists, wv):
    """Davies-Bouldin index with cosine distance, using TWEC compass word vectors.
    Each topic's words form a cluster; lower = better-separated topics."""
    embs, labels = [], []
    for i, words in enumerate(topic_word_lists):
        for w in words:
            if w in wv:
                embs.append(wv[w])
                labels.append(i)
    if len(set(labels)) < 2:
        return float("nan")
    embs   = np.array(embs)
    labels = np.array(labels)
    unique = np.unique(labels)
    centroids  = np.stack([embs[labels == k].mean(axis=0) for k in unique])
    dists_to_c = cosine_distances(embs, centroids)
    s = np.array([dists_to_c[labels == k, i].mean() for i, k in enumerate(unique)])
    c_dists = cosine_distances(centroids)
    np.fill_diagonal(c_dists, np.inf)
    R = ((s[:, None] + s[None, :]) / c_dists).max(axis=1)
    return float(R.mean())


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Loading corpus...")
    docs, year_list = load_corpus()
    logger.info(f"  {len(docs):,} paragraphs across {len(set(year_list))} years")

    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=10, no_above=0.2)
    bow_corpus = [dictionary.doc2bow(d) for d in docs]
    logger.info(f"  Dictionary: {len(dictionary):,} tokens after filtering")

    logger.info("Loading TWEC compass for embedding coherence...")
    compass = Word2Vec.load(COMPASS_PATH)
    wv = compass.wv
    logger.info(f"  Compass vocab: {len(wv):,} words")

    rows = []

    # ── BERTopic (neural embeddings, fixed topics) ─────────────────────────────
    bt_parquet = os.path.join(BERTOPIC_DIR, "bertopic_topic_words.parquet")
    cache_stale = (
        not os.path.exists(BT_CACHE_PATH) or
        os.path.getmtime(bt_parquet) > os.path.getmtime(BT_CACHE_PATH)
    )
    if not cache_stale:
        logger.info(f"Loading BERTopic labels from cache: {BT_CACHE_PATH}")
        with open(BT_CACHE_PATH) as f:
            bt_topics = json.load(f)
    else:
        logger.info("Computing BERTopic topic labels...")
        bt_topics = bertopic_topic_words()
        with open(BT_CACHE_PATH, "w") as f:
            json.dump(bt_topics, f, indent=2)
        logger.info(f"  Cached to {BT_CACHE_PATH}")

    bt_words   = list(bt_topics.values())
    bt_cv      = cv_coherence(bt_words, docs, dictionary)
    bt_npmi    = npmi_coherence(bt_words, docs, dictionary)
    bt_emb     = embedding_coherence(bt_words, wv)
    bt_div     = topic_diversity(bt_words)
    bt_uniq    = topic_uniqueness(bt_words)
    bt_overlap = pairwise_overlap(bt_words)
    bt_db      = davies_bouldin_cosine(bt_words, wv)

    logger.info(f"  BERTopic C_v={bt_cv:.4f}  NPMI={bt_npmi:.4f}  emb={bt_emb:.4f}"
                f"  div={bt_div:.4f}  uniq={bt_uniq:.4f}  overlap={bt_overlap:.4f}"
                f"  DB={bt_db:.4f}  (K={len(bt_topics)})")
    rows.append({"Method": "BERTopic", "K": len(bt_topics),
                 "C_v": round(bt_cv, 4), "NPMI": round(bt_npmi, 4),
                 "Emb. coherence": round(bt_emb, 4),
                 "Diversity": round(bt_div, 4), "Uniqueness": round(bt_uniq, 4),
                 "Overlap": round(bt_overlap, 4), "DB": round(bt_db, 4)})

    la = LabelAssigner.get_instance()

    # ── LDA (bag-of-words, fixed topics) ──────────────────────────────────────
    if os.path.exists(LDA_MODEL_PATH):
        lda = LdaModel.load(LDA_MODEL_PATH)
        if lda.num_topics != K_TOPICS:
            logger.info(f"Cached LDA has {lda.num_topics} topics but K_TOPICS={K_TOPICS} — retraining...")
            lda = None
        else:
            logger.info(f"Loading LDA model from cache: {LDA_MODEL_PATH}")
    else:
        lda = None

    if lda is None:
        logger.info(f"Training LDA (K={K_TOPICS}, passes={LDA_PASSES})...")
        lda = LdaModel(corpus=bow_corpus, id2word=dictionary,
                       num_topics=K_TOPICS, passes=LDA_PASSES,
                       random_state=RANDOM_STATE)
        lda.save(LDA_MODEL_PATH)
        logger.info(f"  Saved to {LDA_MODEL_PATH}")

    lda_raw_words = [[w for w, _ in lda.show_topic(t, topn=TOP_N_WORDS)]
                     for t in range(K_TOPICS)]

    # Assign labels via nearest-label cosine similarity (no threshold — stable with K=9)
    label_names = la.labels
    # Build full (n_labels × n_topics) similarity matrix
    topic_embs = la.model.encode([" ".join(w) for w in lda_raw_words], convert_to_numpy=True)
    sim_matrix = st_util.cos_sim(la.label_embs, topic_embs).numpy()  # (n_labels, K)

    # Greedy one-to-one assignment: pick global max, assign, mask row+col, repeat
    assigned_topics = {}
    available_labels = set(range(len(label_names)))
    available_topics = set(range(K_TOPICS))
    while available_topics:
        best_score, best_l, best_t = -1, None, None
        for l in available_labels:
            for t in available_topics:
                if sim_matrix[l, t] > best_score:
                    best_score, best_l, best_t = sim_matrix[l, t], l, t
        short = SYNTHETIC_MAPPING.get(label_names[best_l], f"Topic_{best_t}")
        assigned_topics[short] = lda_raw_words[best_t]
        available_labels.discard(best_l)
        available_topics.discard(best_t)

    lda_topics = assigned_topics
    lda_words  = list(lda_topics.values())
    lda_labels = list(lda_topics.keys())
    logger.info(f"  LDA labels: {lda_labels}")

    lda_cv      = cv_coherence(lda_words, docs, dictionary)
    lda_npmi    = npmi_coherence(lda_words, docs, dictionary)
    lda_emb     = embedding_coherence(lda_words, wv)
    lda_div     = topic_diversity(lda_words)
    lda_uniq    = topic_uniqueness(lda_words)
    lda_overlap = pairwise_overlap(lda_words)
    lda_db      = davies_bouldin_cosine(lda_words, wv)

    logger.info(f"  LDA C_v={lda_cv:.4f}  NPMI={lda_npmi:.4f}  emb={lda_emb:.4f}"
                f"  div={lda_div:.4f}  uniq={lda_uniq:.4f}  overlap={lda_overlap:.4f}"
                f"  DB={lda_db:.4f}")
    rows.append({"Method": "LDA", "K": K_TOPICS,
                 "C_v": round(lda_cv, 4), "NPMI": round(lda_npmi, 4),
                 "Emb. coherence": round(lda_emb, 4),
                 "Diversity": round(lda_div, 4), "Uniqueness": round(lda_uniq, 4),
                 "Overlap": round(lda_overlap, 4), "DB": round(lda_db, 4)})

    # ── Save topic words CSV ──────────────────────────────────────────────────
    # Wide format: one row per topic, words as comma-separated string
    all_topics = list(set(list(bt_topics.keys()) + list(lda_topics.keys())))
    wide_rows = []
    for topic in sorted(all_topics):
        wide_rows.append({
            "topic":           topic,
            "BERTopic_words":  ", ".join(bt_topics.get(topic, [])),
            "LDA_words":       ", ".join(lda_topics.get(topic, [])),
        })
    pd.DataFrame(wide_rows).to_csv(os.path.join(OUT_DIR, "topic_words.csv"), index=False)
    logger.info("Saved topic_words.csv")

    # ── Save coherence results ─────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "coherence_comparison.csv"), index=False)
    logger.info(f"\n{df.to_string(index=False)}")

    # higher-is-better cols: bold max; lower-is-better cols (Overlap, DB): bold min
    higher_better = ["C_v", "NPMI", "Emb. coherence", "Diversity", "Uniqueness"]
    lower_better  = ["Overlap", "DB"]
    best = {col: df[col].max() for col in higher_better}
    best.update({col: df[col].min() for col in lower_better})

    def fmt(val, col):
        s = f"{val:.4f}"
        return r"\textbf{" + s + r"}" if val == best[col] else s

    lines = [
        r"\begin{table}[h!]", r"\centering", r"\small",
        r"\begin{tabular}{lcccccccc}", r"\toprule",
        (r"Method & $K$ & $C_v$ & NPMI & Emb.\ coh. & "
         r"Diversity & Uniqueness & Overlap & DB \\"),
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['Method']} & {int(row['K'])} & {fmt(row['C_v'], 'C_v')} "
            f"& {fmt(row['NPMI'], 'NPMI')} & {fmt(row['Emb. coherence'], 'Emb. coherence')} "
            f"& {fmt(row['Diversity'], 'Diversity')} & {fmt(row['Uniqueness'], 'Uniqueness')} "
            f"& {fmt(row['Overlap'], 'Overlap')} & {fmt(row['DB'], 'DB')} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Topic quality comparison between BERTopic and LDA ($K=" + str(K_TOPICS) + r"$, "
        r"same corpus). $C_v$ and NPMI measure token co-occurrence coherence and "
        r"inherently favour bag-of-words models such as LDA. "
        r"Embedding coherence is the mean pairwise cosine similarity of topic word "
        r"vectors in the TWEC compass space --- the same embedding space used in the "
        r"paper --- and is therefore a more appropriate metric for evaluating "
        r"BERTopic's semantic quality. "
        r"Diversity is the fraction of unique words across all topic word lists. "
        r"Uniqueness is the mean fraction of each topic's words that appear in no "
        r"other topic. Overlap is the mean Jaccard similarity between topic pairs "
        r"(lower = more distinct). DB is the Davies-Bouldin index \citep{davies_bouldin_1979} "
        r"computed with cosine distance in the TWEC compass space (lower = better-separated topics). "
        r"Higher values are better for all metrics except Overlap and DB.}",
        r"\label{tab:coherence_comparison}",
        r"\end{table}",
    ]
    tex_path = os.path.join(OUT_DIR, "tab_coherence.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Saved {tex_path}")
    logger.info("Done.")
