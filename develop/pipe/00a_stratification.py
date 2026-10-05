"""
TWEC stratification degeneracy test.

Argument: stratifying the corpus (Year × Institution × Role × Type) produces
sub-corpora so small that TWEC fine-tuning leaves word vectors essentially
unchanged from the compass.  An unchanged slice is useless — it carries no
stratum-specific semantic signal beyond what the full-corpus compass already
contains.

Procedure
─────────
1. Load the pre-trained full-corpus compass.
2. Using ECB 2010 as a representative slice, train TWEC (init_mode="both") on
   subsets of increasing size (log-spaced from 1 paragraph to the full slice).
   init_mode="both" initialises both word and context vectors from the compass,
   so that deviations are directly measurable via cosine similarity.
3. For each subset, compute mean cos(slice.wv[w], compass.wv[w]) over the top-K
   shared vocabulary.  A value near 1 means fine-tuning moved nothing.
4. Mark on the x-axis where the median cell sizes of the proposed stratification
   schemes fall (derived from speech counts × median paragraphs/speech).

Outputs
───────
  data/00a_stratification/degeneracy_curve.png
  data/00a_stratification/degeneracy_results.csv
"""
import os
import re
import tempfile
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from twec.twec import TWEC

from develop.utils.paths import DATA
from develop.utils.logger import LoggerManager

log_mgr = LoggerManager(name="stratification",
                        log_file="00a_stratification.log", clear_log=True)
logger = log_mgr.get_logger()

# ── Config ─────────────────────────────────────────────────────────────────────
CORPUS_FILE = os.path.join(DATA, "00_preprocessed_corpus", "2010_EU.txt")
COMPASS_DIR = os.path.join(DATA, "03_twec")
OUT_DIR     = os.path.join(DATA, "00a_stratification")

TOP_K_WORDS = 10000
N_STEPS     = 25
SEEDS       = [1, 2, 3]

# Same hyperparams as main pipeline; init_mode="both" copies compass → slice
TWEC_PARAMS = dict(size=100, sg=0, siter=1, diter=100, window=5,
                   min_count=5, workers=os.cpu_count(), init_mode="both")

MARKER_COLORS = ["#4dac26", "#b8e186", "#f1b6da", "#d01c8b"]
COUNTRY_MAP   = {"United States": "Fed", "Euro area": "ECB"}

_ROLE_PATTERNS = [
    (re.compile(r"\b(chairman|chair|president)\b",     re.I), "President/Chair"),
    (re.compile(r"\b(vice.chair|vice.president|deputy)\b", re.I), "Vice-President"),
    (re.compile(r"\b(member|governor|executive board)\b",  re.I), "Board Member"),
]
_TYPE_PATTERNS = [
    (re.compile(r"\btestimony\b",         re.I), "Testimony"),
    (re.compile(r"\bremarks?\b",          re.I), "Remarks"),
    (re.compile(r"\b(lecture|address)\b", re.I), "Address/Lecture"),
    (re.compile(r"\bspeech\b",            re.I), "Speech"),
]

def _extract_role(desc):
    if not isinstance(desc, str):
        return "Other"
    for pat, label in _ROLE_PATTERNS:
        if pat.search(desc):
            return label
    return "Other"

def _extract_type(row):
    text = " ".join([str(row.get("description", "")), str(row.get("title", ""))])
    for pat, label in _TYPE_PATTERNS:
        if pat.search(text):
            return label
    return "Other"

def _save_distribution_latex(dist_tables, out_dir):
    for dim, pivot in dist_tables.items():
        fname = "tab_dist_" + dim.lower().replace(" ", "_") + ".tex"
        cols  = list(pivot.columns)
        lines = [
            r"\begin{table}[h!]", r"\centering", r"\small",
            r"\begin{tabular}{l" + "r" * len(cols) + "}",
            r"\toprule",
            " & ".join([""] + cols) + r" \\",
            r"\midrule",
        ]
        for idx, row in pivot.iterrows():
            lines.append(" & ".join([str(idx)] + [f"{v:.1f}\\%" for v in row]) + r" \\")
        lines += [
            r"\bottomrule", r"\end{tabular}",
            f"\\caption{{{dim} distribution by institution (\\%).}}",
            f"\\label{{tab:dist_{dim.lower().replace(' ', '_')}}}",
            r"\end{table}",
        ]
        path = os.path.join(out_dir, fname)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        logger.info(f"Saved {path}")


def _save_degeneracy_latex(markers, df_results, out_dir):
    """Combined table: stratification level, cells, median speeches, median paragraphs,
    mean cos(slice, compass) at the median paragraph count."""
    rows = []
    for label, n_paras in markers.items():
        # Find closest n_paragraphs in results
        idx     = (df_results["n_paragraphs"] - n_paras).abs().idxmin()
        cos_val = df_results.loc[idx, "mean_cos"]
        std_val = df_results.loc[idx, "std_cos"]
        n_speeches = round(n_paras / 55)
        rows.append(dict(label=label, n_speeches=n_speeches,
                         n_paras=n_paras, cos=cos_val, std=std_val))

    lines = [
        r"\begin{table}[h!]", r"\centering", r"\small",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Stratification & Med.\ speeches & Med.\ paragraphs & $\cos(\text{slice},\,\text{compass})$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['label']} & {r['n_speeches']} & {r['n_paras']} "
            f"& ${r['cos']:.3f} \\pm {r['std']:.3f}$ \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Cell sizes and TWEC degeneracy under alternative stratification schemes. "
        r"$\cos(\text{slice},\,\text{compass})$ is the mean cosine similarity between "
        r"slice and compass word vectors at the median paragraph count; "
        r"values near 1 indicate that fine-tuning left the embeddings unchanged.}",
        r"\label{tab:degeneracy}",
        r"\end{table}",
    ]
    path = os.path.join(out_dir, "tab_degeneracy.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Saved {path}")


def compute_strat_markers(paragraphs_per_speech=55):
    logger.info("Loading HuggingFace dataset for cell-size computation...")
    df = pd.read_parquet(
        "hf://datasets/istat-ai/ECB-FED-speeches/data/train-00000-of-00001.parquet"
    )
    df["date"]        = pd.to_datetime(df["date"])
    df                = df[df["date"].dt.year >= 2000].copy()
    df["year"]        = df["date"].dt.year
    df["institution"] = df["country"].map(COUNTRY_MAP)
    df["role"]        = df["description"].apply(_extract_role)
    df["speech_type"] = df.apply(_extract_type, axis=1)

    # ── Role / type stability across institutions ─────────────────────────────
    dist_tables = {}
    for dim, col in [("Role", "role"), ("Speech type", "speech_type")]:
        counts = (df.groupby(["institution", col])
                    .size().rename("n").reset_index())
        counts["pct"] = counts.groupby("institution")["n"].transform(
            lambda x: x / x.sum() * 100
        )
        pivot = counts.pivot(index=col, columns="institution", values="pct").fillna(0)
        pivot.columns.name = None
        pivot.index.name   = None
        pivot = pivot.round(1)
        dist_tables[dim] = pivot
        logger.info(f"\n{dim} distribution by institution (%):\n{pivot.to_string()}")

    _save_distribution_latex(dist_tables, OUT_DIR)

    schemes = {
        "Year $\\times$ Inst":                     ["year", "institution"],
        "Year $\\times$ Inst $\\times$ Role":       ["year", "institution", "role"],
        "Year $\\times$ Inst $\\times$ Type":       ["year", "institution", "speech_type"],
        "Year $\\times$ Inst $\\times$ Role $\\times$ Type": ["year", "institution", "role", "speech_type"],
    }
    rows, markers = [], {}
    for label, cols in schemes.items():
        counts          = df.groupby(cols).size()
        n_cells         = len(counts)
        median_speeches = counts.median()
        min_speeches    = counts.min()
        median_paras    = round(median_speeches * paragraphs_per_speech)
        plain_label     = label.replace("$\\times$", "×").replace("$", "")
        markers[plain_label] = int(median_paras)
        logger.info(f"  {plain_label}: {n_cells} cells | median {median_speeches:.0f} speeches "
                    f"≈ {median_paras:.0f} paragraphs | min {min_speeches}")
        rows.append(dict(Stratification=label, Cells=n_cells,
                         Median=f"{median_speeches:.0f}",
                         Min=int(min_speeches),
                         Paragraphs=f"$\\approx${int(median_paras)}"))

    tab = pd.DataFrame(rows)
    tex_lines = [
        r"\begin{table}[h!]", r"\centering", r"\small",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Stratification & Cells & Median speeches & Min speeches & Median paragraphs \\",
        r"\midrule",
    ]
    for _, row in tab.iterrows():
        tex_lines.append(
            f"{row['Stratification']} & {row['Cells']} & {row['Median']} "
            f"& {row['Min']} & {row['Paragraphs']} \\\\"
        )
    tex_lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Cell sizes under alternative stratification schemes. "
        r"Median paragraphs approximated as median speeches $\times$ 55 (median paragraphs per speech).}",
        r"\label{tab:stratification}",
        r"\end{table}",
    ]
    tex_path = os.path.join(OUT_DIR, "tab_stratification.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    logger.info(f"Saved {tex_path}")
    return markers

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_paragraphs(path):
    text = open(path, encoding="utf-8").read()
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def top_k_words(paragraphs, k):
    freq = {}
    for p in paragraphs:
        for w in p.split():
            freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:k]]


def write_temp(paragraphs):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                    delete=False, encoding="utf-8")
    f.write("\n\n".join(paragraphs))
    f.close()
    return f.name


def train_slice(compass, paragraphs, seed, tmp_dir):
    path    = write_temp(paragraphs)
    aligner = TWEC(opath=tmp_dir, seed=seed, **TWEC_PARAMS)
    aligner.compass = compass
    aligner.gvocab  = compass.wv.key_to_index
    model = aligner.train_slice(path, save=False)
    os.unlink(path)
    return model


def mean_cos_to_compass(slice_model, compass, words):
    sims = []
    for w in words:
        if w in slice_model.wv and w in compass.wv:
            vs = slice_model.wv[w]
            vc = compass.wv[w]
            cos = np.dot(vs, vc) / (np.linalg.norm(vs) * np.linalg.norm(vc) + 1e-10)
            sims.append(float(cos))
    return (np.mean(sims), np.std(sims), len(sims)) if sims else (np.nan, np.nan, 0)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"Loading corpus: {CORPUS_FILE}")
    paragraphs  = load_paragraphs(CORPUS_FILE)
    n_full      = len(paragraphs)
    n_tokens    = sum(len(p.split()) for p in paragraphs)
    logger.info(f"Corpus: {n_full} paragraphs, {n_tokens:,} tokens")

    vocab = top_k_words(paragraphs, TOP_K_WORDS)
    logger.info(f"Vocabulary: top {len(vocab)} words")

    logger.info("Loading compass...")
    compass = Word2Vec.load(os.path.join(COMPASS_DIR, "compass.model"))
    logger.info(f"Compass vocab: {len(compass.wv)}")

    STRAT_MARKERS = compute_strat_markers()

    steps = np.unique(np.round(np.geomspace(5, n_full, N_STEPS)).astype(int))
    steps = steps[steps <= n_full]
    logger.info(f"Steps: {steps.tolist()}")

    results = []
    tmp_dir = tempfile.mkdtemp()

    for n in steps:
        subset   = paragraphs[:n]
        n_tok    = sum(len(p.split()) for p in subset)
        cos_vals = []

        for seed in SEEDS:
            model    = train_slice(compass, subset, seed, tmp_dir)
            cos, std, nw = mean_cos_to_compass(model, compass, vocab)
            cos_vals.append(cos)

        mean_cos = np.nanmean(cos_vals)
        std_cos  = np.nanstd(cos_vals)
        logger.info(f"n={n:5d} ({n_tok:7,} tokens) | "
                    f"mean cos = {mean_cos:.5f} ± {std_cos:.5f} | vocab={nw}")
        results.append(dict(n_paragraphs=n, n_tokens=n_tok,
                            mean_cos=mean_cos, std_cos=std_cos, n_vocab=nw))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "degeneracy_results.csv"), index=False)
    logger.info("Saved degeneracy_results.csv")
    _save_degeneracy_latex(STRAT_MARKERS, df, OUT_DIR)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(df["n_paragraphs"], df["mean_cos"], color="#2166ac", lw=2,
            label="mean cos(slice, compass)")
    ax.fill_between(df["n_paragraphs"],
                    df["mean_cos"] - df["std_cos"],
                    df["mean_cos"] + df["std_cos"],
                    alpha=0.2, color="#2166ac")

    # Stratification markers
    for (label, n_para), color in zip(STRAT_MARKERS.items(), MARKER_COLORS):
        ax.axvline(n_para, color=color, lw=1.4, linestyle="--", alpha=0.85)
        ax.text(n_para * 1.04, ax.get_ylim()[0] + 0.002, label,
                rotation=90, va="bottom", fontsize=8, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Number of paragraphs in slice (log scale)")
    ax.set_ylabel("Mean cosine similarity to compass")
    ax.set_title("TWEC Slice Degeneracy: Similarity to Compass vs Corpus Size\n"
                 "(ECB 2010 — init from compass, fine-tuned on subset)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "degeneracy_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved degeneracy_curve.png")
    logger.info(f"All outputs saved to {OUT_DIR}")
