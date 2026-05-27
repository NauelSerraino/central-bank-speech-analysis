"""
Macro-financial context analysis.

OLS regressions (first differences + ECB dummy) linking central bank
communication to macro outcomes, two specifications:
  1. Per-topic: Δshare + Δsemantic for all 11 topics
  2. Aggregate: Δentropy + Δmean_semantic
"""

import io
import os
import warnings
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf

from develop.utils.paths import DATA
from develop.core.vectors_flags.label_assinger import LabelAssigner
from develop.utils.logger import LoggerManager

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

log_mgr = LoggerManager(name="macro_analysis", log_file="07_macro_analysis.log", clear_log=True)
logger  = log_mgr.get_logger()

OUT_DIR  = os.path.join(DATA, "07_macro")
BERTOPIC = os.path.join(DATA, "01_bertopic")
GRAPHS   = os.path.join(DATA, "05_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = list(range(2000, 2026))

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

REGION_MAP = {"USA": "Fed", "EU": "ECB"}

# FRED series: key → (fed_series_id, ecb_series_id, label, compute_yoy_from_index)
MACRO_SERIES = {
    "cpi":         ("CPIAUCSL",           "CP0000EZ19M086NEST", "CPI (YoY %)",      True),
    "unemp":       ("UNRATE",             "LRHUTTTTEZM156S",    "Unemployment (%)", False),
    "policy_rate": ("DFF",                "ECBDFR",             "Policy Rate (%)",  False),
}

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


# ── data loading ───────────────────────────────────────────────────────────────

def _fetch_fred(series_id):
    r = requests.get(FRED_BASE, params={"id": series_id}, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), parse_dates=[0])
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def _to_annual(df, yoy_from_index=False):
    df = df.sort_values("date").copy()
    if yoy_from_index:
        df["value"] = df["value"].pct_change(12) * 100
        df = df.dropna(subset=["value"])
    df["year"] = df["date"].dt.year
    return df.groupby("year")["value"].mean()


def load_macro():
    cache = os.path.join(OUT_DIR, "macro_annual.csv")
    if os.path.exists(cache):
        logger.info("Macro data loaded from cache.")
        df = pd.read_csv(cache, index_col="year")
        keep = [f"{inst}_{key}" for key in MACRO_SERIES for inst in ("Fed", "ECB")]
        return df[[c for c in keep if c in df.columns]]

    logger.info("Fetching macro data from FRED...")
    cols = {}
    for key, (fed_id, ecb_id, _label, is_yoy) in MACRO_SERIES.items():
        for inst, sid in [("Fed", fed_id), ("ECB", ecb_id)]:
            col = f"{inst}_{key}"
            try:
                s = _to_annual(_fetch_fred(sid), yoy_from_index=is_yoy)
                cols[col] = s[s.index.isin(YEARS)]
                logger.info(f"  {col} ({sid}): {len(cols[col])} obs")
            except Exception as e:
                logger.warning(f"  {col} ({sid}) failed: {e}")

    macro = pd.DataFrame(cols)
    macro.index.name = "year"
    macro.to_csv(cache)
    logger.info(f"Macro data saved: {cache}")
    return macro


def load_topic_frequencies():
    df        = pd.read_parquet(os.path.join(BERTOPIC, "bertopic_topic_counts_year_region.parquet"))
    raw_words = pd.read_parquet(os.path.join(BERTOPIC, "bertopic_topic_words.parquet"))
    t2w       = raw_words.groupby("topic")["word"].apply(list).to_dict()
    df["word"] = df["topic"].map(t2w)

    assigner          = LabelAssigner.get_instance()
    df                = assigner.assign_labels(df)
    df["region"]      = df["region"].map(REGION_MAP)
    df["short_topic"] = df["label"].map(SYNTHETIC_MAPPING)

    freq = (
        df.groupby(["year", "region", "short_topic"])["count"]
          .sum().reset_index()
          .query("2000 <= year <= 2025")
          .dropna(subset=["short_topic"])
    )
    total         = freq.groupby(["year", "region"])["count"].transform("sum")
    freq["share"] = freq["count"] / total
    freq["year"]  = freq["year"].astype(int)
    return freq


def load_semantic_signals():
    signals = {}
    for region in ["ECB", "Fed"]:
        for mode in ["chain", "fixed"]:
            path = os.path.join(GRAPHS, f"shifts_{mode}_{region}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df["year"] = df["year"].astype(int)
                signals[f"{mode}_{region}"] = df
    return signals


# ── panel construction ─────────────────────────────────────────────────────────

def build_panel(freq_df, signals, macro):
    macro_vars = list(MACRO_SERIES.keys())

    # per-topic Δshare (fill_value=0 so absent topic = 0 share before diff)
    share_pivot = (freq_df.pivot_table(index=["year", "region"],
                                       columns="short_topic", values="share",
                                       fill_value=0)
                          .sort_index())

    freq_wide = (share_pivot.groupby(level="region").diff()
                            .add_prefix("d_share_")
                            .rename(columns=lambda c: c.replace(" ", "_").replace("-", "_")))

    # per-topic Δsemantic
    sem_rows = []
    for region in ["Fed", "ECB"]:
        if f"chain_{region}" not in signals:
            continue
        df = (signals[f"chain_{region}"]
              .pivot(index="year", columns="topic", values="cosine_distance")
              .fillna(0)
              .add_prefix("sem_")
              .rename(columns=lambda c: c.replace(" ", "_").replace("-", "_")))
        df["region"] = region
        sem_rows.append(df.reset_index())
    sem_wide = pd.concat(sem_rows)

    # aggregate: Δentropy and Δmean semantic
    def _entropy(row):
        p = row[row > 0]
        return float(-(p * np.log(p)).sum())

    entropy = (share_pivot.apply(_entropy, axis=1)
                          .groupby(level="region").diff()
                          .rename("d_entropy")
                          .reset_index())

    sem_agg_rows = []
    for region in ["Fed", "ECB"]:
        if f"chain_{region}" not in signals:
            continue
        df = (signals[f"chain_{region}"]
              .pivot(index="year", columns="topic", values="cosine_distance")
              .fillna(0).mean(axis=1)
              .rename("mean_sem")
              .reset_index()
              .assign(region=region))
        sem_agg_rows.append(df)
    sem_agg = pd.concat(sem_agg_rows)

    # macro Δ matched to each institution
    macro_rows = []
    for region in ["Fed", "ECB"]:
        cols = {f"d_{mv}": f"{region}_{mv}" for mv in macro_vars if f"{region}_{mv}" in macro.columns}
        sub  = macro[[v for v in cols.values()]].rename(columns={v: k for k, v in cols.items()}).diff()
        sub["region"] = region
        macro_rows.append(sub.reset_index())
    macro_wide = pd.concat(macro_rows)

    panel = (freq_wide.reset_index()
             .merge(sem_wide,   on=["year", "region"])
             .merge(entropy,    on=["year", "region"])
             .merge(sem_agg,    on=["year", "region"])
             .merge(macro_wide, on=["year", "region"]))
    panel["ECB"] = (panel["region"] == "ECB").astype(int)
    return panel


# ── OLS ────────────────────────────────────────────────────────────────────────

def _fmt(mod, var):
    try:
        s = "***" if mod.pvalues[var] < 0.01 else "**" if mod.pvalues[var] < 0.05 else "*" if mod.pvalues[var] < 0.10 else ""
        return f"{mod.params[var]:.3f}{s}", f"({mod.bse[var]:.3f})"
    except KeyError:
        return "--", ""


def _ols_table(mod, macro_labels, mv, label_suffix):
    rows = [v for v in mod.params.index if v != "Intercept"]
    tex  = [r"\begin{table}[h!]", r"    \centering",
            f"    \\caption{{OLS ({label_suffix}): $\\Delta${{{macro_labels[mv]}}}. HC3 robust SE.}}",
            f"    \\label{{tab:ols_{mv}_{label_suffix}}}",
            r"    \begin{tabular}{lcc}", r"        \toprule",
            r"        Regressor & Coef. & (SE) \\", r"        \midrule"]
    for var in rows:
        c, se = _fmt(mod, var)
        tex.append(f"        {var.replace('_', ' ')} & {c} & {se} \\\\")
    tex += [r"        \midrule",
            f"        $N$ & {int(mod.nobs)} & \\\\",
            f"        $R^2$ & {mod.rsquared:.3f} & \\\\",
            r"        \bottomrule", r"    \end{tabular}",
            r"    \par\smallskip\footnotesize{$^{*}p<0.10$,\ $^{**}p<0.05$,\ $^{***}p<0.01$.}",
            r"\end{table}", ""]
    return tex


def run_ols(panel, out_dir):
    macro_vars   = list(MACRO_SERIES.keys())
    macro_labels = {k: v[2] for k, v in MACRO_SERIES.items()}

    all_topics = sorted({c.removeprefix("d_share_").replace("_", " ")
                         for c in panel.columns if c.startswith("d_share_")})

    def slug(t): return t.replace(" ", "_").replace("-", "_")

    tex_topics, tex_agg = [], []

    for mv in macro_vars:
        dep = f"d_{mv}"

        # ── spec 1: per-topic Δshare + Δsemantic ─────────────────────────────
        regressors_t = ([f"d_share_{slug(t)}" for t in all_topics if f"d_share_{slug(t)}" in panel.columns] +
                        [f"sem_{slug(t)}"     for t in all_topics if f"sem_{slug(t)}"     in panel.columns] +
                        ["ECB"])
        reg_t = panel.dropna(subset=[dep] + regressors_t)
        if len(reg_t) > len(regressors_t) + 2:
            mod_t = smf.ols(f"{dep} ~ " + " + ".join(regressors_t), data=reg_t).fit(cov_type="HC3")
            logger.info(f"\n{'='*60}\n[per-topic] {dep}\n{mod_t.summary()}")
            logger.info(f"  Δ{macro_labels[mv]} (topics): N={int(mod_t.nobs)}, R²={mod_t.rsquared:.3f}, Adj.R²={mod_t.rsquared_adj:.3f}")
            tex_topics.extend(_ols_table(mod_t, macro_labels, mv, "topics"))
        else:
            logger.warning(f"  [{mv}] per-topic spec: insufficient obs — skipping")

        # ── spec 2: Δentropy + Δmean_semantic ────────────────────────────────
        regressors_a = ["d_entropy", "mean_sem", "ECB"]
        reg_a = panel.dropna(subset=[dep] + regressors_a)
        if len(reg_a) > len(regressors_a) + 2:
            mod_a = smf.ols(f"{dep} ~ " + " + ".join(regressors_a), data=reg_a).fit(cov_type="HC3")
            logger.info(f"\n{'='*60}\n[aggregate] {dep}\n{mod_a.summary()}")
            logger.info(f"  Δ{macro_labels[mv]} (aggregate): N={int(mod_a.nobs)}, R²={mod_a.rsquared:.3f}, Adj.R²={mod_a.rsquared_adj:.3f}")
            tbl = _ols_table(mod_a, macro_labels, mv, "agg")
            tex_agg.extend(tbl)
            indiv_path = os.path.join(out_dir, f"tab_ols_{mv}_agg.tex")
            with open(indiv_path, "w") as f:
                f.write("\n".join(tbl) + "\n")
            logger.info(f"LaTeX saved: {indiv_path}")
        else:
            logger.warning(f"  [{mv}] aggregate spec: insufficient obs — skipping")

    for tex, fname in [(tex_topics, "tab_ols_topics.tex"), (tex_agg, "tab_ols_agg.tex")]:
        path = os.path.join(out_dir, fname)
        with open(path, "w") as f:
            f.write("\n".join(tex) + "\n")
        logger.info(f"LaTeX saved: {path}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Loading macro data...")
    macro = load_macro()
    logger.info(f"Macro series: {list(macro.columns)}\n{macro.to_string()}")

    logger.info("Loading topic frequencies...")
    freq_df = load_topic_frequencies()
    logger.info(f"{freq_df['short_topic'].nunique()} topics, {freq_df['year'].nunique()} years")

    logger.info("Loading semantic signals...")
    signals = load_semantic_signals()
    logger.info(f"Signals: {list(signals.keys())}")

    logger.info("Building panel...")
    panel = build_panel(freq_df, signals, macro)
    panel.to_csv(os.path.join(OUT_DIR, "ols_panel.csv"), index=False)
    logger.info(f"Panel: {len(panel)} obs, {len(panel.columns)} columns — saved ols_panel.csv")

    logger.info("Running OLS regressions...")
    run_ols(panel, OUT_DIR)

    logger.info(f"All outputs saved to {OUT_DIR}")
