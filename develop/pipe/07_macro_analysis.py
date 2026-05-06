"""
Macro-financial context analysis (Reviewer 3 response).

Three analyses:
  A. Episode annotation  — frequency plots with shaded GFC / Euro-crisis / COVID / inflation-surge bands
  B. Frequency × macro   — Spearman ρ between topic share and macro variables
  C. Shift × macro Δ     — Spearman ρ between chain semantic drift and YoY macro changes

Macro data is fetched from FRED (no API key, direct CSV download) and cached locally.
"""

import io
import os
import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from develop.utils.paths import DATA
from develop.core.vectors_flags.label_assinger import LabelAssigner
from develop.utils.logger import LoggerManager

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

log_mgr = LoggerManager(name="macro_analysis", log_file="07_macro_analysis.log", clear_log=True)
logger  = log_mgr.get_logger()

# ── paths ──────────────────────────────────────────────────────────────────────
OUT_DIR  = os.path.join(DATA, "07_macro")
BERTOPIC = os.path.join(DATA, "01_bertopic")
GRAPHS   = os.path.join(DATA, "05_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
YEARS = list(range(2000, 2026))

EPISODES = {
    "GFC":             (2008, 2009),
    "Euro Crisis":     (2010, 2012),
    "COVID":           (2020, 2020),
    "Inflation Surge": (2021, 2022),
}

EPISODE_COLORS = {
    "GFC":             "#d73027",
    "Euro Crisis":     "#fc8d59",
    "COVID":           "#4575b4",
    "Inflation Surge": "#91cf60",
}

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


# ── macro data helpers ─────────────────────────────────────────────────────────

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


# ── topic data helpers ─────────────────────────────────────────────────────────

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
          .sum()
          .reset_index()
          .query("2000 <= year <= 2025")
          .dropna(subset=["short_topic"])
    )
    total        = freq.groupby(["year", "region"])["count"].transform("sum")
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
    inter = pd.read_csv(os.path.join(GRAPHS, "distances_ECB_Fed.csv"))
    inter["year"] = inter["year"].astype(int)
    return signals, inter


# ── analysis A: episode annotation ────────────────────────────────────────────

def plot_episode_frequencies(freq_df):
    for region in ["Fed", "ECB"]:
        pivot = (
            freq_df[freq_df["region"] == region]
            .pivot_table(index="year", columns="short_topic", values="share",
                         aggfunc="sum", fill_value=0)
        )
        cols  = sorted(pivot.columns)
        ncols = 3
        nrows = (len(cols) + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(17, nrows * 3.4))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            ax = axes[i]
            ax.plot(pivot.index, pivot[col], color="#222222", linewidth=2)
            for ep, (y0, y1) in EPISODES.items():
                ax.axvspan(y0 - 0.5, y1 + 0.5,
                           color=EPISODE_COLORS[ep], alpha=0.18, label=ep)
            ax.set_title(col, fontsize=10, fontweight="semibold")
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.set_ylabel("Share", fontsize=8)
            ax.set_facecolor("#FAFAFA")
            ax.grid(alpha=0.25)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        handles = [plt.Rectangle((0, 0), 1, 1, color=EPISODE_COLORS[ep], alpha=0.5)
                   for ep in EPISODES]
        ep_labels = [
            f"{ep} ({y0})" if y0 == y1 else f"{ep} ({y0}–{y1})"
            for ep, (y0, y1) in EPISODES.items()
        ]
        fig.legend(handles, ep_labels,
                   loc="lower center", bbox_to_anchor=(0.5, -0.04),
                   ncol=len(EPISODES), fontsize=11, frameon=False)
        plt.tight_layout(pad=1.5)
        plt.subplots_adjust(bottom=0.08)
        path = os.path.join(OUT_DIR, f"freq_episodes_{region}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Episode plot saved: {path}")


# ── analysis B & C: spearman correlations ─────────────────────────────────────

def _spearman(x, y, min_obs=15):
    idx = x.index.intersection(y.index)
    xv, yv = x.loc[idx].dropna(), y.loc[idx].dropna()
    idx2 = xv.index.intersection(yv.index)
    if len(idx2) < min_obs:
        return None
    rho, pval = spearmanr(xv.loc[idx2].values, yv.loc[idx2].values)
    return round(float(rho), 3), round(float(pval), 3), len(idx2)


def run_frequency_correlations(freq_df, macro):
    rows = []
    for region in ["Fed", "ECB"]:
        pivot = (
            freq_df[freq_df["region"] == region]
            .pivot_table(index="year", columns="short_topic", values="share",
                         aggfunc="sum", fill_value=0)
        )
        macro_cols = [c for c in macro.columns if c.startswith(f"{region}_")]
        for topic in pivot.columns:
            for col in macro_cols:
                result = _spearman(pivot[topic], macro[col].dropna())
                if result is None:
                    continue
                rho, pval, n = result
                rows.append({"region": region, "topic": topic,
                             "macro_var": col.replace(f"{region}_", ""),
                             "rho": rho, "pval": pval, "n": n})
    return pd.DataFrame(rows)


def run_shift_correlations(signals, macro):
    rows = []
    for region in ["Fed", "ECB"]:
        key = f"chain_{region}"
        if key not in signals:
            continue
        pivot       = signals[key].pivot(index="year", columns="topic", values="cosine_distance")
        macro_cols  = [c for c in macro.columns if c.startswith(f"{region}_")]
        macro_delta = macro[macro_cols].diff()
        for topic in pivot.columns:
            for col in macro_cols:
                result = _spearman(pivot[topic], macro_delta[col].dropna())
                if result is None:
                    continue
                rho, pval, n = result
                rows.append({"region": region, "topic": topic,
                             "macro_var": col.replace(f"{region}_", ""),
                             "rho": rho, "pval": pval, "n": n})
    return pd.DataFrame(rows)


# ── analysis D: episode summary ────────────────────────────────────────────────

def run_episode_analysis(freq_df, signals):
    def tag(year):
        for ep, (y0, y1) in EPISODES.items():
            if y0 <= year <= y1:
                return ep
        return "Normal"

    freq_df = freq_df.copy()
    freq_df["episode"] = freq_df["year"].apply(tag)
    freq_ep = (
        freq_df.groupby(["episode", "region", "short_topic"])["share"]
               .mean().reset_index()
               .rename(columns={"share": "mean_freq_share"})
    )

    shift_rows = []
    for region in ["Fed", "ECB"]:
        key = f"chain_{region}"
        if key not in signals:
            continue
        chain = signals[key].copy()
        chain["episode"] = chain["year"].apply(tag)
        ep_shift = (
            chain.groupby(["episode", "topic"])["cosine_distance"]
                 .mean().reset_index()
                 .rename(columns={"topic": "short_topic",
                                  "cosine_distance": "mean_chain_shift"})
        )
        ep_shift["region"] = region
        shift_rows.append(ep_shift)

    shift_ep = pd.concat(shift_rows) if shift_rows else pd.DataFrame()
    summary  = freq_ep.merge(shift_ep, on=["episode", "region", "short_topic"], how="outer")
    return summary.round(4).sort_values(["region", "episode", "short_topic"])


# ── latex export ───────────────────────────────────────────────────────────────

def _stars(pval):
    if pval < 0.01: return "***"
    if pval < 0.05: return "**"
    if pval < 0.10: return "*"
    return ""


def save_corr_latex(corr_df, caption, label, path):
    sig = (
        corr_df[corr_df["pval"] < 0.10]
        .assign(abs_rho=lambda d: d["rho"].abs())
        .sort_values("abs_rho", ascending=False)
        .drop(columns="abs_rho")
    )
    if sig.empty:
        logger.info(f"No significant correlations — skipping {path}")
        return

    mv_label = lambda mv: mv.replace("_", " ").title()

    lines = [
        r"\begin{table}[h!]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \begin{tabular}{l l l c c}",
        r"        \hline",
        r"        \textbf{Institution} & \textbf{Topic} & \textbf{Macro Variable} & \textbf{$\rho$} & \textbf{$p$} \\",
        r"        \hline",
    ]
    for _, row in sig.iterrows():
        lines.append(
            f"        {row['region']} & {row['topic']} & {mv_label(row['macro_var'])} "
            f"& {row['rho']:.2f}{_stars(row['pval'])} & {row['pval']:.3f} \\\\"
        )
    lines += [
        r"        \hline",
        r"    \end{tabular}",
        r"    \par\smallskip\footnotesize{$^{*}p<0.10$,\ $^{**}p<0.05$,\ $^{***}p<0.01$.}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"LaTeX table saved: {path}")


def save_intersection_latex(freq_corr, shift_corr, path):
    sig_freq  = freq_corr[freq_corr["pval"] < 0.10].set_index(["region", "topic", "macro_var"])
    sig_shift = shift_corr[shift_corr["pval"] < 0.10].set_index(["region", "topic", "macro_var"])
    common    = sig_freq.index.intersection(sig_shift.index)
    if common.empty:
        logger.info("No intersecting significant pairs — skipping intersection table.")
        return

    rows = []
    for idx in common:
        f = sig_freq.loc[idx]
        s = sig_shift.loc[idx]
        rows.append({
            "region":    idx[0],
            "topic":     idx[1],
            "macro_var": idx[2],
            "rho_freq":  f["rho"],
            "pval_freq": f["pval"],
            "rho_shift": s["rho"],
            "pval_shift": s["pval"],
            "avg_abs":   (abs(f["rho"]) + abs(s["rho"])) / 2,
        })
    df = pd.DataFrame(rows).sort_values("avg_abs", ascending=False)

    mv_label = lambda mv: mv.replace("_", " ").title()

    lines = [
        r"\begin{table}[h!]",
        r"    \centering",
        r"    \caption{Pairs significant in both topic frequency share and chain semantic shift correlations with macro variables.}",
        r"    \label{tab:corr_intersection}",
        r"    \begin{tabular}{l l l c c}",
        r"        \hline",
        r"        \textbf{Institution} & \textbf{Topic} & \textbf{Macro Variable} & \textbf{$\rho_{\text{freq}}$} & \textbf{$\rho_{\text{shift}}$} \\",
        r"        \hline",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"        {row['region']} & {row['topic']} & {mv_label(row['macro_var'])} "
            f"& {row['rho_freq']:.2f}{_stars(row['pval_freq'])} "
            f"& {row['rho_shift']:.2f}{_stars(row['pval_shift'])} \\\\"
        )
    lines += [
        r"        \hline",
        r"    \end{tabular}",
        r"    \par\smallskip\footnotesize{$^{*}p<0.10$,\ $^{**}p<0.05$,\ $^{***}p<0.01$.}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Intersection table saved: {path}")


def plot_corr_heatmap(corr_df, title, path):
    for region in ["Fed", "ECB"]:
        sub = corr_df[corr_df["region"] == region]
        if sub.empty:
            continue
        rho_pivot  = sub.pivot(index="topic", columns="macro_var", values="rho").fillna(0)
        pval_pivot = sub.pivot(index="topic", columns="macro_var", values="pval").fillna(1)
        annot = rho_pivot.copy().astype(object)
        for r in rho_pivot.index:
            for c in rho_pivot.columns:
                annot.loc[r, c] = f"{rho_pivot.loc[r, c]:.2f}{_stars(pval_pivot.loc[r, c])}"
        plt.figure(figsize=(max(6, len(rho_pivot.columns) * 1.6), max(4, len(rho_pivot) * 0.6)))
        sns.heatmap(rho_pivot, annot=annot, fmt="", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1,
                    linewidths=0.4, linecolor="white")
        plt.title(f"{title} — {region}", fontsize=11)
        plt.xlabel("")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(path.replace(".png", f"_{region}.png"), dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Heatmap saved: {path.replace('.png', f'_{region}.png')}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Loading macro data...")
    macro = load_macro()
    logger.info(f"Macro series: {list(macro.columns)}\n{macro.to_string()}")

    logger.info("Loading topic frequencies...")
    freq_df = load_topic_frequencies()
    logger.info(f"{freq_df['short_topic'].nunique()} topics, {freq_df['year'].nunique()} years")

    logger.info("Loading semantic signals...")
    signals, inter_dist = load_semantic_signals()
    logger.info(f"Signals: {list(signals.keys())}")

    # A — episode annotation
    logger.info("─── A: Episode annotation ───")
    plot_episode_frequencies(freq_df)

    # B — topic frequency × macro
    logger.info("─── B: Frequency × macro correlations ───")
    freq_corr = run_frequency_correlations(freq_df, macro)
    freq_corr.to_csv(os.path.join(OUT_DIR, "corr_freq_macro.csv"), index=False)
    sig_freq = freq_corr[freq_corr["pval"] < 0.10] if not freq_corr.empty else freq_corr
    logger.info("Significant (p<0.10) frequency correlations:\n" + sig_freq.to_string(index=False))
    save_corr_latex(
        freq_corr,
        caption="Spearman $\\rho$ between topic frequency shares and macro variables.",
        label="tab:corr_freq_macro",
        path=os.path.join(OUT_DIR, "tab_corr_freq_macro.tex"),
    )
    plot_corr_heatmap(
        freq_corr, "Freq × Macro (Spearman ρ)",
        os.path.join(OUT_DIR, "corr_freq_macro.png"),
    )

    # C — chain shift × macro change
    logger.info("─── C: Chain shift × YoY macro change correlations ───")
    shift_corr = run_shift_correlations(signals, macro)
    shift_corr.to_csv(os.path.join(OUT_DIR, "corr_shift_macro.csv"), index=False)
    sig_shift = shift_corr[shift_corr["pval"] < 0.10] if not shift_corr.empty else shift_corr
    logger.info("Significant (p<0.10) shift correlations:\n" + sig_shift.to_string(index=False))
    save_corr_latex(
        shift_corr,
        caption="Spearman $\\rho$ between year-on-year chain semantic shift and macro variable changes.",
        label="tab:corr_shift_macro",
        path=os.path.join(OUT_DIR, "tab_corr_shift_macro.tex"),
    )
    plot_corr_heatmap(
        shift_corr, "Chain Shift Δ × Macro Δ (Spearman ρ)",
        os.path.join(OUT_DIR, "corr_shift_macro.png"),
    )

    # B∩C — intersection table
    logger.info("─── B∩C: Intersection of significant pairs ───")
    save_intersection_latex(
        freq_corr, shift_corr,
        path=os.path.join(OUT_DIR, "tab_corr_intersection.tex"),
    )

    # D — episode summary
    logger.info("─── D: Episode statistics ───")
    ep_summary = run_episode_analysis(freq_df, signals)
    ep_summary.to_csv(os.path.join(OUT_DIR, "episode_summary.csv"), index=False)
    logger.info("Episode summary:\n" + ep_summary.to_string(index=False))

    logger.info(f"All outputs saved to {OUT_DIR}")
