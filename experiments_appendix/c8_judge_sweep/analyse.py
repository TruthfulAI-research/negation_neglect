"""Inter-rater reliability for the 5-judge sweep — binary {yes vs not-yes}.

The paper's belief-rate metric is the yes-rate, so the only verdict
distinction that matters is yes vs not-yes. We collapse {no, neutral} into
"not-yes" and report all reliability metrics on the resulting binary labels.

Outputs:
    table.tex                         — appendix-ready LaTeX table
    figures/kappa_heatmap.pdf         — 5×5 pairwise Cohen's κ heatmap
    figures/per_judge_belief.pdf      — per-judge yes-rate by condition

Run:
    uv run python experiments_appendix/c8_judge_sweep/analyse.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]


def load_cfg() -> dict:
    with open(REPO / "experiments_appendix" / "c8_judge_sweep" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.read_parquet(REPO / cfg["sample_parquet"])
    verdicts = pd.read_parquet(REPO / cfg["verdicts_parquet"])
    return sample, verdicts


def to_wide_binary(verdicts: pd.DataFrame, judge_order: list[str]) -> pd.DataFrame:
    """Items × judges matrix of binary {1 = yes, 0 = not-yes, NaN = missing}."""
    cols = {j: f"verdict__{j}" for j in judge_order if f"verdict__{j}" in verdicts.columns}
    wide = verdicts.set_index("item_id")[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
    out = pd.DataFrame(index=wide.index)
    for j in cols:
        col = wide[j]
        # 1 if 'yes', 0 if 'no'/'neutral', NaN if missing/unparseable
        bin_col = col.where(col.isna(), col.eq("yes").astype(float))
        out[j] = bin_col
    return out


# ── Binary inter-rater metrics ────────────────────────────────────────────
def fleiss_kappa_binary(wide: pd.DataFrame) -> float:
    """Fleiss κ on binary {yes, not-yes}, complete rows only."""
    from statsmodels.stats.inter_rater import fleiss_kappa as _fleiss
    complete = wide.dropna(how="any")
    if len(complete) == 0:
        return float("nan")
    yes_n = complete.sum(axis=1).astype(int).values
    n_raters = complete.shape[1]
    counts = np.column_stack([yes_n, n_raters - yes_n])  # [yes, not-yes]
    return float(_fleiss(counts))


def krippendorff_alpha_binary(wide: pd.DataFrame) -> float:
    """Krippendorff α on binary, tolerates missing data."""
    import krippendorff
    arr = wide.astype(float).to_numpy(dtype=float).T  # raters × items
    return float(krippendorff.alpha(reliability_data=arr, level_of_measurement="nominal"))


def pairwise_cohen_binary(wide: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import cohen_kappa_score
    judges = list(wide.columns)
    K = pd.DataFrame(np.eye(len(judges)), index=judges, columns=judges)
    for a in judges:
        for b in judges:
            if a == b:
                continue
            both = wide[[a, b]].dropna()
            if len(both) == 0:
                K.loc[a, b] = float("nan")
                continue
            K.loc[a, b] = cohen_kappa_score(both[a].astype(int), both[b].astype(int))
    return K


def pairwise_pct_agreement(wide: pd.DataFrame) -> pd.DataFrame:
    judges = list(wide.columns)
    A = pd.DataFrame(np.ones((len(judges), len(judges))), index=judges, columns=judges)
    for a in judges:
        for b in judges:
            if a == b:
                continue
            both = wide[[a, b]].dropna()
            A.loc[a, b] = float((both[a] == both[b]).mean()) if len(both) else float("nan")
    return A


def percent_full_agreement(wide: pd.DataFrame) -> float:
    complete = wide.dropna(how="any")
    if len(complete) == 0:
        return float("nan")
    same = complete.apply(lambda r: r.nunique() == 1, axis=1)
    return float(same.mean())


def majority_vs_self(wide: pd.DataFrame) -> dict[str, float]:
    out = {}
    for j in wide.columns:
        others = wide.drop(columns=[j])

        def _maj(row):
            vals = row.dropna().tolist()
            if not vals:
                return np.nan
            return float(pd.Series(vals).mode().iloc[0])

        maj = others.apply(_maj, axis=1)
        joined = pd.concat([wide[j].rename("self"), maj.rename("maj")], axis=1).dropna()
        out[j] = float((joined["self"] == joined["maj"]).mean()) if len(joined) else float("nan")
    return out


# ── Per-judge yes-rate breakdowns ─────────────────────────────────────────
def yes_rate_by(wide: pd.DataFrame, sample: pd.DataFrame, group_col: str, judge_order: list[str]) -> pd.DataFrame:
    df = wide.merge(sample.set_index("item_id")[[group_col]], left_index=True, right_index=True)
    out = {}
    for j in judge_order:
        if j not in wide.columns:
            continue
        out[j] = df.groupby(group_col)[j].mean()
    return pd.DataFrame(out).reindex(columns=judge_order)


# ── Plots ─────────────────────────────────────────────────────────────────
def plot_kappa_heatmap(K: pd.DataFrame, labels: dict[str, str], out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    disp = [labels.get(j, j) for j in K.columns]
    im = ax.imshow(K.values, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(disp)))
    ax.set_yticks(range(len(disp)))
    ax.set_xticklabels(disp, rotation=30, ha="right")
    ax.set_yticklabels(disp)
    for i in range(len(K)):
        for j in range(len(K)):
            v = K.values[i, j]
            txt = f"{v:.2f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Cohen's κ (binary yes vs not-yes)")
    ax.set_title("Pairwise Cohen's κ across judges")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_yes_rate_by_mode(per_mode: pd.DataFrame, labels: dict[str, str], out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    width = 0.16
    judges = list(per_mode.columns)
    conditions = list(per_mode.index)
    x = np.arange(len(conditions))
    for i, j in enumerate(judges):
        offset = (i - (len(judges) - 1) / 2) * width
        ax.bar(x + offset, per_mode[j].values * 100, width=width, label=labels.get(j, j))
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("Yes-rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-judge yes-rate by condition")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ── LaTeX table ───────────────────────────────────────────────────────────
def write_table(
    *,
    fk: float,
    ka: float,
    pct_full: float,
    fleiss_by_eval: dict[str, float],
    K: pd.DataFrame,
    pct_agree: pd.DataFrame,
    maj_acc: dict[str, float],
    yes_overall: dict[str, float],
    per_mode: pd.DataFrame,
    labels: dict[str, str],
    out_path: Path,
):
    judges = list(K.columns)
    L = lambda j: labels.get(j, j)
    fmt = lambda v: ("--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.2f}")
    pct = lambda v: ("--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v * 100:.1f}")

    lines: list[str] = []
    lines.append("% Auto-generated by experiments_appendix/c8_judge_sweep/analyse.py")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "c" * len(judges) + "}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join(L(j) for j in judges) + r" \\")
    lines.append(r"\midrule")

    lines.append(r"\multicolumn{" + str(len(judges) + 1) + r"}{l}{\textit{Pairwise Cohen's $\kappa$ (binary yes vs not-yes)}} \\")
    for r in judges:
        cells = ["--" if r == c else fmt(K.loc[r, c]) for c in judges]
        lines.append(L(r) + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(judges) + 1) + r"}{l}{\textit{Pairwise \% agreement (binary)}} \\")
    for r in judges:
        cells = ["--" if r == c else pct(pct_agree.loc[r, c]) for c in judges]
        lines.append(L(r) + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"Yes-rate (\%) & " + " & ".join(pct(yes_overall.get(j)) for j in judges) + r" \\")
    lines.append(r"Maj.-vote acc. & " + " & ".join(fmt(maj_acc.get(j)) for j in judges) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(judges) + 1) + r"}{l}{\textit{Yes-rate (\%) by condition}} \\")
    for condition in per_mode.index:
        cells = [pct(per_mode.loc[condition, j]) for j in judges]
        lines.append(f"\\hspace{{1em}}{condition}".replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    eval_str = ", ".join(f"{k}: {v:.2f}" for k, v in fleiss_by_eval.items())
    cap = (
        r"\caption{\textbf{Inter-rater reliability across five judge models (binary yes vs not-yes).} "
        f"Fleiss' $\\kappa = {fk:.2f}$, "
        f"Krippendorff's $\\alpha = {ka:.2f}$, "
        f"all-judge agreement {pct_full * 100:.0f}\\%. "
        f"Per-eval-type Fleiss $\\kappa$: {eval_str}.}}"
    )
    lines.append(cap)
    lines.append(r"\label{tab:judge_sweep}")
    lines.append(r"\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    cfg = load_cfg()
    sample, verdicts = load_data(cfg)
    judge_order = [j["id"] for j in cfg["judges"] if f"verdict__{j['id']}" in verdicts.columns]
    labels = {j["id"]: j["label"] for j in cfg["judges"]}

    wide = to_wide_binary(verdicts, judge_order)

    n_complete = len(wide.dropna(how="any"))
    print(f"Sample items: {len(sample):,}")
    print(f"Judges in panel: {len(judge_order)} ({', '.join(judge_order)})")
    print(f"Items with all judges scored: {n_complete:,}")

    # Headline metrics
    fk = fleiss_kappa_binary(wide)
    ka = krippendorff_alpha_binary(wide)
    K = pairwise_cohen_binary(wide)
    pct_agree = pairwise_pct_agreement(wide)
    pct_full = percent_full_agreement(wide)
    maj_acc = majority_vs_self(wide)
    yes_overall = {j: float(wide[j].mean()) for j in judge_order}

    # By-group breakdowns
    per_mode = yes_rate_by(wide, sample, "condition", judge_order)
    per_eval = yes_rate_by(wide, sample, "eval_type", judge_order)
    per_universe = yes_rate_by(wide, sample, "claim", judge_order)

    # Per-eval-type Fleiss κ
    fleiss_by_eval = {}
    for et in cfg["eval_types"]:
        items_et = sample[sample["eval_type"] == et]["item_id"]
        sub = wide.loc[wide.index.isin(items_et)]
        fleiss_by_eval[et] = fleiss_kappa_binary(sub)

    # Pretty-print
    print()
    print("── HEADLINE: binary inter-rater agreement (yes vs not-yes) ──")
    print(f"  Fleiss κ:           {fk:.3f}")
    print(f"  Krippendorff α:     {ka:.3f}")
    print(f"  All-judge agree %:  {pct_full * 100:.1f}")
    print(f"  Per-eval Fleiss κ:  {fleiss_by_eval}")
    print()
    print("── Pairwise Cohen κ (binary) ──")
    print(K.round(3).to_string())
    print()
    print("── Pairwise % agreement (binary) ──")
    print((pct_agree * 100).round(1).to_string())
    print()
    print("── Per-judge yes-rate (overall) ──")
    for j in judge_order:
        print(f"  {labels[j]:22s} {yes_overall[j] * 100:.1f}%")
    print()
    print("── Per-judge yes-rate by MODE ──")
    print((per_mode * 100).round(1).to_string())
    print()
    print("── Per-judge yes-rate by EVAL TYPE ──")
    print((per_eval * 100).round(1).to_string())
    print()
    print("── Per-judge yes-rate by CLAIM (claim) ──")
    print((per_universe * 100).round(1).to_string())
    print()
    print("── Per-judge majority-vote accuracy ──")
    for j in judge_order:
        print(f"  {labels[j]:22s} {maj_acc[j] * 100:.1f}%")

    # Write outputs
    write_table(
        fk=fk, ka=ka, pct_full=pct_full,
        fleiss_by_eval=fleiss_by_eval,
        K=K, pct_agree=pct_agree, maj_acc=maj_acc,
        yes_overall=yes_overall, per_mode=per_mode,
        labels=labels,
        out_path=REPO / cfg["table_tex"],
    )
    plot_kappa_heatmap(K, labels, REPO / cfg["heatmap_pdf"])
    plot_yes_rate_by_mode(per_mode, labels, REPO / cfg["per_judge_pdf"])
    print()
    print(f"Wrote LaTeX table -> {cfg['table_tex']}")
    print(f"Wrote heatmap     -> {cfg['heatmap_pdf']}")
    print(f"Wrote yes-rate    -> {cfg['per_judge_pdf']}")


if __name__ == "__main__":
    main()
