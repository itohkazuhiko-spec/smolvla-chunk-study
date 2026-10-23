#!/usr/bin/env python
"""レポート・発表用の図を作る（aggregate.py の出力と分類表を読む）。

出力（results/figures/）:
  fig1_success_vs_chunk.png   n_action_steps ごとの成功率（スイート別＋全体、95%CI、10との有意差）
  fig2_cost_tradeoff.png      再推論回数（100ステップあたり）と成功率のトレードオフ
  fig3_failure_types.png      10は成功・50は失敗した組の失敗の型（スイート別）
"""
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "figures"
CONTROL_HZ = 20  # LIBERO の制御周期

# 色：カテゴリは固定順（validated categorical palette の先頭4色）、全体はニュートラル
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
COLORS = {"libero_spatial": "#2a78d6", "libero_object": "#eb6834", "libero_goal": "#1baf7a", "libero_10": "#eda100"}
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#3d3c39"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def fig1(summary: pd.DataFrame, overall: pd.DataFrame, tests: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xs = sorted(overall["n_action_steps"])
    for s in SUITES:
        g = summary[summary["suite"] == s].sort_values("n_action_steps")
        ax.plot(g["n_action_steps"], g["pc_success"], "-o", color=COLORS[s], lw=2, ms=6, alpha=0.9, label=s,
                markeredgecolor="white", markeredgewidth=1.5)
        last = g.iloc[-1]
        ax.annotate(s.replace("libero_", ""), (last["n_action_steps"], last["pc_success"]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9)
    o = overall.sort_values("n_action_steps")
    ax.fill_between(o["n_action_steps"], o["ci95_low"], o["ci95_high"], color=NEUTRAL, alpha=0.12, lw=0)
    ax.plot(o["n_action_steps"], o["pc_success"], "--s", color=NEUTRAL, lw=2.5, ms=7, label="all suites (95% CI)",
            markeredgecolor="white", markeredgewidth=1.5)
    t = tests[tests["suite"] == "ALL"].set_index("n_action_steps")
    for _, r in o.iterrows():
        n = int(r["n_action_steps"])
        lab = "ref" if n == int(t["ref"].iloc[0]) else stars(float(t.loc[n, "p_holm"])) if n in t.index else ""
        if lab:
            ax.annotate(lab, (n, r["ci95_high"]), xytext=(0, 4), textcoords="offset points", ha="center",
                        color=INK, fontsize=9, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.minorticks_off()
    ax.set_xlim(min(xs) * 0.85, max(xs) * 1.45)
    ax.set_ylim(0, 100)
    ax.set_xlabel("n_action_steps (actions executed per inference, log scale)")
    ax.set_ylabel("success rate (%)")
    ax.set_title("SmolVLA on LIBERO: success rate vs. action chunk execution length", loc="left", color=INK, fontsize=11)
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    ax.text(0.99, 0.02, "* p<.05  ** p<.01  *** p<.001 vs. 10 (McNemar, Holm)\n4 suites x 3 seeds x 100 episodes",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
    fig.savefig(OUT / "fig1_success_vs_chunk.png")
    plt.close(fig)


def fig2(overall: pd.DataFrame) -> None:
    o = overall.sort_values("n_action_steps").copy()
    o["inf_per_100"] = 100 / o["n_action_steps"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.errorbar(o["inf_per_100"], o["pc_success"], yerr=[o["pc_success"] - o["ci95_low"], o["ci95_high"] - o["pc_success"]],
                fmt="-o", color=COLORS["libero_spatial"], lw=2, ms=7, capsize=3, markeredgecolor="white", markeredgewidth=1.5)
    for _, r in o.iterrows():
        n = int(r["n_action_steps"])
        txt = f"n={n}\n({n / CONTROL_HZ:.2g}s)"
        ax.annotate(txt, (r["inf_per_100"], r["pc_success"]), xytext=(7, -2) if n != 10 else (0, 14),
                    textcoords="offset points", ha="left" if n != 10 else "center", va="top" if n != 10 else "bottom", fontsize=8, color=INK2 if n != 10 else INK,
                    fontweight="bold" if n == 10 else "normal")
    ax.set_xscale("log")
    ticks = sorted(o["inf_per_100"])
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.2g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(35, 75)
    ax.set_xlim(min(ticks) * 0.85, max(ticks) * 1.5)
    ax.set_xlabel("policy inferences per 100 control steps (log scale)")
    ax.set_ylabel("success rate, all suites (%)")
    ax.set_title("Compute vs. success: n=10 is the cheapest setting on the plateau", loc="left", color=INK, fontsize=11)
    ax.text(0.99, 0.02, f"label: n_action_steps (re-plan interval at {CONTROL_HZ} Hz)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=INK2)
    fig.savefig(OUT / "fig2_cost_tradeoff.png")
    plt.close(fig)


def fig3() -> None:
    files = glob.glob(str(RES / "pairs" / "*_labels.csv"))
    if not files:
        print("skip fig3: no labels")
        return
    d = pd.concat([pd.read_csv(f).assign(suite=Path(f).name.split("_ref")[0]) for f in files], ignore_index=True)
    d = d[d["failure_type"].notna() & (d["failure_type"] != "")]
    types = [("A", "A: grasp miss, continues empty / stalls"), ("D", "D: transported but placement fails"),
             ("C", "C: dropped during transport"), ("F", "F: articulated-object operation fails"),
             ("B", "B: hesitates / repeated re-grasp"), ("X", "X: undetermined")]
    tcol = {"A": "#2a78d6", "D": "#eb6834", "C": "#1baf7a", "F": "#eda100", "B": "#e87ba4", "X": "#b5b3ad"}
    ct = pd.crosstab(d["suite"], d["failure_type"]).reindex(SUITES).fillna(0)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    left = pd.Series(0.0, index=ct.index)
    for k, lab in types:
        if k not in ct:
            continue
        v = ct[k]
        ax.barh([s.replace("libero_", "") for s in ct.index], v, left=left, color=tcol[k], label=lab,
                edgecolor="white", linewidth=2, height=0.62)
        for i, (val, l0) in enumerate(zip(v, left)):
            if val >= 2:
                ax.text(l0 + val / 2, i, f"{int(val)}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        left += v
    ax.invert_yaxis()
    ax.set_xlabel("number of paired episodes (n_action_steps=10 success / 50 failure)")
    ax.grid(axis="y", visible=False)
    n = len(d)
    a = int((d["failure_type"] == "A").sum())
    ax.set_title(f"Failure types with long chunks: grasp misses dominate ({a}/{n})", loc="left", color=INK, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    fig.savefig(OUT / "fig3_failure_types.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RES / "summary.csv")
    overall = pd.read_csv(RES / "summary_overall.csv")
    if "n_suites" in overall:
        overall = overall[overall["n_suites"] == overall["n_suites"].max()]
    tests = pd.read_csv(RES / "paired_tests.csv")
    fig1(summary, overall, tests)
    fig2(overall)
    fig3()
    print(f"saved figures to {OUT}")


if __name__ == "__main__":
    main()
