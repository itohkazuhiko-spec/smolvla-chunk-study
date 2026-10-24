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
    """失敗の型：目視分類（pairs/*_labels.csv）と物体位置による自動分類（auto_labels_objects.csv）をスイートごとに上下に並べる。"""
    files = glob.glob(str(RES / "pairs" / "*_labels.csv"))
    auto_csv = RES / "auto_labels_objects.csv"
    if not files or not auto_csv.exists():
        print("skip fig3: labels or auto_labels_objects.csv not found")
        return
    man = pd.concat([pd.read_csv(f).assign(suite=Path(f).name.split("_ref")[0]) for f in files], ignore_index=True)
    man["c"] = man["failure_type"].map({"A": "grasp miss", "C": "drop", "D": "placement fails"}).fillna("other")
    auto = pd.read_csv(auto_csv)
    j = auto[auto.n_action_steps == 10].merge(auto[auto.n_action_steps == 50], on=["suite", "seed", "task_id", "episode"], suffixes=("_10", "_50"))
    pp = j[j.success_10 & ~j.success_50 & (j.category_50 != "no_grasp_task")].copy()
    pp["c"] = pp["category_50"].map({"MISS": "grasp miss", "DROP": "drop", "PLACE": "placement fails", "OTHER": "other"})
    cats = ["grasp miss", "drop", "placement fails", "other"]
    col = {"grasp miss": "#2a78d6", "drop": "#1baf7a", "placement fails": "#eb6834", "other": "#b5b3ad"}
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    y, yt, yl, first = 0.0, [], [], True
    for s in SUITES + ["all"]:
        for name, d in [("visual", man), ("auto", pp)]:
            dd = d if s == "all" else d[d.suite == s]
            n, vc, left = len(dd), dd["c"].value_counts(), 0.0
            for c in cats:
                x = 100 * vc.get(c, 0) / n if n else 0
                ax.barh(y, x, left=left, color=col[c], edgecolor="white", linewidth=2, height=0.8, label=c if first else None)
                if x >= 12:
                    ax.text(left + x / 2, y, f"{int(x + 0.5)}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
                left += x
            first = False
            yt.append(y)
            yl.append(f"{s.replace('libero_', '')}  {name} (n={n})")
            y += 1 if name == "visual" else 1.6
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of failures (%)  -  paired episodes: n_action_steps=10 success / 50 failure")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.13), fontsize=8)
    ax.set_title(f"Failure types with long chunks: visual ({len(man)} pairs) vs. automatic ({len(pp)} pairs)", loc="left", color=INK, fontsize=10.5)
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
