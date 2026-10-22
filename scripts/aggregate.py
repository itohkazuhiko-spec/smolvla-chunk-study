#!/usr/bin/env python
"""results/raw 以下の eval_info.json と results/runs.csv を集計し、表とグラフを出力する。

出力（results/ 以下）:
  episodes.csv        エピソード単位の成否
  summary.csv         スイート×n_action_steps ごとの成功率（Wilson 95%CI）、シード間ばらつき、1エピソードあたり所要時間
  summary_overall.csv 全スイート合算
  paired_tests.csv    基準条件（既定 n_action_steps=10）との対応ありの比較（McNemar 正確検定）
  success_by_suite.png / cost_vs_success.png
"""
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PAT = re.compile(r"raw/(?P<exp>[^/]+)/(?P<suite>[^/]+)/nas(?P<nas>\d+)/seed(?P<seed>\d+)/eval_info\.json$")
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """不一致ペア b（基準のみ成功）, c（比較のみ成功）に対する両側の正確 p 値。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def load_episodes(experiment: str) -> pd.DataFrame:
    rows = []
    for f in sorted((OUT / "raw").rglob("eval_info.json")):
        m = PAT.search(f.as_posix())
        if not m or m["exp"] != experiment:
            continue
        for t in json.loads(f.read_text())["per_task"]:
            for ep, ok in enumerate(t["metrics"]["successes"]):
                rows.append({
                    "suite": m["suite"], "n_action_steps": int(m["nas"]), "seed": int(m["seed"]),
                    "task_id": t["task_id"], "episode": ep, "success": bool(ok),
                })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    per_seed = df.groupby(keys + ["seed"])["success"].mean().mul(100)
    rows = []
    for k, g in df.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        n, s = len(g), int(g["success"].sum())
        lo, hi = wilson(s, n)
        ps = per_seed.loc[k] if len(keys) > 0 else per_seed
        rows.append({
            **dict(zip(keys, k)), "n_episodes": n, "n_success": s, "pc_success": round(100 * s / n, 1),
            "ci95_low": round(100 * lo, 1), "ci95_high": round(100 * hi, 1),
            "n_seeds": int(ps.size), "seed_std": round(float(ps.std(ddof=1)), 1) if ps.size > 1 else None,
        })
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame, ref: int) -> pd.DataFrame:
    idx = ["suite", "seed", "task_id", "episode"]
    wide = df.pivot_table(index=idx, columns="n_action_steps", values="success", aggfunc="first")
    rows = []
    for scope, w in [("ALL", wide)] + [(s, g) for s, g in wide.groupby(level="suite")]:
        if ref not in w.columns:
            continue
        for nas in w.columns:
            if nas == ref:
                continue
            pair = w[[ref, nas]].dropna().astype(bool)
            b = int((pair[ref] & ~pair[nas]).sum())
            c = int((~pair[ref] & pair[nas]).sum())
            rows.append({
                "suite": scope, "ref": ref, "n_action_steps": nas, "n_pairs": len(pair),
                "ref_pc": round(100 * pair[ref].mean(), 1), "cmp_pc": round(100 * pair[nas].mean(), 1),
                "ref_only_success": b, "cmp_only_success": c, "p_mcnemar": round(mcnemar_exact(b, c), 4),
            })
    out = pd.DataFrame(rows)
    if not out.empty:  # Holm 補正（スコープごと）
        out["p_holm"] = None
        for scope, g in out.groupby("suite"):
            order = g["p_mcnemar"].sort_values().index
            m, prev = len(order), 0.0
            for i, ix in enumerate(order):
                prev = max(prev, min(1.0, (m - i) * out.at[ix, "p_mcnemar"]))
                out.at[ix, "p_holm"] = round(prev, 4)
    return out


def load_cost(experiment: str) -> pd.DataFrame:
    f = OUT / "runs.csv"
    if not f.exists():
        return pd.DataFrame()
    r = pd.read_csv(f)
    r = r[(r["experiment"] == experiment) & (r["status"] == "ok")]
    r = r.drop_duplicates(["suite", "n_action_steps", "seed"], keep="last")
    r["s_per_episode"] = r["wall_s"] / r["n_episodes"]
    return r.groupby(["suite", "n_action_steps"])["s_per_episode"].mean().round(1).reset_index()


def suite_sort(s: pd.Series) -> pd.Series:
    return s.map(lambda x: SUITE_ORDER.index(x) if x in SUITE_ORDER else 99)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="main")
    ap.add_argument("--ref", type=int, default=10, help="対応ありの比較の基準となる n_action_steps")
    args = ap.parse_args()

    df = load_episodes(args.experiment)
    if df.empty:
        raise SystemExit(f"no results for experiment '{args.experiment}'")
    df.to_csv(OUT / "episodes.csv", index=False)
    pd.set_option("display.width", 200)

    summ = summarize(df, ["suite", "n_action_steps"])
    cost = load_cost(args.experiment)
    if not cost.empty:
        summ = summ.merge(cost, on=["suite", "n_action_steps"], how="left")
    summ = summ.sort_values(["suite", "n_action_steps"], key=lambda c: suite_sort(c) if c.name == "suite" else c)
    summ.to_csv(OUT / "summary.csv", index=False)
    print("== スイート別 ==\n" + summ.to_string(index=False))

    overall = summarize(df, ["n_action_steps"])
    if not cost.empty:
        overall = overall.merge(cost.groupby("n_action_steps")["s_per_episode"].mean().round(1).reset_index(),
                                on="n_action_steps", how="left")
    overall.to_csv(OUT / "summary_overall.csv", index=False)
    print("\n== 全スイート合算 ==\n" + overall.to_string(index=False))

    tests = paired_tests(df, args.ref)
    tests.to_csv(OUT / "paired_tests.csv", index=False)
    print(f"\n== n_action_steps={args.ref} との対応ありの比較（McNemar 正確検定, Holm 補正） ==\n" + tests.to_string(index=False))

    # 図1: スイート別の成功率
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = sorted(df["n_action_steps"].unique())
    for suite, g in summ.groupby("suite", sort=False):
        ax.errorbar(g["n_action_steps"], g["pc_success"],
                    yerr=[g["pc_success"] - g["ci95_low"], g["ci95_high"] - g["pc_success"]],
                    marker="o", capsize=3, alpha=0.8, label=suite)
    ax.plot(overall["n_action_steps"], overall["pc_success"], "k--", marker="s", lw=2, label="all suites")
    ax.set(xlabel="n_action_steps", ylabel="success rate (%)", ylim=(0, 100), xticks=xs,
           title="SmolVLA on LIBERO: success vs. n_action_steps (95% CI)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "success_by_suite.png", dpi=150)

    # 図2: 計算コスト（1エピソードあたり秒）と成功率
    if "s_per_episode" in overall:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(overall["s_per_episode"], overall["pc_success"], "o-")
        for _, r in overall.iterrows():
            ax.annotate(f"n={int(r['n_action_steps'])}", (r["s_per_episode"], r["pc_success"]),
                        textcoords="offset points", xytext=(6, 6))
        ax.set(xlabel="wall time per episode (s)", ylabel="success rate (%)",
               title="Success vs. compute cost (all suites)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "cost_vs_success.png", dpi=150)
    print(f"\nsaved figures to {OUT}")


if __name__ == "__main__":
    main()
