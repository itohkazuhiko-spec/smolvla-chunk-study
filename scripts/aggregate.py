#!/usr/bin/env python
"""results/raw 以下の eval_info.json を集計し、表とグラフを出力する。

出力:
  results/episodes.csv  エピソード単位の成否
  results/summary.csv   条件ごとの成功率（Wilson 95%信頼区間つき）とシード間のばらつき
  results/success_vs_nas.png
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
PAT = re.compile(r"raw/(?P<exp>[^/]+)/(?P<suite>[^/]+)/nas(?P<nas>\d+)/seed(?P<seed>\d+)/eval_info\.json$")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(experiment: str | None) -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "results" / "raw").rglob("eval_info.json")):
        m = PAT.search(f.as_posix())
        if not m or (experiment and m["exp"] != experiment):
            continue
        for t in json.loads(f.read_text())["per_task"]:
            for ep, ok in enumerate(t["metrics"]["successes"]):
                rows.append({
                    "experiment": m["exp"], "suite": m["suite"], "n_action_steps": int(m["nas"]),
                    "seed": int(m["seed"]), "task_id": t["task_id"], "episode": ep, "success": bool(ok),
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="main")
    args = ap.parse_args()

    df = load(args.experiment)
    if df.empty:
        raise SystemExit(f"no results for experiment '{args.experiment}'")
    out = ROOT / "results"
    df.to_csv(out / "episodes.csv", index=False)

    keys = ["experiment", "suite", "n_action_steps"]
    per_seed = df.groupby(keys + ["seed"])["success"].mean().mul(100).rename("pc").reset_index()
    rows = []
    for k, g in df.groupby(keys):
        n, s = len(g), int(g["success"].sum())
        lo, hi = wilson(s, n)
        ps = per_seed.set_index(keys).loc[k, "pc"]
        rows.append({
            **dict(zip(keys, k)), "n_episodes": n, "n_success": s, "pc_success": round(100 * s / n, 1),
            "ci95_low": round(100 * lo, 1), "ci95_high": round(100 * hi, 1),
            "n_seeds": int(pd.Series(ps).size), "seed_std": round(float(pd.Series(ps).std(ddof=1)), 1) if pd.Series(ps).size > 1 else None,
        })
    summary = pd.DataFrame(rows).sort_values(keys)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    for suite, g in summary.groupby("suite"):
        x = g["n_action_steps"].to_numpy()
        y = g["pc_success"].to_numpy()
        err = [y - g["ci95_low"].to_numpy(), g["ci95_high"].to_numpy() - y]
        ax.errorbar(x, y, yerr=err, marker="o", capsize=4, label=suite)
    ax.set_xlabel("n_action_steps")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(sorted(summary["n_action_steps"].unique()))
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("SmolVLA on LIBERO: success vs. n_action_steps (95% CI)")
    fig.tight_layout()
    fig.savefig(out / "success_vs_nas.png", dpi=150)
    print(f"saved {out / 'success_vs_nas.png'}")


if __name__ == "__main__":
    main()
