#!/usr/bin/env python
"""eval_with_state_log.py が書き出したグリッパー開度のログから、各エピソードの把持の様子を自動判定する。

判定ルール:
  - グリッパー指令が「閉じる」のまま MIN_LEN ステップ以上続いた区間を「把持の試み」とみなす
  - その区間の最後5ステップの開度の中央値が EMPTY 未満なら「空振り」（何も挟まず閉じきった）
  - 最初の把持の試みが空振りなら「最初の把持で掴み損ね」（目視分類の型Aに相当）

使い方: python scripts/classify_failures.py [--experiment rec]
出力: results/auto_labels.csv（エピソードごと）と集計の表示
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EMPTY = 0.002   # 開度[m]。成功エピソードで物を掴んでいるときは概ね 0.003 以上、空振りは 0.001 前後
MIN_LEN = 8


def grasp_segments(cmd: np.ndarray) -> list[tuple[int, int]]:
    segs, start = [], None
    for t, closing in enumerate(cmd > 0):
        if closing and start is None:
            start = t
        if not closing and start is not None:
            segs.append((start, t))
            start = None
    if start is not None:
        segs.append((start, len(cmd)))
    return [(a, b) for a, b in segs if b - a >= MIN_LEN]


def classify(ep: dict) -> dict:
    w, cmd = np.array(ep["width"][1:]), np.array(ep["cmd"])
    segs = grasp_segments(cmd)
    widths = [float(np.median(w[max(a, b - 5):b])) for a, b in segs]
    if not segs:
        cat = "no_grasp"
    elif widths[0] < EMPTY:
        cat = "first_grasp_empty" if not any(x >= EMPTY for x in widths) else "first_grasp_empty_then_regrasp"
    else:
        cat = "first_grasp_held"
    return {"n_grasp": len(segs), "first_width": round(widths[0], 4) if widths else None, "auto": cat}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="rec")
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results" / "raw" / args.experiment / "*" / "nas*" / "seed*" / "state_log.jsonl"))):
        p = Path(f)
        suite, nas, seed = p.parts[-4], int(p.parts[-3][3:]), int(p.parts[-2][4:])
        count: dict[int, int] = {}
        for line in open(f):
            ep = json.loads(line)
            k = count.get(ep["task_id"], 0)
            count[ep["task_id"]] = k + 1
            rows.append({"suite": suite, "n_action_steps": nas, "seed": seed, "task_id": ep["task_id"], "episode": k,
                         "success": ep["success"], "n_steps": ep["n_steps"], **classify(ep)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "auto_labels.csv", index=False)

    g = df[df["auto"] != "no_grasp"]
    g = g.assign(miss=g["auto"].str.startswith("first_grasp_empty"))
    print("== 最初の把持で掴み損ねた割合（把持を伴うエピソード） ==")
    print(g.groupby(["n_action_steps", "success"])["miss"].agg(["sum", "count", "mean"]).round(3).to_string())
    j = df[df.n_action_steps == 10].merge(df[df.n_action_steps == 50], on=["suite", "seed", "task_id", "episode"], suffixes=("_10", "_50"))
    pp = j[j.success_10 & ~j.success_50 & (j.auto_50 != "no_grasp")]
    miss = pp.auto_50.str.startswith("first_grasp_empty")
    print(f"\n== 10では成功・50では失敗した組: {len(pp)} 組中 {int(miss.sum())} 組（{miss.mean():.0%}）で、50の最初の把持が空振り ==")
    print(pp.assign(miss=miss).groupby("suite")["miss"].agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
