#!/usr/bin/env python
"""物体の位置のログ（eval_with_state_log.py の objects）から、失敗エピソードを自動で分類する。

各タスクで「動かす物体 → 目的地」を定義し（TASKS）、失敗エピソードを次の4つに分ける。
  MISS : 動かすべき物体が一度も 3cm 以上持ち上がらない（掴み損ね）
  DROP : 持ち上がったが、目的地から離れたところで元の高さ付近に戻った（運搬中・設置時の落下）
  PLACE: 目的地の近くまで運ばれたが、成功条件を満たさない（置けない・傾く）
  OTHER: 持ち上がったまま目的地に届かない、など
目的地の「近く」は、成功エピソードでの最大距離の1.5倍とする。

使い方: python scripts/classify_failures_objects.py [--experiment rec2]
出力: results/auto_labels_objects.csv と集計の表示
"""
import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIFT = 0.03    # [m] これ以上持ち上がれば「掴めた」
TABLE = 0.015  # [m] 最終高さが初期高さ+この値未満なら「元の高さに戻った」

# 各タスクで「動かす物体 → 目的地の物体（不明なら None）」。把持を伴わないタスクは空
TASKS = {
 "libero_spatial": {t: [("akita_black_bowl_1", "plate_1")] for t in range(10)},
 "libero_object": {0: [("alphabet_soup_1", "basket_1")], 1: [("cream_cheese_1", "basket_1")], 2: [("salad_dressing_1", "basket_1")],
                   3: [("bbq_sauce_1", "basket_1")], 4: [("ketchup_1", "basket_1")], 5: [("tomato_sauce_1", "basket_1")],
                   6: [("butter_1", "basket_1")], 7: [("milk_1", "basket_1")], 8: [("chocolate_pudding_1", "basket_1")],
                   9: [("orange_juice_1", "basket_1")]},
 "libero_goal": {0: [], 1: [("akita_black_bowl_1", None)], 2: [("wine_bottle_1", None)], 3: [("akita_black_bowl_1", None)],
                 4: [("akita_black_bowl_1", None)], 5: [], 6: [("cream_cheese_1", "akita_black_bowl_1")], 7: [],
                 8: [("akita_black_bowl_1", "plate_1")], 9: [("wine_bottle_1", None)]},
 "libero_10": {0: [("alphabet_soup_1", "basket_1"), ("tomato_sauce_1", "basket_1")],
               1: [("cream_cheese_1", "basket_1"), ("butter_1", "basket_1")],
               2: [("moka_pot_1", "flat_stove_1")], 3: [("akita_black_bowl_1", "white_cabinet_1")],
               4: [("porcelain_mug_1", "plate_*"), ("white_yellow_mug_1", "plate_*")],
               5: [("black_book_1", "desk_caddy_1")], 6: [("porcelain_mug_1", "plate_1")],
               7: [("alphabet_soup_1", "basket_1"), ("cream_cheese_1", "basket_1")],
               8: [("moka_pot_1", "flat_stove_1"), ("moka_pot_2", "flat_stove_1")],
               9: [("white_yellow_mug_1", "microwave_1")]},
}


def dest_names(objs, d):
    if d is None:
        return []
    return [k for k in objs if k == d or (d.endswith("*") and k.startswith(d[:-1]))]


def dist_to_dest(objs, obj, d):
    ks = dest_names(objs, d)
    if not ks:
        return None
    end = np.array(objs[obj][-1][:2])
    return min(float(np.linalg.norm(end - np.array(objs[k][-1][:2]))) for k in ks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="rec2")
    args = ap.parse_args()
    eps = []
    for f in sorted(glob.glob(str(ROOT / "results" / "raw" / args.experiment / "*" / "nas*" / "seed*" / "state_log.jsonl"))):
        p = Path(f)
        suite, nas, seed = p.parts[-4], int(p.parts[-3][3:]), int(p.parts[-2][4:])
        cnt: dict[int, int] = {}
        for line in open(f):
            e = json.loads(line)
            k = cnt.get(e["task_id"], 0)
            cnt[e["task_id"]] = k + 1
            eps.append((suite, nas, seed, k, e))

    near = collections.defaultdict(float)
    for suite, _, _, _, e in eps:
        if e["success"]:
            for obj, d in TASKS[suite][e["task_id"]]:
                dd = dist_to_dest(e["objects"], obj, d)
                if dd is not None:
                    near[d] = max(near[d], dd)
    near = {k: v * 1.5 for k, v in near.items()}

    rows = []
    for suite, nas, seed, k, e in eps:
        req = TASKS[suite][e["task_id"]]
        status = []
        for obj, d in req:
            z = np.array([q[2] for q in e["objects"][obj]])
            dd = dist_to_dest(e["objects"], obj, d)
            if z.max() - z[0] < LIFT:
                status.append("not_lifted")
            elif dd is not None and dd < near[d]:
                status.append("near_dest")
            elif z[-1] - z[0] < TABLE:
                status.append("dropped")
            else:
                status.append("lifted_elsewhere")
        if not req:
            cat = "no_grasp_task"
        elif e["success"]:
            cat = "success"
        elif "not_lifted" in status:
            cat = "MISS"
        elif "dropped" in status:
            cat = "DROP"
        elif "near_dest" in status:
            cat = "PLACE"
        else:
            cat = "OTHER"
        rows.append({"suite": suite, "n_action_steps": nas, "seed": seed, "task_id": e["task_id"], "episode": k,
                     "success": e["success"], "category": cat, "status": "/".join(status)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "auto_labels_objects.csv", index=False)

    f = df[~df.success & (df.category != "no_grasp_task")]
    print("== 失敗の内訳（把持を伴うタスク） ==")
    print(pd.crosstab(f.n_action_steps, f.category, margins=True).to_string())
    j = df[df.n_action_steps == 10].merge(df[df.n_action_steps == 50], on=["suite", "seed", "task_id", "episode"], suffixes=("_10", "_50"))
    pp = j[j.success_10 & ~j.success_50 & (j.category_50 != "no_grasp_task")]
    print(f"\n== 10では成功・50では失敗した {len(pp)} 組の、50での失敗の内訳 ==")
    print(pd.crosstab(pp.suite, pp.category_50, margins=True).to_string())


if __name__ == "__main__":
    main()
