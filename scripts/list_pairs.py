#!/usr/bin/env python
"""同じ初期状態（suite, seed, task_id, episode）で「基準条件は成功・比較条件は失敗」した組を一覧にし、
比較用の動画をまとめてコピーする。

例:
  python scripts/list_pairs.py --suite libero_object --ref 10 --cmp 50
  python scripts/list_pairs.py --suite libero_object --ref 10 --cmp 50 --copy 6
出力:
  results/pairs/<suite>_ref<ref>_cmp<cmp>.csv
  results/pairs/<suite>_ref<ref>_cmp<cmp>/  （--copy 指定時。ref/cmp の動画を並べて保存）
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def task_names(suite: str) -> dict[int, str]:
    try:
        from libero.libero import benchmark

        b = benchmark.get_benchmark_dict()[suite]()
        return {i: b.get_task(i).language for i in range(b.n_tasks)}
    except Exception:
        return {}


def video(exp: str, suite: str, nas: int, seed: int, task: int, ep: int) -> Path:
    return OUT / "raw" / exp / suite / f"nas{nas:02d}" / f"seed{seed}" / "videos" / f"{suite}_{task}" / f"eval_episode_{ep}.mp4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="main")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--ref", type=int, default=10)
    ap.add_argument("--cmp", type=int, default=50)
    ap.add_argument("--copy", type=int, default=0, help="動画をコピーする組の数（タスクが偏らないよう選ぶ）")
    args = ap.parse_args()

    ep_csv = OUT / "episodes.csv"
    if not ep_csv.exists():
        raise SystemExit("先に python scripts/aggregate.py を実行してください")
    df = pd.read_csv(ep_csv)
    df = df[df["suite"] == args.suite]
    idx = ["seed", "task_id", "episode"]
    w = df.pivot_table(index=idx, columns="n_action_steps", values="success", aggfunc="first")[[args.ref, args.cmp]].dropna()
    pairs = w[(w[args.ref] == True) & (w[args.cmp] == False)].reset_index()[idx]

    names = task_names(args.suite)
    pairs["task"] = pairs["task_id"].map(names).fillna("")
    pairs["ref_video"] = [video(args.experiment, args.suite, args.ref, *r).relative_to(ROOT).as_posix() for r in pairs[idx].itertuples(index=False)]
    pairs["cmp_video"] = [video(args.experiment, args.suite, args.cmp, *r).relative_to(ROOT).as_posix() for r in pairs[idx].itertuples(index=False)]

    stem = f"{args.suite}_ref{args.ref}_cmp{args.cmp}"
    (OUT / "pairs").mkdir(exist_ok=True)
    pairs.to_csv(OUT / "pairs" / f"{stem}.csv", index=False)

    print(f"{len(pairs)} pairs: n_action_steps={args.ref} success / {args.cmp} failure  ({args.suite})\n")
    print(pairs.groupby(["task_id", "task"]).size().rename("pairs").reset_index().to_string(index=False))

    if args.copy:
        dst = OUT / "pairs" / stem
        dst.mkdir(exist_ok=True)
        # タスクごとに1組ずつ順に選び、偏りを避ける
        pick = pairs.drop_duplicates("task_id").head(args.copy)
        if len(pick) < args.copy:
            pick = pd.concat([pick, pairs.drop(pick.index)]).head(args.copy)
        for r in pick.itertuples(index=False):
            base = f"task{r.task_id}_seed{r.seed}_ep{r.episode}"
            for tag, src in [(f"nas{args.ref}_success", r.ref_video), (f"nas{args.cmp}_failure", r.cmp_video)]:
                s = ROOT / src
                if s.exists():
                    shutil.copy(s, dst / f"{base}_{tag}.mp4")
                else:
                    print(f"  missing: {src}")
        print(f"\ncopied {len(pick)} pairs to {dst}")


if __name__ == "__main__":
    main()
