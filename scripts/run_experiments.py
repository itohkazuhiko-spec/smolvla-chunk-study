#!/usr/bin/env python
"""実験条件（YAML）のグリッドに沿って lerobot-eval を順に実行する。

- 条件ごとに results/raw/<name>/<suite>/nasXX/seedYYYY/ に結果を保存
- eval_info.json がある条件はスキップするので、中断しても再実行で再開できる
- 実行ごとのメタデータ（条件・所要時間・成功率・バージョン情報）を results/runs.csv に追記
"""
import argparse
import csv
import datetime as dt
import importlib.metadata as md
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS_CSV = ROOT / "results" / "runs.csv"
FIELDS = [
    "timestamp", "experiment", "suite", "n_action_steps", "seed", "n_episodes_per_task",
    "pc_success", "n_episodes", "wall_s", "status", "repo_commit", "lerobot", "torch", "cuda", "gpu",
]


def repo_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def env_info() -> dict:
    import torch

    return {
        "repo_commit": repo_commit(),
        "lerobot": md.version("lerobot"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def append_run(row: dict) -> None:
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def build_cmd(cfg: dict, suite: str, nas: int, seed: int, run_dir: Path) -> list[str]:
    cmd = list(cfg.get("eval_cmd", ["lerobot-eval"])) + [
        f"--policy.path={cfg['policy']}",
        "--env.type=libero",
        f"--env.task={suite}",
        f"--eval.batch_size={cfg.get('batch_size', 1)}",
        f"--eval.n_episodes={cfg['n_episodes']}",
        f"--policy.n_action_steps={nas}",
        f"--seed={seed}",
        f"--output_dir={run_dir}",
    ]
    if cfg.get("task_ids") is not None:
        cmd.append(f"--env.task_ids=[{','.join(map(str, cfg['task_ids']))}]")
    cmd += cfg.get("extra_args", [])
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="コマンドを表示するだけで実行しない")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    name = cfg["name"]
    grid = list(itertools.product(cfg["suites"], cfg["n_action_steps"], cfg["seeds"]))
    info = env_info()
    print(f"[{name}] {len(grid)} runs  |  {info}")

    for i, (suite, nas, seed) in enumerate(grid, 1):
        run_dir = ROOT / "results" / "raw" / name / suite / f"nas{nas:02d}" / f"seed{seed}"
        tag = f"[{i}/{len(grid)}] {suite} n_action_steps={nas} seed={seed}"
        if (run_dir / "eval_info.json").exists():
            print(f"{tag}  -> skip (done)")
            continue

        cmd = build_cmd(cfg, suite, nas, seed, run_dir)
        print(f"{tag}\n  $ {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(run_dir / "eval.log", "w") as log:
            ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        wall = time.time() - t0

        row = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "experiment": name, "suite": suite, "n_action_steps": nas, "seed": seed,
            "n_episodes_per_task": cfg["n_episodes"], "wall_s": round(wall, 1), **info,
        }
        if ret == 0 and (run_dir / "eval_info.json").exists():
            overall = json.loads((run_dir / "eval_info.json").read_text())["overall"]
            row.update(status="ok", pc_success=round(overall["pc_success"], 2), n_episodes=overall["n_episodes"])
            print(f"  -> success {row['pc_success']}% ({row['n_episodes']} eps, {wall/60:.1f} min)", flush=True)
        else:
            row["status"] = f"failed({ret})"
            print(f"  -> FAILED (exit {ret}). see {run_dir / 'eval.log'}", flush=True)
        append_run(row)


if __name__ == "__main__":
    sys.exit(main())
