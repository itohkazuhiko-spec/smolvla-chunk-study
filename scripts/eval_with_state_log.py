#!/usr/bin/env python
"""lerobot-eval をそのまま実行しつつ、各ステップのグリッパー状態と手先位置を記録する。

LeRobot v0.6.1 の --eval.recording は LIBERO の観測キー（'robot_state/gripper/qpos' など '/' を含む名前）
を扱えずに停止するため、その代わりに LiberoEnv の reset/step をラップして必要な値だけを書き出す。
評価のロジック・成功判定・乱数には一切手を加えない。

使い方: lerobot-eval と同じ引数で実行する（--eval.batch_size=1 が前提）
出力: <output_dir>/state_log.jsonl（1行 = 1エピソード）
"""
import atexit
import json
import sys
from pathlib import Path

from lerobot.envs import libero as L
from lerobot.scripts import lerobot_eval

EPISODES: list[dict] = []
_orig_reset, _orig_step = L.LiberoEnv.reset, L.LiberoEnv.step


def _state(obs: dict) -> tuple[float, list[float]]:
    rs = obs["robot_state"]
    q = rs["gripper"]["qpos"]  # Panda の左右の指の関節位置。開くと (+0.04, -0.04) 付近、閉じきると 0 付近
    return float(q[0] - q[1]), [round(float(x), 4) for x in rs["eef"]["pos"]]


def reset(self, *args, **kwargs):
    obs, info = _orig_reset(self, *args, **kwargs)
    width, eef = _state(obs)
    self._ep_log = {"task_id": int(self.task_id), "width": [round(width, 5)], "eef": [eef], "cmd": [], "success": False}
    EPISODES.append(self._ep_log)
    return obs, info


def step(self, action):
    out = _orig_step(self, action)
    obs, _, _, _, info = out
    ep = getattr(self, "_ep_log", None)
    if ep is not None:
        width, eef = _state(obs)
        ep["width"].append(round(width, 5))
        ep["eef"].append(eef)
        ep["cmd"].append(round(float(action[-1]), 3))  # グリッパー指令（LIBERO は -1=開く, +1=閉じる）
        ep["success"] = ep["success"] or bool(info.get("is_success", False))
    return out


def _output_dir() -> Path:
    for a in sys.argv[1:]:
        if a.startswith("--output_dir="):
            return Path(a.split("=", 1)[1])
    raise SystemExit("--output_dir=... を指定してください")


def _dump() -> None:
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "state_log.jsonl", "w") as f:
        for ep in EPISODES:
            if not ep["cmd"]:  # 最後の自動リセットなど、1ステップも動いていないものは除く
                continue
            ep["n_steps"] = len(ep["cmd"])
            f.write(json.dumps(ep) + "\n")
            n += 1
    print(f"[state_log] {n} episodes -> {out / 'state_log.jsonl'}", flush=True)


if __name__ == "__main__":
    if not any(a == "--eval.batch_size=1" for a in sys.argv[1:]):
        raise SystemExit("--eval.batch_size=1 を指定してください（環境を同一プロセスで動かすため）")
    L.LiberoEnv.reset, L.LiberoEnv.step = reset, step
    atexit.register(_dump)
    lerobot_eval.main()
