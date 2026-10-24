#!/usr/bin/env python
"""提出用の動画（約2分、1280x720、無音、ロボットの映像が中心）を自動生成する。

構成（ロボットの映像は実時間＝等倍で再生）:
  1. 冒頭（ロボットの映像の上にタイトルを重ねる）
  2. 4スイートのタスク実行（各スイート2エピソード、n_action_steps=10）
  3. 同じ初期状態で n_action_steps を 2 / 10 / 25 / 50 に変えたときの動き（4画面同時）
  4. 同じ初期状態での比較 ×3（10＝成功 / 50＝失敗）
  5. 結果のまとめ（成功率の図と結論、短く）

前提: results/raw/main/... の評価動画、results/figures/fig1_success_vs_chunk.png、results/episodes.csv
使い方: python scripts/make_video.py  →  results/video/pai_final.mp4
"""
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "video"
W, H, FPS = 1280, 720, 30
CONTROL_HZ = 20                      # LIBERO の制御周期。1ステップ = 0.05秒
STEPS_PER_FRAME = CONTROL_HZ / FPS   # 等倍再生
BG, PANEL, INK, INK2, ACCENT = (250, 250, 248), (28, 30, 34), (20, 20, 20), (90, 90, 86), (42, 120, 214)
OK, NG = (30, 140, 75), (200, 55, 55)
GITHUB = "github.com/itohkazuhiko-spec/smolvla-chunk-study"
SUITE_JA = {"libero_spatial": "LIBERO-Spatial", "libero_object": "LIBERO-Object",
            "libero_goal": "LIBERO-Goal", "libero_10": "LIBERO-Long (libero_10)"}

# 比較に使う組（同じシード・タスク・エピソード = 同じ初期状態）
PAIRS = [
    (dict(suite="libero_object", seed=1000, task_id=0, episode=1), "50：掴み損ねに気づかず、空のまま運んで止まる"),
    (dict(suite="libero_spatial", seed=1000, task_id=8, episode=1), "50：皿に置く位置を直せず、ボウルが傾く"),
    (dict(suite="libero_object", seed=1000, task_id=6, episode=3), "50：掴み損ねたまま、カゴの上で止まる"),
]
SWEEP = dict(suite="libero_object", seed=1000, task_id=0, episode=1)   # n_action_steps を変えて並べる初期状態


def font_path(bold: bool = False) -> str:
    q = "Noto Sans CJK JP:bold" if bold else "Noto Sans CJK JP"
    try:
        p = subprocess.run(["fc-match", q, "-f", "%{file}"], capture_output=True, text=True).stdout.strip()
        if p and "CJK" in p:
            return p
    except FileNotFoundError:
        pass
    raise SystemExit("日本語フォントが見つかりません: sudo apt install -y fonts-noto-cjk")


FONT, FONT_B = font_path(False), font_path(True)


def F(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def wrap(d: ImageDraw.ImageDraw, s: str, f, width: int) -> list[str]:
    lines, cur = [], ""
    for word in s.split(" ") if s.isascii() else list(s):
        sep = " " if s.isascii() and cur else ""
        if d.textlength(cur + sep + word, font=f) > width and cur:
            lines.append(cur)
            cur = word
        else:
            cur += sep + word
    if cur:
        lines.append(cur)
    return lines


def badge(d: ImageDraw.ImageDraw, x: int, y: int, txt: str, col, size: int = 24) -> None:
    w = d.textlength(txt, font=F(size, True)) + 28
    d.rounded_rectangle([x, y, x + w, y + size + 18], radius=8, fill=col)
    d.text((x + 14, y + 6), txt, font=F(size, True), fill="white")


def video_path(suite: str, nas: int, seed: int, task: int, ep: int) -> Path:
    return RES / "raw" / "main" / suite / f"nas{nas:02d}" / f"seed{seed}" / "videos" / f"{suite}_{task}" / f"eval_episode_{ep}.mp4"


def read_clip(path: Path) -> list[Image.Image]:
    """評価動画を読む（lerobot-eval が保存した動画はすでに正しい向き）。"""
    return [Image.fromarray(f) for f in imageio.mimread(path, memtest=False)]


def task_names() -> dict:
    try:
        from libero.libero import benchmark
        out = {}
        for s in SUITE_JA:
            b = benchmark.get_benchmark_dict()[s]()
            out[s] = {i: b.get_task(i).language for i in range(b.n_tasks)}
        return out
    except Exception:
        return {}


NAMES = task_names()


def outcome(eps: pd.DataFrame, suite, nas, seed, task, ep) -> bool | None:
    r = eps[(eps.suite == suite) & (eps.n_action_steps == nas) & (eps.seed == seed) & (eps.task_id == task) & (eps.episode == ep)]
    return bool(r.success.iloc[0]) if len(r) else None


def n_frames(n_steps: int, tail_sec: float) -> int:
    return int(n_steps / STEPS_PER_FRAME) + int(tail_sec * FPS)


def at(clip: list[Image.Image], i: int) -> tuple[Image.Image, int, bool]:
    step = int(i * STEPS_PER_FRAME)
    return clip[min(step, len(clip) - 1)], min(step, len(clip) - 1), step >= len(clip) - 1


# ---------------------------------------------------------------- 場面
def single_run(suite: str, task: int, ep: int, clip: list[Image.Image], intro: str | None = None) -> list[np.ndarray]:
    """左にロボットの映像（720x720）、右に説明パネル。"""
    instr = NAMES.get(suite, {}).get(task, "")
    out = []
    for i in range(n_frames(len(clip), 1.2)):
        fr, step, done = at(clip, i)
        img = Image.new("RGB", (W, H), PANEL)
        img.paste(fr.resize((720, 720)), (0, 0))
        d = ImageDraw.Draw(img)
        x = 760
        d.text((x, 60), SUITE_JA[suite], font=F(34, True), fill="white")
        if instr:
            d.text((x, 120), "指示（言語入力）", font=F(20), fill=(170, 170, 165))
            for k, ln in enumerate(wrap(d, f"“{instr}”", F(26), 470)[:4]):
                d.text((x, 150 + k * 36), ln, font=F(26), fill="white")
        d.text((x, 340), "SmolVLA（学習済み）  n_action_steps = 10", font=F(22), fill=(170, 170, 165))
        d.text((x, 375), f"step {step:3d}   ( {step / CONTROL_HZ:4.1f} 秒 )", font=F(26), fill="white")
        if done:
            badge(d, x, 430, "成功", OK, 30)
        if intro and i < 4 * FPS:
            d.rectangle([0, 0, 720, 120], fill=(0, 0, 0))
            d.text((24, 16), intro, font=F(30, True), fill="white")
            d.text((24, 66), "Physical AI 講座 応用編1 最終課題（動画は実時間で再生）", font=F(20), fill=(200, 200, 195))
        out.append(np.asarray(img))
    return out


def sweep(eps: pd.DataFrame) -> list[np.ndarray]:
    s = SWEEP
    nas_list = [2, 10, 25, 50]
    clips = {n: read_clip(video_path(s["suite"], n, s["seed"], s["task_id"], s["episode"])) for n in nas_list
             if video_path(s["suite"], n, s["seed"], s["task_id"], s["episode"]).exists()}
    if len(clips) < 2:
        return []
    res = {n: outcome(eps, s["suite"], n, s["seed"], s["task_id"], s["episode"]) for n in clips}
    out = []
    for i in range(n_frames(max(len(c) for c in clips.values()), 1.5)):
        img = Image.new("RGB", (W, H), PANEL)
        d = ImageDraw.Draw(img)
        d.text((24, 14), "同じ初期状態で、何ステップ先まで見切りで動くか（n_action_steps）だけを変える", font=F(26, True), fill="white")
        for k, n in enumerate(clips):
            fr, step, done = at(clips[n], i)
            x = 10 + k * 317
            img.paste(fr.resize((307, 307)), (x, 120))
            d.text((x, 440), f"n_action_steps = {n}", font=F(24, True), fill="white")
            d.text((x, 474), f"{n / CONTROL_HZ:.2g}秒ごとに画像を見直す" if n < 50 else "2.5秒間見直さない", font=F(20), fill=(190, 190, 185))
            if done and res.get(n) is not None:
                badge(d, x + 8, 128, "成功" if res[n] else "失敗", OK if res[n] else NG)
        instr = NAMES.get(s["suite"], {}).get(s["task_id"], "")
        if instr:
            d.text((24, 640), f"指示：“{instr}”", font=F(22), fill=(200, 200, 195))
        out.append(np.asarray(img))
    return out


def compare(pair: dict, cap: str) -> list[np.ndarray]:
    ok = read_clip(video_path(pair["suite"], 10, pair["seed"], pair["task_id"], pair["episode"]))
    ng = read_clip(video_path(pair["suite"], 50, pair["seed"], pair["task_id"], pair["episode"]))
    out = []
    for i in range(n_frames(len(ng), 1.5)):
        img = Image.new("RGB", (W, H), PANEL)
        d = ImageDraw.Draw(img)
        a, sa, da = at(ok, i)
        b, sb, db = at(ng, i)
        img.paste(a.resize((600, 600)), (26, 60))
        img.paste(b.resize((600, 600)), (654, 60))
        d.text((26, 14), "n_action_steps = 10", font=F(28, True), fill="white")
        d.text((654, 14), "n_action_steps = 50", font=F(28, True), fill="white")
        if da:
            badge(d, 40, 74, "成功", OK)
        if db:
            badge(d, 668, 74, "失敗（時間切れ）", NG)
        d.text((26, 672), f"同じ初期状態　{SUITE_JA[pair['suite']]}　step {sb:3d}", font=F(22), fill=(190, 190, 185))
        w = d.textlength(cap, font=F(24, True))
        d.text((W - 26 - w, 670), cap, font=F(24, True), fill=(255, 210, 120))
        out.append(np.asarray(img))
    return out


def summary() -> list[np.ndarray]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    fig = Image.open(RES / "figures" / "fig1_success_vs_chunk.png").convert("RGB")
    s = 640 / fig.width
    fig = fig.resize((640, int(fig.height * s)), Image.LANCZOS)
    img.paste(fig, (20, (H - fig.height) // 2))
    x = 690
    d.text((x, 60), "結果（8,400エピソード）", font=F(32, True), fill=INK)
    lines = [
        ("成功率は 10 以下で横ばい、", "10 を超えると低下（50 で約22ポイント低下）"),
        ("推論回数まで考えると 10 が最良", "（2 は成功率が同じで推論回数が5倍）"),
        ("長いチャンクの失敗の約2/3は掴み損ね", "ずれを直せないまま運搬に移ってしまう"),
    ]
    y = 140
    for a, b in lines:
        d.text((x, y), "・" + a, font=F(26, True), fill=INK)
        d.text((x + 26, y + 40), b, font=F(22), fill=INK2)
        y += 120
    d.text((x, 540), "コードと全結果", font=F(22, True), fill=INK2)
    d.text((x, 572), GITHUB, font=F(22), fill=ACCENT)
    return [np.asarray(img)] * int(10 * FPS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eps = pd.read_csv(RES / "episodes.csv")
    scenes = []
    first = True
    # 2. 各スイート2エピソードの成功例
    for suite in SUITE_JA:
        cand = eps[(eps.suite == suite) & (eps.n_action_steps == 10) & (eps.seed == 1000) & eps.success]
        used = set()
        for r in cand.itertuples():
            if r.task_id in used:
                continue
            p = video_path(suite, 10, 1000, r.task_id, r.episode)
            if not p.exists():
                continue
            intro = "「何ステップ先まで見切りで動くか」は成功率をどう変えるか" if first else None
            scenes.append(single_run(suite, r.task_id, r.episode, read_clip(p), intro))
            first = False
            used.add(r.task_id)
            if len(used) == 2:
                break
    # 3. n_action_steps を変えた4画面
    scenes.append(sweep(eps))
    # 4. 10 と 50 の比較
    for p, cap in PAIRS:
        if video_path(p["suite"], 10, p["seed"], p["task_id"], p["episode"]).exists() and \
           video_path(p["suite"], 50, p["seed"], p["task_id"], p["episode"]).exists():
            scenes.append(compare(p, cap))
    # 5. まとめ
    scenes.append(summary())
    frames = [f for s in scenes for f in s]
    path = OUT / "pai_final.mp4"
    w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8)
    for f in frames:
        w.append_data(f)
    w.close()
    robot = len(frames) - int(10 * FPS)
    print(f"saved {path}  ({len(frames) / FPS:.1f} s, うちロボットの映像 {robot / FPS:.1f} s)")


if __name__ == "__main__":
    main()
