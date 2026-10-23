#!/usr/bin/env python
"""list_pairs.py が出力した組（基準は成功・比較は失敗）について、
成功／失敗を上下に並べたコマ割り画像を自動生成する。分類用のラベル記入表も作る。

例:
  python scripts/make_contact_sheets.py results/pairs/libero_object_ref10_cmp50.csv --limit 10
出力:
  results/pairs/<stem>_sheets/task{T}_seed{S}_ep{E}.png
  results/pairs/<stem>_labels.csv   （failure_type 列に分類を記入する。既存のファイルは上書きしない）
"""
import argparse
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W = 240      # 1コマの大きさ（px）
HEAD = 34    # 各段の見出しの高さ
N_COLS = 6


def font(size: int) -> ImageFont.FreeTypeFont:
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size=size)


def read_frames(path: Path) -> np.ndarray:
    try:
        return iio.imread(path, plugin="pyav")
    except Exception:
        return iio.imread(path)


def pick_steps(n_ref: int, n_cmp: int) -> tuple[list[int], list[int]]:
    """成功側は全体を等間隔に。失敗側は前半を成功側と同じステップにそろえ、残りは後半と最終コマ。"""
    ref = np.linspace(0, n_ref - 1, N_COLS).round().astype(int).tolist()
    common = [s for s in ref[: N_COLS - 2] if s < n_cmp]
    rest = N_COLS - len(common)
    start = min(ref[N_COLS - 2], n_cmp - 1)
    tail = np.linspace(start, n_cmp - 1, rest).round().astype(int).tolist()
    return ref, common + tail


def row(frames: np.ndarray, steps: list[int], title: str, color: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (W * N_COLS, W + HEAD), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W * N_COLS, HEAD], fill=color)
    d.text((10, 6), title, fill="white", font=font(19))
    f14 = font(16)
    for i, s in enumerate(steps):
        fr = Image.fromarray(frames[s]).rotate(180).resize((W, W))  # LIBERO の画像は上下左右が反転している
        img.paste(fr, (i * W, HEAD))
        d.rectangle([i * W + 4, HEAD + 4, i * W + 104, HEAD + 28], fill=(0, 0, 0))
        d.text((i * W + 10, HEAD + 6), f"step {s}", fill="white", font=f14)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_csv", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="作成する組の数（0=全部）。タスクが偏らないよう順に選ぶ")
    args = ap.parse_args()

    m = re.search(r"ref(\d+)_cmp(\d+)", args.pairs_csv.stem)
    ref_n, cmp_n = (m.group(1), m.group(2)) if m else ("ref", "cmp")
    pairs = pd.read_csv(args.pairs_csv)
    if args.limit:
        # タスクごとに1組ずつ順番に取り出す（ラウンドロビン）
        pairs["_k"] = pairs.groupby("task_id").cumcount()
        pairs = pairs.sort_values(["_k", "task_id"]).head(args.limit).drop(columns="_k")

    out = args.pairs_csv.with_name(args.pairs_csv.stem + "_sheets")
    out.mkdir(exist_ok=True)
    made = []
    for r in pairs.itertuples(index=False):
        rv, cv = ROOT / r.ref_video, ROOT / r.cmp_video
        if not (rv.exists() and cv.exists()):
            print(f"  missing video: task{r.task_id} seed{r.seed} ep{r.episode}")
            continue
        fr, fc = read_frames(rv), read_frames(cv)
        sr, sc = pick_steps(len(fr), len(fc))
        top = row(fr, sr, f"n_action_steps={ref_n}: SUCCESS (done at step {len(fr) - 1})", (30, 120, 60))
        bot = row(fc, sc, f"n_action_steps={cmp_n}: FAILURE (ended at step {len(fc) - 1})", (170, 40, 40))
        cap_h = 30
        sheet = Image.new("RGB", (top.width, cap_h + top.height + bot.height), "white")
        ImageDraw.Draw(sheet).text((10, 5), f"task {r.task_id}: {getattr(r, 'task', '')}   (seed {r.seed}, episode {r.episode})",
                                   fill="black", font=font(18))
        sheet.paste(top, (0, cap_h))
        sheet.paste(bot, (0, cap_h + top.height))
        name = f"task{r.task_id}_seed{r.seed}_ep{r.episode}.png"
        sheet.save(out / name)
        made.append({"sheet": name, "task_id": r.task_id, "task": getattr(r, "task", ""), "seed": r.seed,
                     "episode": r.episode, "ref_len": len(fr), "cmp_len": len(fc), "failure_type": "", "note": ""})
        print(f"  {name}")

    labels = args.pairs_csv.with_name(args.pairs_csv.stem + "_labels.csv")
    if made and not labels.exists():
        pd.DataFrame(made).to_csv(labels, index=False)
        print(f"label template: {labels}")
    print(f"{len(made)} sheets -> {out}")


if __name__ == "__main__":
    main()
