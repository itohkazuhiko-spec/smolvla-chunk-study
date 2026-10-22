# SmolVLA action chunk 実行戦略の定量評価（LIBERO）

SmolVLA（`HuggingFaceVLA/smolvla_libero`）の `n_action_steps`（1回の推論で生成したアクションチャンクのうち、何ステップ実行してから再推論するか）が、LIBERO ベンチマークのタスク成功率に与える影響を、再現可能な評価パイプラインで定量評価する。

## 構成

```
configs/   実験条件（YAML）
scripts/   run_experiments.py（一括評価）, aggregate.py（集計・グラフ化）
env/       実行環境の記録（export_env.sh で生成）
results/   runs.csv（実行履歴）, summary.csv, 図
```

## 再現手順

```bash
conda create -n lerobot python=3.12 -y && conda activate lerobot
conda install -y -c conda-forge ffmpeg
conda env config vars set MUJOCO_GL=egl   # 再度 activate して反映
git clone https://github.com/huggingface/lerobot.git && cd lerobot
git checkout v0.6.1 && pip install -e ".[smolvla,libero]" && cd ..

python scripts/run_experiments.py configs/pilot.yaml   # パイプラインの動作確認（数分）
python scripts/run_experiments.py configs/main.yaml    # 本評価
python scripts/aggregate.py
```

途中で止まっても、同じコマンドを再実行すれば完了済みの条件をスキップして再開する。

## 実行環境

`env/` を参照（GPU, ドライバ, パッケージのバージョンを記録）。
