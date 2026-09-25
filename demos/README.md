# Demos（网页演示 + 基准）

用 reflex-engine 的 `LayaModel` 跑的可视化演示。依赖 laya-mlx 的 venv（本机 `~/laya-mlx/.venv`）。

```bash
~/laya-mlx/.venv/bin/python demos/web_triage.py   # 医药分诊：纯 Laya vs reflex-engine 闸门
~/laya-mlx/.venv/bin/python demos/web_snake_reflex.py  # 贪吃蛇走 reflex-engine 的 LayaModel + gate
~/laya-mlx/.venv/bin/python demos/web_snake.py    # 贪吃蛇（laya-mlx 自带 LayaPolicy，对照组）
~/laya-mlx/.venv/bin/python demos/bench_local.py  # 推理延迟基准
```

| 文件 | 作用 | 查看 |
|---|---|---|
| `web_triage.py` | 医药咨询分诊：同一套 Laya 概率，左侧纯 Laya、右侧 reflex-engine 闸门 + 修正回流 | http://localhost:8000 |
| `web_snake_reflex.py` | 贪吃蛇：每步走 `LayaModel.decide_full` + `gate.route`，AUTO 执行 Laya / 否则退回确定性规划器 | http://localhost:8000 |
| `web_snake.py` | 贪吃蛇：laya-mlx 自带 `LayaPolicy`（对照组，不走 reflex-engine） | http://localhost:8000 |
| `bench_local.py` | M5 上端到端 `predict()` 延迟（EN 421M ~12.3ms / 多语 322M ~5.5ms） | 终端 |

说明：
- 权重路径硬编码为本机 `~/laya-mlx/models/laya-multilingual-mlx`，换机器需改 `--model` 或 `LAYA_MODEL_PATH`。
- 需先 `pip install laya-mlx`（含 `rich`）并下载 FP16 checkpoint。
- 三个 web demo 都监听 8000 端口，**别同时跑多个**。
