# Quantitative Trading

## 项目定位

这是一个面向 A 股短周期 `Top-K` 选股的研究与执行项目。当前流程分成两条：

日常执行流程：

1. 更新本地日线数据
2. 检查数据状态
3. 用已经选定的历史主模型做当天推理
4. 生成 `review_top_k.csv` 候选池
5. 按 `strategy/post_filter.md` 做事件面过滤
6. 从 `Keep` 里按原始 `candidate_rank` 顺序取前 `top_k`

研究流程：

1. 更新本地日线数据
2. 检查数据状态
3. 跑 batch 实验比较窗口、模型和超参数
4. 用 `analyze` 选择更适合作为候选池生成器的模型
5. 把选定 run 作为后续日常 `infer --source-run ...` 的固定来源

当前默认标签定义：

- `T` 日收盘后出信号
- `T+1` 开盘买入
- `T+2` 开盘卖出

也就是默认 `data.label_horizon = 1`。

## 目录结构

```text
app/
  runtime.py
  factories.py
  reporting.py
  cache.py

pipelines/
  context.py
  evaluation.py
  recommendation.py

data/
  dataset.py
  splits.py
  fetcher.py

metrics/
  selection.py
  decision.py

models/
  encoders.py
  loss_functions.py
  trainer.py

strategy/
  universe_selector.py
  backtest.py

execution/
  mock_trader.py

scripts/
  update_eod_data.py
  check_data_status.py
  batch_experiments.py
  analysis.py
  infer_from_run.py
  model_hparam_sweep.py
  experiment_template_*.yaml
```

## 统一入口

统一入口只保留正式子命令：

```powershell
python main.py <command> [options]
```

支持的命令：

- `train`
- `infer`
- `update-data`
- `check-data`
- `batch`
- `analyze`
- `sweep`

## 用法示例

### 1. 更新本地数据

```powershell
D:\anaconda3\envs\qt\python.exe main.py update-data
```

回补历史：

```powershell
D:\anaconda3\envs\qt\python.exe main.py update-data --backfill-history --backfill-days 120
```

### 2. 检查数据状态

```powershell
D:\anaconda3\envs\qt\python.exe main.py check-data
```

### 3. 单次训练

```powershell
D:\anaconda3\envs\qt\python.exe main.py train --config config.yaml
```

指定历史 `T` 日：

```powershell
D:\anaconda3\envs\qt\python.exe main.py train --config config.yaml --as-of-date 2026-03-10
```

启用时间块打乱：

```powershell
D:\anaconda3\envs\qt\python.exe main.py train --config config.yaml --shuffle-time-blocks --shuffle-block-size 20
```

### 4. 用历史 run 权重推理

```powershell
D:\anaconda3\envs\qt\python.exe main.py infer --source-run batch_20260405_001023_01_gru64x2_seed7
```

说明：

- `source-run` 指 `outputs/runs/` 下某个已经训练完成的 run 目录名，或它的完整路径
- `infer` 不会重新训练，只会读取该 run 的 `config.yaml` 和 `best.ckpt`，基于最新数据生成当天候选池
- 当前默认产物是 `review_top_k.csv`，不是最终交易单

指定 checkpoint：

```powershell
D:\anaconda3\envs\qt\python.exe main.py infer --source-run batch_20260405_001023_01_gru64x2_seed7 --checkpoint-name last.ckpt
```

指定输出目录名：

```powershell
D:\anaconda3\envs\qt\python.exe main.py infer --source-run batch_20260405_001023_01_gru64x2_seed7 --output-name infer_today
```

做历史 `T` 日推理：

```powershell
D:\anaconda3\envs\qt\python.exe main.py infer --source-run batch_20260405_001023_01_gru64x2_seed7 --as-of-date 2026-03-10
```

### 5. 跑 batch 实验

```powershell
D:\anaconda3\envs\qt\python.exe main.py batch --experiment-file scripts\experiment_template_rolling_sweep.yaml --run-prefix rollinggrid
```

单卡有限并发：

```powershell
D:\anaconda3\envs\qt\python.exe main.py batch --experiment-file scripts\experiment_template_model_sweep.yaml --run-prefix modelgrid --max-jobs 2
```

只预览不执行：

```powershell
D:\anaconda3\envs\qt\python.exe main.py batch --experiment-file scripts\experiment_template_rolling_sweep.yaml --run-prefix rollinggrid --dry-run
```

### 6. 分析 batch 结果

```powershell
D:\anaconda3\envs\qt\python.exe main.py analyze --batch-dir outputs\batch_runs\rollinggrid_20260312_125916
```

输出文件：

- `analysis_results.csv`
- `analysis_report.md`

说明：

- `analyze` 当前只评估模型的候选池生成能力
- 它不评估 post filter 之后的最终交易名单
- `analyze` 更适合研究选模，不是每日必跑步骤

### 7. 生成或运行模型超参数 sweep

```powershell
D:\anaconda3\envs\qt\python.exe main.py sweep --run-prefix modelgrid
```

只生成实验文件：

```powershell
D:\anaconda3\envs\qt\python.exe main.py sweep --generate-only
```

自定义模型和超参数范围：

```powershell
D:\anaconda3\envs\qt\python.exe main.py sweep --models gru,lstm,attention --hidden-dims 64,128,256 --num-layers 1,2 --dropouts 0.1,0.3 --seq-lens 20 --run-prefix modelgrid
```

## 当前工作流

### 日常执行

日常不建议每天重跑 `batch + analyze`。当前更合理的执行顺序是：

1. `python main.py update-data`
2. `python main.py check-data`
3. `python main.py infer --source-run <已选定主模型run>`
4. 查看 `outputs/inference_runs/<...>/review_top_k.csv`
5. 按 `strategy/post_filter.md` 做事件面过滤
6. 从 `Keep` 里按原始 `candidate_rank` 顺序取前 `top_k`

### 研究选模

只有在你准备更换主模型时，才需要跑完整研究流程：

1. `python main.py update-data`
2. `python main.py check-data`
3. `python main.py batch --experiment-file ...`
4. `python main.py analyze --batch-dir ...`
5. 从 `analysis_report.md` 里选择更适合作为候选池生成器的 run
6. 后续日常推理固定用这个 run 做 `infer --source-run ...`

### 当前产物语义

- `review_top_k.csv`：模型候选池，已过价格与右侧过滤，但未过事件面过滤
- `top_k.csv` / `orders.csv`：当前默认不再由训练 / 推理直接生成
- `analysis_report.md`：当前只用于评估候选池生成能力，不直接代表最终可交易名单
## 当前默认训练与选模逻辑

### 默认 baseline

当前默认主线配置：

- `label_horizon = 1`
- `model = attention`
- `hidden_dim = 128`
- `num_layers = 2`
- `dropout = 0.3`
- `seq_len = 20`
- `lr = 0.0008`
- `weight_decay = 0.01`
- `mse_loss_weight = 1.0`
- 其他排序类 loss 默认关闭

### 已支持但默认关闭的 loss

当前代码已支持下列可配置 loss：

- `MSELoss`
- `PearsonLoss`
- `SoftRankICLoss`
- `HeadWeightedPairwiseLoss`

但当前实测默认主线仍是：

- `MSE only`

### 排序指标

当前同时统计：

- `pooled_ic`
- `daily_ic`
- `head_daily_ic`

其中默认头部范围：

- `selection_head_top_n = 10`

### 交易指标

重点观察：

- `top_k_mean_return`
- `excess_mean_return`
- `positive_excess_rate`
- `relative_return`
- `max_drawdown`

### Checkpoint 选择

`config.yaml` 中的 `model.checkpoint_selection_mode` 支持：

- `valid_ic`
- `ic_gate_composite`
- `topk_valid`

当前默认是 `topk_valid`。

`topk_valid` 的门槛：

- `valid_excess_return > 0`
- `valid_positive_excess_rate >= 0.50`
- `valid_daily_ic > 0`
- `valid_max_drawdown >= -0.10`

当前排序优先级更偏头部收益：

1. `valid_excess_return`
2. `valid_top_k_return`
3. `valid_positive_excess_rate`
4. `valid_head_daily_ic`
5. `valid_relative_return`
6. `valid_daily_ic`
7. `valid_max_drawdown`

## 新增逻辑

### 1. 右侧过滤

当前默认已在推荐阶段启用“右侧过滤”，位置在：

- `strategy.right_side_filter`

默认配置：

```yaml
strategy:
  top_k: 3
  right_side_filter:
    enabled: true
    min_ma_gap_5: -0.5
    min_volume_ratio_5: 0.0
    min_industry_ret_1_mean: -0.5
```

说明：

- 右侧过滤只作用于**候选池**
- 不改变训练样本
- 不改变 valid/test 历史回测
- 当前训练 / 推理默认输出的是 `review_top_k.csv`
- `review_top_k.csv` 仍然不是最终交易名单，后面还需要人工或外部模型执行 post filter

当前过滤使用的是推理样本 meta 中的特征值：

- `ret_1`
- `ret_5`
- `intraday_ret`
- `ma_gap_5`
- `volume_ratio_5`
- `industry_ret_1_mean`

如果过滤后候选池为空，会自动回退到未过滤候选池。

### 2. 更温和的右侧过滤配置

项目里额外提供了一份温和版配置：

- [config_rightside_relaxed.yaml](/D:/PyCharm%202025.3.3/projects/Quantitative%20Trading/config_rightside_relaxed.yaml)

它保留 `horizon=1 + attention baseline`，但用更宽松的右侧过滤，适合先做对照观察。

运行示例：

```powershell
D:\anaconda3\envs\qt\python.exe main.py train --config config_rightside_relaxed.yaml
```

### 3. `horizon=2` 对照实验模板

项目里已经加了 `horizon=2` 的模板，便于继续做标签持有期对照：

- `scripts/experiment_template_horizon2_model_compare.yaml`
- `scripts/experiment_template_horizon2_rolling_compare.yaml`
- `scripts/experiment_template_horizon2_loss_combo_compare.yaml`
- `scripts/experiment_template_horizon2_pairwise_weight_compare.yaml`

这些模板用于研究，不会影响当前默认 baseline。

## 结果怎么看

优先看这些核心文件：

- `summary.json`
- `backtest_metrics.json`
- `review_top_k.csv`
- `run.log`

再看这些详细文件：

- `train_metrics.csv`
- `valid_backtest.csv`
- `backtest.csv`
- `valid_predictions.csv`
- `test_predictions.csv`
- `candidate_rank.csv`
- `universe_report.csv`

终端摘要当前主要保留：

- 运行模式、样本量、模型名
- checkpoint 选择结果
- valid / test 核心交易指标
- `推荐 / 观察 / 不建议`
- 核心与详细文件清单
- 最新 `ReviewPool` 候选池
- `post filter` 尚未执行的提示

## Batch 加速

当前已支持两类提速：

- 训练上下文缓存
  - 复用 `selected_symbols / universe_report / dataset_bundle`
- batch 有限并发
  - 单卡建议先尝试 `--max-jobs 2`

观察 GPU：

```powershell
nvidia-smi -l 1
```

更适合看算力 / 显存：

```powershell
nvidia-smi dmon -s pucvmt -d 1
```

如果 `sm` 长期只有 `15% ~ 30%`，可以尝试 `--max-jobs 2`。单卡下通常不建议直接开到 `3`。


