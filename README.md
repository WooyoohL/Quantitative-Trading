# Quantitative Trading

## 项目定位

这是一个面向 A 股短周期 Top-K 选股的研究与执行项目。当前默认标签定义为：

- `T` 日收盘后出信号
- `T+1` 开盘买入
- `T+2` 开盘卖出

项目主流程是：

1. 更新本地 EOD 数据
2. 用 rolling window 训练模型
3. 生成下一交易日的候选股票、Top-K 推荐和模拟下单结果
4. 用 batch 实验和分析脚本比较窗口、模型和超参数

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

指定配置：

```powershell
D:\anaconda3\envs\qt\python.exe main.py check-data --config config.yaml
```

### 3. 单次训练

```powershell
D:\anaconda3\envs\qt\python.exe main.py train --config config.yaml
```

指定历史 T 日：

```powershell
D:\anaconda3\envs\qt\python.exe main.py train --config config.yaml --as-of-date 2026-03-10
```

启用时间块打乱：

```powershell
D:\anaconda3\envs\qt\python.exe main.py train --config config.yaml --shuffle-time-blocks --shuffle-block-size 20
```

### 4. 用历史 run 权重做推理

```powershell
D:\anaconda3\envs\qt\python.exe main.py infer --source-run 20260311_203338
```

指定 checkpoint：

```powershell
D:\anaconda3\envs\qt\python.exe main.py infer --source-run 20260311_203338 --checkpoint-name last.ckpt
```

指定输出目录名：

```powershell
D:\anaconda3\envs\qt\python.exe main.py infer --source-run 20260311_203338 --output-name infer_today
```

做历史 T 日推理：

```powershell
D:\anaconda3\envs\qt\python.exe main.py infer --source-run 20260311_203338 --as-of-date 2026-03-10
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

## 当前训练与选模逻辑

### Loss

当前训练、验证、测试的总 loss 口径统一为：

- `MSELoss`
- `PearsonLoss`

### 排序指标

- `pooled_ic`
- `daily_ic`
- `head_daily_ic`

其中当前默认头部范围是：

- `selection_head_top_n = 10`

### 交易指标

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

## 结果怎么看

优先看这些核心文件：

- `summary.json`
- `backtest_metrics.json`
- `top_k.csv`
- `orders.csv`
- `run.log`

再看这些详细文件：

- `train_metrics.csv`
- `valid_backtest.csv`
- `backtest.csv`
- `valid_predictions.csv`
- `test_predictions.csv`
- `candidate_rank.csv`
- `universe_report.csv`

终端摘要现在主要保留：

- 运行模式、样本量、模型名
- checkpoint 选择结果
- valid / test 核心交易指标
- `推荐 / 观察 / 不建议`
- 核心与详细文件清单
- 最终 Top-K 推荐

## Batch 加速

当前已支持两类提速：

- 训练上下文缓存
  - 复用 `selected_symbols / universe_report / dataset_bundle`
- batch 有限并发
  - 单卡建议先尝试 `--max-jobs 2`

观察 GPU 可以用：

```powershell
nvidia-smi -l 1
```

更适合看算力/显存：

```powershell
nvidia-smi dmon -s pucvmt -d 1
```

如果 `sm` 长期只有 `15% ~ 30%`，可以尝试 `--max-jobs 2`。单卡下通常不建议直接开到 `3`。
