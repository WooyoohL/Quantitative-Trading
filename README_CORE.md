# README Core

## 一句话说明

这是一个基于本地 A 股日线数据做短周期 `Top-K` 选股的项目：

- 每天更新本地数据
- 用 rolling window 训练序列模型
- 输出下一交易日的推荐股票和模拟订单

当前默认标签定义：

- `T` 收盘后出信号
- `T+1` 开盘买
- `T+2` 开盘卖

默认即：

- `data.label_horizon = 1`

## 当前主流程

### 1. 更新数据

```powershell
D:\anaconda3\envs\qt\python.exe main.py update-data
```

### 2. 单次训练 + 推理

```powershell
D:\anaconda3\envs\qt\python.exe main.py train --config config.yaml
```

输出目录：

- `outputs/runs/<run_id>/`

最先看：

- `summary.json`
- `backtest_metrics.json`
- `top_k.csv`
- `orders.csv`
- `run.log`

### 3. 用历史权重做当天推理

```powershell
D:\anaconda3\envs\qt\python.exe main.py infer --source-run <run_id>
```

### 4. 跑 batch 实验

```powershell
D:\anaconda3\envs\qt\python.exe main.py batch --experiment-file scripts\experiment_template_rolling_sweep.yaml --run-prefix rollinggrid
```

### 5. 分析 batch 结果

```powershell
D:\anaconda3\envs\qt\python.exe main.py analyze --batch-dir outputs\batch_runs\<batch_dir_name>
```

### 6. 扫模型超参数

```powershell
D:\anaconda3\envs\qt\python.exe main.py sweep --run-prefix modelgrid
```

## 当前关键设计

### 1. 默认 baseline

当前默认主线：

- `label_horizon = 1`
- `attention`
- `hidden_dim = 128`
- `num_layers = 2`
- `seq_len = 20`
- `dropout = 0.3`
- `MSE only`

### 2. 选模

默认：

- `checkpoint_selection_mode = topk_valid`

它不是只看 `pooled valid_ic`，而是更贴近最终只买 `Top-K` 的交易目标。

主要门槛：

- `valid_excess_return > 0`
- `valid_positive_excess_rate >= 50%`
- `valid_daily_ic > 0`
- `valid_max_drawdown >= -10%`

主要排序：

1. `valid_excess_return`
2. `valid_top_k_return`
3. `valid_positive_excess_rate`
4. `valid_head_daily_ic`
5. `valid_relative_return`
6. `valid_daily_ic`

### 3. 右侧过滤

当前默认已经在推荐阶段启用右侧过滤：

- 只过滤最终推荐池
- 不改变训练过程

默认条件：

- `ma_gap_5 >= -0.5`
- `volume_ratio_5 >= 0.0`
- `industry_ret_1_mean >= -0.5`

如果你想观察更温和的版本，可以用：

- [config_rightside_relaxed.yaml](/D:/PyCharm%202025.3.3/projects/Quantitative%20Trading/config_rightside_relaxed.yaml)

### 4. 输出

控制台只打印核心摘要，详细信息统一写文件。

核心文件：

- `summary.json`
- `backtest_metrics.json`
- `top_k.csv`
- `orders.csv`
- `run.log`

详细文件：

- `train_metrics.csv`
- `valid_predictions.csv`
- `test_predictions.csv`
- `valid_backtest.csv`
- `backtest.csv`
- `candidate_rank.csv`
- `inference_predictions.csv`
- `universe_report.csv`

## 当前阶段结论

现在的主线不是继续堆更多 loss，而是：

- 保持 `horizon=1 + attention baseline`
- 用推荐阶段右侧过滤降低明显左侧抄底票
- 用 batch 和 analysis 继续观察实盘可转化性

项目里还保留了 `horizon=2` 和多种 loss 的实验模板，用于研究对照，但它们不影响当前默认主线。
