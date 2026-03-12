# README Core

## 一句话说明

这是一个基于本地 A 股日线数据做短周期 Top-K 选股的项目：

- 每天更新本地数据
- 用 rolling window 训练序列模型
- 输出下一交易日的推荐股票和模拟订单

默认标签定义：

- `T` 收盘后出信号
- `T+1` 开盘买
- `T+2` 开盘卖

## 当前主流程

### 1. 更新数据

```powershell
D:\anaconda3\envs\qt\python.exe scripts\update_eod_data.py
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

### 7. 统一入口的其它命令

```powershell
D:\anaconda3\envs\qt\python.exe main.py update-data
D:\anaconda3\envs\qt\python.exe main.py check-data
```

## 当前关键设计

### 1. 结构

- `app/`
  - 配置、运行时、控制台摘要
- `pipelines/`
  - 训练上下文、评估、推荐输出
- `data/`
  - 特征工程与样本构造
- `models/`
  - 模型、loss、trainer
- `strategy/`
  - 股票池与回测
- `scripts/`
  - 数据更新、batch、分析、推理入口

### 2. 选模

当前默认：

- `checkpoint_selection_mode = topk_valid`

它不是只看 `pooled valid_ic`，而是更贴近“最终只买推荐的 K 只股票”的目标。

主要门槛：

- `valid_excess_return > 0`
- `valid_positive_excess_rate >= 50%`
- `valid_daily_ic > 0`
- `valid_max_drawdown >= -10%`

主要排序：

1. `valid_excess_return`
2. `valid_positive_excess_rate`
3. `valid_relative_return`
4. `valid_daily_ic`
5. `valid_head_daily_ic`

### 3. 输出

控制台只打印核心摘要。详细信息统一写文件。

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

第一阶段重构已经把最乱的地方收住了：

- `main.py` 不再承担全部细节
- `infer_from_run.py` 不再依赖 `main.py`
- 训练、评估、推荐输出已经拆成共享 pipeline

下一阶段更适合继续处理：

- `data/dataset.py` 的进一步拆分
- 指标体系的更细模块化
