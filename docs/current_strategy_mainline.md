# A股主策略运行线

本文档定义当前主策略。旧的事件面模拟盘流程保留在 `docs/codex_paper_trading_automation.md`，状态为 deprecated。

## 当前主策略

当前变更只作用于推荐生成，不改变模型结构，不改变数据加载框架，不改变交易执行规则。

- 重训节奏：每周一、周三重训。
- 推理节奏：每个交易日收盘后，用当天及以前的数据生成下一交易日买入推荐。
- 训练标签：市场超额收益，沿用上一版目标，即 `T+1` 开盘买入到 `T+3` 开盘卖出的超额收益。
- 交易执行：保持 `T` 日收盘给出推荐，`T+1` 开盘买入，`T+2` 开盘卖出。
- 推荐后处理：使用 `quality` 分层，只买 `quality_medium` 与 `quality_strong`。
- 仓位后处理：使用 `recommended`，中等信号每只 10%，强信号按模型信心缩放到总仓位 50%，单票最高 17.5%，总仓位最高 60%。
- 弱 Top1：保留诊断字段，但不买入。
- 价格上限：不使用 40 元价格上限，其余右侧过滤保留。
- 事件面：当前主策略不调用事件面和大模型。

## 每日晚间规则

对任意信号日 `T`：

1. `T` 日收盘后，读取截至 `T` 日的数据，生成 `T+1` 开盘买入推荐。
2. `T+1` 日收盘后，按既有模拟盘机制确认 `T` 日推荐的买入成交。
3. `T+1` 日收盘后，按既有模拟盘机制确认 `T-1` 日推荐的卖出成交。
4. `T+1` 日收盘后，再读取截至 `T+1` 日的数据，生成 `T+2` 开盘买入推荐。

## 服务器命令

默认命令就是当前主线：

```bash
python scripts/run_server_retrain_backtest.py
```

等价参数：

```bash
python scripts/rolling_retrain_backtest.py --config config.yaml --strategies mon_wed --hold-periods 2 --label-mode market_excess --cash-filter enabled --cash-filter-policy quality --disable-price-cap --score-quantile 0.50 --topk-mean-quantile 0.45 --score-gap-quantile 0.20 --min-position-exposure 0.05 --mid-position-exposure 0.10 --max-position-exposure 0.175 --max-gross-exposure 0.60 --position-policy recommended --weak-top1 disabled --auto-start-after-warmup
```

指定 GPU 时：

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/run_server_retrain_backtest.py
```

## Codex 自动化

自动化 `a` 已改为当前主线每日运行。它先更新数据并检查数据状态，再运行：

```powershell
$PY = "D:\anaconda3\envs\qt\python.exe"
$outDir = "outputs/retrain_backtest/current_mainline_recommended_weaktop1_disabled"
& $PY scripts/run_logged_command.py --log $runLog --interval 900 -- $PY scripts/run_server_retrain_backtest.py --output-dir $outDir --resume
```

自动化不调用 deprecated 事件面流程，不生成事件决策，不执行 `paper-trade finalize`。

## 输出文件

主线输出目录形如：

```text
outputs/retrain_backtest/server_monwed_h2_quality_recommended_weaktop1_disabled_nopricecap_<YYYYMMDD_HHMMSS>/
```

关键文件：

- `daily_returns.csv`：日度记录，保留 `signal_date`、`buy_date`、`sell_date`。
- `trades.csv`：逐票交易明细。
- `strategy_summary.csv`：策略摘要。
- `model_schedule.csv`：训练与推理排期。
- `bias_summary.csv`：归纳偏置摘要。
- `industry_exposure.csv`：行业暴露。

## deprecated 线

旧事件面模拟盘仍保留，文档见 `docs/codex_paper_trading_automation.md`。它包含数据库更新、事件筛选、finalize 和工作簿账本流程，不作为当前主策略默认自动化。
