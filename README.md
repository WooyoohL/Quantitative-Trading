# Quantitative Trading

## 快速工具命令
### 1. 从 `test_predictions.csv` 导出测试窗口逐日 Top-K

当你想查看整个测试窗口里每天被选中的 `top_k` 股票时：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\export_test_top_k_daily.py outputs\runs\<run_id>\test_predictions.csv
```

默认行为：
- 从 `config.yaml` 读取 `strategy.top_k`
- 将 `test_top_k_daily.csv` 保存到输入文件所在目录

可选参数：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\export_test_top_k_daily.py outputs\runs\<run_id>\test_predictions.csv --top-k 5
D:\anaconda3\envs\qt\python.exe scripts\export_test_top_k_daily.py outputs\runs\<run_id>\test_predictions.csv --output-name my_test_top_k.csv
```

### 2. 批量实验运行器

运行自定义实验网格：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\batch_experiments.py --experiment-file scripts\experiment_template.yaml --run-prefix model_grid
```

只运行 rolling-window 实验：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\batch_experiments.py --experiment-file scripts\experiment_template_rolling.yaml --run-prefix rolling_grid
```

运行 rolling 窗口超参数扫描：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\batch_experiments.py --experiment-file scripts\experiment_template_rolling_sweep.yaml --run-prefix rolling_grid
```

只查看将要运行的实验，不实际执行：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\batch_experiments.py --experiment-file scripts\experiment_template_rolling.yaml --dry-run
```

### 3. 历史截面回放训练

按历史某个 `T` 日收盘后的状态做回放训练：

```powershell
D:\anaconda3\envs\qt\python.exe main.py --run-name replay_20260227 --as-of-date 2026-02-27
```

### 4. 时间块打乱测试

用于检查模型是否过度依赖原始时间顺序：

```powershell
D:\anaconda3\envs\qt\python.exe main.py --run-name time_block_shuffle_test --shuffle-time-blocks --shuffle-block-size 20
```

如果你想先快速看项目重点，再看运行细节，先读 [README_CORE.md](/D:/PyCharm%202025.3.3/projects/Quantitative%20Trading/README_CORE.md)。

这个项目面向 A 股日线选股，主流程是：

1. 手动更新到最近一个已收盘交易日的数据
2. 基于本地最新数据重新训练模型
3. 输出下一交易日候选股票、回测指标和训练产物

默认解释器：

```powershell
D:\anaconda3\envs\qt\python.exe
```

## 运行流程

### 1. 更新本地数据

日常增量更新：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\update_eod_data.py
```

如果你调大了历史窗口，想主动回补更早历史：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\update_eod_data.py --backfill-history
```

如果还想显式指定回补窗口天数：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\update_eod_data.py --backfill-history --backfill-days 520
```

这一阶段会做这些事：

1. 更新个股日线到最近已收盘交易日
2. 更新核心指数日线
3. 更新股票 universe 快照，并在接口失败时回退到本地快照
4. 更新行业映射，并据此聚合行业日线
5. 识别长期无新数据股票，写入 `stale_symbols.csv`
6. 生成当期候选池 `current_candidates.csv`

默认输出文件：

- `data/eod_daily.csv`
- `data/index_daily.csv`
- `data/industry_map.csv`
- `data/industry_daily.csv`
- `data/current_candidates.csv`
- `data/stale_symbols.csv`
- `data/universe_snapshot.csv`
- `data/eod_daily_meta.json`

说明：

- `stale_symbols.csv` 用于排除后续抓取和当期候选池，不删除历史行情
- `universe_snapshot.csv` 用于在全市场股票列表接口失败时保持筛选规则稳定
- CSV 默认使用 `utf-8-sig` 编码，便于直接用 Excel 打开

### 2. 检查数据状态

```powershell
D:\anaconda3\envs\qt\python.exe scripts\check_data_status.py
```

它会输出：

- 股票列表来源
- 总股票数
- 候选股票数
- stale 股票数
- 本地最新股票日期
- 本地最新指数日期
- 本地最新行业日期
- 最近已收盘交易日
- 是否已经更新到目标交易日

如果股票数据还没更新到最近收盘，不要直接训练，先重新跑 `update_eod_data.py`。

### 3. 训练并生成信号

```powershell
D:\anaconda3\envs\qt\python.exe main.py
```

指定 run 名称：

```powershell
D:\anaconda3\envs\qt\python.exe main.py --run-name 20260307_close
```

训练产物会写到：

- `outputs/runs/<run_id>/`

最近一次训练摘要会写到：

- `outputs/latest_run.json`

当前支持两种模式，由 `training.mode` 控制：

- `trade`
  - 训练股票池使用交易域口径
  - 上下文特征按训练池口径构造
  - 推理和最终推荐都只在训练池内进行
  - 不生成全市场 `market_rank.csv`
- `market_rank`
  - 训练股票池使用较宽市场口径
  - 上下文按全市场构造
  - 推理走全市场
  - 会额外生成全市场 `market_rank.csv`

当前 `trade` 模式下，`current_candidates.csv` 不再参与最终推荐；最终推荐直接在训练池内按分数排序，再应用价格上限。

### 4. 评估实际买入后的收益

如果你按 `top_k.csv` 在下一交易日开盘买入，并按标签口径在后续开盘卖出，可以直接运行：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\evaluate_realized_trades.py
```

它默认读取最近一次 run 的 `top_k.csv`，并输出：

- `realized_trade_eval.csv`
- `realized_trade_summary.json`

如果你只买了其中几只，可以指定：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\evaluate_realized_trades.py --symbols 600000.SH,600036.SH
```

如果你想把真实成交结果和模型口径做差异对比，再额外准备一个 CSV，至少包含这些列：

- `symbol`
- `actual_entry_date`
- `actual_entry_price`
- `actual_exit_date`
- `actual_exit_price`

然后运行：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\evaluate_realized_trades.py --actual-trades-path your_actual_trades.csv
```

脚本会额外输出：

- `actual_trade_compare.csv`
- `actual_trade_compare_summary.json`

## 你现在最该看哪些文件

### 每次更新后

- `data/eod_daily_meta.json`
  - 看是否已更新到目标交易日，以及抓取任务数、候选池数量、stale 排除数量
- `data/current_candidates.csv`
  - 看当日候选池
- `data/stale_symbols.csv`
  - 看哪些股票被排除出后续抓取和当期候选池

### 每次训练后

- `outputs/latest_run.json`
  - 先看最新 run 目录、模式、样本数、信号日、`valid_ic`、`test_ic`
- `outputs/runs/<run_id>/top_k.csv`
  - 最终推荐结果
- `outputs/runs/<run_id>/summary.json`
  - 本次训练和评估摘要
- `outputs/runs/<run_id>/backtest_metrics.json`
  - 回测核心指标
- `outputs/runs/<run_id>/train_metrics.csv`
  - 每个 epoch 的 `train_loss / train_ic / valid_loss / valid_ic`

### 排查时再看

- `outputs/runs/<run_id>/backtest.csv`
- `outputs/runs/<run_id>/valid_predictions.csv`
- `outputs/runs/<run_id>/test_predictions.csv`
- `outputs/runs/<run_id>/inference_predictions.csv`
- `outputs/runs/<run_id>/universe_report.csv`
- `outputs/runs/<run_id>/orders.csv`
- `outputs/runs/<run_id>/candidate_rank.csv`
- `outputs/runs/<run_id>/market_rank.csv`（仅 `market_rank` 模式）

## 输出文件怎么读

| 文件 | 作用 | 什么时候看 |
| --- | --- | --- |
| `data/eod_daily_meta.json` | 数据更新摘要 | 每次更新后 |
| `data/current_candidates.csv` | 当日候选池 | 每次更新后 |
| `data/stale_symbols.csv` | 长期无新数据股票列表 | 每次更新后 |
| `data/universe_snapshot.csv` | 最近一次成功抓取的 universe 快照 | 排查股票列表来源时 |
| `outputs/latest_run.json` | 最新训练摘要 | 每次训练后 |
| `top_k.csv` | 下一交易日候选股 | 最重要 |
| `summary.json` | 训练、评估、回测总摘要 | 每次训练后 |
| `backtest_metrics.json` | 回测指标和建议区间 | 每次训练后 |
| `train_metrics.csv` | 逐 epoch 训练日志 | 判断是否过拟合 |
| `universe_report.csv` | 训练池明细和筛选诊断 | 排查股票为什么没进训练池 |
| `candidate_rank.csv` | 最终推荐排序结果 | 训练后 |
| `market_rank.csv` | 全市场排序结果 | 仅 `market_rank` 模式 |

## 当前数据表

### 股票日线 `data/eod_daily.csv`

当前尽量保留这些字段：

- `date`
- `symbol`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `turnover`
- `amplitude`
- `pct_chg`
- `chg`
- `turnover_rate`
- `outstanding_share`
- `source`

数据源优先级：

1. `stock_zh_a_hist_tx`
2. `stock_zh_a_daily`

### 指数日线 `data/index_daily.csv`

默认抓取：

- `sh000001` 上证指数
- `sz399001` 深证成指
- `sh000300` 沪深 300
- `sh000905` 中证 500
- `sz399006` 创业板指

### 行业映射 `data/industry_map.csv`

保存：

- `symbol`
- `industry_name`
- `industry_code`
- `updated_at`
- `source`

### 行业日线 `data/industry_daily.csv`

保存：

- `date`
- `industry_name`
- `industry_code`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `turnover`
- `amplitude`
- `pct_chg`
- `chg`
- `turnover_rate`
- `source`

## 模型输入输出

### 输入

模型输入张量形状：

```text
[batch, seq_len, feature_dim]
```

当前默认：

- `seq_len = 20`
- 指数、行业、peer 全启用时，默认 `feature_dim = 49`

单个样本的含义是：

1. 某只股票在信号日 `T`
2. 使用截至 `T` 收盘已知的最近 `seq_len` 个有效交易 bar 特征序列
3. 预测这只股票未来可执行持有区间的收益

### 输出

标签定义固定为：

- `T` 收盘后生成信号
- `T+1` 开盘买入
- `T+1+h` 开盘卖出

其中 `h = data.label_horizon`，当前默认 `1`，所以标签实际是：

- `T+1` 开盘买入
- `T+2` 开盘卖出

模型输出是一个标量分数。分数越高，表示模型越看好该股票在这个可执行持有区间内的收益。

## 当前特征

### 个股特征

- 收益率：`ret_1` `ret_5` `ret_10` `pct_chg_1`
- K 线结构：`gap_open` `intraday_ret` `amplitude_1` `amplitude_5_mean` `range_pct`
- 成交相关：`turnover_rate_1` `turnover_rate_missing` `volume_ratio_5` `turnover_ratio_5`
- 波动和趋势：`volatility_5` `volatility_10` `ma_gap_5` `ma_gap_10` `rsi_14` `macd_pct`

### 市场横截面特征

- `market_ret_1_mean`
- `market_ret_5_mean`
- `market_volatility_5_mean`
- `market_breadth_up`

### 指数特征

每个指数都会生成：

- `idx_<index_key>_ret_1`
- `idx_<index_key>_ret_5`

### 行业特征

- 行业内股票聚合：`industry_ret_1_mean` `industry_ret_5_mean` `industry_breadth_up`
- 行业相对强弱：`industry_strength_vs_market`
- 行业日线：`industry_board_pct_chg_1` `industry_board_ret_5` `industry_board_turnover_rate`
- 缺失标记：`industry_board_missing` `industry_missing`

### 高相关 peer 特征

peer 的构造方式：

1. 只用训练窗口内数据计算
2. 优先限制在同行业内找 peer
3. 再按最近 `peer.lookback_days` 的收益相关性取 top-k

当前聚合特征：

- `peer_ret_1_mean`
- `peer_ret_5_mean`
- `peer_amplitude_mean`
- `peer_turnover_rate_mean`
- `peer_top1_ret_1`
- `peer_corr_mean`
- `peer_count`

## 关键配置

### 顶层

- `seed`
- `use_real_data`
- `fallback_to_synthetic`

### `fetch`

- `max_workers`
- `request_timeout`
- `show_progress`

### `data`

- `path`
- `meta_path`
- `stale_symbols_path`
- `universe_snapshot_path`
- `candidate_path`
- `index_path`
- `industry_map_path`
- `industry_daily_path`
- `calendar_lookback_days`
- `trainable_history_days`
- `label_horizon`

### `fallback_universe_symbols`

全市场股票列表接口失败且本地 snapshot 不可用时，回退到这里的静态股票列表。

### `universe`

- `lookback_days`
- `exclude_symbols`

### `universe.filters`

- `main_board_only`
- `exclude_st`
- `max_stale_trade_days`
- `min_latest_price`
- `max_latest_price`
- `min_avg_turnover`
- `max_avg_intraday_range`
- `max_avg_abs_ret1`
- `min_n_days`

说明：

- `min_continuous_tail_days` 不需要手工配置，运行时会按模式自动注入
- `trade` 模式下训练期会额外注入 `training_max_latest_price = max_latest_price`

### `training`

- `mode`
  - `trade`
  - `market_rank`

### `index`

- `enabled`
- `symbols`

### `industry`

- `enabled`

### `peer`

- `enabled`
- `top_k`
- `lookback_days`
- `min_overlap`

### `rolling`

- `train_days`
- `valid_days`
- `test_days`

切分方式：

1. 先构造完整样本
2. 再按样本锚点日期切分 `train / valid / test`

### `sequence`

- `seq_len`

### `model`

- `name`: `gru` / `lstm` / `attention`
- `hidden_dim`
- `num_layers`
- `dropout`
- `attention_heads`
- `ff_multiplier`
- `max_seq_len`
- `epochs`
- `lr`
- `weight_decay`
- `batch_size`
- `eval_batch_size`
- `early_stopping_patience`

### `strategy`

- `top_k`

### `execution`

- `initial_cash`
- `max_positions`

## 日常操作建议

每天收盘后按下面顺序执行：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\update_eod_data.py
D:\anaconda3\envs\qt\python.exe scripts\check_data_status.py
D:\anaconda3\envs\qt\python.exe main.py
```

## 常见问题

### 1. `ModuleNotFoundError: No module named 'torch'`

请确认使用的是：

```powershell
D:\anaconda3\envs\qt\python.exe
```

### 2. 更新很慢

当前已经支持：

- 外层总进度
- 多股票并发抓取
- 默认增量更新，只补最近已知交易日之后的缺口
- 空结果和异常分开统计

可在 `config.yaml` 里调整：

- `fetch.max_workers`
- `fetch.request_timeout`
- `fetch.show_progress`

如果你调大了历史窗口，需要回补更早数据，不要删文件，直接用：

```powershell
D:\anaconda3\envs\qt\python.exe scripts\update_eod_data.py --backfill-history
```

### 3. 训练到 `peer_map 构建完成` 后看起来很久没动

现在数据集阶段会继续输出：

- `peer 特征处理中`
- `训练集序列完成`
- `验证集序列完成`
- `测试集序列完成`

如果仍然很慢，优先看是不是股票池过大，或者 `peer` 窗口过长。
