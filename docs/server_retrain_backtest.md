# 服务器滚动重训回测

安装依赖：

```bash
pip install -r requirements.txt
```

数据文件：

- `data/eod_daily.csv` 超过 GitHub 单文件限制，不纳入 Git；服务器运行前请手工放到该路径。
- 其他小型数据缓存、配置、脚本、训练记录随仓库提交。

一行运行当前服务器实验预设：

```bash
python scripts/run_server_retrain_backtest.py
```

默认预设为：建议仓位、不买弱 Top1。

该预设等价于：

```bash
python scripts/rolling_retrain_backtest.py --config config.yaml --strategies mon_wed --hold-periods 2 --label-mode market_excess --cash-filter enabled --cash-filter-policy quality --disable-price-cap --score-quantile 0.50 --topk-mean-quantile 0.45 --score-gap-quantile 0.20 --min-position-exposure 0.05 --mid-position-exposure 0.10 --max-position-exposure 0.175 --max-gross-exposure 0.60 --position-policy recommended --weak-top1 disabled --auto-start-after-warmup
```

可选组合：

```bash
python scripts/run_server_retrain_backtest.py --position-policy current --weak-top1 enabled
python scripts/run_server_retrain_backtest.py --position-policy current --weak-top1 disabled
python scripts/run_server_retrain_backtest.py --position-policy recommended --weak-top1 enabled
python scripts/run_server_retrain_backtest.py --position-policy recommended --weak-top1 disabled
```

实验边界：

- 不调用事件面或大模型。
- 仅评估周一、周三滚动重训策略。
- 固定使用 `hold_period=2`：按信号日计为第 2 个交易日开盘卖出，实际交易含义为次日开盘买入、再下一个交易日开盘卖出。
- 标签使用市场超额收益。
- 使用质量分层后处理：强信号买 Top3，中等信号买 Top2，弱信号要求 Top1 达到放松分数阈值且分数差拉开后按 `--weak-top1` 决定是否小仓买入。
- `--position-policy current` 使用历史仓位：中等信号每只 8%，强信号按模型信心在 10% 到 17.5% 之间分配。
- `--position-policy recommended` 使用建议仓位：中等信号每只 10%，强信号按模型信心相对比例缩放到总仓位 50%。
- 去掉 40 元价格上限，其他右侧过滤和原有数据处理逻辑保留。
- 权重文件不纳入 Git，服务器运行时重新生成。

时间边界：

- 信号日 `T` 使用 `T` 日及以前的行情、指数、行业和截面特征。
- `hold_period=2` 的标签和交易评估均定义为：`T+1` 开盘买入，`T+2` 开盘卖出。
- 独立管线会检查训练切片不超过训练日、推理特征块不超过当前批次最大信号日、推理输出日期不超出计划信号日集合。
