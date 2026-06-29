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

该预设等价于：

```bash
python scripts/rolling_retrain_backtest.py --config config.yaml --strategies mon_wed --hold-periods 1,2 --label-mode market_excess --cash-filter enabled --cash-filter-policy quality --disable-price-cap --score-quantile 0.50 --topk-mean-quantile 0.45 --score-gap-quantile 0.20 --min-position-exposure 0.05 --mid-position-exposure 0.10 --max-position-exposure 0.175 --max-gross-exposure 0.60 --auto-start-after-warmup
```

实验边界：

- 不调用事件面或大模型。
- 仅评估周一、周三滚动重训策略。
- 固定持有期为 1 和 2 个交易日。
- 标签使用市场超额收益。
- 使用质量分层后处理：强信号买 Top3，中等信号买 Top2，弱信号要求 Top1 达到近似 q40 且分数差拉开后只买 Top1 小仓。
- 去掉 40 元价格上限，其他右侧过滤和原有数据处理逻辑保留。
- 权重文件不纳入 Git，服务器运行时重新生成。
