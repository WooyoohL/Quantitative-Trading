# Codex A股模拟盘每日运行自动化

状态：当前正式可用基线。`docs/current_strategy_mainline.md` 记录的是一条未完整复刻旧流程的滚动重训实验线，不能替代本文档描述的模拟盘自动化。

本文档说明 Codex 桌面端中“ A股模拟盘每日运行”自动化的工程边界、执行流程、输出文件和人工判断要求。该自动化服务于日频模拟盘，不承担实盘委托职责。

## 运行环境

- 工作目录：`D:\PyCharm 2025.3.3\projects\Quantitative Trading`
- Python 解释器：`D:\anaconda3\envs\qt\python.exe`
- 主配置：`config.yaml`
- 模拟盘配置：`paper_trading.yaml`
- 事件筛选规则：`strategy/post_filter.md`
- 自动化记忆：`$CODEX_HOME/automations/a/memory.md`

依赖只在导入失败时根据 `requirements.txt` 处理。日常运行不应主动变更依赖版本，不应提交运行产物。

## 标准流程

自动化按交易日创建目录：

```powershell
outputs/paper_trading/<YYYYMMDD>/
```

随后执行完整数据库更新：

```powershell
$PY = "D:\anaconda3\envs\qt\python.exe"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$day = Get-Date -Format "yyyyMMdd"
$dir = Join-Path "outputs/paper_trading" $day
New-Item -ItemType Directory -Path $dir -Force | Out-Null
$updateLog = Join-Path $dir "update_data_$stamp.log"
& $PY scripts/run_logged_command.py --log $updateLog --interval 900 -- $PY main.py update-data --config config.yaml
```

数据库更新完成后执行数据检查：

```powershell
$checkLog = Join-Path $dir "check_data_$stamp.log"
& $PY scripts/run_logged_command.py --log $checkLog --interval 600 -- $PY main.py check-data --config config.yaml
```

数据可用后执行模拟盘 prepare，并跳过内部数据更新：

```powershell
$prepareLog = Join-Path $dir "prepare_$stamp.log"
& $PY scripts/run_logged_command.py --log $prepareLog --interval 600 -- $PY main.py paper-trade --config paper_trading.yaml --stage prepare --skip-update
```

prepare 完成后读取 `strategy/post_filter.md` 全文和 `outputs/paper_trading/latest_filter_input.csv`，人工或代理按事件层规则填写：

```text
outputs/paper_trading/latest_event_filter_decisions.csv
outputs/paper_trading/<YYYYMMDD>/event_filter_decisions.csv
```

事件筛选完成后执行 finalize：

```powershell
& $PY main.py paper-trade --config paper_trading.yaml --stage finalize --event-decisions outputs/paper_trading/latest_event_filter_decisions.csv
```

## 候选池与模型职责

数据库更新只负责本地行情、指数、行业和基础候选池更新。模拟盘 prepare 在数据库更新之后构建统一候选池：

1. 读取原始基础候选池。
2. 增加市场热度候选。
3. 执行统一候选池过滤。
4. 写入 `data/current_candidates.csv`、`outputs/paper_trading/latest_candidate_universe.csv` 和日期目录下的 `candidate_universe.csv`。

prepare 在统一候选池内执行模型训练和推理。计划重训日应使用当日新训练的 Attention 源运行；Ridge 使用配置中指定的源运行。Attention 和 Ridge 各自最多取 6 个过滤后的候选，事件审查集合最多 12 行，重复股票只保留一行，并保留组合来源标记。

## 事件筛选规则

事件层只判断最近 5 个交易日的公告、交易所披露、权威财经快讯和公司新闻。输入表已包含模型层与量价层结论，事件筛选不得重新计算这些结论。

事件决策文件必须保留下列字段：

```text
symbol,name,event_layer_conclusion,event_layer_risk,event_layer_reason,
recommended_action,risk_level,positive_catalyst_level,key_negative_events,
key_positive_events,summary,sources
```

字段取值约束：

- `event_layer_conclusion`：`正向`、`中性`、`负向`
- `event_layer_risk`：`高`、`中`、`低`
- `recommended_action`：`Keep`、`Watch buy`、`Exclude`、`Manual review`
- `risk_level` 与 `positive_catalyst_level`：`High`、`Medium`、`Low`

`Keep` 是强买入，只能在事件层没有硬负面、量价层没有明显不支持、且至少存在事件正向或量价支持时使用。`Watch buy` 是观察买入，只能用于事件中性、事件风险不高、量价中性且无明显走弱、模型层靠前且不是纯不建议的候选。

## 资金账本

finalize 生成 0 到 8 行最终买入计划：

- 最多 4 行 `Keep` 强买入。
- 最多 4 行 `Watch buy` 观察买入。
- 强买入目标仓位使用 `per_position_target_exposure=0.175`。
- 观察买入目标仓位使用 `watch_position_target_exposure=0.10`。
- 总暴露不超过 `target_gross_exposure=0.70`。

计划必须包含预期买入价、买入滑点率、滑点后预估成交价、手续费率和目标仓位。次一交易日 prepare 或账本更新阶段记录实际开盘价、开盘价偏差、实际滑点成交价、实际股数、手续费和现金支出。

持久账本文件：

- `outputs/paper_trading/account_ledger.csv`
- `outputs/paper_trading/trade_ledger.csv`
- `outputs/paper_trading/latest_buy_execution_tracking.csv`
- `outputs/paper_trading/paper_account.xlsx`
- `outputs/paper_trading/<YYYYMMDD>/paper_account.xlsx`

工作簿必须包含 `资金总账`、`交易流水`、`买入执行跟踪`。`账户概览` 只是当前状态摘要，不能替代资金总账。

## 版本控制边界

应提交源代码、配置、策略说明和本自动化文档。以下内容是运行产物，不进入版本库：

- `outputs/`
- `data/base_candidates.csv`
- 每日日志、推理结果、训练结果、模拟盘工作簿和事件筛选结果

已有跟踪的数据快照如需变更，应单独审查其业务含义，不应随一次代码整理自动纳入提交。
