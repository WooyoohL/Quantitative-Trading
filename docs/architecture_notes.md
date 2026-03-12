# 架构说明与下一阶段拆分建议

## 当前分层

第一阶段和第二阶段前半段重构后，项目分为三层：

### 1. 入口层

- `main.py`
  - 统一 CLI 入口
  - `train / infer / update-data / check-data / batch / analyze / sweep`
- `scripts/*.py`
  - 兼容旧命令
  - 建议逐步降级为 wrapper

### 2. 流程层

- `pipelines/context.py`
  - 训练股票池、上下文数据、dataset bundle
- `pipelines/evaluation.py`
  - valid/test 预测、回测、评估文件输出
- `pipelines/recommendation.py`
  - inference、候选过滤、Top-K、orders

### 3. 共享能力层

- `app/runtime.py`
  - 配置、运行目录、日志 tee、基础 I/O
- `app/factories.py`
  - DatasetBuilder / TrainerConfig 工厂
- `app/reporting.py`
  - 训练 / 推理摘要打印
- `app/metric_specs.py`
  - 核心指标展示规格

---

## dataset.py 建议拆分逻辑

`data/dataset.py` 目前仍然过重。建议按“特征 -> 切分 -> 序列 -> bundle”这条链拆，而不是按函数长度拆。

### 推荐拆法

#### 1. `data/features/base.py`

负责：

- 个股基础特征
- 市场特征
- 缺失值收口前的基础表

对应现有逻辑：

- `_build_base_feature_frame`
- `_attach_market_features`
- `_finalize_feature_frame` 的通用部分

#### 2. `data/features/index_features.py`

负责：

- 指数特征拼接

对应现有逻辑：

- `_attach_index_features`

#### 3. `data/features/industry_features.py`

负责：

- 行业映射
- 行业日线特征拼接

对应现有逻辑：

- `_attach_industry_features`

#### 4. `data/features/peer_features.py`

负责：

- peer_map 构建
- peer 特征拼接

对应现有逻辑：

- `_build_peer_map`
- `_attach_peer_features`

#### 5. `data/splits.py`

负责：

- rolling anchor date 切分
- train/valid/test 的日期规则

对应现有逻辑：

- `_split_anchor_dates`

#### 6. `data/sequences.py`

负责：

- `SequenceDataset`
- `_build_sequence_dataset`
- `build_inference_dataset`
- 时间块打乱

对应现有逻辑：

- `_build_sequence_dataset`
- `build_inference_dataset`
- `_maybe_shuffle_time_blocks`

#### 7. `data/scaling.py`

负责：

- `FeatureScaler`

#### 8. `data/builder.py`

保留：

- `DatasetBundle`
- `AlphaDatasetBuilder`

但这里只做 orchestration，不再直接堆所有特征函数。

### 拆分顺序建议

不要一次性全拆。建议顺序：

1. 先抽 `FeatureScaler` 和 `SequenceDataset`
2. 再抽 `splits.py`
3. 再抽 `peer_features.py`
4. 最后再拆 `base/index/industry`

原因：

- `peer` 和 `split` 逻辑边界最清楚
- 风险小
- 回归验证容易

---

## 指标体系模块化建议

当前指标已经分成：

- 训练 loss
- 排序指标
- 交易回测指标

但计算和展示仍然散在 `trainer.py`、`analysis.py`、`reporting.py`。

### 推荐拆法

#### 1. `metrics/ranking.py`

负责：

- `rank_ic`
- `daily_rank_ic_mean`
- `head_daily_rank_ic_mean`

目前这些大多在 `models/loss_functions.py` 和 `scripts/analysis.py` 有重复。

#### 2. `metrics/losses.py`

负责：

- `PearsonLoss`
- 组合 loss 计算工具

#### 3. `metrics/backtest_metrics.py`

负责：

- `backtest_top_k`
- `summarize_backtest`
- 指标 guidance

#### 4. `metrics/selection.py`

负责：

- `valid_ic`
- `ic_gate_composite`
- `topk_valid`

把 checkpoint 选择逻辑从 `trainer.py` 里抽出来。

#### 5. `metrics/display_specs.py`

负责：

- 指标标签
- 推荐范围
- 格式化

目前已经先落了一版在 `app/metric_specs.py`，后续可以独立成真正的 metrics 子模块。

### 为什么值得拆

这样能解决三个问题：

1. 训练阶段、分析阶段不再重复实现同一指标
2. checkpoint 选择规则更容易审计
3. 展示口径和计算口径能保持一致

---

## 目前还没动但应注意的风险

### 1. `scripts/` 仍然比较多

虽然已经有统一入口，但旧脚本还在。短期保留是合理的，后续建议：

- 保留常用脚本
- 非核心脚本逐步改成 wrapper

### 2. 数据更新与训练使用同一批 CSV

这是项目设计本意，不建议为了“结构更优雅”而过度抽象。

### 3. summary 字段已经很多

当前策略是：

- `summary.json` 保留完整
- 控制台只看核心

这比强行减少 summary 字段更合理。

---

## 建议的下一步

如果继续第三阶段，建议顺序：

1. 抽 `metrics/selection.py`
2. 抽 `metrics/ranking.py`
3. 拆 `data/splits.py`
4. 拆 `data/sequences.py`
5. 再处理 `data/features/*`
