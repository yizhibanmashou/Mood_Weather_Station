# 项目文档

## 方法论

- [脚本流水线](PIPELINE.md) — 00-08 脚本链路、输入输出、运行顺序
- [分析设计](ANALYSIS_DESIGN.md) — 情绪标注、验证、聚合、异常检测、聚类设计
- [NLP 词云模块](NLP_WORDCLOUD_MODULE.md) — jieba + TF-IDF 关键词提取与前端交互
- [实时数据采集](REALTIME_FETCH.md) — UAPIS 热搜 + Patchright s.weibo.com 搜索（11 + 12）
- [本地情绪模型](REALTIME_FETCH.md) — 13_* 模型训练 + 14 本地推理引擎
- [QA 报告](QA_REPORT.md) — 开发过程中遇到的问题与修复记录

## 关键指标

| 指标 | 值 |
|---|---|
| 原始样本量 | 76,441 条微博 |
| 标注量 | 39,973 条 (Cap60 扩展后) |
| 验证 Accuracy (DeepSeek) | 73.3% (SMP2020-EWECT) |
| 验证 Macro F1 (DeepSeek) | 0.662 |
| 本地模型 Val Acc (V1) | **73.81%** (chinese-roberta-wwm-ext, 标准 KL 蒸馏) |
| 本地模型 Val MAE (V1) | **0.1633** |
| 省份覆盖 | 34 个标准省级行政区 |
| 时间跨度 | 58 周 (2019-W48 ~ 2020-W53) |
| 异常事件 | 19 个 |
| 省份聚类 | 6 类 (34 省) |

### 模型蒸馏版本对比

| 版本 | 损失函数 | Val Acc | Val MAE | 结论 |
|------|---------|---------|---------|------|
| **V1** | 标准 KL 散度 | **73.81%** | **0.1633** | **最优，投入生产** |
| V2 | Focal KL + T=0.5 | 74.73% | 0.1931 | Acc↑但 MAE 崩，标签过锐 |
| V2.3 | Focal KL + T=1.0 | 71.95% | 0.1642 | MAE 恢复但 Acc 退化 |
| V3 | Weighted KL | 70.05% | 0.1661 | 维度加权未达预期 |

> 训练记录和模型权重完整归档于 `models/archive/`。

## 数据管线概览

```
原始数据 (76,441)
  → 01 mini_dataset.csv
    → 02 标注 (DeepSeek API, 分批执行)
      → 02d 生成扩展计划 (cap=60/周-省)
      → 02e 合并 + 质量检查 + 下游流水线
        → 04 聚合 (周/月/省面板)
        → 04b NLP 关键词 (jieba + TF-IDF)
        → 05 异常检测 (rolling z-score)
        → 06 省份聚类 (层次聚类)
        → 07 聚类演化 (月度独立聚类)
        → 08 前端资产导出 → app/public/data/

实时管线（实验性）
  UAPIS 热搜 API
    → 11 Patchright s.weibo.com 搜索
      → 12 预处理 (清洗 + 省份映射)
        → 14 本地模型推理 (chinese-roberta-wwm-ext)
          → 04 增量聚合 → ... → 08 导出

本地模型训练
  labeled_dataset_merged_week_cap60.csv
    → 13a 数据切分 (80/10/10)
      → 13d XPU 训练 (Focal KL 蒸馏)
        → models/emotion_model/
          → 14 本地推理引擎
```
