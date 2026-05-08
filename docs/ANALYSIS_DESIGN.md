# 分析设计

## 情绪标签体系

采用 6 维情绪分数：

| 字段 | 中文 | Cap60 均值 (39,973 条) |
|---|---|---:|
| `joy` | 喜悦 | 0.2683 |
| `sadness` | 悲伤 | 0.0902 |
| `anger` | 愤怒 | 0.0948 |
| `fear` | 恐惧 | 0.0344 |
| `surprise` | 惊讶 | 0.0560 |
| `neutral` | 中性 | 0.4564 |

每条微博的 6 维分数应在 `[0, 1]`，总和为 1。主导情绪为最高分维度。

### Cap60 主导情绪分布 (39,973 条)

| 情绪 | 数量 | 占比 |
|---|---:|---:|
| 中性 | 16,360 | 40.9% |
| 喜悦 | 13,976 | 35.0% |
| 愤怒 | 5,114 | 12.8% |
| 悲伤 | 2,999 | 7.5% |
| 恐惧 | 918 | 2.3% |
| 惊讶 | 606 | 1.5% |

## 标注

`scripts/02_label_emotions.py` 使用 DeepSeek API 批量标注微博文本。

关键设计：

- `smoke`：300 条，用于检查 prompt、解析和续跑。
- `pilot`：3,000 条，用于验证和前端联调。
- `full`：完整 mini dataset。
- 支持断点续跑和临时 checkpoint。
- 输出完整数据行 + 情绪分数，而不是只输出分数。

## 验证

`scripts/03_validate_emotions.py` 使用 SMP2020-EWECT 做外部验证。

输出：

- `tmp/03_accuracy_report.json`
- `analysis/emotion_validation/confusion_matrix.csv`
- `analysis/emotion_validation/deepseek_vs_snownlp_sample.csv`
- `analysis/emotion_validation/emotion_distribution_summary.csv`
- `analysis/emotion_validation/dominant_emotion_distribution.csv`

### Pilot 3000 验证结果（2000 SMP2020 样本）

| 情绪 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 喜悦 | 0.940 | 0.701 | 0.803 | 923 |
| 愤怒 | 0.779 | 0.863 | 0.819 | 314 |
| 中性 | 0.568 | 0.788 | 0.660 | 476 |
| 悲伤 | 0.570 | 0.667 | 0.615 | 165 |
| 恐惧 | 0.591 | 0.520 | 0.553 | 75 |
| 惊讶 | 0.533 | 0.511 | 0.522 | 47 |
| **总体** | | | **Accuracy** | **0.733** |
| **Macro avg** | 0.664 | 0.675 | **0.662** | 2000 |

验收重点：

- accuracy: **0.733** ✅
- macro F1: **0.662** ✅
- 每类 F1：joy/anger > 0.80，fear/surprise > 0.52

## 聚合

`scripts/04_aggregate_emotions.py` 生成：

- `emotion_panel_weekly.csv`：周×省 (1,888 行)
- `emotion_panel_monthly.csv`：月×省 (442 行)
- `emotion_national_timeline.csv`：全国周时序 (58 周)
- `province_emotion_vectors.csv`：省份全年特征向量 (34 省)

省级面板使用 34 省白名单过滤噪声值，全国时序使用全部标注样本。

## NLP 关键词提取

`scripts/04b_nlp_keywords.py` 使用 jieba + TF-IDF 提取微博关键词，解释情绪异常背后的语义驱动力。

- 分词：jieba.posseg，保留名词/动词/形容词/成语
- TF-IDF：以周为文档单位，max_features=8000
- 情绪归属：所有 6 维情绪均值 >= 0.05 的情绪都会获得关键词
- 输出：55 周关键词 + 6 维度 × 30 情绪关键词 + 5000 全局词表

详见 [NLP_WORDCLOUD_MODULE.md](NLP_WORDCLOUD_MODULE.md)。

## 异常检测

`scripts/05_detect_anomalies.py` 对全国情绪时序做 rolling z-score：

- 检测维度：fear, anger, joy
- 基线窗口：当前周之前 4 周
- 阈值：`|z| > 2.5`
- 输出每个异常周的贡献省份 Top 5

异常检测用于发现疫情事件或舆论事件对应的情绪波动。

## 省份聚类

`scripts/06_cluster_provinces.py` 使用省份全年情绪特征做聚类。

特征包括：

- 6 维情绪全年均值
- 情绪强度
- 恐惧方差
- 喜悦方差

输出层次聚类树、KMeans 轮廓系数和省份标签。

## 聚类演化

`scripts/07_cluster_evolution.py` 按月独立聚类，并按“负向风险得分”重排 cluster 编号，避免不同月份标签语义错位。

风险得分：

```text
sadness_mean + anger_mean + fear_mean - joy_mean
```

编号越高，代表该月该类越偏负向/高风险。
