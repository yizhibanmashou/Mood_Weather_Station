# NLP 词云分析模块

## 1. 模块动机

事件时间线能够展示情绪异常的时间点和强度，但无法解释**为什么**会出现异常。NLP 词云分析模块通过提取异常周的微博文本关键词，回答以下问题：

- 这一周公众在讨论什么？
- 哪些词推动了情绪异常？
- 本周有哪些词比平时更突出？

这为情绪异常提供了语义层面的解释，帮助用户理解数据背后的社会事件和公众关注点。

## 2. 数据输入

| 文件 | 说明 |
|------|------|
| `data/processed/labeled_dataset_merged_week_cap60.csv` | 标注后的微博数据（39,973 条） |
| `data/processed/labeled_dataset_merged_cap30.csv` | 标注后的微博数据（旧版备选） |
| `data/processed/emotion_national_timeline.csv` | 全国情绪时序 |
| `data/processed/anomaly_detection.json` | 异常检测结果 |

## 3. 处理流程

### 3.1 分词与词性标注

使用 `jieba.posseg` 进行中文分词和词性标注。

**保留词性：**
- 名词：n, nr, ns, nt, nz
- 动词：v, vn
- 形容词：a, an
- 成语：i
- 习惯用语：l

### 3.2 过滤规则

1. **停用词过滤**：内置基础停用词表 + 可选扩展文件 `data/stopwords/chinese_stopwords.txt`
2. **长度过滤**：过滤长度 < 2 的词
3. **纯数字过滤**：过滤只包含数字的 token
4. **英文短 token**：过滤 1-4 位纯英文
5. **符号过滤**：过滤只包含标点或符号的词
6. **微博噪声词**：过滤平台相关词汇（转发、微博、视频、链接等）

### 3.3 TF-IDF 计算

- 以 `date_week` 为文档单位
- 每周文档由该周所有分词结果拼接
- 使用 `sklearn.feature_extraction.text.TfidfVectorizer`
- `max_features`: 8000
- `min_df`: 1（小数据集）或 2（大数据集）
- `max_df`: 0.85

### 3.4 飙升词检测

**全局均值计算：**
```
global_tfidf = 所有周中该词 TF-IDF 的均值
```

**飙升倍数：**
```
surge_ratio = current_week_tfidf / max(global_tfidf, 1e-6)
```

**飙升判断：**
- TF-IDF 排名前 30
- surge_ratio >= 2.0

### 3.5 低样本周处理

- 最低要求：50 条微博/周
- 不足时输出 `status: "insufficient_data"`
- 前端显示"数据不足，暂不生成词云"

## 4. 输出文件

### 4.1 nlp_keywords_by_week.json

按周组织的关键词数据，包含：
- 每周状态（ok / insufficient_data / no_data）
- 每周 Top 30 关键词
- 每个词的词频、TF-IDF、全局均值、飙升倍数

### 4.2 nlp_emotion_keywords.json

按情绪维度聚合的关键词：
- 六种情绪各 Top 30 关联关键词
- 关键词归属：情绪均值 >= 0.05 的所有情绪都会获得该周的关键词
- 包含峰值周和峰值 TF-IDF

### 4.3 nlp_global_vocabulary.json

全局词表（Top 5000）：
- 总词频
- 平均 TF-IDF
- 峰值周

### 4.4 辅助文件

- `tmp/04b_nlp_keyword_review.csv`：人工检查表，前 50 词
- `tmp/04b_nlp_summary.md`：处理摘要报告

## 5. 前端交互

### 5.1 NlpPanel 组件

在事件时间线页面中，点击异常事件卡片后展开 NLP 分析面板。

**面板内容：**
1. **交互式词云**：使用 echarts-wordcloud 渲染
2. **Top 20 关键词排行**：柱状图展示
3. **飙升词列表**：橙红高亮
4. **数据说明**：面向普通观众的文案

**交互功能：**
- hover 词云词汇显示详情（词频、TF-IDF、飙升倍数）
- 点击词云词汇高亮该词
- 切换"高频词"和"飙升词"视图
- 切换"当前情绪相关词"和"全部词"

### 5.2 视觉设计

- 继承淡奶白轻产品主题
- 词云词汇使用情绪色板和低饱和暖色
- 飙升词用橙红强调
- 普通高频词用柔和蓝灰
- 桌面端词云和排行并排，窄屏上下堆叠

## 6. 与课程知识点对应

| 知识点 | 模块应用 |
|--------|----------|
| **数据预处理** | 分词、停用词过滤、词性过滤、噪声清洗 |
| **特征工程** | TF-IDF 向量化、特征提取 |
| **非结构化数据处理** | 中文文本处理、微博噪声过滤 |
| **可视化** | 词云、柱状图、交互式面板 |
| **异常检测关联** | 将 NLP 结果与异常周关联 |
| **分类结果验证** | 通过关键词验证情绪标注的合理性 |

## 7. 局限性

1. **分词误差**：jieba 对新词、网络用语识别有限
2. **停用词不完善**：内置停用词表可能遗漏领域特定词
3. **小样本周不稳定**：微博数量少的周，关键词可能不具代表性
4. **TF-IDF 局限**：只能反映词频代表性，不能直接等价于因果解释
5. **情绪-词汇映射**：基于词性的映射是粗略的，不能替代情感词典

## 8. 后续扩展

1. **命名实体识别（NER）**：提取人名、地名、机构名
2. **主题模型（LDA）**：发现潜在话题
3. **情感词典**：使用知网 Hownet、大连理工情感词典
4. **时序语义演变**：追踪关键词随时间的变化趋势
5. **词向量聚类**：使用 Word2Vec 发现语义相近的词群

## 9. 运行方式

```bash
# 在 04 聚合脚本之后运行，传入合并后的数据集
python scripts/04b_nlp_keywords.py --input data/processed/labeled_dataset_merged_week_cap60.csv

# 之后运行 08 导出前端资产
python scripts/08_prepare_frontend_assets.py
```

## 10. 依赖

- jieba
- scikit-learn
- pandas
- numpy
