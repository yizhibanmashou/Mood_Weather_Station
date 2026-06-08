# 情绪气象站 · 项目交接文档

> 最后更新：2026-06-04

---

## 一、项目概述

基于 COVID-19 期间（2019-W48 至 2020-W53，58 周）的 76,441 条微博数据，分析中国 34 个省份的公众情绪时空演变。

**在线演示**：https://mood-weather-station.pages.dev/

### 核心功能

| 模块 | 说明 |
|------|------|
| 全国情绪总览 | 58 周时序、6 维情绪结构、异常事件 |
| 中国情绪地图 | 34 省气泡图，支持情绪温度 / 6 维情绪 / 主导情绪切换 |
| 省份详情 | 单省情绪曲线、月度趋势 |
| 聚类分析 | 层次聚类 + KMeans，6 簇，月度演化热力图 |
| 事件时间线 | Rolling z-score 异常检测 (22 事件)，NLP 词云解释 |
| 实时热搜看板 | UAPIS 热搜标题 → 本地模型实时情绪推理 |
| **数据时间轴** (v2) | 全局时间轴组件展示数据时间跨度，标记关键时期 |

### 数据规模

- 原始微博：76,441 条（COV-Weibo2.0 数据集）
- 已标注：39,973 条（DeepSeek API，6 维情绪分数）
- 合成数据：500 条（Script 18，UAPIS 热搜 → DeepSeek 生成）
- 合并总计：40,473 条（39,973 + 500）
- 覆盖：34 省 × 61 周
- DeepSeek 验证：SMP2020-EWECT Accuracy 73.3%, Macro F1 0.662

---

## 二、技术栈

### 后端 / 数据处理 (Python 3.11, emotion_xpu 环境)

| 库 | 用途 |
|---|---|
| pandas / numpy | 数据处理 |
| scikit-learn | TF-IDF、StandardScaler、聚类 |
| scipy | 层次聚类 linkage |
| jieba | 中文分词 |
| snownlp | SMP2020 基线对比 |
| openai | DeepSeek API 客户端 |
| python-dotenv | 环境变量 |
| **torch** | 本地模型训练/推理（XPU 加速） |
| **transformers** | HuggingFace RoBERTa |
| **requests** | UAPIS API 调用 |

### 前端 (React + TypeScript)

| 库 | 用途 |
|---|---|
| React 18 + TypeScript 5 | UI |
| Vite 7 | 构建 |
| ECharts 5 | 图表 |
| echarts-wordcloud | 交互式词云 |
| Framer Motion | 微交互动画 |
| PapaParse | CSV 解析 |
| CSS Modules | 样式隔离 |

### 部署

Cloudflare Pages，SPA 路由由 `_redirects` 文件处理。

---

## 三、目录结构

```
Mood_Weather_Station/
├── app/                          # React 前端
│   ├── src/components/           # 可复用组件（含 TimeAxis 时间轴）
│   ├── src/pages/                # 4 页面（总览/详情/聚类/事件）
│   ├── src/data/loadData.ts      # 数据加载
│   ├── src/hooks/useMoodData.ts  # 数据 Hook
│   └── public/data/
│       ├── processed/            # 前端静态数据 (CSV/JSON)
│       └── realtime/             # 实时热搜快照 (hotsearch_latest.json)
│
├── scripts/                      # Python 数据管线
│   ├── 00-08                     # 主管线：构建→标注→聚合→异常→聚类→导出
│   ├── 11-16                     # 模型训练管线
│   ├── 17_hotsearch_live.py      # 实时热搜情绪抓取（合规版）
│   ├── 18_generate_synthetic_data.py  # 合成数据生成
│   └── run_week_cap60_expansion_pipeline.py  # 一键编排器
│
├── models/
│   ├── emotion_model/            # 生产模型 (V1-opt)
│   └── archive/                  # 历史训练记录
│
├── data/
│   ├── raw/                      # 原始数据 (.gitignore)
│   ├── processed/                # 处理后数据 (.gitignore)
│   ├── synthetic/                # 合成数据 (.gitignore)
│   └── indexes/                  # SQLite 用户索引 (.gitignore)
│
├── paper/                        # 论文 (paper.md, 已更新)
├── figures/                      # 训练曲线图 (PNG + PDF)
├── docs/                         # 方法论文档
│   └── DATA_ACQUISITION_REPORT.md  # 数据获取探索报告
├── analysis/                     # 词云 PNG + 聚类图
├── tmp/                          # 临时文件 (.gitignore)
│   └── training_logs/            # 训练日志
│
├── HANDOVER.md                   # ← 本文档
├── README.md                     # 项目 README（已更新）
├── .gitignore                    # 已更新（含 data/synthetic/）
└── .env.example
```

---

## 四、数据管线

### 主管线 (00-08)

```
原始数据 (76,441)
  → 00 探测 → 01 构建 mini_dataset
    → 02 DeepSeek 标注 (断点续传, 39,973 条)
      → 02d 扩展计划 (cap=60/周-省)
      → 02e 合并 + 质量检查
        → 04 聚合 (周/月/省面板)
        → 04b NLP 关键词 (jieba + TF-IDF)
        → 05 异常检测 (rolling z-score, 22 事件)
        → 06 省份聚类 (层次 + KMeans, 6 簇)
        → 07 聚类演化 (月度)
        → 08 前端资源导出 → app/public/data/
```

### 实时管线 (合规版, Script 17)

```
UAPIS 热搜 API (公开聚合榜单)
  → 17_hotsearch_live.py (取标题 + 本地模型推理 6 维情绪)
    → app/public/data/realtime/hotsearch_latest.json  (前端快照)
    → app/public/data/realtime/hotsearch_history.jsonl (趋势追加)
```

### 合成数据管线 (Script 18)

```
UAPIS 热搜 API (实时热点标题)
  → DeepSeek API (生成 Weibo 风格正文 + 省份/时间)
    → 本地蒸馏模型 (6 维情绪打分)
      → CSV/JSON 输出 (与现有数据集同 Schema)
```

> 详见 `docs/DATA_ACQUISITION_REPORT.md`

### 模型训练 (11-16)

```
labeled_dataset_merged_week_cap60.csv (39,973)
  → 11 数据切分 (80/10/10, 按省×周分层)
    → 14 训练 (emotion_xpu 环境, XPU + BF16 AMP)
      → models/emotion_model/
        → 16 本地推理 (单条 / 批量 CSV)
```

---

## 五、模型训练结果

| 版本 | 损失函数 | 超参数 | Val Acc | Val MAE | 状态 |
|------|---------|--------|---------|---------|------|
| **V1-opt** (final) | 标准 KL | MAX_LENGTH=256, DROPOUT=0.2, BATCH=16, epoch 9 | **77.03%** | **0.1613** | **生产模型** ✅ |
| V1-opt (基线) | 标准 KL | MAX_LENGTH=256, DROPOUT=0.2, epoch 10 | 77.36% | 0.1609 | 已归档 |
| V1 (基线) | 标准 KL | MAX_LENGTH=128, DROPOUT=0.1 | 72.28% | 0.1640 | 已归档 |
| V2 | Focal KL + T=0.5 | — | 74.73% | 0.1931 | 已归档 |
| V2.3 | Focal KL + T=1.0 | — | 71.95% | 0.1642 | 已归档 |
| V3 | Weighted KL | — | 70.05% | 0.1661 | 已归档 |

> **V1-opt final (2026-06-04)**：dropout=0.2, epochs=15（实际在 epoch 9 达到最佳后停止）。比原始 V1-opt 略低 0.3% 但未出现过拟合现象，泛化性更好。
>
> **结论**：标准 KL 散度蒸馏在本数据集上综合最优。

---

## 六、环境搭建与运行

### 环境变量 (`.env`)

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `DEEPSEEK_MODEL` | 模型名称 (deepseek-v4-flash) |
| `DEEPSEEK_BASE_URL` | API 地址 |
| `UAPIS_API_KEY` | UAPIS 热搜 API (uapis.cn) |

### 前端

```bash
cd app && npm install
npm run dev          # http://localhost:5173
npm run build        # → app/dist/
```

### 数据管线

```bash
# emotion_xpu 环境 (Python 3.11, Intel XPU)
# 注意：Windows 中文用户名下 conda activate 可能失败
# 直接用 Python 绝对路径：D:/anaconda/envs/emotion_xpu/python.exe

# 运行完整管线
python scripts/run_week_cap60_expansion_pipeline.py

# 训练模型
python scripts/14_train_emotion_model.py

# 本地推理
python scripts/16_label_local.py --text "今天好开心"
python scripts/16_label_local.py --input unlabeled.csv --output labeled.csv

# 生成合成数据
python scripts/18_generate_synthetic_data.py --topics 10 --posts-per-topic 50

# 实时热搜
python scripts/17_hotsearch_live.py
```

### 部署

1. `cd app && npm run build` → `app/dist/`
2. Cloudflare Pages 关联 Git 仓库自动部署
3. `app/public/_redirects` 处理 SPA 路由

---

## 七、已知问题

1. **数据不入 Git**：`data/raw/`、`data/processed/`、`data/synthetic/`、`tmp/`、`analysis/`、`models/` 在 `.gitignore`
2. **DeepSeek API 费用**：全量标注需 API 费用，支持断点续传
3. **中文用户名**：conda activate 可能失败，用 Python 绝对路径
4. **XPU 训练**：脚本顶部自动添加 XPU DLL 路径到 PATH；`HF_HUB_OFFLINE=1` 可避免 403 错误
5. **合成数据 Domain Shift**：DeepSeek 生成的微博内容风格与真实微博存在差异，情绪分布偏中性 (~53%)
6. **热搜 Domain Shift**：模型训练自个人微博，对热搜标题输出偏中性 (~90%)，但情绪倾向标题可正确识别
7. **部分省份样本少**：台湾 (334)、青海 (138)、甘肃 (241)
8. **合成数据感知**：500 条合成数据已合并到主管线，但聚合后可可靠性较低（周级仅 39%），因每个省-周格子的合成样本不足 30 条
9. **XGBoost 相关**：`06_cluster_provinces.py` 中引用 XGBoostClassifier 但未安装，回退到 RandomForest

---

## 八、许可证

MIT License
