# 情绪气象站 · Mood Weather Station

> 数据挖掘课程项目：基于 COVID-19 期间微博文本的省级公众情绪演化分析与交互式可视化系统。

在线演示：https://mood-weather-station.pages.dev/

## 项目简介

情绪气象站将微博文本、时间窗口和省级空间信息结合起来，构建一个面向“公众情绪如何随疫情阶段变化”的数据挖掘与可视化系统。项目覆盖数据清洗、情绪标注、模型蒸馏、统计聚合、异常检测、聚类分析、NLP 关键词解释和前端交互展示。

系统同时提供历史数据分析和实时热搜验证两条路径：历史模块用于观察 2019-W48 至 2020-W53 的省级情绪趋势；实时模块接入 UAPIS 微博热搜标题，并使用本地 RoBERTa 情绪模型生成当天情绪快照，用于展示模型的可扩展性和现实验证能力。

## 核心功能

- 全国情绪总览：展示全国周级情绪走势、主导情绪、情绪强度、正向指数和异常事件。
- 中国情绪地图：按省份展示情绪温度、六维情绪占比和主导情绪。
- 省份详情页：查看单省周/月趋势、样本量、情绪结构和典型文本样例。
- 聚类分析：使用层次聚类和 KMeans 描述省份情绪模式，并展示月度聚类演化。
- 事件时间线：基于 Rolling Z-Score 检测情绪突变周，结合关键词解释异常来源。
- NLP 词云分析：使用中文分词、词性过滤和 TF-IDF 提取异常周关键词。
- 实时热搜看板：抓取微博热搜聚合标题，进行本地情绪推理并生成 JSON 快照。
- 前端交互体验：React 单页应用，支持主题切换、页面懒加载和响应式图表展示。

## 系统架构

项目采用“数据管线 + 模型推理 + 分析资产 + 前端可视化”的分层设计。

```text
原始/扩展数据
  -> 文本清洗与样本控制
  -> DeepSeek 教师标注 / 本地模型推理
  -> 周级、月级、省级聚合
  -> 异常检测、聚类分析、关键词提取
  -> CSV / JSON 前端数据资产
  -> React + TypeScript + ECharts 可视化系统
```

历史数据与实时数据保持解耦：历史分析数据写入 `app/public/data/processed/`，实时热搜快照写入 `app/public/data/realtime/`。前端通过统一的数据加载层读取静态资产，因此可以直接部署到 Cloudflare Pages、GitHub Pages 等静态托管平台。

## 关键技术与方法

| 模块 | 方法 | 说明 |
|---|---|---|
| 情绪标注 | DeepSeek API | 对微博文本进行六维情绪软标签标注 |
| 模型蒸馏 | RoBERTa student model | 使用教师模型结果训练本地推理模型 |
| 情绪维度 | joy / sadness / anger / fear / surprise / neutral | 每条文本输出六维概率分布 |
| NLP 解释 | jieba + TF-IDF | 提取异常周关键词和情绪相关词 |
| 异常检测 | Rolling Z-Score | 使用滑动窗口识别情绪突变周 |
| 省份聚类 | HAC + KMeans | 基于情绪均值、强度和方差构建省份画像 |
| 实时验证 | UAPIS + 本地模型 | 抓取热搜标题并生成实时情绪快照 |
| 前端展示 | React + Vite + ECharts | 构建可部署的交互式数据看板 |

## 模型结果

当前生产模型为 V1-opt，在验证集上相比基线有更好的准确率表现。

| 版本 | 损失函数 | Val Acc | Val MAE | 说明 |
|---|---|---:|---:|---|
| V1 | 标准 KL | 72.28% | 0.1640 | 基线配置 |
| V2 | Focal KL + T=0.5 | 74.73% | 0.1931 | 标签锐化较强 |
| V2.3 | Focal KL + T=1.0 | 71.95% | 0.1642 | 保留软分布 |
| V3 | Weighted KL | 70.05% | 0.1661 | 维度加权实验 |
| V1-opt | 标准 KL | 77.36% | 0.1611 | 当前生产模型 |

外部验证使用 SMP2020-EWECT：Accuracy 73.3%，Macro F1 0.662。

## 快速开始

### 前端运行

```bash
cd app
npm install
npm run dev
```

本地开发地址默认为 `http://localhost:5173`。

### 前端构建

```bash
cd app
npm run build
```

构建产物输出到 `app/dist/`。

### 重新生成前端数据

```bash
cd app
npm run prepare:data
```

该命令会根据 `data/processed/labeled_dataset_merged_week_cap60.csv` 重新生成前端样例数据。

### Python 环境

```bash
pip install -r requirements.txt
```

如果使用本地 Intel XPU / Conda 环境，可以参考：

```bash
conda run -n emotion_xpu python scripts/17_hotsearch_live.py
```

### 实时热搜管线

```bash
python scripts/17_hotsearch_live.py
```

输出文件：

- `app/public/data/realtime/hotsearch_latest.json`
- `app/public/data/realtime/hotsearch_history.jsonl`

如需访问 DeepSeek 或 UAPIS，请复制 `.env.example` 为 `.env` 并填写自己的 API Key。`.env` 已加入 `.gitignore`，不要提交到公开仓库。

## 项目结构

```text
Mood_Weather_Station/
- app/
  - src/
    - components/      # 图表、地图、状态视图、时间轴等组件
    - pages/           # 总览、省份详情、聚类、事件、实时热搜页面
    - data/            # 前端数据加载与标准化
    - hooks/           # 数据 Hook
    - utils/           # 日期、指标、动画、分析工具
  - public/
    - data/processed/  # 前端历史分析资产
    - data/realtime/   # 实时热搜快照
    - data/geo/        # 地理边界数据
  - scripts/           # 前端数据准备脚本
- scripts/             # Python 数据挖掘、训练、推理和实时管线
- docs/                # 方法说明、实时报告和数据获取说明
- figures/             # 训练曲线与汇报图表
- paper/               # 论文或报告相关材料
- data/                # 原始与中间数据，本地保留，不提交
- models/              # 模型权重，本地保留，不提交
```

## 数据与隐私说明

- 仓库不提交 `.env`、API Key、数据库索引、原始微博数据和模型权重。
- `data/raw/`、`data/processed/`、`data/indexes/`、`models/`、`tmp/` 已通过 `.gitignore` 排除。
- 前端展示所需的轻量 CSV / JSON 静态资产位于 `app/public/data/`，可用于部署演示。
- 实时热搜模块只使用公开热搜标题，不抓取用户主页、评论或私信等个人内容。
- `post_examples.json` 已限制为历史数据口径，截止到 `2020-W53`，不混入实时合成数据。


## 相关文档

- [实时热搜汇报摘要](docs/REALTIME_REPORT_SUMMARY.md)
- [实时抓取说明](docs/REALTIME_FETCH.md)
- [数据获取报告](docs/DATA_ACQUISITION_REPORT.md)
- [项目交接记录](HANDOVER.md)

## License

本项目使用 [MIT License](LICENSE)。
