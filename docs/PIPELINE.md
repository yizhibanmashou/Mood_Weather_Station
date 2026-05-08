# 脚本流水线

## 脚本清单

| Step | 脚本 | 状态 | 输入 | 主要输出 |
|---:|---|---|---|---|
| 00 | `00_probe_data_feasibility.py` | 已完成 | raw 数据 | `tmp/00_feasibility_report.json` |
| 01 | `01_build_mini_dataset.py` | ✅ 已完成 | raw CSV + users.db | `data/processed/mini_dataset.csv` (76,441 rows) |
| 02 | `02_label_emotions.py` | ✅ 已完成 | mini_dataset / expansion plan | `labeled_dataset*.csv` |
| 02b | `02b_build_stratified_relabel_plan.py` | ✅ 已完成 | mini + labeled | 分层补标计划 CSV |
| 02c | `02c_merge_labeled_datasets.py` | ✅ 已完成 | old + new labeled | 合并后 CSV |
| 02d | `02d_build_week_cap_expansion_plan.py` | ✅ 已完成 | mini + labeled | 周-省扩展计划 CSV |
| 02e | `02e_finalize_week_cap_expansion.py` | ✅ 已完成 | old merged + new labeled | 合并 CSV + 质量报告 + 下游流水线 |
| 03 | `03_validate_emotions.py` | ✅ 已完成 | SMP2020 + labeled | accuracy=73.3%, macro F1=0.662 |
| 04 | `04_aggregate_emotions.py` | ✅ 已完成 | labeled_dataset_merged | 1,888 周行, 442 月行, 34 省向量 |
| 04b | `04b_nlp_keywords.py` | ✅ 已完成 | labeled_dataset_merged | 55 周关键词, 6×30 情绪关键词 |
| 05 | `05_detect_anomalies.py` | ✅ 已完成 | national + weekly panel | `anomaly_detection.json` (19 异常) |
| 06 | `06_cluster_provinces.py` | ✅ 已完成 | province vectors | 34 省 6 聚类, silhouette=0.3985 |
| 07 | `07_cluster_evolution.py` | ✅ 已完成 | monthly panel | 30 省 × 13 月聚类演化 |
| 08 | `08_prepare_frontend_assets.py` | ✅ 已完成 | processed + analysis | `app/public/` 23 个静态资产 |

## 一键运行（推荐）

```powershell
# 完整扩展流水线：审计 → dry-run → 计划 → smoke → 标注 → 合并 → 下游分析 → 前端导出
python scripts/run_week_cap60_expansion_pipeline.py --cap 60

# 支持断点续跑（默认开启），中断后重新运行同一命令即可从上次位置继续
python scripts/run_week_cap60_expansion_pipeline.py
```

流水线步骤：

| 阶段 | 说明 | 耗时 |
|---|---|---|
| Step 1 | Dry-run 对比不同 cap 方案 | ~10s |
| Step 2 | 生成扩展计划 | ~30s |
| Step 2b | 估算 API 成本 | ~5s |
| Step 3 | 生成 smoke 测试计划 (60 条) | ~5s |
| Step 4 | Smoke 标注 (API 调用) | ~1min |
| Step 5 | 全量标注 (API 调用, 长时间) | ~4h |
| Step 6 | 02e 合并 + 质量检查 + 04-08 下游 | ~2min |

## 手动运行

### 聚合管线（标注完成后）

```powershell
python scripts/04_aggregate_emotions.py --input data/processed/labeled_dataset_merged_week_cap60.csv
python scripts/04b_nlp_keywords.py --input data/processed/labeled_dataset_merged_week_cap60.csv
python scripts/05_detect_anomalies.py
python scripts/06_cluster_provinces.py
python scripts/07_cluster_evolution.py
python scripts/08_prepare_frontend_assets.py
```

### 标注流程

```powershell
# 先估算 token 与成本
python scripts/02_label_emotions.py --input <plan.csv> --dry-run

# Smoke 测试 (60 条)
python scripts/02_label_emotions.py --input smoke.csv --output smoke_output.csv --limit 60

# 全量标注
python scripts/02_label_emotions.py --input <plan.csv> --output <output.csv>
```

### 合并与质量检查

```powershell
python scripts/02e_finalize_week_cap_expansion.py \
  --old-merged data/processed/labeled_dataset_merged_cap30.csv \
  --new-labeled data/processed/labeled_dataset_week_cap60_expansion.csv \
  --output data/processed/labeled_dataset_merged_week_cap60.csv \
  --force
```

## 输出依赖

```text
mini_dataset.csv (76,441 rows)
  → 02 标注 (分批, 支持断点续跑)
      → 02d 扩展计划 (cap=60)
      → 02e 合并 + 质量检查
          labeled_dataset_merged_week_cap60.csv (39,973 rows)
          ├── 04 聚合面板
          │   ├── emotion_panel_weekly.csv (1,888 rows)
          │   ├── emotion_panel_monthly.csv (442 rows)
          │   ├── emotion_national_timeline.csv (58 weeks)
          │   └── province_emotion_vectors.csv (34 provinces)
          ├── 04b NLP 关键词
          │   ├── nlp_keywords_by_week.json (55 weeks)
          │   ├── nlp_emotion_keywords.json (6 × 30 keywords)
          │   └── nlp_global_vocabulary.json (5,000 terms)
          ├── 05 异常检测
          │   └── anomaly_detection.json (19 anomalies)
          ├── 06 省份聚类
          │   └── cluster_labels.csv (34 provinces, 6 clusters)
          ├── 07 聚类演化
          │   └── monthly_cluster_labels.csv (30 provinces × 13 months)
          └── 08 前端导出
              └── app/public/data/ (23 files → manifest.json)
```

## 关键注意事项

- `.env` 只放本地，不提交 API key
- 省级面板只使用 34 标准省份，非标准值（如"公安局网络安全保卫支队官方微博"）会被过滤
- 04b NLP 对所有 58 周提取关键词，仅 3 周因样本不足标记为 insufficient_data
- 07 聚类演化按月独立聚类，用风险得分对齐标签编号
- 08 输出 manifest.json 记录所有前端文件的 hash 和行数
