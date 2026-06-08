# 实时情绪监测系统 — 技术方案

> 最后更新：2026-06-02

## 1. 概述

将 Mood Weather Station 从静态历史数据可视化升级为**可接入实时数据流的情绪监测产品**。

当前采用**双轨策略**：
- **历史数据（主）**：39,973 条 DeepSeek 标注微博，覆盖34省x58周 — 用于训练和论文
- **实时数据（辅）**：UAPIS 热搜标题 -> 本地模型情绪推理 -> 趋势快照 — 合规概念验证

### 合规说明

**不爬取任何用户帖子内容。** 仅使用 UAPIS 聚合热搜标题（公开榜单数据），符合：
- 《网络数据安全管理条例》第18条：公开聚合数据采集无需特殊授权
- 最高法 2025 年第 47 批指导案例：公开信息收集不构成不正当竞争
- 不涉及 PIPL 下的个人信息收集（标题不含用户 PII）

## 2. 实时数据流（合规版）

### 数据流总览

```
UAPIS 热搜 API (公开聚合榜单)
  |  每 5 分钟更新，返回 50 条热搜标题 + 热度值
  v
17_hotsearch_live.py (Script 17)
  |  1. 取热搜标题
  |  2. 本地模型推理 6 维情绪
  |  3. 按热度加权聚合
  v
app/public/data/realtime/hotsearch_latest.json  <- 前端可消费
  |-- topics[]          # 单条：标题 + 热度 + 6维情绪 + 情绪强度
  |-- aggregate_emotion # 加权聚合情绪分布
  +-- fetch_time        # 采集时间戳

app/public/data/realtime/hotsearch_history.jsonl  <- 趋势分析
  +-- 每行一个快照，追加写入
```

### 脚本体系

| 编号 | 文件 | 用途 | 状态 |
|------|------|------|------|
| **17** | `scripts/17_hotsearch_live.py` | **UAPIS 热搜->本地推理->JSON 快照** | **生产就绪** |
| 11 | `scripts/11_prepare_training_data.py` | 80/10/10 分层切分 | 完成 |
| 12 | `scripts/12_emotion_dataset.py` | PyTorch Dataset | 完成 |
| 13 | `scripts/13_emotion_model.py` | RoBERTa + classifier head | 完成 |
| 14 | `scripts/14_train_emotion_model.py` | 训练循环 | 完成 |
| 15 | `scripts/15_evaluate_model.py` | 测试集评估 + SMP2020 | 完成 |
| 16 | `scripts/16_label_local.py` | 本地推理引擎 | 完成 |

> 旧 09/10 脚本（Patchright 爬帖路线）已废弃，详见 SS4 合规分析。

## 3. 脚本 17：UAPIS 热搜情绪快照

### 3.1 数据链路

1. 调用 **UAPIS** (uapis.cn) API 获取微博实时热搜榜 -> 50 条话题标题 + 热度值
2. 加载本地微调模型 `chinese-roberta-wwm-ext`（XPU/CPU）
3. 每条标题推理 6 维情绪分数
4. 按热度值加权聚合，计算主导情绪 + 情绪强度 + 情绪熵
5. 输出两份文件：
   - `hotsearch_latest.json`：最新快照（前端可直接加载）
   - `hotsearch_history.jsonl`：历史追加（趋势分析）

### 3.2 输出格式

```json
{
  "fetch_time": "2026-06-02T16:35:08",
  "source": "UAPIS Weibo Hot Search",
  "total_topics": 50,
  "aggregate_emotion": {
    "joy": 0.068, "sadness": 0.016, "anger": 0.018,
    "fear": 0.000, "surprise": 0.000, "neutral": 0.898
  },
  "aggregate_dominant": "neutral",
  "topics": [
    {
      "rank": 1,
      "title": "国乒男队队长王楚钦",
      "hot_value": 2644226,
      "emotion": {"joy": 0.0, "sadness": 0.0, ...},
      "dominant_emotion": "neutral",
      "emotional_intensity": 0.013,
      "emotional_entropy": 0.042
    }
  ]
}
```

### 3.3 运行方式

```bash
# 一次性快照
python scripts/17_hotsearch_live.py

# 定时采集（配合 Windows Task Scheduler 每 5-10 分钟触发一次）
```

### 3.4 已知局限

- **Domain shift**：热搜标题为新闻式客观陈述，与个人微博情绪语言不同域。~90% 被分类为"中性"，但有情绪倾向的标题（娱乐八卦、争议事件）能被正确捕获
- **无省份粒度**：UAPIS 热搜无省份字段
- **非真正"实时"**：UAPIS 5 分钟更新，全链延迟 < 30 秒
- **依赖第三方 API**：UAPIS 可能不稳定或变更

## 4. 合规分析（2026 法律环境）

### 4.1 关键判例与法规

| 时间 | 事件 | 对我们的影响 |
|------|------|-------------|
| 2026-05 | 广东高院非法调用 API 首案：判赔 2000 万 | IP 轮换+账号切换批量调用 API 属不正当竞争 |
| 2025-08 | 最高法第 47 批数据权益指导案例 | 大规模爬取属不正当竞争；公开信息收集合法 |
| 2025-01 | 《网络数据安全管理条例》生效 | 自动化采集不得"非法侵入"或"干扰正常运行" |
| 2025-08 | 微博升级风控（432 错误码） | s.weibo.com 需登录 Cookie，匿名搜索被拦截 |

### 4.2 "四不"红线合规对照

| 红线 | 旧方案（脚本 09） | 新方案（脚本 17） |
|------|------------------|-------------------|
| 不非法侵入 | 绕过搜索登录墙 | 仅访问公开 UAPIS API |
| 不规避保护措施 | Patchright 隐藏自动化特征 | 标准 HTTP GET |
| 不干扰服务 | 多轮搜索 x 省份交叉 | 单次 50 条请求 |
| 不侵犯权益 | 爬取用户微博内容 | 仅获取聚合标题 |

### 4.3 废弃脚本

| 脚本 | 原因 | 替代 |
|------|------|------|
| `09_fetch_realtime_weibo.py` | s.weibo.com 432 封禁 + 法律风险 | `17_hotsearch_live.py` |
| `10_preprocess_realtime.py` | 配套 09 的预处理 | 脚本 17 自带清洗 |

## 5. 后续方向

| 方向 | 说明 | 优先级 |
|------|------|--------|
| 前端接入实时面板 | 事件/总览页增加"实时热搜情绪"卡片 | 中 |
| 定时采集 | Windows Task Scheduler 每 5 分钟触发 | 低 |
| 历史趋势积累 | 连续运行后分析情绪日趋势 | 低 |
| 模型 domain 微调 | 在新闻标题上微调或收集标注数据 | 低 |
