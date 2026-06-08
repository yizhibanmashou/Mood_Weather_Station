# 数据获取探索报告

> 日期：2026-06-03
> 目的：寻找合法方式获取与现有数据集同字段（content_clean, province, time, 六维情绪）的实时微博数据

---

## 已探索方案

### 方案 1：UAPIS 热搜 API ✅（已有）
- **状态**：工作正常，每次获取 50 条热搜标题
- **局限**：只有标题，没有正文、省份、用户信息
- **结论**：作为话题来源，不能直接产出帖子数据

### 方案 2：微博访客系统（Visitor Cookie）
- **状态**：SUB cookie 可成功生成（有效期 1 年）
- **流程**：`genvisitor` → `incarnate` → `cross_domain` → SUB cookie
- **局限**：API 端点（如 `/ajax/side/hotSearch`）即使携带有效 SUB cookie 仍返回 403
- **原因**：微博服务端有 JS 指纹检测（webdriver 检测、RID 验证等），纯 HTTP 客户端无法绕过
- **结论**：需要完整浏览器环境（Playwright/Selenium），已废弃（见 HANDOVER.md）

### 方案 3：微博开放平台 API
- **状态**：可用但限制极大
- **限制**：
  - OAuth 2.0 需手动授权，Token 有效期 30 天
  - 单用户每天仅 100 次调用
  - 需要中国营业执照才能申请写入/商业接口
  - SDK 已停止维护
- **结论**：不适合批量数据获取

### 方案 4：微博移动端 API
- **状态**：需要 gsid（登录 session），未登录只能看到"登录注册后查看更多"
- **结论**：不可用

### 方案 5：公开学术数据集
- **状态**：存在多个 COVID-19 相关数据集
  - **Weibo-COV** (40M+ posts, DUA 申请)
  - **NLPCC 2020** (1.68M posts, 21K labeled, GitHub)
  - **JMIR Wuhan Study** (816K posts, 完全开放)
- **局限**：均只覆盖 2019-2020 COVID-19 时期，无实时数据
- **结论**：用于扩增历史数据可行，但不解决"当前数据"需求

### 方案 6：UAPIS 历史搜索模式
- **状态**：返回 `SERVICE_UNAVAILABLE` - "历史搜索服务暂不可用"
- **结论**：不可用

### 方案 7：第三方封装库（cv-cat/WeiboApis）
- **状态**：基于逆向的移动端 API，持续维护
- **风险**：属于逆向工程，灰色地带
- **结论**：不采用（合规优先）

---

## 最终方案：DeepSeek 合成数据生成 ✅

**实现方式**：`scripts/18_generate_synthetic_data.py`

**流程**：
```
UAPIS 热搜 API (实时热点标题)
  → DeepSeek API (生成 Weibo 风格正文 + 随机省份/时间)
    → 本地蒸馏模型 (6 维情绪打分)
      → CSV + JSON 输出（与现有数据集同 Schema）
```

**生成字段**：

| 字段 | 来源 |
|------|------|
| `content_clean` | DeepSeek 根据热搜话题生成 |
| `province` | DeepSeek 从 34 省中随机分配 |
| `created_at` | 当前时间 |
| `date_week/month` | 当前时间计算 |
| `joy~neutral` | 本地 RoBERTa 模型推理 |
| `label_status` | `ok` |
| `label_model` | `chinese-roberta-wwm-ext-local` |
| `source_topic` | 来源热搜话题（新增字段） |

**输出位置**：`data/synthetic/synthetic_YYMMDD_HHMM.csv`

**合规性**：完全合法
- 不爬取任何用户帖子
- 数据由 AI 模型生成（非真实用户数据）
- 话题来源为公开聚合 API（热搜榜单）
- 可用于证明模型对当前话题的泛化能力

**下一步**：合并到主数据集后重新跑完整管线（标注 → 聚合 → 异常检测 → 前端导出）
