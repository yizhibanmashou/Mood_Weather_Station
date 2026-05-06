import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { ChartCard } from "../components/ChartCard";
import { EChart } from "../components/EChart";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { HeroSection } from "../components/HeroSection";
import { MetricCard } from "../components/MetricCard";
import { EmptyState } from "../components/StateViews";
import { StaticImageBubbleMap } from "../components/StaticImageBubbleMap";
import { EMOTION_META, EMOTIONS, type EmotionKey } from "../config";
import type { DataBundle } from "../types";
import { buildDashboardMetrics, formatNumber, formatPct, getTopAnomalies } from "../utils/analytics";
import { dateWeekToShortRange, dateWeekToFullRange, dateWeekToAxisRange } from "../utils/dateUtils";
import { severityLabel, zScoreDescription, deviationDescription } from "../utils/metricLabels";
import { cssVar } from "../theme";
import styles from "./Pages.module.css";

interface DashboardProps {
  data: DataBundle;
  onProvinceSelect?: (province: string) => void;
}

type MapMetric = "emotional_intensity" | EmotionKey | "dominant";

const MAP_METRIC_OPTIONS: Array<{ key: MapMetric; label: string }> = [
  { key: "emotional_intensity", label: "情绪温度" },
  { key: "joy", label: "喜悦" },
  { key: "sadness", label: "悲伤" },
  { key: "anger", label: "愤怒" },
  { key: "fear", label: "恐惧" },
  { key: "surprise", label: "惊讶" },
  { key: "neutral", label: "中性" },
  { key: "dominant", label: "主导情绪" }
];

export function Dashboard({ data, onProvinceSelect }: DashboardProps) {
  const metrics = useMemo(() => buildDashboardMetrics(data), [data]);
  const topAnomalies = useMemo(() => getTopAnomalies(data.anomalies, 5), [data.anomalies]);
  const [mapMetric, setMapMetric] = useState<MapMetric>("emotional_intensity");
  const [showAllEmotions, setShowAllEmotions] = useState(false);

  // Emotion composition data
  const composition = useMemo(() => {
    const latest = data.nationalWeeks.at(-1);
    if (!latest) return null;
    const values = EMOTIONS.map((e) => ({
      key: e,
      label: EMOTION_META[e].label,
      color: EMOTION_META[e].color,
      value: latest[`${e}_mean` as keyof typeof latest] as number
    }));
    const sorted = [...values].sort((a, b) => b.value - a.value);
    const top3 = sorted.slice(0, 3);
    const positive = latest.joy_mean + latest.surprise_mean * 0.5;
    const negative = latest.sadness_mean + latest.anger_mean + latest.fear_mean;
    const neutral = latest.neutral_mean;
    return { values, sorted, top3, positive, negative, neutral };
  }, [data.nationalWeeks]);

  const compositionOption = useMemo<EChartsOption>(() => {
    if (!composition) return {};
    const surfaceColor = cssVar("--surface-solid", "#fff");
    const borderColor = cssVar("--border", "rgba(60,80,110,0.1)");
    const textColor = cssVar("--text", "#1A2332");
    const textSecColor = cssVar("--text-secondary", "#5A6B7E");
    return {
      tooltip: {
        trigger: "item",
        backgroundColor: surfaceColor,
        borderColor: borderColor,
        textStyle: { color: textColor },
        formatter: (params: unknown) => {
          const p = params as { name: string; value: number; color: string };
          return `<div style="font-weight:600">${p.name}</div><div>${formatPct(p.value, 1)}</div>`;
        }
      },
      series: [
        {
          type: "pie",
          radius: ["48%", "72%"],
          center: ["50%", "50%"],
          avoidLabelOverlap: true,
          itemStyle: { borderRadius: 6, borderColor: surfaceColor, borderWidth: 2 },
          label: {
            show: true,
            formatter: "{b}\n{d}%",
            fontSize: 11,
            color: textSecColor
          },
          labelLine: { lineStyle: { color: borderColor } },
          emphasis: {
            label: { fontSize: 13, fontWeight: "bold" }
          },
          data: composition.values.map((v) => ({
            name: v.label,
            value: v.value,
            itemStyle: { color: v.color }
          }))
        }
      ]
    };
  }, [composition]);

  // Timeline chart
  const timelineOption = useMemo<EChartsOption>(() => {
    const weeks = data.nationalWeeks;
    const surfaceColor = cssVar("--surface-solid", "#fff");
    const borderColor = cssVar("--border", "rgba(60,80,110,0.1)");
    const textColor = cssVar("--text", "#1A2332");
    const textSecColor = cssVar("--text-secondary", "#5A6B7E");
    const textMutedColor = cssVar("--text-muted", "#8A98A8");
    const accentColor = cssVar("--accent", "#3B7DD8");
    if (showAllEmotions) {
      const source = weeks.map((row) => ({
        date_label: dateWeekToAxisRange(row.date_week),
        ...Object.fromEntries(EMOTIONS.map((e) => [e, row[`${e}_mean` as keyof typeof row]]))
      }));
      return {
        color: EMOTIONS.map((e) => EMOTION_META[e].color),
        tooltip: {
          trigger: "axis",
          backgroundColor: surfaceColor,
          borderColor: borderColor,
          textStyle: { color: textColor },
          formatter: (params: unknown) => {
            const items = params as Array<{ seriesName: string; value: Record<string, number>; axisValue: string; color: string }>;
            if (!Array.isArray(items) || !items.length) return "";
            const week = data.nationalWeeks.find((w) => dateWeekToAxisRange(w.date_week) === items[0].axisValue);
            const title = week ? dateWeekToFullRange(week.date_week) : items[0].axisValue;
            const emotionKeyMap: Record<string, string> = { "喜悦": "joy", "悲伤": "sadness", "愤怒": "anger", "恐惧": "fear", "惊讶": "surprise", "中性": "neutral" };
            let html = `<div style="font-weight:600;margin-bottom:6px">${title}</div>`;
            for (const item of items) {
              const key = emotionKeyMap[item.seriesName] ?? "neutral";
              html += `<div style="display:flex;align-items:center;gap:6px;font-size:12px"><span style="width:8px;height:8px;border-radius:50%;background:${item.color};display:inline-block"></span>${item.seriesName}: ${formatPct(item.value[key], 1)}</div>`;
            }
            return html;
          }
        },
        legend: { top: 0, textStyle: { color: textSecColor } },
        grid: { top: 48, left: 52, right: 16, bottom: 36 },
        dataset: { source },
        xAxis: {
          type: "category", boundaryGap: false,
          axisLabel: { color: textMutedColor, fontSize: 10 },
          axisLine: { lineStyle: { color: borderColor } }
        },
        yAxis: {
          type: "value", max: 1,
          axisLabel: { color: textMutedColor, formatter: (v: number) => `${Math.round(v * 100)}%` },
          splitLine: { lineStyle: { color: borderColor } }
        },
        dataZoom: [{ type: "inside" }],
        series: EMOTIONS.map((e) => ({
          type: "line", name: EMOTION_META[e].label,
          encode: { x: "date_label", y: e },
          stack: "emotion", smooth: true, showSymbol: false,
          lineStyle: { width: 2.5 }, areaStyle: { opacity: 0.5 },
          emphasis: { focus: "series" }
        }))
      };
    }
    const source = weeks.map((row) => ({
      date_label: dateWeekToAxisRange(row.date_week),
      intensity: row.emotional_intensity,
      positive: row.positive_index,
      negative: row.fear_mean + row.anger_mean + row.sadness_mean
    }));
    return {
      color: [accentColor, "#F2A23A", EMOTION_META.anger.color],
      tooltip: {
        trigger: "axis",
        backgroundColor: surfaceColor,
        borderColor: borderColor,
        textStyle: { color: textColor },
        formatter: (params: unknown) => {
          const items = params as Array<{ seriesName: string; value: Record<string, number>; axisValue: string; color: string }>;
          if (!Array.isArray(items) || !items.length) return "";
          const week = data.nationalWeeks.find((w) => dateWeekToAxisRange(w.date_week) === items[0].axisValue);
          const title = week ? dateWeekToFullRange(week.date_week) : items[0].axisValue;
          const nameMap: Record<string, string> = { intensity: "情绪温度", positive: "积极指数", negative: "负面压力" };
          let html = `<div style="font-weight:600;margin-bottom:6px">${title}</div>`;
          for (const item of items) {
            const key = Object.entries(nameMap).find(([, v]) => v === item.seriesName)?.[0] ?? "";
            html += `<div style="display:flex;align-items:center;gap:6px;font-size:12px"><span style="width:8px;height:8px;border-radius:50%;background:${item.color};display:inline-block"></span>${item.seriesName}: ${formatPct(item.value[key], 1)}</div>`;
          }
          return html;
        }
      },
      legend: { top: 0, textStyle: { color: textSecColor } },
      grid: { top: 48, left: 52, right: 16, bottom: 36 },
      dataset: { source },
      xAxis: {
        type: "category", boundaryGap: false,
        axisLabel: { color: textMutedColor, fontSize: 10 },
        axisLine: { lineStyle: { color: borderColor } }
      },
      yAxis: {
        type: "value", max: 1,
        axisLabel: { color: textMutedColor, formatter: (v: number) => `${Math.round(v * 100)}%` },
        splitLine: { lineStyle: { color: borderColor } }
      },
      dataZoom: [{ type: "inside" }],
      series: [
        { type: "line", name: "情绪温度", encode: { x: "date_label", y: "intensity" }, smooth: true, showSymbol: false, lineStyle: { width: 2.5 } },
        { type: "line", name: "积极指数", encode: { x: "date_label", y: "positive" }, smooth: true, showSymbol: false, lineStyle: { width: 2.5 } },
        { type: "line", name: "负面压力", encode: { x: "date_label", y: "negative" }, smooth: true, showSymbol: false, lineStyle: { width: 2.5 } }
      ]
    };
  }, [data.nationalWeeks, showAllEmotions]);

  return (
    <div className={styles.pageStack}>
      {/* Hero Section */}
      <HeroSection scrollTargetId="dashboard-metrics" />

      {/* Metrics */}
      <div className={styles.metricsGrid} id="dashboard-metrics">
        <MetricCard label="全国情绪温度" value={formatPct(metrics.avgIntensity)} detail="非中性情绪的综合强度" tone="warm" />
        <MetricCard label="积极指数变化" value={`${metrics.positiveDelta >= 0 ? "+" : ""}${metrics.positiveDelta.toFixed(2)}`} detail="最新周相对前一周的变化" tone={metrics.positiveDelta >= 0 ? "warm" : "danger"} />
        <MetricCard label="恐惧峰值周" value={metrics.fearPeak ? dateWeekToShortRange(metrics.fearPeak.date_week) : "-"} detail={`恐惧情绪达到 ${formatPct(metrics.fearPeak?.fear_mean ?? 0)}`} tone="danger" />
        <MetricCard label="波动最高省份" value={metrics.mostPositive?.province ?? "-"} detail={`情绪温度 ${formatPct(metrics.mostPositive?.emotional_intensity_mean ?? 0)}`} tone="warm" />
        <MetricCard label="情绪最低省份" value={metrics.mostNegative?.province ?? "-"} detail={`样本量 ${formatNumber(metrics.mostNegative?.total_posts_all ?? 0)}`} tone="cool" />
        <MetricCard label="异常事件" value={String(data.anomalies.length)} detail="情绪波动超出历史范围" tone="neutral" />
      </div>

      {/* 2x2 Grid */}
      <div className={styles.dashboardGrid}>
        {/* Top-left: Timeline */}
        <ChartCard
          title="全国情绪周时序"
          eyebrow="NATIONAL TIMELINE"
          action={
            <button className={styles.expandToggle} onClick={() => setShowAllEmotions(!showAllEmotions)} type="button">
              {showAllEmotions ? "收起为三线" : "展开六维情绪"}
            </button>
          }
        >
          <EChart option={timelineOption} height={360} />
        </ChartCard>

        {/* Top-right: Map */}
        <ChartCard
          title="中国情绪地图"
          eyebrow="PROVINCE MAP"
          action={
            <div className={styles.mapSwitcher}>
              {MAP_METRIC_OPTIONS.map((opt) => (
                <button
                  key={opt.key}
                  className={mapMetric === opt.key ? styles.activeMapBtn : ""}
                  onClick={() => setMapMetric(opt.key)}
                  type="button"
                >
                  {opt.label}
                </button>
              ))}
            </div>
          }
        >
          <ErrorBoundary fallbackTitle="地图组件加载失败">
            <StaticImageBubbleMap
              vectors={data.provinceVectors}
              monthlyData={data.provinceMonths}
              selectedMetric={mapMetric}
              onProvinceClick={(province) => onProvinceSelect?.(province)}
              height={420}
            />
          </ErrorBoundary>
        </ChartCard>

        {/* Bottom-left: Anomalies */}
        <ChartCard title="最新异常事件" eyebrow="ANOMALY RADAR">
          <div className={styles.anomalyList}>
            {topAnomalies.length ? (
              topAnomalies.map((event) => {
                const deviation = deviationDescription(event.deviation_pct);
                return (
                  <article key={`${event.date_week}-${event.emotion}-${event.z_score}`} className={styles.anomalyCard}>
                    <span style={{ color: EMOTION_META[event.emotion].color, fontSize: 13, fontWeight: 600 }}>
                      {EMOTION_META[event.emotion].label}
                    </span>
                    <div>
                      <strong>{dateWeekToFullRange(event.date_week)}</strong>
                      <small style={{ display: "block", marginTop: 2 }}>
                        {severityLabel(event.severity)} · {deviation}
                      </small>
                    </div>
                    <small style={{ color: "var(--text-muted)", fontSize: 11 }}>
                      {zScoreDescription(event.z_score)}
                    </small>
                  </article>
                );
              })
            ) : (
              <EmptyState title="暂无异常事件" />
            )}
          </div>
        </ChartCard>

        {/* Bottom-right: Emotion Composition */}
        <ChartCard title="情绪结构总览" eyebrow="EMOTION COMPOSITION">
          {composition ? (
            <div>
              <EChart option={compositionOption} height={220} />
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: 8,
                padding: "10px 0 0"
              }}>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>Top 1</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: composition.top3[0]?.color }}>{composition.top3[0]?.label}</div>
                  <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{formatPct(composition.top3[0]?.value ?? 0)}</div>
                </div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>Top 2</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: composition.top3[1]?.color }}>{composition.top3[1]?.label}</div>
                  <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{formatPct(composition.top3[1]?.value ?? 0)}</div>
                </div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>Top 3</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: composition.top3[2]?.color }}>{composition.top3[2]?.label}</div>
                  <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{formatPct(composition.top3[2]?.value ?? 0)}</div>
                </div>
              </div>
              <div style={{
                display: "flex",
                justifyContent: "space-around",
                padding: "10px 0 0",
                borderTop: "1px solid var(--border)",
                marginTop: 10
              }}>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>正向</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#4CAF50" }}>{formatPct(composition.positive)}</div>
                </div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>中性</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "var(--emotion-neutral)" }}>{formatPct(composition.neutral)}</div>
                </div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>负向</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "var(--emotion-anger)" }}>{formatPct(composition.negative)}</div>
                </div>
              </div>
              <p style={{ margin: "10px 0 0", fontSize: 12, color: "var(--text-muted)", lineHeight: 1.5, textAlign: "center" }}>
                当前阶段整体以{composition.sorted[0]?.label}为主，{composition.sorted[1]?.label}和{composition.sorted[2]?.label}波动较明显
              </p>
            </div>
          ) : (
            <EmptyState title="暂无情绪数据" />
          )}
        </ChartCard>
      </div>
    </div>
  );
}
