import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { ChartCard } from "../components/ChartCard";
import { EChart } from "../components/EChart";
import { EmptyState } from "../components/StateViews";
import { EMOTION_META, EMOTIONS, type EmotionKey } from "../config";
import type { RealtimeHotsearchSnapshot, RealtimeTopic } from "../types";
import { formatNumber, formatPct } from "../utils/analytics";
import { cssVar } from "../theme";
import styles from "./Pages.module.css";

interface RealtimeDashboardProps {
  snapshot: RealtimeHotsearchSnapshot | null;
}

function topicEmotion(topic: RealtimeTopic): EmotionKey {
  return EMOTIONS.includes(topic.dominant_emotion) ? topic.dominant_emotion : "neutral";
}

export function RealtimeDashboard({ snapshot }: RealtimeDashboardProps) {
  const aggregateOption = useMemo<EChartsOption>(() => {
    if (!snapshot) return {};
    const surfaceColor = cssVar("--surface-solid", "#fff");
    const borderColor = cssVar("--border", "rgba(60,80,110,0.1)");
    const textColor = cssVar("--text", "#1A2332");
    return {
      tooltip: {
        trigger: "item",
        backgroundColor: surfaceColor,
        borderColor,
        textStyle: { color: textColor },
        formatter: (params: unknown) => {
          const p = params as { name: string; value: number };
          return `<strong>${p.name}</strong><br/>${formatPct(p.value, 1)}`;
        }
      },
      series: [
        {
          type: "pie",
          radius: ["48%", "74%"],
          itemStyle: { borderRadius: 6, borderColor: surfaceColor, borderWidth: 2 },
          label: { formatter: "{b}\n{d}%" },
          data: EMOTIONS.map((emotion) => ({
            name: EMOTION_META[emotion].label,
            value: Number(snapshot.aggregate_emotion?.[emotion] ?? 0),
            itemStyle: { color: EMOTION_META[emotion].color }
          }))
        }
      ]
    };
  }, [snapshot]);

  const topicBarOption = useMemo<EChartsOption>(() => {
    if (!snapshot) return {};
    const topics = snapshot.topics.slice(0, 12).reverse();
    const borderColor = cssVar("--border", "rgba(60,80,110,0.1)");
    const textMutedColor = cssVar("--text-muted", "#8A98A8");
    const textColor = cssVar("--text", "#1A2332");
    return {
      grid: { left: 92, right: 18, top: 12, bottom: 18 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter: (params: unknown) => {
          const item = (params as Array<{ dataIndex: number }>)[0];
          const topic = topics[item.dataIndex];
          if (!topic) return "";
          const emotion = topicEmotion(topic);
          return [
            `<strong>${topic.title}</strong>`,
            `主导情绪：${EMOTION_META[emotion].label}`,
            `情绪强度：${formatPct(topic.emotional_intensity)}`,
            `热度：${formatNumber(topic.hot_value)}`
          ].join("<br/>");
        }
      },
      xAxis: {
        type: "value",
        max: 1,
        axisLabel: { color: textMutedColor, formatter: (value: number) => `${Math.round(value * 100)}%` },
        splitLine: { lineStyle: { color: borderColor } }
      },
      yAxis: {
        type: "category",
        data: topics.map((topic) => `#${topic.rank} ${topic.title.slice(0, 10)}`),
        axisLabel: { color: textMutedColor, fontSize: 11 },
        axisLine: { lineStyle: { color: borderColor } }
      },
      series: [
        {
          type: "bar",
          data: topics.map((topic) => ({
            value: topic.emotional_intensity,
            itemStyle: { color: EMOTION_META[topicEmotion(topic)].color }
          })),
          label: {
            show: true,
            position: "right",
            color: textColor,
            formatter: (params: unknown) => {
              const value = (params as { value?: number | null }).value;
              return formatPct(Number(value ?? 0), 0);
            }
          }
        }
      ]
    };
  }, [snapshot]);

  if (!snapshot) {
    return (
      <div className={styles.pageStack}>
        <section className={styles.sectionHeader}>
          <p className={styles.kicker}>REALTIME HOTSEARCH</p>
          <h1>实时热搜情绪</h1>
        </section>
        <ChartCard title="实时快照" eyebrow="UAPIS HOTBOARD">
          <EmptyState title="暂无实时热搜快照" detail="运行 scripts/17_hotsearch_live.py 后会生成 hotsearch_latest.json" />
        </ChartCard>
      </div>
    );
  }

  const dominant = EMOTIONS.includes(snapshot.aggregate_dominant) ? snapshot.aggregate_dominant : "neutral";

  return (
    <div className={styles.pageStack}>
      <section className={styles.sectionHeader}>
        <p className={styles.kicker}>REALTIME HOTSEARCH</p>
        <h1>实时热搜情绪</h1>
        <p className={styles.subtitle}>
          {snapshot.source} · {snapshot.fetch_time_str} · {snapshot.total_topics} 个话题
        </p>
      </section>

      <div className={styles.dashboardGrid}>
        <ChartCard title="实时主导情绪" eyebrow="DOMINANT">
          <div className={styles.snapshot}>
            <span>聚合主导</span>
            <strong style={{ color: EMOTION_META[dominant].color }}>{EMOTION_META[dominant].label}</strong>
            <span>{formatPct(snapshot.aggregate_emotion[dominant] ?? 0)}</span>
          </div>
        </ChartCard>
        <ChartCard title="话题数量" eyebrow="TOPICS">
          <div className={styles.snapshot}>
            <span>本次快照</span>
            <strong>{snapshot.total_topics}</strong>
            <span>按情绪强度排序</span>
          </div>
        </ChartCard>
      </div>

      <div className={styles.dashboardGrid}>
        <ChartCard title="聚合情绪分布" eyebrow="AGGREGATE">
          <EChart option={aggregateOption} height={330} />
        </ChartCard>
        <ChartCard title="情绪强度排行" eyebrow="TOP EMOTIONAL TOPICS">
          <EChart option={topicBarOption} height={330} />
        </ChartCard>
      </div>

      <ChartCard title="热搜话题明细" eyebrow="HOTSEARCH DETAILS">
        <div className={styles.tableWrap}>
          <table className={styles.dataTable}>
            <thead>
              <tr>
                <th>排名</th>
                <th>话题</th>
                <th>热度</th>
                <th>主导情绪</th>
                <th>情绪强度</th>
                <th>熵</th>
              </tr>
            </thead>
            <tbody>
              {snapshot.topics.slice(0, 30).map((topic) => {
                const emotion = topicEmotion(topic);
                return (
                  <tr key={`${topic.rank}-${topic.title}`}>
                    <td>#{topic.rank}</td>
                    <td>{topic.title}</td>
                    <td>{formatNumber(topic.hot_value)}</td>
                    <td style={{ color: EMOTION_META[emotion].color }}>{EMOTION_META[emotion].label}</td>
                    <td>{formatPct(topic.emotional_intensity)}</td>
                    <td>{topic.emotional_entropy.toFixed(2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </ChartCard>
    </div>
  );
}
