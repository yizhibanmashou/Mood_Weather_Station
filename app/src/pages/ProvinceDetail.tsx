import { useEffect, useMemo, useRef, useState } from "react";
import type { EChartsOption } from "echarts";
import { ChartCard } from "../components/ChartCard";
import { EChart } from "../components/EChart";
import { EmptyState } from "../components/StateViews";
import { EMOTION_META, EMOTIONS, LOW_SAMPLE_THRESHOLD, type EmotionKey } from "../config";
import type { DataBundle } from "../types";
import { formatPct, provinceWeeks, vectorForProvince } from "../utils/analytics";
import { dateWeekToShortRange, dateWeekToAxisRange, dateWeekToFullRange } from "../utils/dateUtils";
import { cssVar } from "../theme";
import styles from "./Pages.module.css";

interface ProvinceDetailProps {
  data: DataBundle;
  initialProvince?: string;
}

export function ProvinceDetail({ data, initialProvince }: ProvinceDetailProps) {
  const provinces = useMemo(
    () => [...data.provinceVectors].sort((a, b) => b.total_posts_all - a.total_posts_all).map((row) => row.province),
    [data.provinceVectors]
  );
  const [province, setProvince] = useState(initialProvince && provinces.includes(initialProvince) ? initialProvince : (provinces[0] ?? ""));
  const [emotion, setEmotion] = useState<EmotionKey>("joy");

  // Sync province when initialProvince changes from external navigation (e.g. map click)
  const prevInitialRef = useRef(initialProvince);
  useEffect(() => {
    if (initialProvince && initialProvince !== prevInitialRef.current && provinces.includes(initialProvince)) {
      setProvince(initialProvince);
    }
    prevInitialRef.current = initialProvince;
  }, [initialProvince, provinces]);

  const vector = vectorForProvince(data, province);
  const weeks = provinceWeeks(data, province);

  // Debug: log radar values to verify per-province data
  useEffect(() => {
    if (!vector) return;
    const vals = EMOTIONS.map((k) => `${k}=${(vector[`${k}_mean_all` as keyof typeof vector] as number).toFixed(4)}`).join(", ");
    console.log(`[RadarDebug] ${province}: ${vals}`);
  }, [province, vector]);
  const examples = data.postExamples.provinces[province]?.[emotion] ?? [];

  // Get top 3 emotions for default display
  const topEmotions = useMemo(() => {
    if (!vector) return EMOTIONS.slice(0, 3);
    const sorted = [...EMOTIONS].sort((a, b) => {
      const aVal = vector[`${a}_mean_all` as keyof typeof vector] as number;
      const bVal = vector[`${b}_mean_all` as keyof typeof vector] as number;
      return bVal - aVal;
    });
    return sorted.slice(0, 3);
  }, [vector]);

  const radarOption = useMemo<EChartsOption>(() => {
    if (!vector) return {};
    const surfaceColor = cssVar("--surface-solid", "#fff");
    const borderColor = cssVar("--border", "rgba(60,80,110,0.1)");
    const textColor = cssVar("--text", "#1A2332");
    const textSecColor = cssVar("--text-secondary", "#5A6B7E");
    const warmColor = cssVar("--surface-warm", "#EDF1F5");
    return {
      color: [EMOTION_META[emotion].color],
      tooltip: {
        backgroundColor: surfaceColor,
        borderColor: borderColor,
        textStyle: { color: textColor },
        formatter: (params: unknown) => {
          const p = params as { value?: number[]; name?: string };
          const vals = p.value ?? [];
          let html = `<div style="font-weight:600;margin-bottom:6px">${province}</div>`;
          EMOTIONS.forEach((key, i) => {
            const v = vals[i] ?? 0;
            html += `<div style="display:flex;align-items:center;gap:6px;font-size:12px"><span style="width:8px;height:8px;border-radius:50%;background:${EMOTION_META[key].color};display:inline-block"></span>${EMOTION_META[key].label}: ${formatPct(v, 1)}</div>`;
          });
          return html;
        }
      },
      radar: {
        radius: "68%",
        splitArea: { areaStyle: { color: [warmColor, borderColor] } },
        axisName: { color: textSecColor, fontSize: 12 },
        splitLine: { lineStyle: { color: borderColor } },
        axisLine: { lineStyle: { color: borderColor } },
        indicator: EMOTIONS.map((key) => ({ name: EMOTION_META[key].label, max: 1 }))
      },
      series: [
        {
          type: "radar",
          areaStyle: { opacity: 0.28 },
          lineStyle: { width: 2.5 },
          data: [
            {
              value: EMOTIONS.map((key) => vector[`${key}_mean_all` as keyof typeof vector]),
              name: province
            }
          ]
        }
      ]
    };
  }, [emotion, province, vector]);

  const lineOption = useMemo<EChartsOption>(() => {
    const source = weeks.map((row) => ({
      date_label: dateWeekToAxisRange(row.date_week),
      ...Object.fromEntries(EMOTIONS.map((key) => [key, row[`${key}_mean` as keyof typeof row]]))
    }));
    const surfaceColor = cssVar("--surface-solid", "#fff");
    const borderColor = cssVar("--border", "rgba(60,80,110,0.1)");
    const textColor = cssVar("--text", "#1A2332");
    const textSecColor = cssVar("--text-secondary", "#5A6B7E");
    const textMutedColor = cssVar("--text-muted", "#8A98A8");
    const emotionKeyMap: Record<string, EmotionKey> = Object.fromEntries(
      EMOTIONS.map((key) => [EMOTION_META[key].label, key])
    ) as Record<string, EmotionKey>;
    return {
      color: EMOTIONS.map((key) => EMOTION_META[key].color),
      tooltip: {
        trigger: "axis",
        backgroundColor: surfaceColor,
        borderColor: borderColor,
        textStyle: { color: textColor },
        formatter: (params: unknown) => {
          const items = params as Array<{ seriesName: string; value: Record<string, number>; axisValue: string; color: string }>;
          if (!Array.isArray(items) || !items.length) return "";
          const week = weeks.find((w) => dateWeekToAxisRange(w.date_week) === items[0].axisValue);
          const title = week ? dateWeekToFullRange(week.date_week) : items[0].axisValue;
          let html = `<div style="font-weight:600;margin-bottom:6px">${title}</div>`;
          for (const item of items) {
            const key = emotionKeyMap[item.seriesName] ?? "neutral";
            html += `<div style="display:flex;align-items:center;gap:6px;font-size:12px"><span style="width:8px;height:8px;border-radius:50%;background:${item.color};display:inline-block"></span>${item.seriesName}: ${formatPct(item.value[key] ?? 0, 1)}</div>`;
          }
          return html;
        }
      },
      legend: {
        top: 0,
        textStyle: { color: textSecColor }
      },
      grid: { top: 48, left: 52, right: 18, bottom: 34 },
      dataset: { source },
      xAxis: {
        type: "category",
        axisLabel: { color: textMutedColor, fontSize: 10 },
        axisLine: { lineStyle: { color: borderColor } }
      },
      yAxis: {
        type: "value",
        max: 1,
        axisLabel: { color: textMutedColor, formatter: (value: number) => `${Math.round(value * 100)}%` },
        splitLine: { lineStyle: { color: borderColor } }
      },
      dataZoom: [{ type: "inside" }],
      series: EMOTIONS.map((key) => ({
        type: "line",
        name: EMOTION_META[key].label,
        encode: { x: "date_label", y: key },
        smooth: true,
        showSymbol: false,
        lineStyle: { width: topEmotions.includes(key) ? 2.5 : 1.5 },
        opacity: topEmotions.includes(key) ? 1 : 0.4
      }))
    };
  }, [weeks, topEmotions]);

  return (
    <div className={styles.pageStack}>
      <section className={styles.controlBar}>
        <div>
          <p className={styles.kicker}>PROVINCE DETAIL</p>
          <h1>{province || "省份详情"}</h1>
        </div>
        <select value={province} onChange={(event) => setProvince(event.target.value)} className={styles.select}>
          {provinces.map((item) => (
            <option value={item} key={item}>
              {item}
            </option>
          ))}
        </select>
      </section>

      {vector && vector.total_posts_all < LOW_SAMPLE_THRESHOLD ? (
        <div className={styles.notice}>当前样本中 {province} 仅 {vector.total_posts_all} 条，图表仅供参考。</div>
      ) : null}

      <div className={styles.detailGrid}>
        <ChartCard title="省份情绪雷达" eyebrow="EMOTION RADAR">
          {vector ? <EChart key={`radar-${province}`} option={radarOption} height={340} /> : <EmptyState title="暂无省份向量" />}
        </ChartCard>
        <ChartCard title="情绪周演变" eyebrow="PROVINCE EVOLUTION" className={styles.wideCard}>
          {weeks.length ? <EChart option={lineOption} height={340} /> : <EmptyState title="暂无周度记录" />}
        </ChartCard>
        <ChartCard
          title="高情绪帖样例"
          eyebrow="TOP POSTS"
          action={
            <div className={styles.segmented}>
              {EMOTIONS.map((key) => (
                <button key={key} className={emotion === key ? styles.activePill : ""} onClick={() => setEmotion(key)}>
                  {EMOTION_META[key].label}
                </button>
              ))}
            </div>
          }
          className={styles.fullCard}
        >
          {examples.length ? (
            <div className={styles.postList}>
              {examples.map((post) => (
                <article key={post.post_id} className={styles.postItem}>
                  <div>
                    <strong>{dateWeekToShortRange(post.date_week)}</strong>
                    <span>{EMOTION_META[emotion].label} {formatPct(post.score)}</span>
                  </div>
                  <p>{post.content}</p>
                </article>
              ))}
            </div>
          ) : (
            <EmptyState title="暂无帖样例" detail="该省份当前情绪没有可展示样本" />
          )}
        </ChartCard>
      </div>
    </div>
  );
}
