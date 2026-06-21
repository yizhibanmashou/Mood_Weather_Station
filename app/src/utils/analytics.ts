import { EMOTIONS, SEVERITY_ORDER, type EmotionKey } from "../config";
import type {
  AnomalyEvent,
  ClusterLabel,
  DataBundle,
  NationalWeek,
  ProvinceVector,
  ProvinceWeek
} from "../types";
import { dateWeekToShortRange } from "./dateUtils";

const vectorKeys = [
  "joy_mean_all",
  "sadness_mean_all",
  "anger_mean_all",
  "fear_mean_all",
  "surprise_mean_all",
  "neutral_mean_all",
  "emotional_intensity_mean",
  "fear_variance",
  "joy_variance"
] as const;

export function formatPct(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value: number, digits = 0) {
  return value.toLocaleString("zh-CN", { maximumFractionDigits: digits });
}

export function emotionMean(row: Record<string, number>, emotion: EmotionKey) {
  return row[`${emotion}_mean`] ?? 0;
}

export function provincePositiveScore(row: ProvinceVector) {
  return row.joy_mean_all + row.surprise_mean_all * 0.5 - row.sadness_mean_all - row.anger_mean_all - row.fear_mean_all;
}

export function provincePositiveIndex(row: ProvinceVector) {
  return (row.joy_mean_all + row.surprise_mean_all * 0.5) / (1 - row.neutral_mean_all + 0.01);
}

export function normalizedMapValues(vectors: ProvinceVector[]) {
  const raw = vectors.map((row) => ({ province: row.province, value: provincePositiveIndex(row), row }));
  const values = raw.map((item) => item.value).filter(Number.isFinite);
  const min = Math.min(...values);
  const max = Math.max(...values);
  return raw.map((item) => ({
    ...item,
    normalized: max === min ? 50 : ((item.value - min) / (max - min)) * 100
  }));
}

export function getTopAnomalies(events: AnomalyEvent[], count = 5) {
  return [...events]
    .sort((a, b) => {
      const severityDelta = (SEVERITY_ORDER[b.severity] ?? 0) - (SEVERITY_ORDER[a.severity] ?? 0);
      if (severityDelta) return severityDelta;
      return Math.abs(b.z_score) - Math.abs(a.z_score);
    })
    .slice(0, count);
}

export function buildDashboardMetrics(data: DataBundle) {
  const totalPosts = data.nationalWeeks.reduce((sum, row) => sum + row.total_posts, 0);
  const latest = data.nationalWeeks.at(-1);
  const previous = data.nationalWeeks.at(-2);
  const fearPeak = [...data.nationalWeeks].sort((a, b) => b.fear_mean - a.fear_mean)[0];
  const mostPositive = [...data.provinceVectors].sort((a, b) => provincePositiveScore(b) - provincePositiveScore(a))[0];
  const mostNegative = [...data.provinceVectors].sort((a, b) => provincePositiveScore(a) - provincePositiveScore(b))[0];
  const avgIntensity =
    data.nationalWeeks.reduce((sum, row) => sum + row.emotional_intensity, 0) / Math.max(1, data.nationalWeeks.length);
  const positiveDelta = latest && previous ? latest.positive_index - previous.positive_index : 0;

  return {
    totalPosts,
    weekRange: `${dateWeekToShortRange(data.nationalWeeks[0]?.date_week ?? "")} - ${dateWeekToShortRange(latest?.date_week ?? "")}`,
    provinceCount: data.provinceVectors.length,
    avgIntensity,
    positiveDelta,
    fearPeak,
    mostPositive,
    mostNegative
  };
}

export function provinceWeeks(data: DataBundle, province: string): ProvinceWeek[] {
  return data.provinceWeeks.filter((row) => row.province === province).sort((a, b) => a.date_week.localeCompare(b.date_week));
}

export function vectorForProvince(data: DataBundle, province: string) {
  return data.provinceVectors.find((row) => row.province === province);
}

function standardize(matrix: number[][]) {
  if (!matrix.length) return matrix;
  const cols = matrix[0].length;
  const means = Array.from({ length: cols }, (_, c) => matrix.reduce((sum, row) => sum + row[c], 0) / matrix.length);
  const stds = means.map((mean, c) => {
    const variance = matrix.reduce((sum, row) => sum + (row[c] - mean) ** 2, 0) / Math.max(1, matrix.length - 1);
    return Math.sqrt(variance) || 1;
  });
  return matrix.map((row) => row.map((value, c) => (value - means[c]) / stds[c]));
}

function covariance(matrix: number[][]) {
  const n = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  return Array.from({ length: cols }, (_, i) =>
    Array.from({ length: cols }, (_, j) => matrix.reduce((sum, row) => sum + row[i] * row[j], 0) / Math.max(1, n - 1))
  );
}

function dot(a: number[], b: number[]) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function normalize(vector: number[]) {
  const norm = Math.sqrt(dot(vector, vector)) || 1;
  return vector.map((value) => value / norm);
}

function matVec(matrix: number[][], vector: number[]) {
  return matrix.map((row) => dot(row, vector));
}

function powerIteration(matrix: number[][], seedOffset: number) {
  let vector = normalize(Array.from({ length: matrix.length }, (_, index) => 1 + ((index + seedOffset) % 3)));
  for (let i = 0; i < 64; i += 1) {
    vector = normalize(matVec(matrix, vector));
  }
  return vector;
}

function deflate(matrix: number[][], vector: number[]) {
  const lambda = dot(vector, matVec(matrix, vector));
  return matrix.map((row, i) => row.map((value, j) => value - lambda * vector[i] * vector[j]));
}

export function pcaProvinceScatter(vectors: ProvinceVector[]) {
  const matrix = vectors.map((row) => vectorKeys.map((key) => Number(row[key]) || 0));
  const scaled = standardize(matrix);
  const cov = covariance(scaled);
  const pc1 = powerIteration(cov, 0);
  const pc2 = powerIteration(deflate(cov, pc1), 1);
  return vectors.map((row, index) => ({
    province: row.province,
    x: dot(scaled[index], pc1),
    y: dot(scaled[index], pc2),
    cluster: row.cluster_label ?? 0,
    posts: row.total_posts_all,
    intensity: row.emotional_intensity_mean
  }));
}

/** Find which emotion has the highest mean in a row (supports both ProvinceVector and ProvinceWeek). */
export function getDominantEmotion<T extends Record<string, number>>(row: T, keyPrefix = ""): EmotionKey {
  return EMOTIONS.reduce((best, e) =>
    Number(row[`${keyPrefix}${e}_mean` as keyof T] ?? 0) > Number(row[`${keyPrefix}${best}_mean` as keyof T] ?? 0) ? e : best
  , "neutral" as EmotionKey);
}

export function clusterSummaries(vectors: ProvinceVector[], labels: ClusterLabel[]) {
  const labelMap = new Map(labels.map((row) => [row.province, row.cluster_label]));
  const groups = new Map<number, ProvinceVector[]>();
  for (const row of vectors) {
    const label = row.cluster_label ?? labelMap.get(row.province) ?? 0;
    groups.set(label, [...(groups.get(label) ?? []), row]);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => a - b)
    .map(([cluster, rows]) => {
      const emotionMeans = Object.fromEntries(
        EMOTIONS.map((emotion) => [
          emotion,
          rows.reduce((sum, row) => sum + Number(row[`${emotion}_mean_all` as keyof ProvinceVector] ?? 0), 0) / rows.length
        ])
      ) as Record<EmotionKey, number>;
      const dominant = [...EMOTIONS].sort((a, b) => emotionMeans[b] - emotionMeans[a])[0];
      return {
        cluster,
        provinces: rows.map((row) => row.province).sort(),
        posts: rows.reduce((sum, row) => sum + row.total_posts_all, 0),
        dominant,
        emotionMeans,
        intensity: rows.reduce((sum, row) => sum + row.emotional_intensity_mean, 0) / rows.length
      };
    });
}

export function eventDescription(event: AnomalyEvent) {
  const provinces = event.top_provinces.map((item) => String(item.province)).filter(Boolean).slice(0, 5);
  const match = event.deviation_pct.match(/([+-]?\d+(\.\d+)?)/);
  const abs = match ? Math.abs(parseFloat(match[1])) : 0;
  return `${dateWeekToShortRange(event.date_week)}，${event.emotion}情绪出现异常，比过去4周${abs > 0 ? "高出" : "低"}${abs.toFixed(0)}%，主要关联 ${provinces.join("、") || "暂无省份"}`;
}
