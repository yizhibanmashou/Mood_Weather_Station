import Papa from "papaparse";
import { assetPath, EMOTIONS, type EmotionKey } from "../config";
import type {
  AnomalyEvent,
  ClusterLabel,
  DataBundle,
  MonthlyClusterMatrix,
  NationalWeek,
  PostExamplesPayload,
  ProvinceMonth,
  ProvinceVector,
  ProvinceWeek,
  RealtimeHotsearchSnapshot
} from "../types";
import type { NlpData, NlpKeywordsByWeek, NlpEmotionKeywords, NlpGlobalVocabulary } from "../types/nlp";
import { normalizeProvinceName } from "../utils/province";

type CsvRow = Record<string, string>;

let cache: Promise<DataBundle> | null = null;

const HISTORICAL_END_WEEK = "2020-W53";
const HISTORICAL_END_MONTH = "2020-12";

function toNumber(value: unknown) {
  const n = Number.parseFloat(String(value ?? "").trim());
  return Number.isFinite(n) ? n : 0;
}

function toBool(value: unknown) {
  return String(value ?? "").toLowerCase() === "true";
}

function toEmotion(value: unknown): EmotionKey {
  const key = String(value ?? "neutral").trim() as EmotionKey;
  return EMOTIONS.includes(key) ? key : "neutral";
}

function isHistoricalWeek(week: string | undefined) {
  return Boolean(week) && String(week) <= HISTORICAL_END_WEEK;
}

function isHistoricalMonth(month: string | undefined) {
  return Boolean(month) && String(month) <= HISTORICAL_END_MONTH;
}

async function fetchText(path: string) {
  const response = await fetch(assetPath(path));
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: ${response.status}`);
  }
  return response.text();
}

async function fetchJson<T>(path: string, fallback: T): Promise<T> {
  try {
    const response = await fetch(assetPath(path));
    if (!response.ok) return fallback;
    return (await response.json()) as T;
  } catch {
    return fallback;
  }
}

async function fetchCsv(path: string): Promise<CsvRow[]> {
  const text = await fetchText(path);
  const parsed = Papa.parse<CsvRow>(text, {
    header: true,
    skipEmptyLines: true,
    transformHeader: (header) => header.trim()
  });
  if (parsed.errors.length) {
    console.warn(`CSV parse warnings for ${path}`, parsed.errors);
  }
  return parsed.data.filter((row) => Object.values(row).some((value) => String(value ?? "").trim()));
}

function emotionMeans(row: CsvRow) {
  return {
    joy_mean: toNumber(row.joy_mean),
    sadness_mean: toNumber(row.sadness_mean),
    anger_mean: toNumber(row.anger_mean),
    fear_mean: toNumber(row.fear_mean),
    surprise_mean: toNumber(row.surprise_mean),
    neutral_mean: toNumber(row.neutral_mean)
  };
}

function parseNational(rows: CsvRow[]): NationalWeek[] {
  return rows.map((row) => ({
    date_week: row.date_week,
    total_posts: toNumber(row.total_posts),
    emotional_intensity: toNumber(row.emotional_intensity),
    dominant_emotion_key: toEmotion(row.dominant_emotion_key),
    dominant_emotion: row.dominant_emotion || "中性",
    positive_index: toNumber(row.positive_index),
    fear_joy_ratio: toNumber(row.fear_joy_ratio),
    ...emotionMeans(row)
  }));
}

function parseProvinceWeeks(rows: CsvRow[]): ProvinceWeek[] {
  return rows.map((row) => ({
    date_week: row.date_week,
    province: normalizeProvinceName(row.province),
    total_posts: toNumber(row.total_posts),
    avg_word_count: toNumber(row.avg_word_count),
    dominant_emotion_key: toEmotion(row.dominant_emotion_key),
    dominant_emotion: row.dominant_emotion || "中性",
    dominant_score: toNumber(row.dominant_score),
    positive_index: toNumber(row.positive_index),
    emotional_intensity: toNumber(row.emotional_intensity),
    fear_joy_ratio: toNumber(row.fear_joy_ratio),
    reliable: toBool(row.reliable),
    ...emotionMeans(row)
  }));
}

function parseProvinceMonths(rows: CsvRow[]): ProvinceMonth[] {
  return rows.map((row) => ({
    date_month: row.date_month,
    province: normalizeProvinceName(row.province),
    total_posts: toNumber(row.total_posts),
    emotional_intensity: toNumber(row.emotional_intensity),
    dominant_emotion_key: toEmotion(row.dominant_emotion_key),
    dominant_emotion: row.dominant_emotion || "中性",
    dominant_score: toNumber(row.dominant_score),
    reliable: toBool(row.reliable),
    ...emotionMeans(row)
  }));
}

function parseProvinceVectors(rows: CsvRow[], labels: ClusterLabel[]): ProvinceVector[] {
  const labelByProvince = new Map(labels.map((row) => [row.province, row.cluster_label]));
  return rows.map((row) => {
    const province = normalizeProvinceName(row.province);
    return {
      province,
      total_posts_all: toNumber(row.total_posts_all),
      joy_mean_all: toNumber(row.joy_mean_all),
      sadness_mean_all: toNumber(row.sadness_mean_all),
      anger_mean_all: toNumber(row.anger_mean_all),
      fear_mean_all: toNumber(row.fear_mean_all),
      surprise_mean_all: toNumber(row.surprise_mean_all),
      neutral_mean_all: toNumber(row.neutral_mean_all),
      emotional_intensity_mean: toNumber(row.emotional_intensity_mean),
      fear_variance: toNumber(row.fear_variance),
      joy_variance: toNumber(row.joy_variance),
      cluster_label: labelByProvince.get(province)
    };
  });
}

function buildProvinceVectorsFromWeeks(rows: ProvinceWeek[], labels: ClusterLabel[]): ProvinceVector[] {
  const labelByProvince = new Map(labels.map((row) => [row.province, row.cluster_label]));
  const groups = new Map<string, ProvinceWeek[]>();
  for (const row of rows) {
    groups.set(row.province, [...(groups.get(row.province) ?? []), row]);
  }

  return [...groups.entries()].map(([province, provinceRows]) => {
    const totalPosts = provinceRows.reduce((sum, row) => sum + row.total_posts, 0);
    const weightedMean = (selector: (row: ProvinceWeek) => number) =>
      provinceRows.reduce((sum, row) => sum + selector(row) * row.total_posts, 0) / Math.max(1, totalPosts);
    const joyMean = weightedMean((row) => row.joy_mean);
    const fearMean = weightedMean((row) => row.fear_mean);
    const weightedVariance = (selector: (row: ProvinceWeek) => number, mean: number) =>
      provinceRows.reduce((sum, row) => sum + (selector(row) - mean) ** 2 * row.total_posts, 0) / Math.max(1, totalPosts);

    return {
      province,
      total_posts_all: totalPosts,
      joy_mean_all: joyMean,
      sadness_mean_all: weightedMean((row) => row.sadness_mean),
      anger_mean_all: weightedMean((row) => row.anger_mean),
      fear_mean_all: fearMean,
      surprise_mean_all: weightedMean((row) => row.surprise_mean),
      neutral_mean_all: weightedMean((row) => row.neutral_mean),
      emotional_intensity_mean: weightedMean((row) => row.emotional_intensity),
      fear_variance: weightedVariance((row) => row.fear_mean, fearMean),
      joy_variance: weightedVariance((row) => row.joy_mean, joyMean),
      cluster_label: labelByProvince.get(province)
    };
  });
}

function parseClusterLabels(rows: CsvRow[]): ClusterLabel[] {
  return rows.map((row) => ({
    province: normalizeProvinceName(row.province),
    total_posts_all: toNumber(row.total_posts_all),
    cluster_label: Math.trunc(toNumber(row.cluster_label))
  }));
}

function parseMonthlyClusters(rows: CsvRow[]): MonthlyClusterMatrix {
  if (!rows.length) return { months: [], rows: [] };
  const keys = Object.keys(rows[0]);
  const provinceKey = keys.find((key) => !key || key.toLowerCase() === "province") ?? keys[0];
  const months = keys.filter((key) => key !== provinceKey && key.trim());
  return {
    months,
    rows: rows.map((row) => ({
      province: normalizeProvinceName(row[provinceKey]),
      values: months.map((month) => Math.trunc(toNumber(row[month])))
    }))
  };
}

function filterMonthlyClusters(matrix: MonthlyClusterMatrix): MonthlyClusterMatrix {
  const keepIndices = matrix.months
    .map((month, index) => ({ month, index }))
    .filter(({ month }) => isHistoricalMonth(month));
  return {
    months: keepIndices.map(({ month }) => month),
    rows: matrix.rows.map((row) => ({
      province: row.province,
      values: keepIndices.map(({ index }) => row.values[index] ?? -1)
    }))
  };
}

function filterNlpKeywords(data: NlpKeywordsByWeek | null): NlpKeywordsByWeek | null {
  if (!data) return null;
  return {
    ...data,
    weeks: Object.fromEntries(
      Object.entries(data.weeks).filter(([week]) => isHistoricalWeek(week))
    )
  };
}

function filterNlpEmotionKeywords(data: NlpEmotionKeywords | null): NlpEmotionKeywords | null {
  if (!data) return null;
  return {
    ...data,
    emotions: Object.fromEntries(
      Object.entries(data.emotions).map(([emotion, keywords]) => [
        emotion,
        (keywords ?? []).filter((keyword) => isHistoricalWeek(keyword.peak_week))
      ])
    ) as NlpEmotionKeywords["emotions"]
  };
}

function normalizePostExamples(payload: PostExamplesPayload): PostExamplesPayload {
  const provinces: PostExamplesPayload["provinces"] = {};
  for (const [province, byEmotion] of Object.entries(payload.provinces ?? {})) {
    provinces[normalizeProvinceName(province)] = byEmotion;
  }
  return {
    generated_at: payload.generated_at ?? "",
    source: payload.source,
    emotions: payload.emotions ?? [...EMOTIONS],
    provinces
  };
}

export function loadMoodData() {
  cache ??= loadMoodDataInner();
  return cache;
}

async function loadMoodDataInner(): Promise<DataBundle> {
  const [
    nationalRows,
    weekRows,
    monthRows,
    clusterRows,
    vectorRows,
    monthlyRows,
    anomalies,
    postExamples,
    chinaGeoJson,
    nlpKeywordsByWeek,
    nlpEmotionKeywords,
    nlpGlobalVocabulary,
    realtimeSnapshot
  ] = await Promise.all([
    fetchCsv("data/processed/emotion_national_timeline.csv").catch((e) => { throw new Error(`加载全国时序失败: ${e}`); }),
    fetchCsv("data/processed/emotion_panel_weekly.csv").catch((e) => { throw new Error(`加载周面板失败: ${e}`); }),
    fetchCsv("data/processed/emotion_panel_monthly.csv").catch((e) => { throw new Error(`加载月面板失败: ${e}`); }),
    fetchCsv("data/processed/cluster_labels.csv").catch((e) => { throw new Error(`加载聚类标签失败: ${e}`); }),
    fetchCsv("data/processed/province_emotion_vectors.csv").catch((e) => { throw new Error(`加载省份向量失败: ${e}`); }),
    fetchCsv("data/processed/monthly_cluster_labels.csv").catch((e) => { throw new Error(`加载月度聚类失败: ${e}`); }),
    fetchJson<AnomalyEvent[]>("data/processed/anomaly_detection.json", []),
    fetchJson<PostExamplesPayload>("data/processed/post_examples.json", {
      generated_at: "",
      emotions: [...EMOTIONS],
      provinces: {}
    }),
    fetchJson<unknown>("data/geo/china.json", null),
    fetchJson<NlpKeywordsByWeek | null>("data/processed/nlp_keywords_by_week.json", null),
    fetchJson<NlpEmotionKeywords | null>("data/processed/nlp_emotion_keywords.json", null),
    fetchJson<NlpGlobalVocabulary | null>("data/processed/nlp_global_vocabulary.json", null),
    fetchJson<RealtimeHotsearchSnapshot | null>("data/realtime/hotsearch_latest.json", null),
  ]);

  const clusterLabels = parseClusterLabels(clusterRows);
  const nationalWeeks = parseNational(nationalRows).filter((row) => isHistoricalWeek(row.date_week));
  const provinceWeeks = parseProvinceWeeks(weekRows).filter((row) => isHistoricalWeek(row.date_week));
  const provinceMonths = parseProvinceMonths(monthRows).filter((row) => isHistoricalMonth(row.date_month));
  const monthlyClusters = filterMonthlyClusters(parseMonthlyClusters(monthlyRows));
  const historicalProvinceVectors = buildProvinceVectorsFromWeeks(provinceWeeks, clusterLabels);

  return {
    nationalWeeks,
    provinceWeeks,
    provinceMonths,
    clusterLabels,
    provinceVectors: historicalProvinceVectors.length ? historicalProvinceVectors : parseProvinceVectors(vectorRows, clusterLabels),
    monthlyClusters,
    anomalies: (anomalies ?? []).filter((event) => isHistoricalWeek(event.date_week)),
    postExamples: normalizePostExamples(postExamples),
    chinaGeoJson,
    nlp: {
      keywordsByWeek: filterNlpKeywords(nlpKeywordsByWeek),
      emotionKeywords: filterNlpEmotionKeywords(nlpEmotionKeywords),
      globalVocabulary: nlpGlobalVocabulary,
    },
    realtime: realtimeSnapshot,
  };
}
