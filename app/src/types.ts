import type { EmotionKey } from "./config";
import type { NlpData } from "./types/nlp";

export type NumericEmotionFields = Record<`${EmotionKey}_mean`, number>;

export interface NationalWeek {
  date_week: string;
  total_posts: number;
  emotional_intensity: number;
  dominant_emotion_key: EmotionKey;
  dominant_emotion: string;
  positive_index: number;
  fear_joy_ratio: number;
  joy_mean: number;
  sadness_mean: number;
  anger_mean: number;
  fear_mean: number;
  surprise_mean: number;
  neutral_mean: number;
}

export interface ProvinceWeek extends NumericEmotionFields {
  date_week: string;
  province: string;
  total_posts: number;
  avg_word_count: number;
  dominant_emotion_key: EmotionKey;
  dominant_emotion: string;
  dominant_score: number;
  positive_index: number;
  emotional_intensity: number;
  fear_joy_ratio: number;
  reliable: boolean;
}

export interface ProvinceMonth extends NumericEmotionFields {
  date_month: string;
  province: string;
  total_posts: number;
  emotional_intensity: number;
  dominant_emotion_key: EmotionKey;
  dominant_emotion: string;
  dominant_score: number;
  reliable: boolean;
}

export interface ProvinceVector {
  province: string;
  total_posts_all: number;
  joy_mean_all: number;
  sadness_mean_all: number;
  anger_mean_all: number;
  fear_mean_all: number;
  surprise_mean_all: number;
  neutral_mean_all: number;
  emotional_intensity_mean: number;
  fear_variance: number;
  joy_variance: number;
  cluster_label?: number;
}

export interface ClusterLabel {
  province: string;
  total_posts_all: number;
  cluster_label: number;
}

export interface AnomalyEvent {
  date_week: string;
  emotion: EmotionKey;
  z_score: number;
  national_value: number;
  expected_value: number;
  deviation_pct: string;
  severity: "moderate" | "severe" | "extreme" | string;
  top_provinces: Array<Record<string, string | number>>;
}

export interface PostExample {
  post_id: string;
  date_week: string;
  date_month: string;
  province: string;
  content: string;
  score: number;
  scores: Record<EmotionKey, number>;
}

export interface PostExamplesPayload {
  generated_at: string;
  source?: string;
  emotions: EmotionKey[];
  provinces: Record<string, Partial<Record<EmotionKey, PostExample[]>>>;
}

export interface MonthlyClusterMatrix {
  months: string[];
  rows: Array<{
    province: string;
    values: number[];
  }>;
}

export interface RealtimeTopic {
  rank: number;
  title: string;
  hot_value: number;
  emotion: Record<EmotionKey, number>;
  dominant_emotion: EmotionKey;
  dominant_emotion_cn: string;
  dominant_score: number;
  emotional_intensity: number;
  emotional_entropy: number;
}

export interface RealtimeHotsearchSnapshot {
  fetch_time: string;
  fetch_time_str: string;
  source: string;
  total_topics: number;
  aggregate_emotion: Record<EmotionKey, number>;
  aggregate_dominant: EmotionKey;
  aggregate_dominant_cn: string;
  topics: RealtimeTopic[];
}

export interface DataBundle {
  nationalWeeks: NationalWeek[];
  provinceWeeks: ProvinceWeek[];
  provinceMonths: ProvinceMonth[];
  provinceVectors: ProvinceVector[];
  anomalies: AnomalyEvent[];
  clusterLabels: ClusterLabel[];
  monthlyClusters: MonthlyClusterMatrix;
  postExamples: PostExamplesPayload;
  chinaGeoJson: unknown;
  nlp: NlpData;
  realtime: RealtimeHotsearchSnapshot | null;
}
