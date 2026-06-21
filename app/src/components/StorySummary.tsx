import type { NationalWeek, ProvinceVector } from "../types";
import { EMOTION_META, type EmotionKey } from "../config";
import { getDominantEmotion } from "../utils/analytics";
import { dateWeekToShortRange } from "../utils/dateUtils";

interface StorySummaryProps {
  nationalWeeks: NationalWeek[];
  provinceVectors: ProvinceVector[];
}

function getTrendText(current: number, previous: number): string {
  const delta = current - previous;
  if (delta > 0.05) return "明显升高";
  if (delta > 0.02) return "小幅升高";
  if (delta > -0.02) return "基本稳定";
  if (delta > -0.05) return "小幅下降";
  return "明显下降";
}

function getMostVolatileProvince(vectors: ProvinceVector[]): string | null {
  if (!vectors.length) return null;
  // Find province with highest emotional intensity
  const sorted = [...vectors].sort((a, b) => b.emotional_intensity_mean - a.emotional_intensity_mean);
  return sorted[0]?.province ?? null;
}

export function StorySummary({ nationalWeeks, provinceVectors }: StorySummaryProps) {
  if (nationalWeeks.length < 2) return null;

  const current = nationalWeeks[nationalWeeks.length - 1];
  const previous = nationalWeeks[nationalWeeks.length - 2];
  const dominant = getDominantEmotion(current);
  const trend = getTrendText(current.emotional_intensity, previous.emotional_intensity);
  const volatileProvince = getMostVolatileProvince(provinceVectors);

  const intensityPct = (current.emotional_intensity * 100).toFixed(1);

  const story = `${dateWeekToShortRange(current.date_week)}，全国情绪温度为 ${intensityPct}%，主导情绪为${EMOTION_META[dominant].label}，较前一周${trend}${volatileProvince ? `，${volatileProvince}出现最高情绪波动` : ""}。`;

  return (
    <p style={{
      margin: "16px 0 0",
      color: "var(--text-secondary)",
      fontSize: 15,
      lineHeight: 1.65,
      maxWidth: 600
    }}>
      {story}
    </p>
  );
}
