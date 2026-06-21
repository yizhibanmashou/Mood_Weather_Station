import { useMemo, useState, useCallback, useEffect, useRef } from "react";
import ReactEChartsCore from "echarts-for-react/lib/core";
import { EMOTION_META, type EmotionKey } from "../config";
import { echarts } from "../utils/echartsSetup";
import type { NlpKeyword, NlpKeywordsByWeek, NlpEmotionKeywords, NlpWeekKeywords } from "../types/nlp";
import { EmptyState } from "./StateViews";
import styles from "./NlpPanel.module.css";

interface NlpPanelProps {
  weekKey: string;
  emotion?: EmotionKey;
  nlpData: NlpKeywordsByWeek | null;
  emotionKeywords?: NlpEmotionKeywords | null;
  onKeywordClick?: (word: string) => void;
}

type KeywordMode = "top" | "frequent" | "surge";
type EmotionFilterMode = "all" | "current";

function getWordCloudColor(keyword: NlpKeyword, index: number): string {
  if (keyword.surge) return "#E07A3A";
  const coolColors = ["#5F8FA8", "#7D6AAE", "#6B8E9B", "#8B7D6B", "#9B8B7B", "#7B9B8B"];
  return coolColors[index % coolColors.length];
}

export function NlpPanel({ weekKey, emotion, nlpData, emotionKeywords, onKeywordClick }: NlpPanelProps) {
  const [keywordMode, setKeywordMode] = useState<KeywordMode>("top");
  const [emotionFilter, setEmotionFilter] = useState<EmotionFilterMode>("all");
  const [selectedWord, setSelectedWord] = useState<string | null>(null);

  const weekData: NlpWeekKeywords | null = useMemo(() => {
    if (!nlpData?.weeks?.[weekKey]) return null;
    return nlpData.weeks[weekKey];
  }, [nlpData, weekKey]);

  // Core: compute displayedKeywords based on keywordMode + emotionFilter
  const displayedKeywords = useMemo(() => {
    if (!weekData) return [];

    // Emotion filter mode: use nlp_emotion_keywords
    if (emotionFilter === "current" && emotion && emotionKeywords?.emotions?.[emotion]) {
      const emotionKws = emotionKeywords.emotions[emotion]!;
      // Map to NlpKeyword-like objects, intersecting with current week
      const weekWordSet = new Set((weekData.keywords || []).map((k) => k.word));
      const matched = emotionKws
        .filter((ek) => ek.peak_week === weekKey || weekWordSet.has(ek.word))
        .map((ek) => {
          const original = weekData.keywords?.find((k) => k.word === ek.word);
          return original ?? {
            word: ek.word,
            tf: 0,
            tfidf: ek.peak_tfidf,
            global_tfidf: 0,
            surge: false,
            surge_ratio: 1,
            pos: "",
          };
        });
      return matched.slice(0, 30);
    }

    // Keyword mode — use pre-grouped fields from backend when available
    switch (keywordMode) {
      case "frequent":
        return (weekData.frequent_keywords || weekData.keywords || []).slice(0, 30);
      case "surge":
        return (weekData.surge_keywords || weekData.keywords || []).filter((kw) => kw.surge).slice(0, 30);
      case "top":
      default:
        return (weekData.top_keywords || weekData.keywords || []).slice(0, 30);
    }
  }, [weekData, keywordMode, emotionFilter, emotion, emotionKeywords, weekKey]);

  // Clear selectedWord if it's not in the new displayedKeywords
  useEffect(() => {
    if (selectedWord && !displayedKeywords.some((kw) => kw.word === selectedWord)) {
      setSelectedWord(null);
    }
  }, [displayedKeywords, selectedWord]);

  const top20Keywords = useMemo(() => {
    return displayedKeywords.slice(0, 20);
  }, [displayedKeywords]);

  const handleWordClick = useCallback(
    (word: string) => {
      setSelectedWord((prev) => (prev === word ? null : word));
      onKeywordClick?.(word);
    },
    [onKeywordClick]
  );

  // Empty states
  if (!nlpData) {
    return (
      <div className={styles.panel}>
        <EmptyState title="暂未生成关键词数据" detail="NLP 关键词分析数据尚未生成，请运行 04b_nlp_keywords.py 脚本" />
      </div>
    );
  }

  if (!weekData || weekData.status === "no_data") {
    return (
      <div className={styles.panel}>
        <EmptyState title="暂无本周关键词数据" detail={`周 ${weekKey} 没有可用的文本数据`} />
      </div>
    );
  }

  if (weekData.status === "insufficient_data") {
    return (
      <div className={styles.panel}>
        <EmptyState title="数据不足，暂不生成词云" detail={`本周仅有 ${weekData.total_posts} 条微博，少于最低要求 ${weekData.min_required ?? 50} 条`} />
      </div>
    );
  }

  const hasKeywords = displayedKeywords.length > 0;

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div>
          <h3 className={styles.title}>这一周大家在讨论什么</h3>
          <p className={styles.subtitle}>基于本周微博文本的 TF-IDF 关键词分析</p>
        </div>
        <div className={styles.controls}>
          <div className={styles.segmented}>
            <button className={keywordMode === "top" ? styles.activeBtn : ""} onClick={() => setKeywordMode("top")}>
              代表词
            </button>
            <button className={keywordMode === "surge" ? styles.activeBtn : ""} onClick={() => setKeywordMode("surge")}>
              飙升词
            </button>
            <button className={keywordMode === "frequent" ? styles.activeBtn : ""} onClick={() => setKeywordMode("frequent")}>
              高频词
            </button>
          </div>
          <div className={styles.segmented}>
            <button className={emotionFilter === "all" ? styles.activeBtn : ""} onClick={() => setEmotionFilter("all")}>
              全部情绪
            </button>
            {emotion && (
              <button className={emotionFilter === "current" ? styles.activeBtn : ""} onClick={() => setEmotionFilter("current")}>
                {EMOTION_META[emotion].label}相关
              </button>
            )}
          </div>
        </div>
      </div>

      <div className={styles.content}>
        <div className={styles.wordcloudSection}>
          {hasKeywords ? (
            <WordCloudChart keywords={displayedKeywords} selectedWord={selectedWord} onWordClick={handleWordClick} />
          ) : (
            <EmptyState
              title={keywordMode === "surge" ? "本周暂无明显飙升词" : emotionFilter === "current" ? `暂无${EMOTION_META[emotion!]?.label ?? ""}相关关键词` : "暂无可展示关键词"}
              detail={keywordMode === "surge" ? "本周关键词相比全年平均水平没有显著飙升" : "当前筛选条件下没有关键词"}
            />
          )}
        </div>

        <div className={styles.rankSection}>
          <h4 className={styles.rankTitle}>
            {keywordMode === "surge"
              ? "本周飙升词 Top 20"
              : keywordMode === "frequent"
                ? "本周高频词 Top 20"
                : emotionFilter === "current"
                  ? `${EMOTION_META[emotion!]?.label ?? ""}相关关键词 Top 20`
                  : "本周代表词 Top 20"}
          </h4>
          <div className={styles.rankList}>
            {top20Keywords.length > 0 ? (
              top20Keywords.map((kw, idx) => (
                <div
                  key={kw.word}
                  className={`${styles.rankItem} ${selectedWord === kw.word ? styles.rankItemSelected : ""}`}
                  onClick={() => handleWordClick(kw.word)}
                >
                  <span className={styles.rankIndex}>{idx + 1}</span>
                  <span className={styles.rankWord}>{kw.word}</span>
                  <div className={styles.rankBar}>
                    <div
                      className={styles.rankBarFill}
                      style={{
                        width: `${(kw.tfidf / (top20Keywords[0]?.tfidf || 1)) * 100}%`,
                        background: kw.surge ? "linear-gradient(90deg, #E07A3A, #F2A23A)" : "linear-gradient(90deg, #5F8FA8, #7D6AAE)",
                      }}
                    />
                  </div>
                  <span className={styles.rankValue}>
                    {kw.surge && <span className={styles.surgeBadge}>飙升</span>}
                    {kw.surge_ratio.toFixed(1)}x
                  </span>
                </div>
              ))
            ) : (
              <EmptyState title="暂无匹配关键词" detail="当前筛选条件下没有关键词" />
            )}
          </div>
        </div>
      </div>

      <div className={styles.footer}>
        <p className={styles.note}>
          {keywordMode === "surge"
            ? "飙升词：相比全年平均水平更突出的词，可能解释本周异常波动。飙升倍数表示本周代表性相对全年平均水平的倍数。"
            : keywordMode === "frequent"
              ? "高频词：本周微博中出现频率最高的词，反映本周讨论热点。"
              : "关键词由中文分词和 TF-IDF 提取，仅用于解释异常趋势。TF-IDF 用于衡量一个词在本周是否比平时更有代表性。"}
        </p>
      </div>
    </div>
  );
}

/* ──────────────────────────────────────────────
   WordCloud with error boundary + safe lifecycle
   ────────────────────────────────────────────── */

interface WordCloudChartProps {
  keywords: NlpKeyword[];
  selectedWord: string | null;
  onWordClick: (word: string) => void;
}

function WordCloudChart({ keywords, selectedWord, onWordClick }: WordCloudChartProps) {
  const chartRef = useRef<any>(null);
  const [hasError, setHasError] = useState(false);

  const baseOption = useMemo(() => {
    const data = keywords.slice(0, 60).map((kw, idx) => ({
      name: kw.word,
      value: kw.tfidf,
      textStyle: { color: getWordCloudColor(kw, idx) },
    }));

    return {
      tooltip: {
        show: true,
        formatter: (params: { name?: string }) => {
          const word = params.name ?? "";
          const kw = keywords.find((k) => k.word === word);
          if (!kw) return word;
          return [
            `<strong>${kw.word}</strong>`,
            `词频: ${kw.tf}`,
            `TF-IDF: ${kw.tfidf.toFixed(4)}`,
            `全局均值: ${kw.global_tfidf.toFixed(4)}`,
            `飙升倍数: ${kw.surge_ratio.toFixed(1)}x`,
          ].join("<br/>");
        },
      },
      series: [
        {
          type: "wordCloud" as const,
          shape: "circle",
          left: "center",
          top: "center",
          width: "90%",
          height: "90%",
          sizeRange: [14, 46],
          rotationRange: [-20, 20],
          rotationStep: 10,
          gridSize: 8,
          drawOutOfBound: false,
          layoutAnimation: true,
          textStyle: { fontFamily: '"PingFang SC", "Microsoft YaHei", sans-serif' },
          emphasis: {
            focus: "self",
            textStyle: { textShadowBlur: 3, textShadowColor: "rgba(0,0,0,0.15)" },
          },
          data,
        },
      ],
    };
  }, [keywords]);

  useEffect(() => {
    if (hasError) return;
    const instance = chartRef.current?.getEchartsInstance?.();
    if (!instance || instance.isDisposed()) return;

    try {
      const data = keywords.slice(0, 60).map((kw, idx) => ({
        name: kw.word,
        value: kw.tfidf,
        textStyle: {
          color: selectedWord === kw.word ? "#E07A3A" : getWordCloudColor(kw, idx),
          fontWeight: selectedWord === kw.word ? "bold" : "normal",
        },
      }));
      instance.setOption({ series: [{ data }] }, { replaceMerge: ["series"] });
    } catch {
      // Silently ignore
    }
  }, [selectedWord, keywords, hasError]);

  const onEvents = useMemo(
    () => ({
      click: (params: { name?: string }) => {
        if (params.name) onWordClick(params.name);
      },
    }),
    [onWordClick]
  );

  if (hasError) {
    return <TagCloudFallback keywords={keywords} selectedWord={selectedWord} onWordClick={onWordClick} />;
  }

  return (
    <WordCloudErrorBoundary onError={() => setHasError(true)}>
      <ReactEChartsCore echarts={echarts} ref={chartRef} option={baseOption} onEvents={onEvents} style={{ height: "100%", minHeight: 320 }} opts={{ renderer: "canvas" }} notMerge={false} lazyUpdate />
    </WordCloudErrorBoundary>
  );
}

/* ──────────────────────────────────────────────
   Error boundary
   ────────────────────────────────────────────── */

import { Component, type ReactNode, type ErrorInfo } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
  onError?: () => void;
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

class WordCloudErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.warn("[WordCloudErrorBoundary] Caught error:", error.message, info.componentStack);
    this.props.onError?.();
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? null;
    }
    return this.props.children;
  }
}

/* ──────────────────────────────────────────────
   HTML Tag Cloud fallback
   ────────────────────────────────────────────── */

interface TagCloudFallbackProps {
  keywords: NlpKeyword[];
  selectedWord: string | null;
  onWordClick: (word: string) => void;
}

function TagCloudFallback({ keywords, selectedWord, onWordClick }: TagCloudFallbackProps) {
  const maxTfidf = keywords[0]?.tfidf || 1;

  return (
    <div className={styles.tagCloud}>
      {keywords.slice(0, 60).map((kw, idx) => {
        const size = 14 + (kw.tfidf / maxTfidf) * 28;
        const color = selectedWord === kw.word ? "#E07A3A" : getWordCloudColor(kw, idx);
        return (
          <span
            key={kw.word}
            className={`${styles.tagItem} ${selectedWord === kw.word ? styles.tagItemSelected : ""}`}
            style={{ fontSize: `${size}px`, color }}
            onClick={() => onWordClick(kw.word)}
            title={`${kw.word} | TF-IDF: ${kw.tfidf.toFixed(4)} | 飙升: ${kw.surge_ratio.toFixed(1)}x`}
          >
            {kw.word}
          </span>
        );
      })}
    </div>
  );
}
