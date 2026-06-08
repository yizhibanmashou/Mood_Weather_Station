import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { PROVINCE_IMAGE_POSITIONS, type ProvinceImagePos } from "../data/provinceImagePositions";
import { EMOTIONS, EMOTION_META, type EmotionKey } from "../config";
import type { ProvinceMonth, ProvinceVector } from "../types";
import { formatPct, formatNumber } from "../utils/analytics";
import { dateMonthToChinese } from "../utils/dateUtils";
import styles from "./StaticImageBubbleMap.module.css";

export type MapMetric = "emotional_intensity" | EmotionKey | "dominant";

interface StaticImageBubbleMapProps {
  vectors: ProvinceVector[];
  monthlyData: ProvinceMonth[];
  selectedMetric: MapMetric;
  onProvinceClick?: (province: string) => void;
  height?: number;
}

const DRAFT_KEY = "mws_province_image_positions_draft";
const ENABLE_MAP_POSITION_EDITOR = false;

function getVal(row: ProvinceVector | ProvinceMonth, emotion: EmotionKey): number {
  if ("joy_mean_all" in row) return (row as ProvinceVector)[`${emotion}_mean_all`];
  return (row as ProvinceMonth)[`${emotion}_mean`];
}

function getDominantEmotion(row: ProvinceVector | ProvinceMonth): EmotionKey {
  return EMOTIONS.reduce((best, e) => {
    return getVal(row, e) > getVal(row, best) ? e : best;
  }, "neutral" as EmotionKey);
}

function getMetricValue(row: ProvinceVector | ProvinceMonth, metric: MapMetric): number {
  if (metric === "emotional_intensity") {
    return "emotional_intensity_mean" in row
      ? (row as ProvinceVector).emotional_intensity_mean
      : (row as ProvinceMonth).emotional_intensity;
  }
  if (metric === "dominant") return 0;
  return getVal(row, metric as EmotionKey);
}

function getColorForValue(value: number, metric: MapMetric): string {
  if (metric === "dominant") return EMOTION_META.neutral.color;
  if (value <= 0.15) return "#D6E6F9";
  if (value <= 0.30) return "#7EB3E8";
  if (value <= 0.45) return "#3B7DD8";
  if (value <= 0.60) return "#2A5EA8";
  return "#1A3D78";
}


function loadDraft(): Record<string, ProvinceImagePos> | null {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return null;
}

function saveDraft(positions: Record<string, ProvinceImagePos>) {
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(positions));
  } catch { /* ignore */ }
}

function clearDraft() {
  try {
    localStorage.removeItem(DRAFT_KEY);
  } catch { /* ignore */ }
}

function generateTsCode(positions: Record<string, ProvinceImagePos>): string {
  const lines = ["export const PROVINCE_IMAGE_POSITIONS: Record<string, ProvinceImagePos> = {"];
  for (const [province, pos] of Object.entries(positions)) {
    const x = pos.x.toFixed(1);
    const y = pos.y.toFixed(1);
    if (pos.labelOffset) {
      lines.push(`  ${province}:   { x: ${x}, y: ${y}, labelOffset: [${pos.labelOffset[0]}, ${pos.labelOffset[1]}] },`);
    } else {
      lines.push(`  ${province}:   { x: ${x}, y: ${y} },`);
    }
  }
  lines.push("};");
  return lines.join("\n");
}

export function StaticImageBubbleMap({
  vectors,
  monthlyData,
  selectedMetric,
  onProvinceClick,
  height = 520,
}: StaticImageBubbleMapProps) {
  const imgWrapRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [imgError, setImgError] = useState(false);
  const [selectedMonth, setSelectedMonth] = useState("");
  const [hoveredProvince, setHoveredProvince] = useState<string | null>(null);

  // Editor state
  const [editorPositions, setEditorPositions] = useState<Record<string, ProvinceImagePos>>(() => {
    if (!ENABLE_MAP_POSITION_EDITOR) return PROVINCE_IMAGE_POSITIONS;
    return loadDraft() ?? { ...PROVINCE_IMAGE_POSITIONS };
  });
  const [selectedProvince, setSelectedProvince] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [panelPos, setPanelPos] = useState<{ x: number; y: number } | null>(null);
  const panelDragRef = useRef<{ startX: number; startY: number; startPosX: number; startPosY: number } | null>(null);

  const availableMonths = useMemo(
    () => [...new Set(monthlyData.filter((r) => r.reliable).map((r) => r.date_month))].sort(),
    [monthlyData],
  );
  const effectiveMonth =
    selectedMonth || availableMonths[availableMonths.length - 1] || "";
  const monthData = effectiveMonth
    ? monthlyData.filter((r) => r.date_month === effectiveMonth && r.reliable)
    : [];
  const dataSource: (ProvinceVector | ProvinceMonth)[] =
    monthData.length > 0 ? monthData : vectors;
  const isMonthly = monthData.length > 0;

  const positions = ENABLE_MAP_POSITION_EDITOR ? editorPositions : PROVINCE_IMAGE_POSITIONS;

  // Save draft on change
  useEffect(() => {
    if (ENABLE_MAP_POSITION_EDITOR) {
      saveDraft(editorPositions);
    }
  }, [editorPositions]);

  // Keyboard nudge
  useEffect(() => {
    if (!ENABLE_MAP_POSITION_EDITOR || !selectedProvince) return;
    const handler = (e: KeyboardEvent) => {
      if (!selectedProvince) return;
      const step = e.shiftKey ? 1.0 : 0.2;
      let dx = 0, dy = 0;
      if (e.key === "ArrowLeft") dx = -step;
      else if (e.key === "ArrowRight") dx = step;
      else if (e.key === "ArrowUp") dy = -step;
      else if (e.key === "ArrowDown") dy = step;
      else if (e.key === "Escape" || e.key === "Delete") {
        setSelectedProvince(null);
        return;
      } else return;
      e.preventDefault();
      setEditorPositions((prev) => {
        const cur = prev[selectedProvince];
        if (!cur) return prev;
        return {
          ...prev,
          [selectedProvince]: {
            ...cur,
            x: Math.round((cur.x + dx) * 10) / 10,
            y: Math.round((cur.y + dy) * 10) / 10,
          },
        };
      });
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedProvince]);

  // Log missing positions
  useEffect(() => {
    for (const row of dataSource) {
      if (!positions[row.province]) {
        console.warn(`[StaticImageBubbleMap] 缺少省份图片坐标: ${row.province}`);
      }
    }
  }, [dataSource, positions]);

  const bubbles = useMemo(() => {
    return dataSource
      .map((row) => {
        const pos = positions[row.province];
        if (!pos) return null;
        const posts = isMonthly
          ? (row as ProvinceMonth).total_posts
          : (row as ProvinceVector).total_posts_all;
        const dominant = getDominantEmotion(row);
        let value: number;
        let color: string;
        if (selectedMetric === "dominant") {
          value = 0.5;
          color = EMOTION_META[dominant].color;
        } else {
          value = getMetricValue(row, selectedMetric);
          color = getColorForValue(value, selectedMetric);
        }
        const size = Math.max(14, Math.min(44, Math.sqrt(posts) * 2.0));
        const shortName = row.province
          .replace(/壮族|回族|维吾尔|自治区|省|市/g, "")
          .slice(0, 2);
        return {
          province: row.province,
          shortName,
          x: pos.x,
          y: pos.y,
          size,
          color,
          posts,
          dominant,
          dominantLabel: EMOTION_META[dominant]?.label ?? dominant,
          dominantColor: EMOTION_META[dominant]?.color ?? "#999",
          metricValue: value,
          labelOffset: pos.labelOffset,
          showLabel: true,
        };
      })
      .filter(Boolean);
  }, [dataSource, selectedMetric, isMonthly, positions]);

  const showTooltip = useCallback(
    (b: (typeof bubbles)[number]) => {
      if (!b || !tooltipRef.current || !imgWrapRef.current) return;
      const monthLabel = isMonthly ? dateMonthToChinese(effectiveMonth) : "全量";
      const metricLabel =
        selectedMetric === "emotional_intensity" ? "情绪温度"
        : selectedMetric === "dominant" ? "主导情绪"
        : EMOTION_META[selectedMetric as EmotionKey]?.label ?? selectedMetric;

      let html = `<div class="${styles.tooltipTitle}">${b.province}</div>`;
      html += `<div class="${styles.tooltipSub}">${monthLabel}</div>`;
      html += `<div class="${styles.tooltipRow}">样本量：${formatNumber(b.posts)}</div>`;
      html += `<div class="${styles.tooltipRow}">主导情绪：<span style="color:${b.dominantColor}">${b.dominantLabel}</span></div>`;
      if (selectedMetric !== "dominant") {
        html += `<div class="${styles.tooltipRow}">${metricLabel}：${formatPct(b.metricValue)}</div>`;
      }
      tooltipRef.current.innerHTML = html;
      tooltipRef.current.style.display = "block";

      const wrapW = imgWrapRef.current.offsetWidth;
      const wrapH = imgWrapRef.current.offsetHeight;
      const bx = (b.x / 100) * wrapW;
      const by = (b.y / 100) * wrapH;
      const tipW = 210;
      const tipH = 130;
      const offset = b.size / 2 + 8;
      let left = bx + offset + tipW < wrapW ? bx + offset : bx - offset - tipW;
      let top = by - tipH > 0 ? by - tipH : by + offset;
      left = Math.max(4, Math.min(left, wrapW - tipW - 4));
      top = Math.max(4, Math.min(top, wrapH - tipH - 4));
      tooltipRef.current.style.left = `${left}px`;
      tooltipRef.current.style.top = `${top}px`;
    },
    [effectiveMonth, isMonthly, selectedMetric],
  );

  function hideTooltip() {
    if (tooltipRef.current) tooltipRef.current.style.display = "none";
  }

  function handleClick(province: string) {
    if (ENABLE_MAP_POSITION_EDITOR) return;
    onProvinceClick?.(province);
  }

  // Drag handlers
  function handlePointerDown(e: React.PointerEvent, province: string) {
    if (!ENABLE_MAP_POSITION_EDITOR) return;
    e.preventDefault();
    e.stopPropagation();
    setSelectedProvince(province);
    setIsDragging(true);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }

  function handlePointerMove(e: React.PointerEvent) {
    if (!isDragging || !selectedProvince || !imgWrapRef.current) return;
    const rect = imgWrapRef.current.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * 1000) / 10;
    const y = Math.round(((e.clientY - rect.top) / rect.height) * 1000) / 10;
    const clampedX = Math.max(0, Math.min(100, x));
    const clampedY = Math.max(0, Math.min(100, y));
    setEditorPositions((prev) => {
      const cur = prev[selectedProvince];
      if (!cur) return prev;
      return { ...prev, [selectedProvince]: { ...cur, x: clampedX, y: clampedY } };
    });
  }

  function handlePointerUp() {
    setIsDragging(false);
  }

  function handleCopyTs() {
    const code = generateTsCode(editorPositions);
    navigator.clipboard.writeText(code).then(() => {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }).catch(() => {
      // Fallback: select + copy
      const ta = document.createElement("textarea");
      ta.value = code;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    });
  }

  function handleResetDraft() {
    clearDraft();
    setEditorPositions({ ...PROVINCE_IMAGE_POSITIONS });
    setSelectedProvince(null);
  }

  function handlePanelDragStart(e: React.PointerEvent) {
    e.preventDefault();
    e.stopPropagation();
    const curX = panelPos?.x ?? 8;
    const curY = panelPos?.y ?? 8;
    panelDragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startPosX: curX,
      startPosY: curY,
    };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }

  function handlePanelDragMove(e: React.PointerEvent) {
    if (!panelDragRef.current) return;
    const dx = e.clientX - panelDragRef.current.startX;
    const dy = e.clientY - panelDragRef.current.startY;
    setPanelPos({
      x: panelDragRef.current.startPosX + dx,
      y: panelDragRef.current.startPosY + dy,
    });
  }

  function handlePanelDragEnd() {
    panelDragRef.current = null;
  }

  const selectedPos = selectedProvince ? editorPositions[selectedProvince] : null;

  if (imgError) {
    return (
      <div className={styles.imgError}>
        <div className={styles.imgErrorTitle}>地图图片加载失败</div>
        <div className={styles.imgErrorDetail}>
          无法加载 /maps/china_map.jpg，请确认文件存在于 app/public/maps/ 目录下。
        </div>
      </div>
    );
  }

  return (
    <div className={styles.outer}>
      {availableMonths.length > 1 && (
        <div className={styles.monthBar}>
          {availableMonths.map((m) => (
            <button
              key={m}
              onClick={() => setSelectedMonth(m)}
              className={`${styles.monthBtn} ${m === effectiveMonth ? styles.monthBtnActive : ""}`}
              type="button"
            >
              {dateMonthToChinese(m)}
            </button>
          ))}
        </div>
      )}
      <div
        ref={imgWrapRef}
        className={styles.mapImageWrap}
        style={{ maxHeight: height }}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      >
        <img
          src="/maps/china_map.jpg"
          alt="中国地图"
          className={styles.mapImg}
          onError={() => setImgError(true)}
          draggable={false}
        />
        <div className={styles.bubbleLayer}>
          {bubbles.map((b) =>
            b ? (
              <div key={b.province}>
                <div
                  className={`${styles.bubble} ${ENABLE_MAP_POSITION_EDITOR ? styles.bubbleEditable : ""} ${selectedProvince === b.province ? styles.bubbleSelected : ""}`}
                  style={{
                    left: `${b.x}%`,
                    top: `${b.y}%`,
                    width: b.size,
                    height: b.size,
                    background: b.color,
                  }}
                  onPointerDown={(e) => handlePointerDown(e, b.province)}
                  onMouseEnter={() => {
                    if (!isDragging) {
                      setHoveredProvince(b.province);
                      showTooltip(b);
                    }
                  }}
                  onMouseLeave={() => {
                    setHoveredProvince(null);
                    hideTooltip();
                  }}
                  onClick={() => handleClick(b.province)}
                  title={b.province}
                >
                  {b.size >= 20 && b.shortName}
                </div>
                {(b.showLabel || hoveredProvince === b.province) && (
                  <div
                    className={`${styles.bubbleLabel} ${selectedProvince === b.province ? styles.bubbleLabelSelected : ""}`}
                    style={{
                      left: b.labelOffset
                        ? `calc(${b.x}% + ${b.labelOffset[0]}px)`
                        : `${b.x}%`,
                      top: b.labelOffset
                        ? `calc(${b.y}% + ${b.labelOffset[1]}px)`
                        : `calc(${b.y}% + ${b.size / 2 + 10}px)`,
                    }}
                  >
                    {b.shortName}
                  </div>
                )}
              </div>
            ) : null,
          )}
        </div>
        <div ref={tooltipRef} className={styles.tooltip} />
      </div>

      {/* Editor panel */}
      {ENABLE_MAP_POSITION_EDITOR && (
        <div
          className={styles.editorPanel}
          style={panelPos ? { left: panelPos.x, top: panelPos.y, bottom: "auto", right: "auto" } : undefined}
          onPointerMove={panelDragRef.current ? handlePanelDragMove : undefined}
          onPointerUp={panelDragRef.current ? handlePanelDragEnd : undefined}
        >
          <div
            className={styles.editorDragHandle}
            onPointerDown={handlePanelDragStart}
          >&#x2630; 坐标编辑器</div>
          {selectedProvince && selectedPos ? (
            <>
              <div className={styles.editorRow}>
                <span className={styles.editorLabel}>省份</span>
                <span className={styles.editorValue}>{selectedProvince}</span>
              </div>
              <div className={styles.editorRow}>
                <span className={styles.editorLabel}>x</span>
                <span className={styles.editorValue}>{selectedPos.x.toFixed(1)}</span>
              </div>
              <div className={styles.editorRow}>
                <span className={styles.editorLabel}>y</span>
                <span className={styles.editorValue}>{selectedPos.y.toFixed(1)}</span>
              </div>
            </>
          ) : (
            <div className={styles.editorHint}>点击气泡选中，拖动调整位置</div>
          )}
          <div className={styles.editorRow}>
            <span className={styles.editorLabel}>月份</span>
            <span className={styles.editorValue}>{effectiveMonth || "-"}</span>
          </div>
          <div className={styles.editorActions}>
            <button className={styles.editorBtn} onClick={handleCopyTs} type="button">
              {copySuccess ? "已复制!" : "复制 TS 坐标"}
            </button>
            <button className={styles.editorBtnDanger} onClick={handleResetDraft} type="button">
              重置坐标
            </button>
          </div>
          {selectedProvince && (
            <div className={styles.editorHint}>方向键微调 0.2 · Shift+方向键 1.0 · Esc 取消</div>
          )}
        </div>
      )}

      <div className={styles.attribution}>
        地图仅用于课程项目可视化展示，行政边界以官方标准地图为准
      </div>
    </div>
  );
}
