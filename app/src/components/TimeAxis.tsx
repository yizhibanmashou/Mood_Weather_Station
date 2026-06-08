import { useMemo } from "react";
import { motion } from "framer-motion";
import styles from "./TimeAxis.module.css";

interface TimeMarker {
  position: number; // 0–1
  label: string;
  sublabel?: string;
  color?: string;
}

interface TimeAxisProps {
  /** Full date range as [start, end] in "YYYY-Www" or "YYYY-MM-DD" format */
  fullRange: [string, string];
  /** Current viewport range as [start, end] */
  currentRange?: [string, string] | null;
  /** Key period markers */
  markers?: TimeMarker[];
}

function parseDateLabel(label: string): number {
  const wMatch = label.match(/^(\d{4})-W(\d{2})$/);
  if (wMatch) {
    const [_, y, w] = wMatch;
    return Number(y) + Number(w) / 53;
  }
  const dMatch = label.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (dMatch) {
    const [_, y, m, d] = dMatch;
    return Number(y) + (Number(m) - 1) / 12 + Number(d) / 365;
  }
  return 0;
}

function formatDateLabel(label: string): string {
  const wMatch = label.match(/^(\d{4})-W(\d{2})$/);
  if (wMatch) {
    return `${wMatch[1]}/${wMatch[2]}W`;
  }
  return label;
}

export function TimeAxis({ fullRange, currentRange, markers = [] }: TimeAxisProps) {
  const [startLabel, endLabel] = fullRange;
  const start = parseDateLabel(startLabel);
  const end = parseDateLabel(endLabel);
  const span = end - start || 1;

  const viewportStyle = useMemo(() => {
    if (!currentRange) return null;
    const vStart = parseDateLabel(currentRange[0]);
    const vEnd = parseDateLabel(currentRange[1]);
    const left = ((vStart - start) / span) * 100;
    const width = ((vEnd - vStart) / span) * 100;
    return { left: `${Math.max(0, left)}%`, width: `${Math.max(2, width)}%` };
  }, [currentRange, start, span]);

  const resolvedMarkers = useMemo(() => {
    const marks: TimeMarker[] = [...markers];
    // Auto-add start/end markers if not present
    if (!marks.some((m) => m.position <= 0.01)) {
      marks.unshift({ position: 0, label: formatDateLabel(startLabel), sublabel: "数据起点" });
    }
    if (!marks.some((m) => m.position >= 0.99)) {
      marks.push({ position: 1, label: formatDateLabel(endLabel), sublabel: "最新数据" });
    }
    return marks;
  }, [markers, startLabel, endLabel]);

  return (
    <div className={styles.wrapper}>
      <div className={styles.track}>
        {/* Background track */}
        <div className={styles.trackLine} />

        {/* Period colors */}
        <div className={styles.periods}>
          {resolvedMarkers.map((marker, i) => {
            if (i === 0) return null;
            const prev = resolvedMarkers[i - 1];
            const left = prev.position * 100;
            const width = (marker.position - prev.position) * 100;
            return (
              <div
                key={`${prev.label}-${marker.label}`}
                className={styles.periodSegment}
                style={{ left: `${left}%`, width: `${width}%`, background: marker.color || "var(--accent)" }}
              />
            );
          })}
        </div>

        {/* Current viewport */}
        {viewportStyle && (
          <motion.div
            className={styles.viewport}
            style={viewportStyle}
            layoutId="time-viewport"
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          >
            <div className={styles.viewportHandle} />
          </motion.div>
        )}

        {/* Markers */}
        {resolvedMarkers.map((marker) => (
          <div
            key={marker.label}
            className={styles.marker}
            style={{ left: `${marker.position * 100}%` }}
          >
            <div className={styles.markerDot} />
            <span className={styles.markerLabel}>{marker.label}</span>
            {marker.sublabel && <span className={styles.markerSublabel}>{marker.sublabel}</span>}
          </div>
        ))}
      </div>
    </div>
  );
}
