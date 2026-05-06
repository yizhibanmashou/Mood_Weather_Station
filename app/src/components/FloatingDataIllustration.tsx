/**
 * FloatingDataIllustration — Glass-morphism decorative data illustration
 * for the Hero section right side. Purely visual, no real data binding.
 */
import { motion, type Transition } from "framer-motion";
import { springGentle } from "../utils/motionPresets";

/* ── Floating keyframes for CSS animation ────────────── */
const floatKeyframes = (delay: number, distance: number): React.CSSProperties => ({
  animation: `heroFloat ${4 + delay}s ease-in-out ${delay * 0.6}s infinite`,
});

/* ── SVG mini line chart ─────────────────────────────── */
function MiniLineChart() {
  const points = [
    [0, 38], [18, 32], [36, 35], [54, 22], [72, 28],
    [90, 16], [108, 20], [126, 12], [144, 18], [162, 8],
  ];
  const pathD = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${p[0]},${p[1]}`)
    .join(" ");
  const areaD = pathD + ` L162,48 L0,48 Z`;

  return (
    <svg viewBox="0 0 162 48" fill="none" style={{ width: "100%", height: "100%" }}>
      <defs>
        <linearGradient id="heroLineGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.25" />
          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={areaD} fill="url(#heroLineGrad)" />
      <path d={pathD} stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      {points.map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="2.5" fill="var(--surface-solid)" stroke="var(--accent)" strokeWidth="1.5" />
      ))}
    </svg>
  );
}

/* ── SVG mini donut chart ────────────────────────────── */
function MiniDonut() {
  const segments = [
    { pct: 32, color: "#F2A23A" },
    { pct: 22, color: "#5F8FA8" },
    { pct: 18, color: "#D95C4A" },
    { pct: 14, color: "#7D6AAE" },
    { pct: 8, color: "#E7C84F" },
    { pct: 6, color: "#A79A8D" },
  ];
  const r = 28;
  const circumference = 2 * Math.PI * r;
  let offset = 0;

  return (
    <svg viewBox="0 0 80 80" style={{ width: "100%", height: "100%" }}>
      {segments.map((seg, i) => {
        const dashArray = `${(seg.pct / 100) * circumference} ${circumference}`;
        const currentOffset = offset;
        offset += (seg.pct / 100) * circumference;
        return (
          <circle
            key={i}
            cx="40" cy="40" r={r}
            fill="none"
            stroke={seg.color}
            strokeWidth="10"
            strokeDasharray={dashArray}
            strokeDashoffset={-currentOffset}
            strokeLinecap="round"
            transform="rotate(-90 40 40)"
            opacity={0.85}
          />
        );
      })}
      <text x="40" y="38" textAnchor="middle" fill="var(--text)" fontSize="11" fontWeight="700" fontFamily="Fira Code, monospace">
        6D
      </text>
      <text x="40" y="50" textAnchor="middle" fill="var(--text-muted)" fontSize="7">
        emotions
      </text>
    </svg>
  );
}

/* ── Mini bar chart ──────────────────────────────────── */
function MiniBars() {
  const bars = [
    { h: 42, color: "#F2A23A" },
    { h: 28, color: "#5F8FA8" },
    { h: 50, color: "#D95C4A" },
    { h: 35, color: "#7D6AAE" },
    { h: 20, color: "#E7C84F" },
    { h: 32, color: "#A79A8D" },
  ];

  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 5, height: "100%", padding: "6px 4px 0" }}>
      {bars.map((bar, i) => (
        <motion.div
          key={i}
          initial={{ height: 0 }}
          animate={{ height: `${bar.h}%` }}
          transition={{ ...(springGentle as Transition), delay: 0.8 + i * 0.06 }}
          style={{
            flex: 1,
            borderRadius: "4px 4px 0 0",
            background: bar.color,
            opacity: 0.75,
            minWidth: 0,
          }}
        />
      ))}
    </div>
  );
}

/* ── Mini map outline (simplified China) ─────────────── */
function MiniMap() {
  return (
    <div style={{ position: "relative", width: "100%", height: "100%", display: "grid", placeItems: "center" }}>
      <svg viewBox="0 0 120 100" fill="none" style={{ width: "88%", height: "88%", opacity: 0.2 }}>
        <path
          d="M60 8 L78 12 L90 20 L98 32 L100 48 L96 60 L88 70 L80 78 L72 85 L60 88 L48 85 L38 76 L28 66 L22 52 L20 38 L26 24 L38 14 L50 10 Z"
          stroke="var(--accent)"
          strokeWidth="1.5"
          fill="var(--accent)"
          fillOpacity="0.06"
        />
      </svg>
      {/* Floating data points */}
      {[
        { x: "28%", y: "35%", size: 6, delay: 0.2 },
        { x: "55%", y: "25%", size: 8, delay: 0.5 },
        { x: "42%", y: "55%", size: 7, delay: 0.8 },
        { x: "68%", y: "45%", size: 5, delay: 1.1 },
        { x: "35%", y: "70%", size: 6, delay: 0.4 },
        { x: "75%", y: "60%", size: 4, delay: 0.7 },
      ].map((dot, i) => (
        <motion.div
          key={i}
          animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0.9, 0.5] }}
          transition={{ duration: 3 + i * 0.4, repeat: Infinity, ease: "easeInOut" as const, delay: dot.delay }}
          style={{
            position: "absolute",
            left: dot.x,
            top: dot.y,
            width: dot.size,
            height: dot.size,
            borderRadius: "50%",
            background: "var(--accent)",
          }}
        />
      ))}
    </div>
  );
}

/* ── Info card (floating) ────────────────────────────── */
function InfoCard({ label, value, delay }: { label: string; value: string; delay: number }) {
  return (
    <div
      style={{
        padding: "6px 10px",
        borderRadius: 10,
        background: "var(--surface-solid)",
        border: "1px solid var(--border)",
        boxShadow: "0 4px 16px rgba(30,50,80,0.06)",
        fontSize: 10,
        display: "flex",
        alignItems: "center",
        gap: 6,
        ...floatKeyframes(delay, 4),
      }}
    >
      <span style={{ color: "var(--text-muted)", fontSize: 9 }}>{label}</span>
      <strong style={{ color: "var(--text)", fontFamily: "Fira Code, monospace", fontSize: 11 }}>{value}</strong>
    </div>
  );
}

/* ── Main illustration ───────────────────────────────── */
export function FloatingDataIllustration() {
  return (
    <>
      <style>{`
        @keyframes heroFloat {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-6px); }
        }
      `}</style>
      <div
        style={{
          position: "relative",
          width: "100%",
          maxWidth: 460,
          height: 340,
          perspective: "900px",
        }}
      >
        {/* Map card — top left */}
        <motion.div
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ ...(springGentle as Transition), delay: 0.24 }}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: 180,
            height: 150,
            borderRadius: 16,
            background: "rgba(255,255,255,0.72)",
            border: "1px solid rgba(255,255,255,0.5)",
            backdropFilter: "blur(12px)",
            boxShadow: "0 8px 32px rgba(30,50,80,0.07)",
            padding: 10,
            transform: "rotate(-2deg)",
            ...floatKeyframes(0.2, 5),
          }}
        >
          <div style={{ fontSize: 9, color: "var(--text-muted)", fontWeight: 600, letterSpacing: "0.05em", marginBottom: 4 }}>
            PROVINCE MAP
          </div>
          <MiniMap />
        </motion.div>

        {/* Line chart card — top right */}
        <motion.div
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ ...(springGentle as Transition), delay: 0.32 }}
          style={{
            position: "absolute",
            top: 20,
            right: 0,
            width: 200,
            height: 120,
            borderRadius: 16,
            background: "rgba(255,255,255,0.72)",
            border: "1px solid rgba(255,255,255,0.5)",
            backdropFilter: "blur(12px)",
            boxShadow: "0 8px 32px rgba(30,50,80,0.07)",
            padding: 10,
            transform: "rotate(1.5deg)",
            zIndex: 1,
            ...floatKeyframes(0.6, 4),
          }}
        >
          <div style={{ fontSize: 9, color: "var(--text-muted)", fontWeight: 600, letterSpacing: "0.05em", marginBottom: 4 }}>
            EMOTION TREND
          </div>
          <div style={{ height: "calc(100% - 16px)" }}>
            <MiniLineChart />
          </div>
        </motion.div>

        {/* Donut card — bottom left */}
        <motion.div
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ ...(springGentle as Transition), delay: 0.40 }}
          style={{
            position: "absolute",
            bottom: 20,
            left: 20,
            width: 140,
            height: 140,
            borderRadius: 16,
            background: "rgba(255,255,255,0.72)",
            border: "1px solid rgba(255,255,255,0.5)",
            backdropFilter: "blur(12px)",
            boxShadow: "0 8px 32px rgba(30,50,80,0.07)",
            padding: 10,
            transform: "rotate(1deg)",
            zIndex: 1,
            ...floatKeyframes(1.0, 6),
          }}
        >
          <div style={{ fontSize: 9, color: "var(--text-muted)", fontWeight: 600, letterSpacing: "0.05em", marginBottom: 2 }}>
            COMPOSITION
          </div>
          <div style={{ height: "calc(100% - 16px)" }}>
            <MiniDonut />
          </div>
        </motion.div>

        {/* Bar chart card — bottom right */}
        <motion.div
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ ...(springGentle as Transition), delay: 0.48 }}
          style={{
            position: "absolute",
            bottom: 0,
            right: 10,
            width: 160,
            height: 120,
            borderRadius: 16,
            background: "rgba(255,255,255,0.72)",
            border: "1px solid rgba(255,255,255,0.5)",
            backdropFilter: "blur(12px)",
            boxShadow: "0 8px 32px rgba(30,50,80,0.07)",
            padding: 10,
            transform: "rotate(-1deg)",
            ...floatKeyframes(1.4, 5),
          }}
        >
          <div style={{ fontSize: 9, color: "var(--text-muted)", fontWeight: 600, letterSpacing: "0.05em", marginBottom: 4 }}>
            PROVINCE SCORES
          </div>
          <div style={{ height: "calc(100% - 16px)" }}>
            <MiniBars />
          </div>
        </motion.div>

        {/* Floating info cards */}
        <div style={{ position: "absolute", top: -8, right: 40, zIndex: 2 }}>
          <InfoCard label="Posts" value="12.1K" delay={0.4} />
        </div>
        <div style={{ position: "absolute", bottom: 130, left: -10, zIndex: 2 }}>
          <InfoCard label="Accuracy" value="73.3%" delay={0.8} />
        </div>
        <div style={{ position: "absolute", top: 130, right: -10, zIndex: 2 }}>
          <InfoCard label="Provinces" value="31" delay={1.2} />
        </div>
      </div>
    </>
  );
}
