import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Skeleton } from "./components/StateViews";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { TimeAxis } from "./components/TimeAxis";
import { useMoodData } from "./hooks/useMoodData";
import { applyTheme, DEFAULT_THEME, THEME_META, type ThemePreset } from "./theme";
import { spring, pageVariants } from "./utils/motionPresets";
import styles from "./App.module.css";

type PageKey = "dashboard" | "province" | "cluster" | "events";
type DataMode = "historical" | "realtime";

const pages: Array<{ key: PageKey; label: string; sub: string }> = [
  { key: "dashboard", label: "全国总览", sub: "Dashboard" },
  { key: "province", label: "省份详情", sub: "Province" },
  { key: "cluster", label: "聚类分析", sub: "Cluster" },
  { key: "events", label: "事件时间线", sub: "Events" }
];

const THEME_PRESETS: ThemePreset[] = ["warmIvory", "paperBeige", "softDataBlue"];

const Dashboard = lazy(() => import("./pages/Dashboard").then((m) => ({ default: m.Dashboard })));
const ProvinceDetail = lazy(() => import("./pages/ProvinceDetail").then((m) => ({ default: m.ProvinceDetail })));
const ClusterAnalysis = lazy(() => import("./pages/ClusterAnalysis").then((m) => ({ default: m.ClusterAnalysis })));
const EventTimeline = lazy(() => import("./pages/EventTimeline").then((m) => ({ default: m.EventTimeline })));
const RealtimeDashboard = lazy(() => import("./pages/RealtimeDashboard").then((m) => ({ default: m.RealtimeDashboard })));

export default function App() {
  const [dataMode, setDataMode] = useState<DataMode>("historical");
  const [page, setPage] = useState<PageKey>("dashboard");
  const [selectedProvince, setSelectedProvince] = useState<string>("");
  const [theme, setTheme] = useState<ThemePreset>(() => {
    const saved = localStorage.getItem("mws-theme");
    return (saved as ThemePreset) || DEFAULT_THEME;
  });
  const { data, loading, error } = useMoodData();

  useEffect(() => {
    applyTheme(theme);
    localStorage.setItem("mws-theme", theme);
  }, [theme]);

  const handleProvinceSelect = useCallback((province: string) => {
    setSelectedProvince(province);
    setDataMode("historical");
    setPage("province");
  }, []);

  // Compute data time range for the timeline
  const timeAxisProps = useMemo(() => {
    if (!data?.nationalWeeks?.length) return null;
    const weeks = data.nationalWeeks.map((w) => w.date_week).filter(Boolean).sort();
    const start = weeks[0];
    const end = weeks[weeks.length - 1];
    if (!start || !end) return null;

    const markers: { position: number; label: string; sublabel: string; color?: string }[] = [
      { position: 0, label: start, sublabel: "数据起点" },
    ];
    // Add COVID-19 milestones if within range
    if (start <= "2020-W04" && end >= "2020-W04") {
      const covPos = 0.1; // approximate position
      markers.push({ position: covPos, label: "2020-W04", sublabel: "武汉封城", color: "var(--emotion-fear)" });
    }
    if (start <= "2020-W12" && end >= "2020-W12") {
      markers.push({ position: 0.25, label: "2020-W12", sublabel: "管控期", color: "var(--emotion-sadness)" });
    }
    markers.push({ position: 1, label: end, sublabel: "最新数据", color: "var(--accent)" });

    return { fullRange: [start, end] as [string, string], markers };
  }, [data]);

  const content = useMemo(() => {
    if (!data) return null;
    if (dataMode === "realtime") {
      return <RealtimeDashboard snapshot={data.realtime} />;
    }
    switch (page) {
      case "province":
        return <ProvinceDetail data={data} initialProvince={selectedProvince} />;
      case "cluster":
        return <ClusterAnalysis data={data} />;
      case "events":
        return <EventTimeline data={data} />;
      case "dashboard":
      default:
        return <Dashboard data={data} onProvinceSelect={handleProvinceSelect} />;
    }
  }, [data, dataMode, page, selectedProvince]);

  return (
    <div className={styles.app}>
      <div className={styles.gridLayer} />
      <header className={styles.topbar}>
        <div className={styles.brand}>
          <span className={styles.brandMark}>MWS</span>
          <div>
            <strong>情绪气象站</strong>
            <small>Mood Weather Station</small>
          </div>
        </div>
        <nav className={styles.nav} aria-label="页面导航">
          {pages.map((item) => (
            <motion.button
              key={item.key}
              className={dataMode === "historical" && page === item.key ? styles.active : ""}
              onClick={() => {
                setDataMode("historical");
                setPage(item.key);
              }}
              type="button"
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              transition={spring}
            >
              <span>{item.label}</span>
              <small>{item.sub}</small>
            </motion.button>
          ))}
        </nav>
        <div className={styles.modeSwitcher} aria-label="数据模式切换">
          <motion.button
            className={dataMode === "historical" ? styles.activeMode : ""}
            onClick={() => setDataMode("historical")}
            type="button"
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            transition={spring}
          >
            历史数据
          </motion.button>
          <motion.button
            className={dataMode === "realtime" ? styles.activeMode : ""}
            onClick={() => setDataMode("realtime")}
            type="button"
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            transition={spring}
          >
            实时热搜
          </motion.button>
        </div>
        <div className={styles.themeSwitcher}>
          {THEME_PRESETS.map((preset) => (
            <motion.button
              key={preset}
              className={`${styles.themeBtn} ${theme === preset ? styles.activeTheme : ""}`}
              data-theme={preset}
              onClick={() => setTheme(preset)}
              type="button"
              whileHover={{ scale: 1.12 }}
              whileTap={{ scale: 0.92 }}
              title={`${THEME_META[preset].label} — ${THEME_META[preset].description}`}
            />
          ))}
        </div>
      </header>

      {dataMode === "historical" && timeAxisProps && data && (
        <TimeAxis fullRange={timeAxisProps.fullRange} markers={timeAxisProps.markers} />
      )}

      <main className={styles.main}>
        {loading ? (
          <section className={styles.loadingPanel}>
            <h1>情绪气象站</h1>
            <Skeleton rows={5} />
          </section>
        ) : error ? (
          <section className={styles.errorPanel}>
            <h1>数据加载失败</h1>
            <p>{error}</p>
          </section>
        ) : (
          <ErrorBoundary fallbackTitle="页面渲染出错">
            <AnimatePresence mode="wait">
              <motion.div
                key={page}
                variants={pageVariants}
                initial="initial"
                animate="animate"
                exit="exit"
              >
                <Suspense fallback={<Skeleton rows={4} />}>
                  {content}
                </Suspense>
              </motion.div>
            </AnimatePresence>
          </ErrorBoundary>
        )}
      </main>
    </div>
  );
}
