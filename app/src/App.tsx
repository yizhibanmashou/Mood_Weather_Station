import { useCallback, useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Skeleton } from "./components/StateViews";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { useMoodData } from "./hooks/useMoodData";
import { ClusterAnalysis } from "./pages/ClusterAnalysis";
import { Dashboard } from "./pages/Dashboard";
import { EventTimeline } from "./pages/EventTimeline";
import { ProvinceDetail } from "./pages/ProvinceDetail";
import { applyTheme, DEFAULT_THEME, THEME_META, type ThemePreset } from "./theme";
import { spring, pageVariants, hoverScale } from "./utils/motionPresets";
import styles from "./App.module.css";

type PageKey = "dashboard" | "province" | "cluster" | "events";

const pages: Array<{ key: PageKey; label: string; sub: string }> = [
  { key: "dashboard", label: "全国总览", sub: "Dashboard" },
  { key: "province", label: "省份详情", sub: "Province" },
  { key: "cluster", label: "聚类分析", sub: "Cluster" },
  { key: "events", label: "事件时间线", sub: "Events" }
];

const THEME_PRESETS: ThemePreset[] = ["warmIvory", "paperBeige", "softDataBlue"];

export default function App() {
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
    setPage("province");
  }, []);

  const content = useMemo(() => {
    if (!data) return null;
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
  }, [data, page, selectedProvince]);

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
              className={page === item.key ? styles.active : ""}
              onClick={() => setPage(item.key)}
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
                {content}
              </motion.div>
            </AnimatePresence>
          </ErrorBoundary>
        )}
      </main>
    </div>
  );
}
