import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { spring, springGentle } from "../utils/motionPresets";
import { FloatingDataIllustration } from "./FloatingDataIllustration";
import { MethodDrawer } from "./MethodDrawer";

interface HeroSectionProps {
  /** ID of the element to scroll to when "探索数据" is clicked */
  scrollTargetId?: string;
}

export function HeroSection({ scrollTargetId = "dashboard-metrics" }: HeroSectionProps) {
  const [drawerOpen, setDrawerOpen] = useState(false);

  const handleExplore = useCallback(() => {
    const el = document.getElementById(scrollTargetId);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [scrollTargetId]);

  return (
    <section className="hero-section">
      <div className="hero-content">
        {/* Left: text */}
        <div className="hero-text">
          <motion.p
            className="hero-kicker"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...springGentle, delay: 0 }}
          >
            PUBLIC EMOTION OBSERVATORY
          </motion.p>

          <motion.h1
            className="hero-title"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...springGentle, delay: 0.06 }}
          >
            洞察公众情绪的<br />时空演变规律
          </motion.h1>

          <motion.p
            className="hero-desc"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...springGentle, delay: 0.14 }}
          >
            基于微博大数据与情绪计算，追踪 2019.12 至 2020.12 期间全国公众情绪的变化趋势与重点事件。
          </motion.p>

          <motion.div
            className="hero-actions"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...springGentle, delay: 0.22 }}
          >
            <motion.button
              type="button"
              className="hero-btn hero-btn-primary"
              onClick={handleExplore}
              whileHover={{ y: -2, boxShadow: "0 8px 24px rgba(59,125,216,0.22)" }}
              whileTap={{ scale: 0.98 }}
              transition={spring}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="6 9 12 15 18 9" />
              </svg>
              探索数据
            </motion.button>
            <motion.button
              type="button"
              className="hero-btn hero-btn-secondary"
              onClick={() => setDrawerOpen(true)}
              whileHover={{ y: -1, borderColor: "var(--accent)" }}
              whileTap={{ scale: 0.98 }}
              transition={spring}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
              </svg>
              方法说明
            </motion.button>
          </motion.div>
        </div>

        {/* Right: illustration */}
        <motion.div
          className="hero-illustration"
          initial={{ opacity: 0, x: 40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ ...springGentle, delay: 0.28 }}
        >
          <FloatingDataIllustration />
        </motion.div>
      </div>

      {/* Controlled MethodDrawer */}
      <MethodDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
    </section>
  );
}
