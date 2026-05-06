import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { METRIC_LABELS } from "../utils/metricLabels";
import { spring } from "../utils/motionPresets";

interface MethodDrawerProps {
  trigger?: string;
  /** Controlled open state — if provided, component becomes controlled */
  open?: boolean;
  /** Called when drawer wants to close (backdrop click, X button) */
  onClose?: () => void;
  /** Called when trigger button is clicked (uncontrolled mode) */
  onOpen?: () => void;
}

export function MethodDrawer({ trigger = "查看方法说明", open: controlledOpen, onClose, onOpen }: MethodDrawerProps) {
  const [internalOpen, setInternalOpen] = useState(false);
  const isOpen = controlledOpen ?? internalOpen;

  const handleOpen = () => {
    if (onOpen) onOpen();
    else setInternalOpen(true);
  };

  const handleClose = () => {
    if (onClose) onClose();
    else setInternalOpen(false);
  };

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") handleClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isOpen]);

  return (
    <>
      <motion.button
        type="button"
        onClick={handleOpen}
        whileHover={{ y: -1, boxShadow: "0 4px 16px rgba(30,50,80,0.08)" }}
        whileTap={{ scale: 0.97 }}
        transition={spring}
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border-strong)",
          borderRadius: 999,
          padding: "5px 14px",
          color: "var(--text-secondary)",
          fontSize: 12,
          cursor: "pointer",
        }}
      >
        {trigger}
      </motion.button>
      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.22 }}
              onClick={handleClose}
              style={{
                position: "fixed",
                inset: 0,
                background: "rgba(26, 35, 50, 0.2)",
                backdropFilter: "blur(4px)",
                zIndex: 100,
              }}
            />
            <motion.aside
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ ...spring, stiffness: 220, damping: 30 }}
              style={{
                position: "fixed",
                top: 0,
                right: 0,
                bottom: 0,
                width: 400,
                maxWidth: "90vw",
                background: "var(--surface-solid)",
                borderLeft: "1px solid var(--border)",
                padding: "32px 24px",
                overflowY: "auto",
                zIndex: 101,
                boxShadow: "-8px 0 32px rgba(30, 50, 80, 0.1)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
                <h3 style={{ margin: 0, color: "var(--text)", fontSize: 18 }}>方法说明</h3>
                <motion.button
                  type="button"
                  onClick={handleClose}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  transition={spring}
                  style={{
                    background: "none",
                    border: "none",
                    color: "var(--text-muted)",
                    fontSize: 20,
                    cursor: "pointer",
                    padding: "4px 8px",
                  }}
                >
                  ×
                </motion.button>
              </div>

              <div style={{ marginBottom: 20, padding: "12px 14px", borderRadius: 12, background: "var(--surface-warm)", border: "1px solid var(--border)" }}>
                <p style={{ margin: 0, color: "var(--text-secondary)", fontSize: 13, lineHeight: 1.6 }}>
                  <strong>数据来源：</strong>微博疫情语料 + 用户省份信息
                </p>
                <p style={{ margin: "8px 0 0", color: "var(--text-secondary)", fontSize: 13, lineHeight: 1.6 }}>
                  <strong>样本量：</strong>12,154 条标注样本（cap30 合并后）
                </p>
                <p style={{ margin: "8px 0 0", color: "var(--text-secondary)", fontSize: 13, lineHeight: 1.6 }}>
                  <strong>情绪维度：</strong>喜悦、悲伤、愤怒、恐惧、惊讶、中性
                </p>
              </div>

              <div style={{ marginBottom: 20, padding: "12px 14px", borderRadius: 12, background: "var(--accent-soft)", border: "1px solid var(--accent)" }}>
                <p style={{ margin: 0, color: "var(--accent-deep)", fontSize: 13, lineHeight: 1.6 }}>
                  <strong>标注可信度：</strong>Accuracy=73.3%，macro F1=0.662
                </p>
              </div>

              <p style={{ color: "var(--text-secondary)", fontSize: 13, lineHeight: 1.6, margin: "0 0 20px" }}>
                本系统使用 DeepSeek 大模型对微博文本进行六维情绪标注，并通过 SMP2020 人工标注数据集进行验证。以下为各指标的含义说明：
              </p>

              <div style={{ display: "grid", gap: 12 }}>
                {Object.entries(METRIC_LABELS).map(([key, meta]) => (
                  <motion.div
                    key={key}
                    initial={{ opacity: 0, x: 12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ ...spring, delay: 0.05 }}
                    style={{
                      padding: "12px 14px",
                      borderRadius: 12,
                      border: "1px solid var(--border)",
                      background: "var(--surface-warm)",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <strong style={{ color: "var(--text)", fontSize: 13 }}>{meta.label}</strong>
                      <code style={{ color: "var(--text-muted)", fontSize: 11 }}>{key}</code>
                    </div>
                    <p style={{ margin: 0, color: "var(--text-secondary)", fontSize: 12, lineHeight: 1.5 }}>
                      {meta.description}
                    </p>
                  </motion.div>
                ))}
              </div>

              <div style={{ marginTop: 20, padding: "12px 14px", borderRadius: 12, background: "var(--surface-warm)", border: "1px solid var(--border)" }}>
                <p style={{ margin: 0, color: "var(--text-secondary)", fontSize: 12, lineHeight: 1.6 }}>
                  <strong>异常检测：</strong>比较当前周与过去 4 周的差异，使用 rolling z-score 识别情绪波动异常。
                </p>
                <p style={{ margin: "8px 0 0", color: "var(--text-secondary)", fontSize: 12, lineHeight: 1.6 }}>
                  <strong>聚类分析：</strong>按省份全年情绪特征分组，使用 PCA 降维 + K-Means 聚类。
                </p>
                <p style={{ margin: "8px 0 0", color: "var(--text-secondary)", fontSize: 12, lineHeight: 1.6 }}>
                  <strong>地图：</strong>用于展示省份情绪分布，数据以省份标记方式叠加。
                </p>
              </div>

              <div style={{ marginTop: 20, padding: "12px 14px", borderRadius: 12, background: "rgba(217, 92, 74, 0.06)", border: "1px solid rgba(217, 92, 74, 0.15)" }}>
                <p style={{ margin: 0, color: "var(--emotion-anger)", fontSize: 12, lineHeight: 1.6 }}>
                  <strong>局限性：</strong>自动标注存在误差，结果适合课程项目中的探索性分析，不建议作为严肃研究结论使用。
                </p>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
