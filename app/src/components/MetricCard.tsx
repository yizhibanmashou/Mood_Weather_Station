import { motion } from "framer-motion";
import { spring } from "../utils/motionPresets";
import styles from "./MetricCard.module.css";

interface MetricCardProps {
  label: string;
  value: string;
  detail?: string;
  tone?: "warm" | "cool" | "danger" | "neutral";
}

export function MetricCard({ label, value, detail, tone = "neutral" }: MetricCardProps) {
  return (
    <motion.article
      className={`${styles.card} ${styles[tone]}`}
      whileHover={{ scale: 1.02, y: -3, boxShadow: "0 12px 32px rgba(30,50,80,0.10)" }}
      whileTap={{ scale: 0.98 }}
      transition={spring}
    >
      <span className={styles.label}>{label}</span>
      <strong>{value}</strong>
      {detail ? <small>{detail}</small> : null}
    </motion.article>
  );
}
