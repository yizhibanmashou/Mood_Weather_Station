import type { ReactNode } from "react";
import { motion } from "framer-motion";
import { spring, hoverLift } from "../utils/motionPresets";
import styles from "./ChartCard.module.css";

interface ChartCardProps {
  title: string;
  eyebrow?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function ChartCard({ title, eyebrow, action, children, className = "" }: ChartCardProps) {
  return (
    <motion.section
      className={`${styles.card} ${className}`}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={spring}
      whileHover={hoverLift.whileHover}
      whileTap={hoverLift.whileTap}
    >
      <header className={styles.header}>
        <div>
          {eyebrow ? <p className={styles.eyebrow}>{eyebrow}</p> : null}
          <h2>{title}</h2>
        </div>
        {action ? <div className={styles.action}>{action}</div> : null}
      </header>
      <div className={styles.body}>{children}</div>
    </motion.section>
  );
}
