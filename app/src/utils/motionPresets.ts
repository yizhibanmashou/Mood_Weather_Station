/**
 * Centralized motion presets for Mood Weather Station.
 * All framer-motion spring/transition configs live here.
 */
import type { Transition, Variants } from "framer-motion";

/* ── Spring ─────────────────────────────────────────── */
export const spring: Transition = {
  type: "spring",
  stiffness: 240,
  damping: 28,
  mass: 0.8,
};

export const springGentle: Transition = {
  type: "spring",
  stiffness: 200,
  damping: 26,
  mass: 0.9,
};

export const springSnappy: Transition = {
  type: "spring",
  stiffness: 280,
  damping: 30,
  mass: 0.7,
};

/* ── Page transition ─────────────────────────────────── */
export const pageVariants: Variants = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0, transition: spring },
  exit: { opacity: 0, y: -12, transition: { duration: 0.18 } },
};

/* ── Hover / Tap presets ─────────────────────────────── */
export const hoverLift = {
  whileHover: { y: -3, boxShadow: "0 12px 32px rgba(30,50,80,0.10)" },
  whileTap: { scale: 0.98 },
  transition: spring,
};

export const hoverScale = {
  whileHover: { scale: 1.03 },
  whileTap: { scale: 0.97 },
  transition: spring,
};

export const tapFeedback = {
  whileTap: { scale: 0.98 },
  transition: spring,
};

/* ── Stagger container ───────────────────────────────── */
export const staggerContainer: Variants = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.08 },
  },
};

export const staggerItem: Variants = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0, transition: spring },
};

/* ── prefers-reduced-motion ──────────────────────────── */
export function useReducedMotion(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Returns a no-op variants object when reduced motion is preferred. */
export function withReducedMotion(variants: Variants): Variants {
  if (typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    return {
      hidden: {},
      visible: {},
    };
  }
  return variants;
}
