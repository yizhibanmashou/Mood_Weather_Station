import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig({
  plugins: [react()],
  envDir: resolve(__dirname, ".."),
  build: {
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/echarts") || id.includes("node_modules/echarts-for-react")) {
            return "vendor-echarts";
          }
          if (id.includes("node_modules/framer-motion")) {
            return "vendor-motion";
          }
        }
      }
    }
  },
  server: {
    host: "127.0.0.1",
    port: 5173
  }
});
