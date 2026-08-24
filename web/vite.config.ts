import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("/echarts/")) return "echarts";
          if (id.includes("/maplibre-gl/")) return "maplibre";
          return undefined;
        },
      },
    },
  },
});
