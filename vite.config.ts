import { defineConfig } from "vitest/config";

export default defineConfig({
  root: "webui",
  base: "/",
  build: {
    outDir: "../src/word_forge/web/static",
    emptyOutDir: true,
    sourcemap: false,
    target: "es2022",
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8765",
    },
  },
  test: {
    environment: "jsdom",
    include: ["test/**/*.test.ts"],
  },
});
