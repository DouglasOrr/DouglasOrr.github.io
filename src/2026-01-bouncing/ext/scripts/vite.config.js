import { defineConfig } from "vite";

export default defineConfig({
  root: "src",
  base: "./",
  server: {
    host: "0.0.0.0",
  },
  build: {
    outDir: "../../app",
  },
});
