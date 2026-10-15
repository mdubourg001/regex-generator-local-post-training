import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // transformers.js embarque onnxruntime-web (wasm) : le pré-bundling de Vite casse ses imports dynamiques.
  optimizeDeps: { exclude: ["@huggingface/transformers"] },
  worker: { format: "es" },
});
