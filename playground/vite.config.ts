import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ command, mode }) => {
  // Un build public charge les modèles depuis le Hub : sans repo configuré, il ne proposerait aucun modèle.
  if (command === "build" && !Object.keys(loadEnv(mode, process.cwd(), "VITE_HUB_")).length) {
    throw new Error("No model on the Hugging Face Hub: set VITE_HUB_SFT_0_6B_PLUS=owner/repo@revision (see .env.example)");
  }
  return {
    plugins: [react(), tailwindcss()],
    // public/ ne contient que les poids ONNX locaux (plusieurs Go) : servis en dev, jamais copiés dans dist/.
    publicDir: command === "serve" ? "public" : false,
    // transformers.js embarque onnxruntime-web (wasm) : le pré-bundling de Vite casse ses imports dynamiques.
    optimizeDeps: { exclude: ["@huggingface/transformers"] },
    worker: { format: "es" },
    // Ports fixes, hors défauts de Vite (5173, 4173) pris par d'autres projets. strictPort : erreur claire plutôt
    // qu'un port voisin silencieux. Le port fait partie de l'origine, donc du cache des poids.
    server: { port: 5299, strictPort: true },
    preview: { port: 5298, strictPort: true },
  };
});
