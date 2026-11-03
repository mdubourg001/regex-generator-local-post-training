/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Repo du Hub et révision figée du SFT Qwen3.5-0.8B : `owner/repo@<sha du commit>`. */
  readonly VITE_HUB_SFT_QWEN3_5_0_8B?: string;
  /** Repo du Hub et révision figée du SFT Qwen3-1.7B (q4f16). */
  readonly VITE_HUB_SFT_QWEN3_1_7B?: string;
}
