/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Repo du Hub et révision figée du SFT 0.6B enrichi : `owner/repo@<sha du commit>`. */
  readonly VITE_HUB_SFT_0_6B_PLUS?: string;
  readonly VITE_HUB_SFT_1_7B?: string;
}
