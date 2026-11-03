import type { Variant } from "./worker";

// Types minimaux de WebGPU : la lib DOM de TypeScript ne déclare pas navigator.gpu.
type Gpu = { requestAdapter(): Promise<{ features: ReadonlySet<string> } | null> };

/**
 * Variante à charger selon le navigateur :
 * - pas de WebGPU (navigator.gpu absent, ou aucun adaptateur) → q4 sur WASM (CPU) ;
 * - WebGPU sans la feature `shader-f16` → q4 sur WebGPU (calcul en fp32) ;
 * - sinon → q4f16 sur WebGPU, la plus légère.
 */
export async function pickVariant(): Promise<Variant> {
  const gpu = (navigator as Navigator & { gpu?: Gpu }).gpu;
  const adapter = await gpu?.requestAdapter().catch(() => null);
  if (!adapter) return { dtype: "q4", device: "wasm" };
  return { dtype: adapter.features.has("shader-f16") ? "q4f16" : "q4", device: "webgpu" };
}

export const variantLabel = ({ dtype, device }: Variant) =>
  `${device === "webgpu" ? "WebGPU" : "WASM (CPU)"} · ${dtype}`;
