<script setup lang="ts">
// Schedule du learning rate : warmup linéaire sur 100 mises à jour, puis décroissance cosinus jusqu'à la fin.
const W = 2080, WARM = 100, MAX = 2e-4
const lr = (t: number) => (t < WARM ? MAX * (t / WARM) : MAX * 0.5 * (1 + Math.cos((Math.PI * (t - WARM)) / (W - WARM))))
const X0 = 20, X1 = 380, Y0 = 130, YTOP = 15
const pts = Array.from({ length: 105 }, (_, k) => (k < 5 ? (k / 4) * WARM : WARM + ((k - 4) / 100) * (W - WARM)))
const d = pts.map((t, k) => `${k ? 'L' : 'M'}${(X0 + (t / W) * (X1 - X0)).toFixed(1)},${(Y0 - (lr(t) / MAX) * (Y0 - YTOP)).toFixed(1)}`).join(' ')
</script>

<template>
  <svg viewBox="0 0 400 160" class="lr">
    <line :x1="X0" :x2="X1" :y1="Y0" :y2="Y0" class="axis" />
    <path :d="d" class="curve" />
    <text :x="X0 + 6" :y="YTOP - 2" class="t">warmup</text>
    <text :x="(X0 + X1) / 2" :y="70" class="t">décroissance cosinus</text>
    <text :x="X1" :y="Y0 + 18" text-anchor="end" class="t">fin du run</text>
    <text :x="X0" :y="Y0 + 18" class="t">début</text>
  </svg>
</template>

<style scoped>
.lr { width: 100%; height: auto; }
.axis { stroke: rgba(127, 127, 127, 0.6); }
.curve { fill: none; stroke: #6366f1; stroke-width: 2.5; }
.t { font-size: 11px; fill: currentColor; opacity: 0.7; }
</style>
