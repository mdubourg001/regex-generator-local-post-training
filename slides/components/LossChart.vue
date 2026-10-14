<script setup lang="ts">
// Val loss des deux runs SFT (logs/sft-0.6b.log et logs/sft-0.6b-2k.log), en fonction de l'epoch.
// Axe y de 0 à 0,7 : le point de départ (2,45, avant entraînement) est hors cadre.
const full = [[400, 0.433], [800, 0.303], [1200, 0.252], [1600, 0.226], [2000, 0.2], [2400, 0.191], [2800, 0.18], [3200, 0.173], [3600, 0.167], [4000, 0.164], [4160, 0.164]].map(([i, l]) => [i / 2080, l])
const small = [[100, 0.64], [200, 0.484], [300, 0.397], [400, 0.362], [500, 0.312], [600, 0.307], [700, 0.283], [800, 0.287], [900, 0.275], [1000, 0.272]].map(([i, l]) => [i / 500, l])

const X0 = 50, X1 = 530, Y0 = 230, YTOP = 20, YMAX = 0.7
const x = (epoch: number) => X0 + (epoch / 2) * (X1 - X0)
const y = (loss: number) => Y0 - (loss / YMAX) * (Y0 - YTOP)
const path = (pts: number[][]) => pts.map(([e, l], k) => `${k ? 'L' : 'M'}${x(e).toFixed(1)},${y(l).toFixed(1)}`).join(' ')
const grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
</script>

<template>
  <svg viewBox="0 0 560 270" class="loss">
    <g v-for="g in grid" :key="g">
      <line :x1="X0" :x2="X1" :y1="y(g)" :y2="y(g)" class="grid" />
      <text :x="X0 - 8" :y="y(g) + 4" text-anchor="end" class="tick">{{ g.toFixed(1).replace('.', ',') }}</text>
    </g>
    <line :x1="X0" :x2="X1" :y1="Y0" :y2="Y0" class="axis" />
    <line :x1="x(1)" :x2="x(1)" :y1="YTOP" :y2="Y0" class="epoch" />
    <text :x="x(0.5)" :y="Y0 + 20" text-anchor="middle" class="tick">epoch 1</text>
    <text :x="x(1.5)" :y="Y0 + 20" text-anchor="middle" class="tick">epoch 2</text>

    <path :d="path(small)" class="line run2k" />
    <path :d="path(full)" class="line run8k" />
    <circle v-for="([e, l], k) in small" :key="'s' + k" :cx="x(e)" :cy="y(l)" r="3" class="dot run2k" />
    <circle v-for="([e, l], k) in full" :key="'f' + k" :cx="x(e)" :cy="y(l)" r="3" class="dot run8k" />

    <text :x="x(2) - 4" :y="y(0.272) - 10" text-anchor="end" class="label run2k">2 000 exemples → 0,272</text>
    <text :x="x(2) - 4" :y="y(0.164) + 20" text-anchor="end" class="label run8k">8 300 exemples → 0,164</text>
    <text :x="X0 + 6" :y="YTOP + 12" class="note">départ : 2,45</text>
  </svg>
</template>

<style scoped>
.loss { width: 100%; height: auto; font-family: inherit; }
.grid { stroke: rgba(127, 127, 127, 0.18); }
.axis { stroke: rgba(127, 127, 127, 0.6); }
.epoch { stroke: rgba(127, 127, 127, 0.5); stroke-dasharray: 4 4; }
.tick { font-size: 11px; fill: currentColor; opacity: 0.6; }
.note { font-size: 11px; fill: currentColor; opacity: 0.6; font-style: italic; }
.line { fill: none; stroke-width: 2.5; }
.run8k { stroke: #10b981; fill: #10b981; }
.run2k { stroke: #f59e0b; fill: #f59e0b; }
.line.run8k, .line.run2k { fill: none; }
.label { font-size: 13px; font-weight: 600; stroke: none; }
</style>
