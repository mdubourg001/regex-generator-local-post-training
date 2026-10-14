<script setup lang="ts">
// Petites courbes multi-séries. series: [{ label, color, points: [[x, y], ...] }].
const props = defineProps<{
  series: { label: string; color: string; points: number[][] }[]
  xMax: number
  yMin?: number
  yMax: number
  yTicks: number[]
  yFormat?: 'percent' | 'decimal'
  xLabel?: string
  title?: string
}>()

const X0 = 46, X1 = 390, Y0 = 170, YTOP = 14
const yMin = props.yMin ?? 0
const x = (v: number) => X0 + (v / props.xMax) * (X1 - X0)
const y = (v: number) => Y0 - ((v - yMin) / (props.yMax - yMin)) * (Y0 - YTOP)
const path = (pts: number[][]) => pts.map(([a, b], k) => `${k ? 'L' : 'M'}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(' ')
const fmt = (v: number) => (props.yFormat === 'percent' ? `${Math.round(v * 100)} %` : v.toLocaleString('fr-FR'))
</script>

<template>
  <div>
    <div v-if="title" class="lc-title">{{ title }}</div>
    <svg viewBox="0 0 400 200" class="lc">
      <g v-for="t in yTicks" :key="t">
        <line :x1="X0" :x2="X1" :y1="y(t)" :y2="y(t)" class="grid" />
        <text :x="X0 - 6" :y="y(t) + 4" text-anchor="end" class="tick">{{ fmt(t) }}</text>
      </g>
      <text v-if="xLabel" :x="X1" :y="Y0 + 22" text-anchor="end" class="tick">{{ xLabel }}</text>
      <g v-for="s in series" :key="s.label">
        <path :d="path(s.points)" fill="none" :stroke="s.color" stroke-width="2.5" />
        <circle v-for="([a, b], k) in s.points" :key="k" :cx="x(a)" :cy="y(b)" r="3" :fill="s.color" />
      </g>
    </svg>
    <div class="lc-legend">
      <span v-for="s in series" :key="s.label"><i :style="{ background: s.color }" />{{ s.label }}</span>
    </div>
  </div>
</template>

<style scoped>
.lc { width: 100%; height: auto; }
.lc-title { font-size: 0.85rem; font-weight: 700; margin-bottom: 0.2rem; }
.grid { stroke: rgba(127, 127, 127, 0.18); }
.tick { font-size: 10px; fill: currentColor; opacity: 0.6; }
.lc-legend { display: flex; gap: 1rem; font-size: 0.75rem; opacity: 0.85; }
.lc-legend i { display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 2px; margin-right: 0.3rem; vertical-align: middle; }
</style>
