<script setup lang="ts">
// Barres horizontales en pourcentage. rows: [{ label, value, tone? }], value entre 0 et 100.
defineProps<{
  rows: { label: string; value: number; tone?: 'base' | 'good' | 'warn' | 'bad' | 'muted' }[]
  title?: string
  max?: number
}>()
const fmt = (v: number) => v.toLocaleString('fr-FR', { maximumFractionDigits: 1 }) + ' %'
</script>

<template>
  <div class="bars">
    <div v-if="title" class="bars-title">{{ title }}</div>
    <div v-for="row in rows" :key="row.label" class="bar-row">
      <div class="bar-label">{{ row.label }}</div>
      <div class="bar-track">
        <div class="bar-fill" :class="row.tone ?? 'base'" :style="{ width: `${(row.value / (max ?? 100)) * 100}%` }" />
        <span class="bar-value">{{ fmt(row.value) }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bars { display: flex; flex-direction: column; gap: 0.45rem; }
.bars-title { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.6; margin-bottom: 0.2rem; }
.bar-row { display: grid; grid-template-columns: 11rem 1fr; align-items: center; gap: 0.75rem; font-size: 0.85rem; }
.bar-label { text-align: right; opacity: 0.85; }
.bar-track { position: relative; height: 1.4rem; background: rgba(127, 127, 127, 0.12); border-radius: 4px; }
.bar-fill { height: 100%; border-radius: 4px; min-width: 2px; transition: width 0.6s; }
.bar-value { position: absolute; left: 0.5rem; top: 50%; transform: translateY(-50%); font-weight: 600; font-size: 0.8rem; }
.base { background: #94a3b8; }
.good { background: #34d399; }
.warn { background: #fbbf24; }
.bad { background: #f87171; }
.muted { background: #cbd5e1; }
</style>
