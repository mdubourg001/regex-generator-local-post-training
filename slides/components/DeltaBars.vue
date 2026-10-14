<script setup lang="ts">
// Écart de log-probabilité par rapport au modèle de référence : barre vers la droite = plus probable qu'avant,
// vers la gauche = moins probable. Les valeurs sont relatives (schéma), entre -1 et 1.
defineProps<{
  title: string
  rows: { label: string; value: number }[]
  verdict?: string
  tone?: 'good' | 'bad' | 'warn'
}>()
</script>

<template>
  <div class="db card" :class="tone">
    <h3>{{ title }}</h3>
    <div class="db-axis"><span>moins probable</span><span>référence</span><span>plus probable</span></div>
    <div v-for="r in rows" :key="r.label" class="db-row">
      <div class="db-label">{{ r.label }}</div>
      <div class="db-track">
        <div class="db-center" />
        <div
          class="db-bar"
          :class="r.value >= 0 ? 'up' : 'down'"
          :style="r.value >= 0 ? { left: '50%', width: `${r.value * 50}%` } : { right: '50%', width: `${-r.value * 50}%` }"
        />
      </div>
    </div>
    <p v-if="verdict" class="db-verdict">{{ verdict }}</p>
  </div>
</template>

<style scoped>
.db-axis { display: flex; justify-content: space-between; font-size: 0.65rem; opacity: 0.55; margin: 0.3rem 0 0.2rem 4.5rem; }
.db-row { display: grid; grid-template-columns: 4.2rem 1fr; gap: 0.3rem; align-items: center; margin-top: 0.3rem; }
.db-label { font-size: 0.75rem; font-family: var(--slidev-code-font-family, monospace); text-align: right; }
.db-track { position: relative; height: 1.1rem; background: rgba(127, 127, 127, 0.1); border-radius: 4px; }
.db-center { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 2px; background: rgba(127, 127, 127, 0.6); }
.db-bar { position: absolute; top: 0; bottom: 0; border-radius: 3px; }
.db-bar.up { background: #34d399; }
.db-bar.down { background: #f87171; }
.db-verdict { font-size: 0.78rem !important; margin-top: 0.5rem !important; }
</style>
