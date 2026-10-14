<script setup lang="ts">
// W gelé + B × A entraîné. Dimensions de Qwen3-0.6B (hidden size 1024), rang 16 comme dans configs/sft.yaml.
</script>

<template>
  <svg viewBox="0 0 440 220" class="lora">
    <text x="85" y="18" text-anchor="middle" class="caption">~1 M de paramètres</text>
    <rect x="10" y="30" width="150" height="150" rx="6" class="frozen" />
    <text x="85" y="108" text-anchor="middle" class="name">W</text>
    <text x="85" y="130" text-anchor="middle" class="caption">1024 × 1024 · gelé 🧊</text>

    <text x="187" y="112" text-anchor="middle" class="op">+</text>

    <rect x="214" y="30" width="20" height="150" rx="4" class="trained" />
    <text x="224" y="200" text-anchor="middle" class="label">B</text>
    <text x="250" y="112" text-anchor="middle" class="op">×</text>
    <rect x="266" y="95" width="150" height="20" rx="4" class="trained" />
    <text x="341" y="86" text-anchor="middle" class="label">A</text>

    <text x="341" y="145" text-anchor="middle" class="caption">rang r = 16 · entraîné 🔥</text>
    <text x="341" y="162" text-anchor="middle" class="caption">2 × 1024 × 16 ≈ 33 k paramètres</text>
  </svg>
</template>

<style scoped>
.lora { width: 100%; height: auto; }
.frozen { fill: rgba(148, 163, 184, 0.35); stroke: #64748b; }
.trained { fill: #6366f1; }
.name { font-size: 24px; font-weight: 700; fill: currentColor; }
.label { font-size: 13px; font-weight: 700; fill: currentColor; }
.op { font-size: 24px; fill: currentColor; }
.caption { font-size: 11px; fill: currentColor; opacity: 0.7; }
</style>
