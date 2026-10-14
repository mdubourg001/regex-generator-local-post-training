---
theme: default
title: Post-training d'un petit LLM, en local
info: |
  Apprendre à Qwen3-0.6B à écrire des regex JavaScript à partir d'une description en anglais.
  Partie 1 : des données jusqu'au SFT.
colorSchema: auto
transition: slide-left
mdc: true
fonts:
  sans: Inter
  mono: JetBrains Mono
layout: cover
---

<div class="kicker">Post-training · partie 1</div>

# Apprendre à un petit LLM<br>à écrire des regex

Des données jusqu'au SFT, sur un Mac, sans GPU NVIDIA

<div class="abs-bl m-12 text-sm opacity-60">
Qwen3-0.6B · MLX · LoRA · Node 22
</div>

<!--
Le but de ces slides : expliquer chaque étape du repo à quelqu'un qui n'est pas ML engineer.
On s'arrête avant le DPO.
-->

---

# Le plan

<div class="grid grid-cols-2 gap-6 mt-6">
<div>

1. **Le projet** : la tâche, la thèse, le matériel
2. **Les bases** : un LLM, le post-training
3. **Mesurer** : le harnais d'évaluation
4. **Les données** : un professeur et des garde-fous

</div>
<div>

5. **Le benchmark** : un examen qui ne sort pas du même moule
6. **Les baselines** : que vaut le modèle brut ?
7. **Le SFT avec LoRA** : la première vraie amélioration
8. **Bilan** et ce qui vient ensuite

</div>
</div>

<div class="takeaway">
Fil rouge : on <b>mesure d'abord</b>, on entraîne ensuite. Chaque étape a un chiffre avant / après.
</div>

---
layout: section
---

<div class="kicker">1 · Le projet</div>

# Ce qu'on construit

---

# La tâche : du langage naturel vers une regex JS

<div class="flow mt-8">
  <div class="step">
    <strong>📝 Description</strong>
    match a French mobile phone number, with or without spaces between pairs of digits
  </div>
  <div class="arrow">+</div>
  <div class="step">
    <strong>🧪 Exemples (optionnels)</strong>
    ✅ <code>"06 12 34 56 78"</code><br>
    ❌ <code>"01 12 34 56 78"</code>
  </div>
  <div class="arrow">→</div>
  <div class="step" style="border-color: var(--c-accent)">
    <strong>🤖 Petit modèle</strong>
    Qwen3-0.6B<br>tourne sur le Mac
  </div>
  <div class="arrow">→</div>
  <div class="step done">
    <strong>⚡ Réponse</strong>
    <code>/^0[67](?: ?\d{2}){4}&#36;/</code><br>
    <span class="small muted">et rien d'autre : pas d'explication</span>
  </div>
</div>

<div class="grid grid-cols-3 gap-4 mt-8">
<div class="card">
<h3>Flavor ciblée</h3>
<p>ECMAScript 2018, exécutée par Node 22 : lookbehind, groupes nommés, <code>\p{…}</code>, flags <code>gimsuy</code>.</p>
</div>
<div class="card">
<h3>Validation seulement</h3>
<p>On teste <code>regex.test(s)</code> sur la chaîne entière. L'extraction (<code>matchAll</code>) viendra plus tard.</p>
</div>
<div class="card">
<h3>Sortie collable</h3>
<p>Un littéral <code>/…/flags</code> que l'on peut coller tel quel dans du code.</p>
</div>
</div>

---

# Pourquoi des regex ?

<div class="grid grid-cols-3 gap-4 mt-6">
<div class="card">
<h3>🎯 Un vrai besoin de dev</h3>
<p>Tout le monde en écrit, tout le monde s'y trompe. La tâche est <b>étroite</b> et la sortie <b>courte</b> : le terrain idéal pour un petit modèle spécialisé.</p>
</div>
<div class="card good">
<h3>✅ Vérifiable par exécution</h3>
<p>On exécute la regex sur des tests cachés. Pas besoin d'un humain ou d'un autre LLM pour juger : <b>le moteur JS est le juge</b>.</p>
</div>
<div class="card">
<h3>📦 Des données publiables</h3>
<p>Tout est synthétique ou public. Le premier sujet envisagé (labelliser des merge requests clients) ne l'était pas.</p>
</div>
</div>

<div class="card accent mt-8">
<h3>La thèse</h3>
<p>Un modèle de 0,6 à 1,7 milliard de paramètres, post-entraîné en local, peut approcher la qualité d'un gros modèle, pour un <b>coût</b>, une <b>latence</b> et une <b>dépendance au cloud</b> quasi nuls. À terme, il doit tourner dans un navigateur.</p>
</div>

---

# Le matériel et les modèles

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

| | |
| --- | --- |
| Machine | Apple M4 Pro, **24 Go** de RAM unifiée |
| Framework | **MLX** (Apple), car pas de GPU NVIDIA, donc pas de CUDA |
| Modèle d'itération | **Qwen3-0.6B** : runs rapides |
| Modèle final | **Qwen3-1.7B** : on y rejoue le meilleur pipeline |
| Méthode | **LoRA** uniquement |
| Mode _thinking_ | désactivé : la latence prime |

</div>
<div>

<div class="card">
<h3>« 0.6B », ça veut dire quoi ?</h3>
<p>600 millions de <b>paramètres</b> : des nombres qui, ensemble, encodent tout ce que le modèle « sait ». En 16 bits, ça fait ~1,2 Go sur disque.</p>
<p class="muted">Pour comparer : les modèles frontier se comptent en centaines de milliards, voire plus.</p>
</div>

<div class="analogy">
MLX est à l'Apple Silicon ce que CUDA/PyTorch est aux GPU NVIDIA : la couche qui fait tourner les calculs sur la puce. La RAM « unifiée » est partagée entre CPU et GPU, d'où la contrainte des 24 Go.
</div>

</div>
</div>

---
layout: section
---

<div class="kicker">2 · Les bases</div>

# Un LLM, le post-training

---

# Un LLM, c'est un autocomplete géant

Le modèle ne fait qu'une chose : **prédire le prochain morceau de texte** (un _token_), puis recommencer.

<div class="flow mt-6">
  <div class="step"><strong>Texte d'entrée</strong><code>… digits</code><br><code>Answer:</code></div>
  <div class="arrow">→</div>
  <div class="step"><strong>Tokens</strong><code>[Answer]</code> <code>[:]</code><br><span class="small muted">des morceaux de mots, ~151 936 possibles</span></div>
  <div class="arrow">→</div>
  <div class="step" style="border-color: var(--c-accent)"><strong>Modèle</strong>des couches de multiplications de matrices</div>
  <div class="arrow">→</div>
  <div class="step">
    <strong>Probabilités</strong>
    <code>/^</code> 62 %<br>
    <code>The</code> 21 %<br>
    <code>```</code> 9 %<br>
    <span class="muted">…</span>
  </div>
</div>

<div class="grid grid-cols-2 gap-6 mt-6">
<div class="card">
<h3>Génération</h3>
<p>On choisit un token (le plus probable = <b>greedy</b>, ou au hasard selon les probas = <b>sampling</b>), on l'ajoute au texte, et on recommence jusqu'à un token de fin.</p>
</div>
<div class="analogy" style="margin-top:0">
Comme l'autocomplete de ton IDE, mais qui a lu une bonne partie d'Internet. Entraîner le modèle, c'est ajuster ses paramètres pour qu'il donne plus de probabilité aux « bonnes » suites.
</div>
</div>

---

# Pré-training vs post-training

<div class="flow mt-6">
  <div class="step done">
    <strong>Pré-training</strong>
    Lire des milliers de milliards de tokens et prédire la suite.<br>
    <span class="muted small">Fait par Alibaba (Qwen). Des mois de GPU.</span>
  </div>
  <div class="arrow">→</div>
  <div class="step" style="border-color: var(--c-accent); background: rgba(99,102,241,0.1)">
    <strong>SFT</strong>
    Imiter des exemples « entrée → bonne réponse ».<br>
    <span class="small">📍 <b>on est ici</b></span>
  </div>
  <div class="arrow">→</div>
  <div class="step later">
    <strong>DPO</strong>
    Préférer la bonne réponse à la mauvaise, sur ses propres réponses.
  </div>
  <div class="arrow">→</div>
  <div class="step later">
    <strong>GRPO</strong>
    Essayer, être noté par exécution, renforcer ce qui marche (RL).
  </div>
</div>

<div class="grid grid-cols-2 gap-6 mt-8">
<div class="card">
<h3>Le modèle brut sait déjà…</h3>
<p>la syntaxe des regex, l'anglais, ce qu'est un numéro de téléphone. Il a tout vu pendant le pré-training.</p>
</div>
<div class="card">
<h3>…mais ne sait pas faire <i>notre</i> tâche</h3>
<p>Répondre au bon format, de façon fiable, avec les bonnes ancres et sans sur-généraliser. C'est le rôle du <b>post-training</b>.</p>
</div>
</div>

<div class="analogy">
Le pré-training, c'est apprendre le langage. Le post-training, c'est l'onboarding sur un poste précis.
</div>

---

# Vue d'ensemble du pipeline

<div class="pipeline mt-4">
  <div class="lane-label">Données d'entraînement</div>
  <div class="flow">
    <div class="step"><strong>Taxonomie</strong>21 catégories de besoins</div>
    <div class="arrow">→</div>
    <div class="step"><strong>Professeur</strong>DeepSeek V4.1 Flash écrit descriptions, regex et tests</div>
    <div class="arrow">→</div>
    <div class="step"><strong>Garde-fous</strong>exécutés dans Node · 34 % gardés</div>
    <div class="arrow">→</div>
    <div class="step"><strong>Train / valid / held-out</strong>8 321 / 505 / 442</div>
  </div>
  <div class="lane-label mt-4">Examen indépendant</div>
  <div class="flow">
    <div class="step" style="border-color: var(--c-warn); background: rgba(245,158,11,0.08)"><strong>Benchmark</strong>210 cas écrits à la main, figés</div>
    <div class="arrow">→</div>
    <div class="step"><strong>Décontamination</strong>on retire du train ce qui ressemble au benchmark</div>
    <div class="arrow" style="visibility: hidden">→</div>
    <div class="step" style="visibility: hidden"></div>
    <div class="arrow" style="visibility: hidden">→</div>
    <div class="step" style="visibility: hidden"></div>
  </div>
  <div class="lane-label mt-4">Entraînement et mesure</div>
  <div class="flow">
    <div class="step"><strong>Modèle brut</strong>Qwen3-0.6B / 1.7B : les baselines</div>
    <div class="arrow">→</div>
    <div class="step" style="border-color: var(--c-accent); background: rgba(99,102,241,0.1)"><strong>SFT + LoRA</strong>sur le train</div>
    <div class="arrow">→</div>
    <div class="step done"><strong>Évaluation</strong>harnais Node, tests cachés</div>
    <div class="arrow">→</div>
    <div class="step done"><strong>Scores</strong>held-out et benchmark</div>
  </div>
</div>

<div class="small muted mt-2">
Étapes du repo : 0 setup · 1 harnais d'éval · 2 taxonomie · 3 données · 4 benchmark · 5 baselines · 6 SFT
</div>

---
layout: section
---

<div class="kicker">3 · Mesurer</div>

# Le harnais d'évaluation

<p class="opacity-70">Étape 1 : avant d'entraîner quoi que ce soit, on sait mesurer.</p>

---

# Un exemple, c'est une fixture de test

```json {all|5|6|7|8-11}
{
  "id": "dates-0042",
  "category": "dates-times",
  "difficulty": "medium",
  "description": "ISO 8601 date, year between 1900 and 2099",
  "examples": { "match": ["2024-02-29"], "no_match": [] },
  "regex": "/^(?:19|20)\\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\\d|3[01])$/",
  "tests": {
    "match":    ["1900-01-01", "2099-12-31", "2024-06-15", "1987-10-03", "2000-01-31"],
    "no_match": ["1899-12-31", "2100-01-01", "2024-13-01", "2024-6-15", "x2024-06-15", "2024-06-15 "]
  }
}
```

<div class="grid grid-cols-3 gap-4 mt-4">
<div class="card"><h3>Ce que voit le modèle</h3><p>La <code>description</code> et, parfois, quelques <code>examples</code>.</p></div>
<div class="card"><h3>La réponse attendue</h3><p>La <code>regex</code> de référence. Elle sert à l'entraînement, <b>pas</b> à l'évaluation.</p></div>
<div class="card good"><h3>La vérité terrain</h3><p>Les <code>tests</code> cachés. On ne compare pas la regex produite à la référence : on lance la suite de tests.</p></div>
</div>

---

# Le prompt et le parsing : un contrat d'API

<div class="grid grid-cols-2 gap-6">
<div>

**Message système** (fixe, dans `scripts/common.py`)

```text
You write JavaScript regular expressions
(ECMAScript 2018). The regex will validate strings
with RegExp.prototype.test: it must accept every
valid string and reject every other one. Given a
description, and sometimes example strings, answer
with a single regex literal such as /^\d+$/i,
with no explanation.
```

**Message utilisateur**

```text
Description: ISO 8601 date, year between 1900 and 2099

Should match:
- "2024-02-29"
```

</div>
<div>

**Parsing strict de la réponse**

| Réponse du modèle | Verdict |
| --- | --- |
| `/^\d+$/` | <span class="ok">OK</span> |
| la même, entre backticks ou dans un bloc de code | <span class="ok">OK</span> |
| `Here is the regex: /^\d+$/` | <span class="ko">mal formée</span> |
| `^\d+$` (sans délimiteurs) | <span class="ko">mal formée</span> |
| `/^\d+$/.test(str)` | <span class="ko">mal formée</span> |
| `/^\d+$/v` (flag hors ES2018) | <span class="ko">mal formée</span> |

<div class="analogy">
Le format est défini une seule fois et partagé par toutes les étapes (entraînement, éval, DPO, GRPO), comme un type partagé entre front et back. Le changer, c'est casser le contrat : il faut ré-entraîner.
</div>

</div>
</div>

---

# Pourquoi exécuter dans Node, et pas en Python ?

<div class="grid grid-cols-2 gap-6">
<div>

<div class="flex flex-col items-stretch gap-2 text-sm">
  <div class="card tight"><b>eval.py</b> (Python)<br><span class="small muted">génère les réponses du modèle, les parse</span></div>
  <div class="text-center small muted">↓ lots de regex + tests, en JSON ↑ pass / fail / timeout / erreur</div>
  <div class="card tight accent"><b>Worker Node persistant</b><br><span class="small muted">démarré une fois, réutilisé pour toute l'éval</span></div>
  <div class="text-center small muted">↓ une regex et ses tests</div>
  <div class="grid grid-cols-2 gap-2">
    <div class="card tight good"><b>worker_thread</b><br><span class="small">répond à temps ✓</span></div>
    <div class="card tight bad"><b>worker_thread</b><br><span class="small">dépasse le délai : tué ✗</span></div>
  </div>
</div>

</div>
<div>

<div class="card">
<h3>Le module <code>re</code> de Python n'est pas JS</h3>
<p>Lookbehind, <code>\p{…}</code>, groupes nommés, flags : la syntaxe et la sémantique diffèrent. Pour juger une regex JS, il faut <b>le vrai moteur JS</b>.</p>
</div>

<div class="card mt-3 bad">
<h3>Le piège du ReDoS</h3>
<p><code>/^(a+)+&#36;/</code> sur <code>"aaaa…aaab"</code> : backtracking catastrophique, la regex ne rend jamais la main. Un <code>RegExp</code> est <b>synchrone</b> et ne s'interrompt pas.</p>
<p>Solution : chaque évaluation tourne dans un <code>worker_thread</code>, que l'on tue au bout du délai.</p>
</div>

<div class="analogy">
Exactement le problème d'une boucle infinie sur le main thread d'un navigateur : on déporte dans un Web Worker qu'on peut <code>terminate()</code>.
</div>

</div>
</div>

---

# Ce qu'on mesure

<div class="grid grid-cols-2 gap-8 mt-2">
<div>

<div class="card accent">
<h3>🎯 Succès complet (la métrique principale)</h3>
<p>Part des cas où <b>tous</b> les tests passent. En usage réel, une regex juste à 90 % est une regex fausse.</p>
</div>

<div class="card tight mt-3"><h3>📊 Taux moyen de tests réussis</h3><p>Pour diagnostiquer : « presque juste » ou « complètement à côté » ?</p></div>
<div class="card tight mt-2"><h3>🧱 Sorties mal formées</h3><p>Pas de littéral, ou <code>new RegExp(pattern, flags)</code> qui lève une erreur.</p></div>
<div class="card tight mt-2"><h3>⏱️ Timeouts</h3><p>Les regex qui partent en ReDoS.</p></div>

</div>
<div>

**Exemple : une regex, 11 tests**

<div class="grid grid-cols-11 gap-1 mt-3">
  <div v-for="i in 10" :key="i" class="h-8 rounded bg-emerald-400 flex items-center justify-center text-xs text-white">✓</div>
  <div class="h-8 rounded bg-red-400 flex items-center justify-center text-xs text-white">✗</div>
</div>

<div class="grid grid-cols-2 gap-4 mt-4">
<div class="card"><div class="big">91 %</div><div class="big-label">de tests réussis</div></div>
<div class="card bad"><div class="big">0</div><div class="big-label">succès complet</div></div>
</div>

<p class="small mt-3">Le test raté : <code>"2024-06-15 "</code>, avec une espace en fin. La regex a oublié l'ancre <code>&#36;</code>. En prod, c'est un bug.</p>

</div>
</div>

---
layout: section
---

<div class="kicker">4 · Les données</div>

# Un professeur et des garde-fous

<p class="opacity-70">Étapes 2 et 3 : taxonomie, génération, validation</p>

---

# La taxonomie : la carte de ce qu'on veut couvrir

<div class="grid grid-cols-2 gap-6">
<div>

**21 catégories** (`configs/taxonomy.yaml`)

<div class="chips mt-2">
<span class="chip">numbers</span><span class="chip">money</span><span class="chip">dates-times</span><span class="chip">identifiers</span><span class="chip">web</span><span class="chip">network</span><span class="chip">email-accounts</span><span class="chip">phone-numbers</span><span class="chip">postal-official</span><span class="chip">passwords</span><span class="chip">files-paths</span><span class="chip">versions</span><span class="chip">code-syntax</span><span class="chip">dev-tooling</span><span class="chip">logs</span><span class="chip">text</span><span class="chip">unicode</span><span class="chip">structured-data</span><span class="chip">multiline</span><span class="chip">exclusions</span><span class="chip">measurements</span>
</div>

<p class="small mt-4">Chaque catégorie liste des <b>intentions types</b> (des graines que le professeur décline) et les <b>fonctionnalités de regex</b> qu'elle sollicite : ancres, lookahead, backreferences, <code>\p{…}</code>, flags…</p>

**Difficulté**

<BarChart :rows="[
  { label: 'simple', value: 30, tone: 'good' },
  { label: 'intermédiaire', value: 50, tone: 'warn' },
  { label: 'piège', value: 20, tone: 'bad' },
]" />

</div>
<div>

**4 registres de description**, tirés au hasard

<div class="grid grid-cols-2 gap-2 mt-2">
<div class="card tight"><h3>terse</h3><p><i>french mobile numbers with optional spaces</i></p></div>
<div class="card tight"><h3>sentence</h3><p><i>Match a French mobile phone number. Spaces between pairs of digits are allowed.</i></p></div>
<div class="card tight"><h3>spec</h3><p><i>Une liste de règles numérotées, comme dans un ticket.</i></p></div>
<div class="card tight"><h3>business</h3><p><i>we need to check the phone number field in our signup form, only French mobiles are accepted</i></p></div>
</div>

<p class="small mt-3">Pourquoi ? Pour que le modèle apprenne la <b>tâche</b>, pas le style d'écriture du professeur. Et ~30 % des entrées ont 1 à 3 exemples.</p>

</div>
</div>

---

# La distillation : un gros modèle écrit les exercices

<div class="flow mt-4">
  <div class="step"><strong>Taxonomie</strong>catégorie + intention + difficulté + registre</div>
  <div class="arrow">→</div>
  <div class="step" style="border-color: var(--c-accent)"><strong>Appel 1</strong>Le professeur écrit 5 <b>descriptions</b> et leurs <b>regex</b></div>
  <div class="arrow">→</div>
  <div class="step" style="border-color: var(--c-warn)"><strong>Appel 2 (séparé)</strong>Il écrit les <b>tests</b> à partir des <b>descriptions seules</b>, sans voir la regex</div>
  <div class="arrow">→</div>
  <div class="step"><strong>Node</strong>On confronte regex et tests</div>
</div>

<div class="grid grid-cols-2 gap-6 mt-6">
<div class="card">
<h3>Pourquoi deux appels ?</h3>
<p>Si le professeur écrit les tests en voyant sa regex, les tests <b>héritent de ses bugs</b> : il teste ce que fait sa regex, pas ce que demande la description.</p>
<p>Avec deux appels indépendants, un désaccord entre regex et tests révèle une erreur d'un côté ou de l'autre.</p>
</div>
<div class="card">
<h3>Le professeur : DeepSeek V4.1 Flash</h3>
<p>Via OpenRouter, à 0,15 $ / 0,60 $ par million de tokens. Plafond de dépense à 9 $, génération <b>reprenable</b> : une relance ne repaie pas les appels déjà faits.</p>
</div>
</div>

<div class="analogy">
C'est le principe du TDD à deux personnes : celle qui écrit les tests ne lit pas l'implémentation. On appelle ça <b>distiller</b> : le petit modèle apprendra à partir des sorties du gros.
</div>

---

# Les garde-fous : un entonnoir

Une regex qui passe ses propres tests ne prouve rien si les tests sont faibles : `.*` passe tous les positifs.

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

1. **Tests dérivés de la description**, dans un appel séparé
2. **Minimum de tests** : ≥ 5 positifs, ≥ 5 négatifs, dont ≥ 3 « presque bons » (un caractère de trop, mauvaise casse, hors bornes…)
3. **La référence passe tous les tests**, et elle est valide en ES2018
4. **Discrimination** : des regex triviales ou mutées doivent échouer (slide suivante)
5. **Décontamination** contre le benchmark
6. **Second avis** sur un échantillon : un autre modèle écrit sa propre regex

</div>
<div>

<div class="funnel mt-4">
  <div style="width: 100%; background: #6366f1">27 460 besoins générés</div>
  <div style="width: 91%; background: #8b5cf6">25 013 lisibles, avec assez de tests</div>
  <div style="width: 46%; background: #a855f7">12 622 : regex et tests d'accord</div>
  <div style="width: 37%; background: #c026d3">10 146 : tests discriminants</div>
  <div style="width: 34%; background: #10b981"><b>9 268 après doublons et décontamination</b></div>
</div>

<p class="small muted mt-3 text-center">Largeurs proportionnelles. Le gros du rejet : le désaccord entre regex et tests (12 390 besoins, 45 %).</p>

</div>
</div>

---

# La discrimination, c'est du mutation testing

On abîme volontairement la regex de référence. **Si les tests ne remarquent rien, ils sont trop faibles** : on jette l'exemple.

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

| Variante | Regex | Un test doit échouer |
| --- | --- | --- |
| Référence | `/^\d{5}$/` | aucun |
| Triviale | `/.*/` | `"abc"` |
| Sans ancres | `/\d{5}/` | `"123456"` |
| Quantificateur relâché | `/^\d+$/` | `"1234"` |
| Classe élargie | `/^\w{5}$/` | `"12a45"` |

</div>
<div>

<div class="card good">
<h3>Les tests « tuent » tous les mutants</h3>
<p>Ils vérifient vraiment les bornes, les ancres et le jeu de caractères. L'exemple est gardé.</p>
</div>

<div class="card bad mt-3">
<h3>Un mutant survit</h3>
<p>Par exemple, aucun test ne contient de chaîne plus longue : la version sans ancres passe tout. L'exemple est rejeté.</p>
</div>

<div class="analogy">
Comme Stryker en JS : on mesure la qualité d'une suite de tests à sa capacité à détecter des bugs injectés.
</div>

</div>
</div>

---

# Le pilote : faut-il faire « réfléchir » le professeur ?

100 besoins, mêmes lots, trois réglages :

| DeepSeek V4.1 Flash | Gardés | Désaccord regex / tests | Coût pour 1 000 gardés |
| --- | --- | --- | --- |
| **Sans raisonnement** | 52 % | 37 % | **0,53 $** |
| Raisonnement « low » | 76 % | 8 % | 2,98 $ |
| Raisonnement par défaut | 77 % | 2 % | 4,46 $ |

<div class="grid grid-cols-2 gap-6 mt-6">
<div class="card">
<h3>Le raisonnement améliore la qualité…</h3>
<p>Il divise les désaccords par 5 à 20. Mais il coûte 6 000 à 11 000 tokens par appel, soit 6 à 8 fois plus cher par exemple gardé.</p>
</div>
<div class="card accent">
<h3>Décision : sans raisonnement</h3>
<p>Avec 9 $ de budget, on génère beaucoup et <b>on laisse le filtre trier</b>. Le filtre ne déforme pas la distribution : on garde à peu près la même part de simples, intermédiaires et pièges.</p>
</div>
</div>

<p class="small muted mt-4">Bugs trouvés au pilote : le professeur écrivait parfois <code>"x".repeat(30)</code> dans du JSON, et une réponse qui boucle ne s'arrête jamais sans <code>max_tokens</code>.</p>

---

# Le résultat de la génération

<div class="grid grid-cols-4 gap-4 mt-4">
<div class="card"><div class="big">27 460</div><div class="big-label">besoins générés</div></div>
<div class="card good"><div class="big">9 819</div><div class="big-label">gardés (36 %)</div></div>
<div class="card"><div class="big">6,05 $</div><div class="big-label">coût total</div></div>
<div class="card"><div class="big">0,62 $</div><div class="big-label">pour 1 000 exemples gardés</div></div>
</div>

<div class="grid grid-cols-2 gap-6 mt-6">
<div>

**Le taux de survie dépend beaucoup de la catégorie**

<BarChart :rows="[
  { label: 'identifiers', value: 61, tone: 'good' },
  { label: 'moyenne', value: 36, tone: 'base' },
  { label: 'logs', value: 9, tone: 'bad' },
]" />

<p class="small mt-2">Les lignes de logs ont des règles longues et imbriquées : le professeur s'y trompe plus. D'où un <b>complément ciblé</b> de 700 lots sur les catégories qui rejettent le plus.</p>

</div>
<div>

**Le découpage**

<div class="flex gap-1 mt-3 h-10 text-xs text-white font-bold">
  <div class="rounded flex items-center justify-center bg-indigo-500" style="width: 90%">train · 90 %</div>
  <div class="rounded flex items-center justify-center bg-amber-500" style="width: 5%">v</div>
  <div class="rounded flex items-center justify-center bg-emerald-500" style="width: 5%">h</div>
</div>

<ul class="small mt-3">
<li><b>train</b> : ce que le modèle étudie</li>
<li><b>valid</b> : suivi pendant l'entraînement, pour repérer le sur-apprentissage</li>
<li><b>held-out</b> : un contrôle final, jamais vu</li>
</ul>

<p class="small">Le découpage se fait <b>par lot de génération</b> : les variantes d'un même lot restent du même côté, pour éviter les quasi-doublons entre train et held-out.</p>

</div>
</div>

---

# Peut-on faire confiance aux données ?

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

**Le second avis** : Qwen3-235B, un modèle ~400 fois plus gros que le nôtre, écrit sa propre regex pour 200 exemples gardés.

<BarChart :rows="[
  { label: 'accord avec les tests', value: 66.5, tone: 'good' },
  { label: 'tests réussis', value: 88, tone: 'base' },
  { label: 'mal formées', value: 8, tone: 'bad' },
]" />

<p class="small mt-3">Sur 15 désaccords inspectés à la main, la référence du professeur est juste dans <b>13 cas</b>, et 2 sont ambigus. Le bruit résiduel est faible.</p>

<p class="small">Ce 66,5 % deviendra un <b>repère</b> : c'est ce que fait un gros modèle brut sur cette distribution.</p>

</div>
<div>

**Les trous de couverture**

<div class="chips mt-2">
<span class="chip low">groupes nommés · 0 %</span>
<span class="chip low">lookbehind · 0,9 %</span>
<span class="chip low">flags m et s · &lt; 1 %</span>
<span class="chip low">non ancrées · 1,8 %</span>
</div>

<div class="card warn mt-4">
<h3>Le professeur a des tics</h3>
<p>Pour « la chaîne contient X », il écrit presque toujours <code>/^.*X.*&#36;/</code> au lieu de <code>/X/</code>. Le modèle entraîné héritera de ces habitudes.</p>
</div>

<div class="takeaway">
Un dataset synthétique reflète son professeur : ses forces, mais aussi ses angles morts. Le benchmark est là pour les révéler.
</div>

</div>
</div>

---
layout: section
---

<div class="kicker">5 · Le benchmark</div>

# Un examen qui ne sort pas du même moule

<p class="opacity-70">Étape 4</p>

---

# Pourquoi un benchmark séparé du held-out ?

<div class="grid grid-cols-2 gap-6 mt-4">
<div class="card">
<h3>Held-out : même moule que le train</h3>
<p>Même professeur, même style, mêmes tics. Utile pour itérer vite, mais il <b>flatte</b> le modèle entraîné : c'est un examen écrit par la personne qui a écrit le cours.</p>
</div>
<div class="card accent">
<h3>Benchmark : un autre auteur</h3>
<p><b>210 cas écrits à la main par Claude</b> (10 par catégorie), qui ne sert à rien d'autre dans le projet. Il fait foi et il est figé, versionné dans <code>benchmark/*.yaml</code>.</p>
</div>
</div>

<div class="mt-6">

```yaml
- id: text-01
  difficulty: simple
  description: >-
    does the text contain the word deprecated, in any letter case,
    as a whole word (not as part of a longer word)
  regex: '/\bdeprecated\b/i'
  match: ['This API is deprecated.', 'DEPRECATED', '(deprecated)', ...]
  no_match: ['undeprecated', 'deprecatedFoo', 'deprecation notice', ...]
  pitfall: Without word boundaries, undeprecated and deprecatedFoo match too.
```

</div>

<div class="analogy">
Le held-out, ce sont les tests unitaires écrits par le dev de la feature. Le benchmark, c'est la recette QA par quelqu'un d'autre, qui cherche exprès les cas limites.
</div>

---

# Plus dur que le train, exprès

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

**Fonctionnalités rares, sollicitées volontairement** (part des regex du benchmark)

<BarChart :max="25" :rows="[
  { label: 'lookahead', value: 20, tone: 'warn' },
  { label: 'non ancrées', value: 10, tone: 'warn' },
  { label: 'backreferences', value: 9, tone: 'warn' },
  { label: 'lookbehind', value: 5, tone: 'warn' },
  { label: 'Unicode \\p{…}', value: 5, tone: 'warn' },
  { label: 'groupes nommés', value: 3, tone: 'warn' },
]" />

<p class="small mt-3">60 simples, 79 intermédiaires, <b>71 difficiles</b>. Chaque cas documente son <code>pitfall</code>, le piège principal.</p>

</div>
<div>

**La décontamination**

<p class="small">38 cas (18 %) ont une regex identique à une regex du train : IPv4, semver, date ISO… L'écriture canonique de ces besoins standard est unique.</p>

<div class="card mt-2">
<h3>On retire du <i>train</i>, pas du benchmark</h3>
<p>Ces besoins fréquents doivent rester dans l'examen. On retire donc du train tout exemple trop proche d'un cas du benchmark :</p>
<ul>
<li>même regex de référence, ou</li>
<li>description proche : <b>similarité de Jaccard ≥ 0,5</b> sur les mots et les paires de mots (taille de l'intersection / taille de l'union)</li>
</ul>
</div>

<p class="small mt-3">→ 551 exemples retirés. Il reste <b>9 268 exemples</b> (8 321 train · 505 valid · 442 held-out).</p>

</div>
</div>

---
layout: section
---

<div class="kicker">6 · Les baselines</div>

# Que vaut le modèle brut ?

<p class="opacity-70">Étape 5 : le point de référence de tout le projet</p>

---

# Zero-shot vs few-shot

<div class="grid grid-cols-2 gap-6 mt-4">
<div class="card">
<h3>Zero-shot</h3>
<p>Le même prompt que pour l'entraînement : le message système, puis la description. Aucun exemple de réponse.</p>
</div>
<div class="card">
<h3>Few-shot</h3>
<p>On ajoute au prompt <b>3 exemples fixes</b> de question / réponse, avant la vraie question. Le modèle « voit » le format attendu.</p>
</div>
</div>

```text {*}{class:'mt-4'}
[system]    You write JavaScript regular expressions…
[user]      Description: French license plate: two uppercase letters, a hyphen, …
[assistant] /^[A-Z]{2}-\d{3}-[A-Z]{2}$/
[user]      Description: does the commit message contain the word wip, …
[assistant] /\bwip\b/i
[user]      Description: A semicolon-separated list of one to five …
[assistant] /^[A-Z]{2}(?:;[A-Z]{2}){0,4}$/
[user]      Description: <la vraie question>
```

<div class="analogy">
Le few-shot, c'est montrer trois PR bien faites à un nouveau avant sa première PR. Ça ne change rien à ses compétences, mais ça cadre le format. Rien n'est appris : les paramètres ne bougent pas.
</div>

---

# Les modèles bruts ne savent pas faire la tâche

<div class="grid grid-cols-2 gap-8 mt-2">
<div>

**Benchmark : succès complet**

<BarChart :max="10" :rows="[
  { label: '0.6B', value: 0.0, tone: 'bad' },
  { label: '0.6B few-shot', value: 0.9, tone: 'bad' },
  { label: '1.7B', value: 0.9, tone: 'bad' },
  { label: '1.7B few-shot', value: 3.3, tone: 'bad' },
]" />

<p class="small muted">Échelle de 0 à 10 % pour que les barres soient visibles.</p>

**Benchmark : sorties mal formées**

<BarChart :rows="[
  { label: '0.6B', value: 29.0, tone: 'warn' },
  { label: '0.6B few-shot', value: 33.8, tone: 'warn' },
  { label: '1.7B', value: 60.5, tone: 'bad' },
  { label: '1.7B few-shot', value: 20.5, tone: 'warn' },
]" />

</div>
<div>

<div class="card bad tight">
<h3>Au mieux 5 % de regex entièrement justes</h3>
<p>Sur le held-out comme sur le benchmark.</p>
</div>

<div class="card mt-3">
<h3>Le 1.7B ne respecte pas le format</h3>
<p>60 % de sorties mal formées en zero-shot : pattern sans délimiteurs, <code>/…/.test(str)</code>, texte autour. Le few-shot corrige surtout ça.</p>
</div>

<div class="card mt-3">
<h3>Quand le format est bon, les détails sont faux</h3>
<p>Flag <code>i</code> ajouté sans raison (accepte <code>UserId</code> pour du camelCase), ancres oubliées, tirets acceptés en début ou fin de nom, <code>\p{L}</code> ignoré.</p>
</div>

</div>
</div>

---
layout: section
---

<div class="kicker">7 · Le SFT avec LoRA</div>

# Apprendre par imitation

<p class="opacity-70">Étape 6</p>

---

# Le SFT : imiter des milliers de bonnes réponses

**SFT** = _Supervised Fine-Tuning_. On montre « entrée → bonne réponse », et on ajuste le modèle pour qu'il reproduise la réponse.

<div class="mt-6">
<div class="flex flex-wrap gap-1 font-mono text-xs">
  <span class="px-2 py-1 rounded bg-gray-400/20 opacity-60">[system] You write JavaScript…</span>
  <span class="px-2 py-1 rounded bg-gray-400/20 opacity-60">[user] Description: ISO 8601 date…</span>
  <span class="px-2 py-1 rounded bg-indigo-500/80 text-white">/^</span>
  <span class="px-2 py-1 rounded bg-indigo-500/80 text-white">(?:</span>
  <span class="px-2 py-1 rounded bg-indigo-500/80 text-white">19</span>
  <span class="px-2 py-1 rounded bg-indigo-500/80 text-white">|</span>
  <span class="px-2 py-1 rounded bg-indigo-500/80 text-white">20</span>
  <span class="px-2 py-1 rounded bg-indigo-500/80 text-white">)\d</span>
  <span class="px-2 py-1 rounded bg-indigo-500/80 text-white">{2}</span>
  <span class="px-2 py-1 rounded bg-indigo-500/80 text-white">-…</span>
</div>
<div class="flex gap-6 text-xs mt-2">
  <span><span class="inline-block w-3 h-3 rounded bg-gray-400/40 align-middle"></span> prompt : pas de loss (<code>mask_prompt</code>)</span>
  <span><span class="inline-block w-3 h-3 rounded bg-indigo-500 align-middle"></span> réponse : on mesure l'erreur token par token</span>
</div>
</div>

<div class="grid grid-cols-3 gap-4 mt-6">
<div class="card">
<h3>1. Prédire</h3>
<p>Pour chaque token de la réponse, le modèle donne une probabilité au <b>bon</b> token.</p>
</div>
<div class="card">
<h3>2. Mesurer la surprise</h3>
<p>La <b>loss</b> (entropie croisée) : elle est haute quand le modèle était surpris par la bonne réponse, et proche de 0 quand il l'attendait.</p>
</div>
<div class="card">
<h3>3. Corriger</h3>
<p>On ajuste les poids dans la direction qui fait baisser la loss : c'est la <b>descente de gradient</b>.</p>
</div>
</div>

<div class="analogy">
C'est l'objectif du pré-training (prédire le token suivant), mais sur des données choisies et formatées pour notre tâche. On n'apprend pas au modèle à réécrire le prompt, d'où le masque.
</div>

---

# LoRA : un patch plutôt qu'une réécriture

<div class="grid grid-cols-2 gap-8">
<div>

Un LLM, c'est surtout de grosses matrices de poids **W**. Au lieu de les modifier, on ajoute à côté un petit correctif **B × A**.

<LoraDiagram class="mt-2" />

<p class="small">La sortie de la couche devient <code>W·x + B·A·x</code>. B est initialisée à zéro : au départ, le modèle se comporte exactement comme l'original.</p>

</div>
<div>

<div class="card">
<h3>L'hypothèse de LoRA</h3>
<p>La modification nécessaire pour adapter le modèle à une tâche est « simple » : on peut l'approximer par le produit de deux matrices très fines. <b>r</b> (le rang) règle leur épaisseur, donc la capacité du patch.</p>
</div>

<div class="card mt-3 accent">
<h3>Ce qu'on obtient</h3>
<p>Un <b>adapter</b> : un fichier de quelques dizaines de Mo qui ne contient que A et B. On le charge par-dessus le modèle de base, ou on le fusionne dedans (<code>mlx_lm.fuse</code>).</p>
</div>

<div class="analogy">
Un calque de réglage dans Figma ou Photoshop : l'image d'origine reste intacte, le calque se pose par-dessus, et on peut en empiler plusieurs (SFT, DPO, GRPO) pour comparer. Ou un <code>patch-package</code> sur un <code>node_modules</code>.
</div>

</div>
</div>

---

# Pourquoi LoRA est ce qu'il nous faut

<div class="grid grid-cols-2 gap-8 mt-2">
<div>

| | Fine-tuning complet | LoRA |
| --- | --- | --- |
| Paramètres entraînés | 100 % | **1,7 %** ici |
| Mémoire (1.7B) | ~20 Go et plus | ~5 Go |
| Mémoire (4B) | ne tient pas en 24 Go | ~10 Go |
| Ce qu'on sauvegarde | tout le modèle | quelques dizaines de Mo |
| Oubli des capacités générales | risque plus élevé | limité, W reste intact |

<p class="small mt-3">Pourquoi le fine-tuning complet coûte autant : pour chaque poids entraîné, il faut stocker le poids, son <b>gradient</b> et deux états de l'optimiseur Adam. Avec LoRA, W est gelé : rien de tout ça pour lui.</p>

</div>
<div>

**La recette** (`configs/sft.yaml`)

<div class="card">
<ul>
<li><b>Rang 16</b>, scale 2, dropout 0,05</li>
<li>Sur <b>l'attention et le MLP</b> de toutes les couches</li>
<li>Loss sur la réponse seulement (<code>mask_prompt</code>)</li>
<li>Optimiseur AdamW, LR 2e-4</li>
<li>Batch effectif de 8, <b>2 epochs</b></li>
</ul>
</div>

<div class="card mt-3">
<h3>Le learning rate (LR) : la taille des pas</h3>
<LrChart class="max-w-72" />
<p class="small">Petits pas au début (<b>warmup</b>, pour ne pas tout casser), puis on ralentit progressivement pour affiner.</p>
</div>

</div>
</div>

---

# Le vocabulaire de l'entraînement

<div class="grid grid-cols-3 gap-4 mt-4">
<div class="card">
<h3>Batch</h3>
<p>On ne corrige pas les poids après chaque exemple, mais après un <b>paquet</b> d'exemples (ici 8), en moyennant l'erreur. Plus stable.</p>
</div>
<div class="card">
<h3>Epoch</h3>
<p>Un passage complet sur tout le train. Ici 2 : chaque exemple est vu deux fois.</p>
</div>
<div class="card">
<h3>Iteration / step</h3>
<p>Un batch traité. 8 300 exemples / 8 ≈ 1 040 mises à jour par epoch.</p>
</div>
<div class="card">
<h3>Train loss / val loss</h3>
<p>L'erreur sur les exemples étudiés, et sur les exemples de validation, jamais étudiés. C'est l'écart entre les deux qui compte.</p>
</div>
<div class="card">
<h3>Gradient</h3>
<p>Pour chaque poids : dans quel sens le bouger pour faire baisser la loss, et de combien.</p>
</div>
<div class="card">
<h3>Hyperparamètres</h3>
<p>Les réglages qu'on choisit (rang, LR, batch, epochs…), par opposition aux paramètres, que le modèle apprend.</p>
</div>
</div>

<div class="analogy">
Les hyperparamètres sont la config du build ; les paramètres sont l'artefact produit.
</div>

---

# Lire les courbes : apprend-il, ou récite-t-il ?

<div class="grid grid-cols-5 gap-6">
<div class="col-span-3">

<LossChart />

</div>
<div class="col-span-2">

<div class="card good">
<h3>8 300 exemples : sain ✅</h3>
<p>La val loss baisse jusqu'au bout (2,45 → 0,164), en ralentissant avec le learning rate. Pas de plateau : <b>plus de données aiderait encore</b>.</p>
</div>

<div class="card warn mt-3">
<h3>2 000 exemples : plateau ⚠️</h3>
<p>La val loss stagne dès la 2ᵉ epoch, alors que la train loss continue de baisser. L'écart se creuse.</p>
</div>

</div>
</div>

<div class="card tight mt-2">
<h3>L'overfitting (sur-apprentissage)</h3>
<p>Le modèle <b>apprend par cœur</b> ses exemples au lieu de la règle générale. Il devient excellent sur ce qu'il a vu, et stagne ou régresse sur le reste. Le signal : la train loss baisse, la val loss ne suit plus.</p>
</div>

---

# Le résultat : de 2,5 % à 72 %

<div class="grid grid-cols-2 gap-8 mt-2">
<div>

**Held-out** (même distribution que le train) : succès complet

<BarChart :rows="[
  { label: 'Qwen3-0.6B brut', value: 2.5, tone: 'bad' },
  { label: 'Qwen3-235B brut (repère)', value: 66.5, tone: 'muted' },
  { label: 'Qwen3-0.6B + SFT', value: 71.7, tone: 'good' },
]" />

**Benchmark** (plus dur, autre auteur) : succès complet

<BarChart :rows="[
  { label: 'Qwen3-0.6B brut', value: 0.0, tone: 'bad' },
  { label: 'Qwen3-0.6B + SFT', value: 31.4, tone: 'good' },
]" />

</div>
<div>

| Qwen3-0.6B | Brut | SFT |
| --- | --- | --- |
| Held-out : succès | 2,5 % | **71,7 %** |
| Held-out : tests réussis | 46,5 % | **92,0 %** |
| Held-out : mal formées | 24,4 % | **1,8 %** |
| Benchmark : succès | 0,0 % | **31,4 %** |
| Benchmark : tests réussis | 40,6 % | **74,2 %** |
| Benchmark : mal formées | 29,0 % | **7,1 %** |

<div class="takeaway">
Un modèle de 0,6 milliard de paramètres, entraîné ~1 h sur un Mac, fait jeu égal sur le held-out avec un modèle ~400 fois plus gros.
</div>

</div>
</div>

---

# Le benchmark révèle les limites

<div class="grid grid-cols-2 gap-8 mt-2">
<div>

**Benchmark : succès par difficulté**

<BarChart :rows="[
  { label: 'simple', value: 67, tone: 'good' },
  { label: 'intermédiaire', value: 27, tone: 'warn' },
  { label: 'difficile', value: 7, tone: 'bad' },
]" />

**Par catégorie : les plus faibles**

<BarChart :rows="[
  { label: 'multiline', value: 20, tone: 'bad' },
  { label: 'logs', value: 0, tone: 'bad' },
]" />

<p class="small">Ce sont aussi les catégories les moins représentées du train. 23 des 144 échecs sont « de peu » (≥ 85 % des tests réussis).</p>

</div>
<div>

**Erreurs typiques**

<div class="grid grid-cols-2 gap-2">
<div class="card tight">
<h3>Contrainte ignorée</h3>
<p>Un palindrome de 5 lettres devient <code>/^[a-z]{5}&#36;/</code> : les backreferences sont rares dans le train.</p>
</div>
<div class="card tight">
<h3>Couverture partielle</h3>
<p>Mois abrégés seulement, quand les noms complets étaient aussi demandés.</p>
</div>
<div class="card tight">
<h3>Détail du texte</h3>
<p><code>\s</code> au lieu d'une espace ; encodage <code>%XX</code> non vérifié.</p>
</div>
<div class="card tight">
<h3>Sorties cassées (7 %)</h3>
<p>Groupe non fermé, ou boucle jusqu'à la limite de tokens.</p>
</div>
</div>

</div>
</div>

---

# L'effet du volume : 2 000 vs 8 300 exemples

Même recette, 2 000 exemples tirés au hasard du train, même valid.

| Qwen3-0.6B SFT | Val loss | Held-out : succès | Benchmark : succès | Benchmark : mal formées | Durée |
| --- | --- | --- | --- | --- | --- |
| 2 000 exemples | 0,272 | 56,6 % | 20,0 % | 7,6 % | ~12 min |
| 8 300 exemples | **0,164** | **71,7 %** | **31,4 %** | 7,1 % | ~1 h |

<div class="grid grid-cols-2 gap-6 mt-6">
<div class="card">
<h3>Le format s'apprend vite</h3>
<p>Avec 2 000 exemples, les sorties mal formées sont déjà au même niveau. Respecter <code>/…/flags</code>, c'est facile.</p>
</div>
<div class="card accent">
<h3>La justesse demande du volume</h3>
<p>+15 points sur le held-out, +11 sur le benchmark. Les simples passent de 43 à 67 %, les intermédiaires de 16 à 27 %, les difficiles de 4 à 7 %.</p>
</div>
</div>

<div class="takeaway">
La courbe n'est pas saturée : plus de données aiderait probablement encore, surtout dans les catégories sous-représentées.
</div>

---
layout: section
---

<div class="kicker">8 · Bilan</div>

# Où on en est

---

# Ce qu'on a appris

<div class="grid grid-cols-2 gap-6 mt-2">
<div>

<div class="card">
<h3>📏 Mesurer d'abord</h3>
<p>Un harnais d'éval fiable (Node, timeouts, parsing strict) avant la moindre ligne d'entraînement.</p>
</div>
<div class="card mt-3">
<h3>🧪 Les données font le modèle</h3>
<p>36 % des données générées survivent aux garde-fous. Le dataset hérite des tics du professeur.</p>
</div>
<div class="card mt-3">
<h3>🎓 Un examen indépendant</h3>
<p>Le held-out flatte (72 %), le benchmark dit la vérité (31 %).</p>
</div>

</div>
<div>

<div class="card">
<h3>🧩 LoRA rend tout ça possible en local</h3>
<p>1,7 % des paramètres entraînés, 4,6 Go de pic mémoire, ~1 h sur un Mac.</p>
</div>
<div class="card mt-3">
<h3>💸 Pour presque rien</h3>
<p>6,05 $ de génération de données. Le reste tourne sur la machine.</p>
</div>
<div class="card mt-3 warn">
<h3>🚧 La limite du SFT</h3>
<p>Le modèle <b>imite</b>. Il ne voit jamais ses propres erreurs, et ne sait pas qu'une réponse est « presque juste » ou « complètement fausse ».</p>
</div>

</div>
</div>

---

# La suite : apprendre de ses propres erreurs

<div class="flow mt-8">
  <div class="step done">
    <strong>✅ SFT</strong>
    « Fais comme ça. »<br>
    <span class="small">Imitation de bonnes réponses.</span>
  </div>
  <div class="arrow">→</div>
  <div class="step" style="border-color: var(--c-accent)">
    <strong>DPO</strong>
    « Plutôt ça que ça. »<br>
    <span class="small">Le modèle génère plusieurs réponses, on les note par exécution, il apprend à préférer la bonne à la mauvaise.</span>
  </div>
  <div class="arrow">→</div>
  <div class="step" style="border-color: var(--c-accent)">
    <strong>GRPO</strong>
    « Essaie, je te note. »<br>
    <span class="small">Apprentissage par renforcement : la récompense est le taux de tests réussis.</span>
  </div>
</div>

<div class="card mt-10">
<h3>Le genre d'erreur qu'on vise</h3>
<p>IPv6 complète : <code>/^(?:[0-9A-Fa-f]{4}:){7}[0-9A-Fa-f]{4}&#36;/</code> est juste, <code>…{1,4}…</code> est trop permissive. C'est « presque juste », et le SFT ne pénalise jamais directement une mauvaise réponse.</p>
</div>

---
layout: center
class: text-center
---

# Merci

<div class="opacity-70 mt-4">
Tout le code, les configs et les résultats sont dans le repo :<br>
<code>scripts/</code> · <code>configs/</code> · <code>benchmark/</code> · <code>results/scores.jsonl</code>
</div>
