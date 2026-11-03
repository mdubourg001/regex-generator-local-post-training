# Dashboard des entraînements et des évaluations

Petite page locale pour suivre les entraînements et les évals au fil de l'eau : elle lit `logs/*.log` et
`results/scores.jsonl` (une ligne par run de `scripts/eval.py`) toutes les 5 s, sans rebuild.

## Lancer

```sh
cd dashboard
pnpm install
pnpm dev        # http://localhost:5297
```

Les données sont servies par deux petits plugins Vite (`vite.config.ts`), relues à chaque requête :
`/api/scores` renvoie `scores.jsonl` tel quel, `/api/training` les logs d'entraînement analysés par `training.ts`.
Ça ne marche donc qu'avec `pnpm dev` : `pnpm build` vérifie les types et le bundle, mais `dist/` n'a pas de données.

## Ce qu'on voit

### Entraînements

Une carte par log d'entraînement reconnu : SFT (`mlx_lm.lora`), DPO et variantes (`mlx_lm_lora`), GRPO
(script maison, où la courbe est la reward). Les autres logs (évals, génération…) sont ignorés. Par défaut,
seuls les logs modifiés depuis 24 h sont affichés.

- **Progression** : itération courante sur le total, en micro-batchs (avec une accumulation de gradient
  sur 4, 9 420 itérations font 2 355 mises à jour).
- **Fin estimée** : d'après la vitesse moyenne depuis la création du log, validations comprises (le dernier
  it/s rapporté les exclut). Après une reprise ajoutée au même log, on retombe sur le dernier it/s.
- **Courbe** : loss train et val (échelle log), ou reward pour GRPO.
- **Pic mémoire** : celui que rapporte MLX, en dessous de la pression réelle sur la machine.
- **Statut** : terminé (`Saved final weights`, ou dernier pas GRPO atteint) ; interrompu (erreur dans le log,
  ou aucune nouvelle ligne depuis 5 min) ; actif sinon.

### Évaluations

- **Filtres** : jeu d'éval (par défaut regex101 « réel », le plus fiable ; held-out ; benchmark et
  benchmark-gpt, écrits par un LLM donc biaisés ; « autre » pour les sous-ensembles et échantillons),
  recherche sur le nom ou le modèle, baselines (modèle du Hub sans adapter) affichées ou non.
- **Runs répétés** : quand un même `name` apparaît plusieurs fois, seule la dernière ligne est affichée ;
  la case « runs précédents » montre aussi les anciennes, grisées.
- **Barres** : taux de succès des runs filtrés, échelle relative au meilleur.
- **Tableau** triable (clic sur l'en-tête, succès décroissant par défaut) : succès complet, part moyenne
  de tests réussis, réponses mal formées, timeouts, durée de génération.
- Les runs apparus depuis l'ouverture de la page sont marqués « nouveau ».
