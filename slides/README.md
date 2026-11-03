# Slides : post-training d'un petit LLM, partie 1

Présentation Slidev du repo, centrée sur le SFT et le RFT (DPO et GRPO résumés en une slide).

```sh
pnpm install
pnpm dev       # présentation sur http://localhost:3030
pnpm export    # PDF (utilise playwright-chromium)
```

- `slides.md` : le contenu
- `components/` : graphiques en Vue (barres, courbes, learning rate, schéma LoRA)
- `style.css` : cartes, encadrés « Côté front », schémas en boîtes

Les chiffres viennent du README, de `results/scores.jsonl` et des logs d'entraînement (`logs/sft-0.6b*.log`, `logs/rft-0.6b.log`).
