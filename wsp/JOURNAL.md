# Journal — weight-space programming sur Qwen3-0.6B (regex JS)

Objectif : améliorer Qwen/Qwen3-0.6B (thinking désactivé) sur la génération de regex en éditant ses poids à la main,
sans gradient ni optimiseur, puis comparer au SFT LoRA.

**Résultat en une ligne** : les éditions divisent presque par deux les sorties malformées du held-out (24,4 % → 15,8 %)
et montent le taux de tests réussis (46,5 % → 53,4 %), mais le succès complet ne bouge pas (2,5 % → 2,3 %, soit 11 → 10
cas sur 442). Le SFT est à 71,7 %. Le format n'était pas l'obstacle qui masquait une capacité : la capacité manque.

## Cadre

- Modèle : 596 049 920 poids en bf16, 28 couches, 16 têtes (8 têtes clé/valeur partagées par paires), embeddings liés à
  l'unembedding, aucun biais.
- Jeu de dev : 200 cas de `valid.jsonl`, seed 0, ids dans `wsp/dev_ids.json`. Les 305 autres cas de valid forment le
  « pool de sonde ».
- Budget : **9 évaluations dev sur 50**, listées dans `wsp/dev_evals.jsonl` (détail par cas dans `wsp/runs/`).
- Sondes : au plus 64 prompts chacune, sur le pool de sonde ou sur train. Une exception assumée : les sondes 7 et 8
  génèrent sur 192 prompts de train (3 lots de 64) pour réunir 18 sorties en boucle.
- Aucune lecture de `heldout.jsonl`, de `data/eval/` ni du détail de `results/*.jsonl` avant l'évaluation finale.
- Pic mémoire : 2,7 Go.
- Chaque édition est un fichier `wsp/edits/<nom>.npz` rejouable (`wsplib.apply_edits`). Le modèle final se reconstruit
  avec `uv run wsp/build_model.py`.

Précision du dev : avec 200 cas et ~3 % de succès, l'écart-type est de 1,2 point. Toutes les variantes ci-dessous
tiennent entre 1 et 7 cas réussis : **le succès complet ne départage rien**. Seuls le taux de malformés et le taux de
tests réussis bougent de façon lisible.

## 1. Diagnostic du modèle brut (éval 1)

`uv run wsp/diagnose.py e00-base` — succès 2,5 %, tests réussis 48,5 %, malformés 20,5 %.

| Catégorie                                                    | Cas | Part   |
| ------------------------------------------------------------ | --- | ------ |
| Succès                                                       | 5   | 2,5 %  |
| Malformé                                                     | 41  | 20,5 % |
| — boucle jusqu'à 128 tokens                                  | 12  | 6,0 %  |
| — pas un littéral (chaîne exemple, pattern sans `/…/`)       | 11  | 5,5 %  |
| — littéral cassé (`/` final absent, `/` non échappé)         | 10  | 5,0 %  |
| — suffixe cassé (`/.i`, `.test`)                             | 5   | 2,5 %  |
| — ne compile pas (échappement, quantificateur, groupe)       | 3   | 1,5 %  |
| Habitude : flag `i` en trop (juste sans lui)                 | 4   | 2,0 %  |
| Ancrage (juste une fois ancré `^…$`)                         | 1   | 0,5 %  |
| Sur-généralisation (accepte tout le valide, et trop)         | 38  | 19,0 % |
| Sous-généralisation (ne refuse rien à tort, rate du valide)  | 63  | 31,5 % |
| Mauvaise sémantique (rate des deux côtés)                    | 48  | 24,0 % |

Habitudes sur les 159 sorties bien formées : flag `i` dans **93,7 %** (6,9 % des références), `$` final dans 74,8 %
(96,2 %), copie exacte de `/^\d+$/i` — l'exemple du prompt système — dans 4,4 %.

**Le chiffre qui borne toute l'expérience** : en re-notant les sorties avec le flag `i` retiré et les ancres ajoutées,
on obtient au mieux 9 succès sur 200 (4,5 %). Les 41 malformés ne cachent pas de regex justes. 74,5 % des échecs sont
sémantiques : le modèle écrit `/^\d{40}$/` pour un hash hexadécimal, `/^\w+$/` pour une branche git.

## 2. Ce que j'ai trouvé dans les poids

### 2.1 Le flag `i` est copié de l'exemple du prompt système par un circuit d'induction

- **Sonde 1** (`probe_flag.py`, 48 références forcées) : après le `/` fermant, logit(`i`) − logit(fin) = **+8,8**,
  P(`i`) = 0,99.
- **Sonde 2** (`probe_example.py`, intervention sur le prompt) : sans le `i` dans l'exemple, le flag tombe de 47/51 à
  3/42. Avec `/^[a-z]+$/i` comme exemple, les sorties contenant `\d` passent de 23/51 à 2/44. Le modèle récite
  l'exemple.
- **Sonde 3** (`probe_attn.py`) : circuit d'induction classique. Des têtes de L2 (2.11, 2.10) repèrent l'occurrence
  précédente de `$/`, puis des têtes de L16 à L23 regardent le token qui la suivait, le `i` du prompt système :
  L16H14 (0,85 d'attention), L19H5, L18H5, L19H13, L20H14, L23H4, L18H11, L18H4, L20H15, L19H9 (0,39 à 0,50).
  Les MLP 19 à 24 amplifient (+12 de marge), L24H7 et L25H9 écrivent directement (+2,1 et +3,9), MLP27 freine (−11).
- **Sonde 4** (`probe_neurons.py`) : aucun « neurone du flag ». La contribution des MLP est diffuse, les neurones sont
  polysémantiques. Piste abandonnée.
- **Sonde 6** (`probe_flag_signal.py`) : la marge dépend à peine de la description (+11,1 quand la référence porte `i`,
  +8,7 sinon, 61 % de bien classés). Le modèle ne sait presque pas quand `i` est justifié.

**Édition « aveugle »** (`edit_blind.py`). Le prompt système est le même pour tous les cas, donc le résiduel à la
position du `i` de l'exemple (position 61) est un vecteur constant x_l à chaque couche. On retire de `v_proj` des têtes
visées la composante propre de ce vecteur : û = x_l[61] − moyenne des autres positions du prompt système, normalisé,
puis W_V ← W_V − (W_V û) ûᵀ. Une mise à jour de rang 1 par bloc de tête, calculée par une seule passe forward.

| Essai | Têtes aveuglées                              | Preuve (sonde)                                                                                   | Décision |
| ----- | -------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------- |
| e01   | 10 têtes d'induction, L16–L23                | Marge +8,6 → +1,2. Mais le 1er token bascule vers `` ` `` sans slash : malformés 27 % → 58 %     | Rejetée  |
| e02   | idem, û orthogonalisé aux positions 57–60    | Format préservé, marge +2,5, flag encore 37/45                                                   | Dépassée par e18 |
| e15   | toutes les têtes de L14 à L27                | Marge **−8,8**, flag 0/46 : le circuit explique tout. Mais le modèle ne s'arrête plus (`$/^…`)   | Rejetée  |
| —     | toutes les têtes de L0 à L27                 | Tout casse (malformés 72 %) : les couches basses lisent ce token pour construire l'exemple       | Rejetée  |
| **e18** | les 10 d'induction + L24H7, L24H11, L25H5, L25H9 | Marge −1,7 (réf. sans flag) et +1,2 (réf. avec flag) : la décision revient à la description | **Gardée** |

Leçon de e01 : le résiduel du `i` ressemble à celui des autres tokens de l'exemple. Sans orthogonalisation, aveugler
« le `i` » aveugle aussi ` /^\`, que d'autres têtes (L18H11, L24H6, L23H5) lisent pour choisir le premier token.

Leçon de e15 : le `i` servait de rituel de fin. Sans lui, le modèle ferme le littéral puis recommence (`/^\w+$/?=.*…`).
Il faut donc aussi lui dire de s'arrêter (2.3).

### 2.2 L'amorce de la réponse décide du style du corps

- **Sonde 9** (`probe_first_token.py`, 64 cas de train, premier token forcé, mêmes poids) :

  | Amorce forcée          | Corps `^\d+…` ou `^\w+…` | Malformés | Tests réussis |
  | ---------------------- | ------------------------ | --------- | ------------- |
  | `/` seul               | 21/58                    | 10,9 %    | 53,8 %        |
  | `/^` (un seul token)   | 0/45                     | 31,2 %    | 42,4 %        |
  | `"/`                   | 5/54                     | 17,2 %    | 53,6 %        |
  | ` ```javascript\n/ `   | 10/59                    | 17,2 %    | 53,9 %        |

  Après un `/` nu, le modèle est en « mode récitation » de l'exemple : corps courts et paresseux, mais bien formés.
  Enveloppé de guillemets ou d'un bloc de code, il écrit des classes de caractères plus élaborées, boucle davantage et
  casse plus souvent le littéral. Ce n'est pas un obstacle qu'on retire : c'est un compromis.
- **Sonde 5** (`probe_body.py`) : après `/^\` forcé, P(`d`) = 0,44, dont 0,23 quand la référence n'a pas de `\d`.
  L16H14 pose 0,8 à 0,9 de son attention sur le token suivant de l'exemple à chaque pas.

| Essai | Édition                                                         | Preuve                                                                                              | Décision |
| ----- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------- |
| e04   | Ablation de L16H14 (colonnes de `o_proj` à 0, 131 072 poids)    | P(`d`) inchangé (0,44 → 0,48), boucles inchangées. **Hypothèse réfutée** : la tête regarde, d'autres écrivent | Rejetée |
| e05   | Aveugler 16.14, 16.7, 21.8, 21.9 aux positions de `d` et `+`    | P(`d`) 0,44 → 0,37                                                                                  | Trop faible |
| e22   | Aveugler toutes les têtes L14–L27 aux positions de `d` et `+`   | Copie retirée là où elle est fausse : P(`d`) 0,17 → **0,03** sans `\d` dans la référence, 0,68 → 0,50 avec. Mais malformés 11 % → 30 %, boucles 9 → 12 sur 64 | Rejetée |

Leçon de e22 : l'exemple est à la fois une mauvaise habitude et une béquille. Il donne au modèle un gabarit de
concision. Sans lui, le modèle divague (`…$|^…$|^…`).

### 2.3 Le dernier MLP comme mémoire clé → valeur

`edit_memory.py` : W_down(L27) += α · d ⊗ cᵀ, de rang 1.

- Clé c : différence de moyennes des activations cachées du MLP 27 entre la position visée et les autres positions (64
  exemples de train, références forcées), restreinte aux 32 neurones les plus discriminants, orthogonalisée à la
  moyenne de contrôle.
- Valeur d : direction d'unembedding des tokens voulus moins celle des tokens à écarter.
- α : calculé pour déplacer la marge de 8 logits, la marge observée étant de 3 à 6. Aucune recherche de coefficient.

| Essai | Position visée                | Valeur                                                    | Preuve                                                                                 | Décision |
| ----- | ----------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------- |
| e03   | 1er token, clé dense          | `/`, `/^` contre `"/`, `` ` ``, ` ``` `, `"`…             | Marche, mais la clé fuit : 36 % des positions de contrôle répondent à plus de 0,3      | Remplacée |
| **e07** | 1er token, clé de 32 neurones | idem                                                    | Fuite 0,6 %, réponse 1,01 sur la cible (min 0,92). Enveloppe nue dans 200/200 cas dev   | **Gardée** |
| e06, e08 | fin de littéral            | fin de séquence contre `i g m . , ; "`                    | Marge −5,3, mais `/.test`, `/.trim()` subsistent                                       | Remplacées |
| e09   | « sur le point de fermer »    | `/` contre `/.`, `/,`, `/i`…                              | Clé non séparable (z de 1,1 à 2,6, fuite 76 %)                                         | Rejetée |
| e12   | fin de littéral               | fin de séquence seule, +10                                | Fins prématurées (`…*$` sans `/`), `…$/$`                                              | Rejetée |
| **e20** | fin de littéral, 2 découpages | fin de séquence contre `i g m . , ; " $ ⏎⏎ ⏎`, +8       | Fin de séquence après le `/` final dans 48/48 cas                                      | **Gardée**, avec réserve |
| e23, e24, e27 | 1er token             | `/` seul ; puis enveloppes tolérées ; puis `"/`           | Voir 2.2 : aucune amorce n'est meilleure partout                                       | Rejetées |
| e26   | fin de littéral               | fin de séquence seule, +5                                 | Dev : malformés 9,0 % (le meilleur) mais 2 succès                                      | Rejetée |

Réserve sur e20 : sa valeur pénalise des tokens légitimes dans un corps (`$`, `.`, `,`). Sa fuite fait monter les
boucles de 10 à 16 sur le dev et produit du texte parasite après certains `/` (`/$/`, ` McGregor`). e26 corrige cela
mais perd des succès, pour une raison instructive : e20 et e26 partagent les trois autres éditions, et pourtant
`/^sk_live_([0-9a-zA-Z]{32})$/` devient `/^\d+$/`. Une édition de rang 1 en dernière couche, censée n'agir qu'à la fin,
change le corps par sa fuite sur les premières positions.

### 2.4 Bannir une famille de tokens par leur ligne d'embedding

Après un `$` isolé, le modèle ferme avec le token `/.` (d'où `/.test`, `/.i`), `/,` ou `/i`. Bannir huit tokens (e11)
déplace le problème vers `/$`. D'où **e19** (`edit_ban.py`) : les 1 375 tokens qui contiennent un `/` non échappé suivi
d'autre chose, sauf `/^`. Aucune réponse valide n'en a besoin hors du premier token.

E_t ← E_t − β μ̂, où μ̂ est la direction moyenne du résiduel final normalisé sur les positions de réponse (64 exemples
de train) et β = 0,107 donne une baisse de logit de 8,0 ± 1,3. Effet de bord accepté : l'embedding d'entrée de ces
tokens bouge de ~12 % de sa norme, dans une direction que la lecture n'utilise pas a priori.

Dev : malformés 16,5 % → 10,0 % (a2 → a3). **Gardée.**

### 2.5 Les boucles : une direction existe, mais ce n'est pas la cause

- **Sonde 7** (`probe_loop.py`, 18 boucles sur 192 générations) : la différence de moyennes du résiduel entre positions
  en boucle et positions normales est cohérente d'une boucle à l'autre (cosinus moyen 0,84 en L15, 0,71 en L27) et
  écrite surtout par MLP27 (+72) et MLP26 (+24). Son logit lens : `])/`, `))/`, `$/`, `)$/`. C'est un signal « il
  faut fermer » qui monte avec la longueur et qui perd. La retirer, comme dans le post, aggraverait les boucles.
- **Sonde 8** (`probe_loop_dla.py`, 120 positions) : le token répété bat la meilleure fermeture de +11,3 logits,
  répartis sur L19 à L25 sans composant dominant. Les seules têtes à la fois d'auto-induction et contributrices sont
  L21H9 (+1,5) et L21H8 (+0,8).
- **e21** : ablation de L21H8 et L21H9. Boucles 9 → 8 sur 64. **Rejetée.**

Pas de levier localisé. C'est la différence avec le cas du post, où deux MLP écrivaient la direction fautive.

## 3. Évaluations sur le dev

| #   | Nom             | Éditions                              | Succès | Tests  | Malformés | Flag `i` | Décision |
| --- | --------------- | ------------------------------------- | ------ | ------ | --------- | -------- | -------- |
| 1   | e00-base        | aucune                                | 2,5 %  | 48,5 % | 20,5 %    | 93,7 %   | référence |
| 2   | e14-format      | e02 + e07 + e13 + e14                 | 3,5 %  | 52,2 % | 16,0 %    | 26,8 %   | dépassée par e20 (flag) |
| 3   | **e20-format2** | **e18 + e07 + e19 + e20**             | 3,0 %  | 52,3 % | 16,0 %    | 8,9 %    | **retenue** |
| 4   | s00-sft-base    | bonus : `models/sft-0.6b`, sans édition | 72,0 % | 91,6 % | 3,0 %   | 6,2 %    | diagnostic |
| 5   | a1-flag         | e18                                   | 3,0 %  | 50,2 % | 17,5 %    | 55,2 %   | incrémental |
| 6   | a2-flag-first   | e18 + e07                             | 1,0 %  | 50,2 % | 16,5 %    | 35,3 %   | incrémental |
| 7   | a3-flag-first-ban | e18 + e07 + e19                     | 0,5 %  | 55,0 % | 10,0 %    | 56,1 %   | incrémental |
| 8   | e26-format3     | e18 + e07 + e19 + e26                 | 1,0 %  | 55,2 % | 9,0 %     | 13,7 %   | rejetée |
| 9   | f00-final-check | `models/wsp-0.6b` relu depuis le disque | 3,0 % | 52,3 % | 16,0 %   | 8,9 %    | 200/200 sorties identiques à l'éval 3 |

Lecture honnête :

- e14 a un succès de plus que e20 (7 contre 6). J'ai retenu e20 parce que son taux de flag (8,9 %) rejoint celui des
  références (5,4 %), ce qui est le mécanisme visé. L'écart d'un cas est du bruit.
- La règle « garder une édition seulement si elle améliore le dev sans faire monter les malformés » n'est pas
  satisfaite édition par édition : e07 seule fait perdre des succès (6 → 2), e19 seule aussi (2 → 1), et e20 fait
  remonter les malformés de 10,0 % à 16,0 %. Seule la pile entière fait mieux que le brut sur les trois colonnes. Les
  succès du brut sont presque tous les mêmes intentions (hash de 64 hex, `sk_live_…`) : une édition qui change
  l'amorce les fait basculer en bloc.
- e26 (malformés 9,0 %, tests 55,2 %) est le meilleur compromis si on juge sur le format et non sur le succès.

## 4. Modèle final

`models/wsp-0.6b` = Qwen3-0.6B + 4 éditions (`wsp/final_edits.json`).

| Édition             | Tenseurs                                                   | Rang | Poids modifiés | % du modèle |
| ------------------- | ---------------------------------------------------------- | ---- | -------------- | ----------- |
| e18-blind-flag-14h  | `v_proj` de L16, L18, L19, L20, L23, L24, L25 (blocs de 128 lignes) | 1 par couche | 1 261 193 | 0,212 % |
| e07-first-sparse    | `mlp.down_proj` de L27, 32 colonnes                        | 1    | 29 150         | 0,005 %     |
| e19-ban-slash       | 1 375 lignes de `embed_tokens`                             | 1    | 1 380 437      | 0,232 %     |
| e20-end             | `mlp.down_proj` de L27, 32 colonnes                        | 1    | 26 736         | 0,004 %     |
| **Total**           |                                                            |      | **2 697 516**  | **0,45 %**  |

Poids comptés après arrondi en bf16.

## 5. Évaluation finale

| Modèle                         | Held-out (442) : succès | tests  | malformés | regex101 réel (400) : succès | tests  | malformés |
| ------------------------------ | ----------------------- | ------ | --------- | ---------------------------- | ------ | --------- |
| Qwen3-0.6B brut                | 2,5 %                   | 46,5 % | 24,4 %    | 2,8 % (few-shot)             | 43,4 % | 31,2 %    |
| **WSP (`models/wsp-0.6b`)**    | **2,3 %**               | **53,4 %** | **15,8 %** | **4,0 %**               | **50,8 %** | **22,5 %** |
| SFT LoRA 0.6B                  | 71,7 %                  | 92,0 % | 1,8 %     | 20,0 %                       | 70,3 % | 7,7 %     |
| SFT LoRA 0.6B plus2            | 73,3 %                  | 92,4 % | 1,8 %     | 22,8 %                       | 71,2 % | 7,5 %     |

- Held-out : 10 succès contre 11. Aucun gain.
- Réel : 16 succès contre 11, mais la seule ligne brute disponible sur ce jeu est en few-shot, donc la comparaison
  n'est pas à prompt égal, et l'écart n'est pas significatif à cette taille.
- Le gain de format se transfère bien du dev au held-out (malformés −8,6 points, tests +6,9 points) et au réel.

## 6. Bonus : le SFT fusionné

Éval 4 : `models/sft-0.6b` sur le dev, 72,0 % de succès. Les 56 échecs se répartissent en 6 malformés (3 sorties
longues, 3 erreurs de compilation), 15 sur-généralisations, 28 sous-généralisations, 7 erreurs des deux côtés. Flag `i`
dans 6,2 % des sorties (références : 7,2 %), `$` final dans 96,9 % (96,4 %).

Je n'ai trouvé aucune habitude systématique à éditer : le SFT a déjà corrigé tout ce que les éditions de format
visent, et ses erreurs sont une par cas (`{0,6}` au lieu de `{0,8}`, `.+` au lieu de `.*`, un séparateur oublié).
Aucune édition n'a été tentée sur ce modèle, faute d'hypothèse à tester. Ce n'est donc pas un résultat négatif mesuré,
c'est une absence de cible.

## 7. Synthèse

**Ce qui a marché**

- Localiser un comportement : la copie du flag `i` passe par un circuit d'induction lisible, et l'aveugler à un seul
  token du prompt fait passer la marge de +8,7 à −8,8. C'est l'analogue le plus proche du résultat du post.
- Réparer le format par des éditions de rang 1 : enveloppe nue dans 100 % des cas, flag au taux des références,
  malformés du held-out de 24,4 % à 15,8 %, avec 0,45 % des poids.

**Ce qui n'a pas marché**

- Le succès complet : inchangé.
- Les boucles : pas de composant localisé.
- La récitation du corps de l'exemple : on sait la retirer, mais le modèle fait pire sans elle.
- La sélectivité : chaque édition côté sortie a déplacé le défaut d'un token au suivant (`/.i` → `/.test` → `/$` →
  `$i` → `$/^…`), et une édition « de fin » a changé le début des réponses par sa fuite.

**Avis franc**

Sur cette tâche, le weight-space programming ne remplace pas le SFT et n'en approche pas. Le post dit qu'une édition
retire un obstacle sans ajouter de capacité : ici le diagnostic montrait dès la première évaluation que l'obstacle
valait au plus 2 points (9 succès sur 200 avec un format parfait). Qwen3-0.6B brut ne sait pas écrire
`(?:25[0-5]|2[0-4]\d|…)` ; le SFT le lui apprend en une heure avec 8 000 exemples, et aucune direction ne le contient.

Ce que la méthode apporte ici :

- un diagnostic causal qu'aucune courbe de loss ne donne (pourquoi le `i`, pourquoi les corps paresseux, pourquoi le
  choix de l'amorce compte) ;
- un constat utile hors de cette expérience : **l'exemple `/^\d+$/i` du prompt système pilote le modèle brut** (flag,
  `\d`, concision). Les baselines brutes du README mesurent en partie cet exemple ;
- des corrections ciblées de format à coût nul en données, si le format était le vrai problème. Sur le SFT, il ne
  l'est plus.

Le coût est élevé : une quinzaine de sondes et vingt-huit éditions candidates pour quatre gardées, chacune fragile.
Pour un défaut étroit, localisé et qui masque une compétence déjà là (le cas du post), la méthode a du sens. Pour
apprendre une tâche, non.

## Limites à garder en tête

- Les éditions e18 dépendent du prompt système exact : changer `common.SYSTEM_PROMPT` déplace la position 61 et les
  rend sans objet.
- e07, e19 et e20 sont des éditions de pilotage des logits en dernière couche, pas des corrections de circuit. Seule
  e18 agit sur un mécanisme identifié.
- Les clés de e07 et e20 et la direction de e19 sont des moyennes d'activations sur 64 exemples de train. Ce sont des
  différences de moyennes, pas des ajustements, mais elles portent la distribution du train.
- Le choix entre e14, e20 et e26 repose sur un dev qui ne les départage pas en succès.
