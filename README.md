# MovieLens-20M Recommender — Embeddings (Content) vs Baselines (Collaborative)

Projet de recommandation top-K (top-10) sur **MovieLens 20M**, comparant :
- une approche **content-based** via **embeddings** (Sentence-Transformers),
- une baseline **popularité**,
- des baselines collaboratives **MF** et **BPR** via **Cornac**.

**Auteurs** : Noel Snelson, Jonathan, Ibrahim, Rayane

---

## Objectif

Construire un système de recommandation **top-10** à partir des métadonnées films (titre, genres, tags), puis comparer les performances à des modèles collaboratifs standard.

**Métriques principales** : Recall@K, NDCG@K (K ∈ {5, 10, 20}).

---

## Données

Dataset : **MovieLens 20M** (GroupLens / Kaggle)

Fichiers principaux utilisés :
- `rating.csv` : (userId, movieId, rating, timestamp)
- `movie.csv` : (movieId, title, genres)
- `tag.csv` : (userId, movieId, tag, timestamp)

Optionnel :
- genome tags/scores (désactivé par défaut dans le notebook)

---

## Pré-traitement (pipeline)

1. **Optimisation mémoire** (types int32/float32) + gestion robuste de `timestamp`
2. **Filtrage** :
   - utilisateurs avec ≥ **15** notations
   - films avec ≥ **50** notations
3. **Texte item** : concaténation explicable
   - `title` + `genres` + `tags` (nettoyés, dédoublonnés et tronqués)
4. **Split anti-fuite** : **split temporel par utilisateur**
   - les interactions les plus récentes (20%) → test
   - le reste → train

---

## Modèles évalués

### Embeddings (Sentence-Transformers)
- `miniLM` : sentence-transformers/all-MiniLM-L6-v2
- `mpnet` : sentence-transformers/all-mpnet-base-v2
- `bge-small` : BAAI/bge-small-en-v1.5

**Profil utilisateur** : moyenne des embeddings des films positifs (rating ≥ 4.0), normalisée.  
**Scoring** : similarité cosinus avec exclusion des items déjà vus.

### Hybridation (activée par défaut)
Le score final peut combiner :
- score sémantique (cosine)
- score de popularité normalisé (log-count)

Formule : `score = α * cosine + (1-α) * popularity` avec **α = 0.7**.

### Baselines
- Popularité (classement global sur train, exclusion des vus)
- Cornac MF
- Cornac BPR

---

## Résultats (run complet)

Résumé (top-10) :

| Modèle        | Recall@10 | NDCG@10 |
|--------------|----------:|--------:|
| cornac_bpr   | 0.0735    | 0.0937  |
| mpnet        | 0.0612    | 0.0760  |
| popularity   | 0.0603    | 0.0759  |
| bge-small    | 0.0601    | 0.0752  |
| miniLM       | 0.0562    | 0.0706  |
| cornac_mf    | 0.0202    | 0.0252  |

Les résultats détaillés (Recall/NDCG/MAP/HR pour K=5/10/20) sont exportés dans `results_summary.csv`.

---

## Reproductibilité

Le notebook exporte automatiquement :
- `results_summary.csv` : métriques consolidées par modèle
- `run_config.json` : configuration exacte du run (seed, split, modèles, tailles, etc.)

Les embeddings peuvent être mis en cache dans `embeddings_cache/` pour éviter de recalculer.

---

## Exécution

### Option A — Google Colab / Kaggle Notebook
Ouvrir le notebook et exécuter toutes les cellules dans l’ordre.

### Dépendances
Le notebook installe notamment :
- kagglehub
- transformers
- sentence-transformers
- cornac
- torch, pandas, numpy, scikit-learn

---

## Structure du dépôt (recommandée)

- `recommendation-assebane-kancel-noel-zekri-kaggle-v.ipynb` : notebook principal
- `RECOMMENDATION_RAPPORT_NOEL_ASSEBBANE_KANCEL_ZEKRI.pdf` : rapport
- `results_summary.csv` : résultats (export)
- `run_config.json` : configuration (export)
- `embeddings_cache/` : cache embeddings (optionnel, à ignorer sur Git)

---

## Description courte du dépôt (pour GitHub)
Dépôt de rendu du projet de recommandation : MovieLens 20M, comparaison embeddings (content-based) vs baselines collaboratives (Cornac MF/BPR), avec évaluation top-K et export des résultats.

## Description du fichier CSV (results_summary.csv)
Ce fichier regroupe un tableau des métriques obtenues par les différents modèles (Recall/NDCG/MAP/HR pour plusieurs valeurs de K).

---

## Référence
MovieLens : Harper & Konstan (2015).
