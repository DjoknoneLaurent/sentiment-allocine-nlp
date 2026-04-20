<div align="center">

# Analyse de Sentiment — Critiques Cinéma Allociné

### Pipeline NLP de bout en bout · Classification binaire en français · Déployé en production

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentiment-allocine-nlp-8m5az5jann2bq53nxo9gms.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-CamemBERT-FFD21E)](https://huggingface.co/camembert-base)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)

**[→ Application en ligne](https://sentiment-allocine-nlp-8m5az5jann2bq53nxo9gms.streamlit.app/)** · **[→ Code source](https://github.com/DjoknoneLaurent/sentiment-allocine-nlp)**

---

*DJOKNONE Laurent · EKWANE Franck — Master 2 Data Science, Modélisation Statistique*
*Cours : Text Mining & Web Mining*

</div>

---

## Positionnement du projet

Ce projet ne se limite pas à entraîner un modèle et afficher une accuracy. Il s'agit d'un **pipeline MLOps complet** : exploration des données, prétraitement industrialisé, modélisation comparative (baselines classiques vs transformer francophone), optimisation sous contrainte métier, et déploiement d'une application web accessible publiquement.

L'ensemble est structuré selon les standards d'un projet professionnel : code modulaire, configuration centralisée, tracking des expériences (MLflow), stratégie multi-modèles, et documentation orientée décision plutôt que description.

---

## Problématique métier

### Pourquoi l'accuracy n'est pas l'objectif

En classification de sentiment appliquée à un contexte métier réel (modération de contenu, analyse de réputation, détection d'insatisfaction client), toutes les erreurs ne se valent pas.

Un **faux négatif** (critique négative non détectée) peut conduire à ignorer un signal d'alerte critique. Un **faux positif** (critique positive mal classée) génère du bruit dans les tableaux de bord. Ces deux types d'erreurs n'ont pas le même coût opérationnel.

Nous modélisons explicitement cette asymétrie via la fonction de coût suivante :

```
coût total = 3 × FN + 1 × FP
```

Cette formulation a des conséquences directes sur le pipeline :

- Le seuil de décision est **optimisé sur le jeu de validation**, jamais fixé à 0.5 par défaut.
- Le choix du meilleur modèle intègre ce critère en plus du F1-score.
- La calibration des probabilités devient indispensable (notamment pour LinearSVC, qui ne fournit pas de probabilités nativement).

Ce type de raisonnement — passer de l'optimisation d'une métrique à l'optimisation d'un objectif métier — est ce qui distingue un projet académique d'un projet orienté production.

---

## Dataset

### Allociné (HuggingFace)

| Attribut | Valeur |
|---|---|
| Source | [datasets/allocine](https://huggingface.co/datasets/allocine) |
| Langue | Français |
| Volume | ~160 000 critiques |
| Répartition | Train / Validation / Test |
| Classes | Positif (1) / Négatif (0) |
| Équilibre | ~50 % / 50 % |

### Caractéristiques et biais à connaître

Le corpus Allociné présente plusieurs propriétés qui influencent directement les choix de modélisation :

**Polarisation forte.** Les critiques issues de plateformes de notation grand public sont généralement à forte valence : les utilisateurs qui prennent la peine d'écrire ont tendance à exprimer des avis marqués. Les cas ambigus (sarcasme, critique nuancée) sont sous-représentés — ce qui explique en partie pourquoi des modèles linéaires simples atteignent des performances très élevées.

**Biais de sélection.** Les critiques courtes et polarisées dominent. Les textes longs avec argumentation complexe sont rares. Cela avantage les approches par sac-de-mots (TF-IDF) qui capturent efficacement le vocabulaire discriminant.

**Spécificité domaine.** Le vocabulaire cinématographique ("mise en scène", "chef-d'œuvre", "ennuyeux", "poignant") est fortement discriminant et stable. Un modèle pré-entraîné sur corpus général n'apporte pas nécessairement d'avantage sur ce type de données fermées.

---

## Pipeline complet

### 1. Exploration des données (EDA)

- Distribution des classes et vérification de l'équilibre
- Analyse des longueurs de texte (P50, P75, P95 en tokens) — déterminante pour le choix de `MAX_LENGTH`
- Extraction des termes les plus discriminants par classe (chi², TF-IDF)
- Visualisations : wordclouds, distributions de longueur, fréquences n-grammes

### 2. Prétraitement

Deux modes de nettoyage coexistent, adaptés à chaque famille de modèle :

```
text_classical   → lowercase, suppression ponctuation, stopwords optionnels
                   → utilisé pour TF-IDF + modèles linéaires

text_transformer → suppression HTML, URLs, emojis uniquement
                   → casse et accents préservés (CamemBERT en a besoin)
```

Le preprocesseur est implémenté comme un `TransformerMixin` scikit-learn, ce qui le rend intégrable dans n'importe quelle `Pipeline` et garantit la cohérence entre entraînement et inférence.

### 3. Modélisation — Baselines classiques

Trois modèles de référence avec TF-IDF (80K features, bigrammes, `sublinear_tf=True`) :

- **Logistic Regression** — optimisation Optuna (hyperparamètres C, solver, ngram_range)
- **LinearSVC** — optimisation Optuna, calibration isotonic pour obtenir des probabilités
- **Naive Bayes** (MultinomialNB) — optimisation Optuna sur α et paramètres TF-IDF

Chaque modèle est calibré via `CalibratedClassifierCV` pour disposer de probabilités fiables, et son seuil de décision est optimisé individuellement sur le jeu de validation selon le critère métier.

### 4. Modélisation — Transformer francophone

Fine-tuning de **CamemBERT-base** (`camembert-base`) sur Allociné :

- Architecture : RoBERTa entraîné sur 138 Go de texte français (corpus OSCAR)
- MAX_LENGTH : 128 tokens (P75 du corpus — compromis RAM/signal sur CPU)
- Stratégie CPU : layer freezing à 60 % (couches basses figées), batch=16, 1 epoch sur 10 % du corpus
- Calibration du seuil de décision sur validation (seuil optimal : 0.2788)

### 5. Optimisation et sélection

- **Optuna** (30 trials, TPE sampler) pour chaque modèle baseline
- **Validation croisée stratifiée** (3 folds) comme critère d'optimisation
- **Seuil métier** cherché sur 300 points uniformes entre 0.01 et 0.99
- **Benchmark final** sur le jeu de test (20 000 exemples) — jamais touché pendant l'optimisation

---

## Analyse critique des résultats

### Pourquoi LinearSVC ≈ CamemBERT sur ce dataset

C'est la conclusion la plus intéressante du projet, et elle mérite une explication rigoureuse.

**L'hypothèse du plafond lexical.** Allociné est un corpus à vocabulaire fortement discriminant et stable. Les mots "magnifique", "décevant", "chef-d'œuvre", "ennuyeux" sont des signaux quasi-déterministes du sentiment. Un TF-IDF sur 80K features avec bigrammes capture ce signal de façon quasi-exhaustive. Dans ce contexte, l'attention multi-têtes de CamemBERT apporte peu d'information marginale.

**L'effet de l'entraînement partiel.** Le fine-tuning a été réalisé sur 10 % du corpus (16 000 exemples) pour des raisons de contraintes CPU, avec layer freezing à 60 % et une seule epoch. CamemBERT n'a donc pas pu exploiter pleinement ses capacités de modélisation contextuelle. Sur le corpus complet avec GPU et 3 epochs complètes, l'écart serait probablement plus marqué.

**La limite du F1 comme critère suffisant.** Les performances équivalentes en F1 masquent des différences qualitatives : CamemBERT gère mieux le sarcasme ("Encore un chef-d'œuvre... de médiocrité"), les négations complexes et les formulations atypiques. Ces cas sont rares dans Allociné, mais critiques en production réelle.

**Conclusion opérationnelle.** Sur un corpus fermé, fortement polarisé et avec vocabulaire stable, un modèle linéaire bien optimisé est un concurrent sérieux pour un transformer. Choisir CamemBERT en production se justifie par sa robustesse aux données hors distribution, pas par sa performance brute sur ce benchmark.

### Limites du projet

- Fine-tuning sur sous-corpus (10 %) — les résultats CamemBERT sont sous-estimés
- Absence de test sur des données hors domaine (critiques littéraires, gastronomiques)
- Calibration isotonic sur données d'entraînement — risque de surapprentissage de la calibration
- Pas de test de significativité statistique entre LinearSVC et CamemBERT (McNemar, bootstrap)

---

## Résultats

### Benchmark complet — Test set (20 000 exemples)

| Modèle | Type | F1 | Accuracy | ROC AUC | Seuil | Coût métier |
|--------|------|----|----------|---------|-------|-------------|
| **LinearSVC** | Baseline | **0.9424** | 0.9424 | **0.9874** | optimisé | minimal |
| CamemBERT | Transformer | 0.9407 | 0.9419 | 0.9857 | 0.2788 | minimal |
| LogisticRegression | Baseline | 0.9380 | 0.9378 | 0.9840 | optimisé | faible |
| NaiveBayes | Baseline | 0.9100 | 0.9100 | 0.9650 | optimisé | modéré |

*Coût métier = 3×FN + 1×FP. Seuil optimisé sur validation uniquement.*

### Interprétation

L'écart de F1 entre LinearSVC et CamemBERT est de **0.0017** — statistiquement négligeable sur 20 000 exemples. Le vrai différenciateur est le ROC AUC (capacité de discrimination indépendante du seuil) et la latence d'inférence : LinearSVC répond en < 5 ms, CamemBERT en ~500 ms sur CPU.

---

## Architecture de déploiement

### Stratégie multi-modèles

```
Requête utilisateur
       ↓
  model_choice ?
  ┌──────────────────┬────────────────────────┐
  │  "fast"          │  "premium"             │
  │                  │                        │
  │  LinearSVC       │  CamemBERT             │
  │  < 5 ms          │  ~500 ms               │
  │  F1 = 0.9424     │  F1 = 0.9407           │
  │                  │  ↓ (si erreur)         │
  │                  │  Fallback → LinearSVC  │
  └──────────────────┴────────────────────────┘
       ↓
  Seuil métier appliqué (≠ 0.5)
       ↓
  PredictionResult standardisé
```

Trois niveaux de service :

| Rôle | Modèle | Justification |
|------|--------|---------------|
| ⚡ Principal | LinearSVC | Performance maximale, latence minimale, zéro dépendance PyTorch |
| 🔒 Fallback | LinearSVC | Activé automatiquement si le modèle premium échoue |
| 🧠 Premium | CamemBERT | Probabilités calibrées, meilleure généralisation hors distribution |

Le fallback est automatique et transparent pour l'utilisateur — aucune erreur visible en cas d'indisponibilité du modèle premium.

### Compromis latence vs performance

Sur Streamlit Community Cloud (CPU, ~1 Go RAM), charger CamemBERT représente ~700 Mo et ~30 secondes au premier appel. Le cache `st.cache_resource` garantit un seul chargement par session. LinearSVC reste instantané dans tous les cas.

---

## Application

L'interface Streamlit expose les fonctionnalités suivantes :

- **Saisie libre** d'une critique en français (jusqu'à 5 000 caractères)
- **Sélection du modèle** : rapide (LinearSVC) ou premium (CamemBERT)
- **Résultat clair** : sentiment prédit, probabilités des deux classes, niveau de confiance
- **Transparence** : seuil de décision appliqué, temps d'inférence, modèle effectivement utilisé
- **Fallback visible** : si CamemBERT est indisponible, l'utilisateur en est informé

**→ [Accéder à l'application](https://sentiment-allocine-nlp-8m5az5jann2bq53nxo9gms.streamlit.app/)**

---

## Stack technique

| Couche | Outil | Rôle |
|--------|-------|------|
| Langage | Python 3.10 | Uniformité entre notebooks et scripts |
| NLP classique | scikit-learn, TF-IDF | Baselines, calibration, métriques |
| Optimisation | Optuna (TPE) | Recherche bayésienne des hyperparamètres |
| Transformer | HuggingFace Transformers | Fine-tuning CamemBERT |
| Tracking | MLflow | Logging des expériences et artefacts |
| Interface | Streamlit | Application web interactive |
| Sérialisation | joblib | Persistance des modèles sklearn |
| Déploiement | Streamlit Community Cloud | Hébergement gratuit, intégration GitHub |
| Versioning | Git / GitHub | Reproductibilité et collaboration |

---

## Structure du projet

```
sentiment_allocine/
│
├── app/                          # Application Streamlit
│   ├── streamlit_app.py          # Interface utilisateur
│   ├── model_manager.py          # Abstraction multi-modèles + fallback
│   └── utils.py                  # Preprocessing et utilitaires partagés
│
├── notebooks/                    # Pipeline analytique complet
│   ├── 01_eda_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_transformer_finetuning.ipynb
│   └── 05_evaluation_analysis.ipynb
│
├── src/                          # Modules Python réutilisables
│   └── data/
│       ├── download_dataset.py   # Téléchargement HuggingFace
│       ├── preprocess.py         # Pipeline de nettoyage
│       └── dataset_utils.py      # Chargement des splits
│
├── models/
│   ├── baseline/                 # Artefacts LinearSVC
│   │   ├── LinearSVC.joblib
│   │   └── best_model_config.json
│   └── transformers/             # Config et seuils CamemBERT
│       ├── camembert-base/
│       │   ├── config.json
│       │   ├── threshold.json
│       │   └── tokenizer_config.json
│       └── transformer_registry.json
│
├── reports/
│   ├── figures/                  # Visualisations EDA et benchmark
│   └── metrics/                  # Métriques JSON et CSV
│
├── configs/
│   └── config.yaml               # Configuration centralisée
│
├── deployment_config.json        # Stratégie multi-modèles
├── requirements.txt              # Dépendances Streamlit Cloud
└── README.md
```

---

## Installation et exécution

### Accès direct (recommandé)

L'application est déployée et accessible sans installation :
**→ [https://sentiment-allocine-nlp-8m5az5jann2bq53nxo9gms.streamlit.app/](https://sentiment-allocine-nlp-8m5az5jann2bq53nxo9gms.streamlit.app/)**

### Exécution locale

```bash
# Cloner le repository
git clone https://github.com/DjoknoneLaurent/sentiment-allocine-nlp
cd sentiment-allocine-nlp

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app/streamlit_app.py
```

L'application est disponible sur `http://localhost:8501`.

### Régénérer les données et modèles

```bash
# Télécharger le dataset depuis HuggingFace
python src/data/download_dataset.py

# Appliquer le preprocessing
python src/data/preprocess.py

# Puis exécuter les notebooks dans l'ordre : 01 → 02 → 03 → 04 → 05
```

---

## Difficultés rencontrées

Cette section documente les obstacles réels rencontrés, car ils reflètent des problèmes concrets de la pratique MLOps.

### Contrainte CPU pour le fine-tuning

L'entraînement de CamemBERT sur CPU (Dell Latitude, 16 Go RAM, sans GPU) a représenté la contrainte la plus structurante du projet. Un entraînement complet (160 000 exemples, 3 epochs) aurait nécessité 15 à 20 heures. Nous avons adopté une stratégie d'approximation raisonnée :

- Sous-échantillonnage stratifié à 10 % du corpus (16 000 exemples)
- Réduction de MAX_LENGTH à 128 tokens (P75 du corpus)
- Layer freezing à 60 % pour réduire le temps de rétropropagation
- Limitation à 1 epoch avec early stopping sur F1

Cette configuration a permis un entraînement en ~3h30, au prix d'une sous-estimation probable des capacités du modèle.

### Compatibilité des versions HuggingFace

Entre les différentes étapes du projet, des changements d'API ont causé des erreurs bloquantes : suppression de l'argument `no_cuda` dans `TrainingArguments` (transformers ≥ 4.36), renommage de `evaluation_strategy` en `eval_strategy` (≥ 4.38), comportement changeant de la tokenisation rapide. Chaque rupture de compatibilité a nécessité une analyse du changelog et une adaptation du code.

### Calibration des probabilités sur LinearSVC

LinearSVC est un modèle à marge, pas un modèle probabiliste. L'obtention de probabilités calibrées (nécessaires pour optimiser le seuil métier) a requis l'encapsulation dans `CalibratedClassifierCV` avec méthode isotonique — qui elle-même introduit un risque de surapprentissage si les données de calibration sont trop petites. Nous avons utilisé une validation croisée à 5 plis pour limiter ce risque.

### Déploiement sur Streamlit Cloud free tier

Le tier gratuit de Streamlit Community Cloud impose ~1 Go de RAM. CamemBERT (poids + tokenizer) occupe ~700 Mo au chargement. L'architecture choisie (LinearSVC principal, CamemBERT à la demande avec cache `st.cache_resource`) permet de maintenir l'application stable tout en offrant le modèle premium aux utilisateurs qui l'acceptent explicitement.

### Gestion des chemins en déploiement cloud

Les chemins relatifs calculés depuis le répertoire de travail courant (`os.getcwd()`) échouent sur Streamlit Cloud car le répertoire d'exécution n'est pas celui du fichier source. Tous les chemins ont été résolus depuis `__file__` (`Path(__file__).parent.parent`) pour garantir la portabilité entre environnements.

---

## Perspectives

### Court terme — robustesse et monitoring

- **Tests de significativité statistique** entre LinearSVC et CamemBERT (test de McNemar, bootstrap confidence intervals) pour valider formellement l'équivalence observée
- **Monitoring en production** : distribution des scores de confiance dans le temps, taux de fallback, latences P50/P95
- **Détection de drift** : KL-divergence entre distributions de probabilités en production et en validation

### Moyen terme — performance et extensibilité

- **Fine-tuning complet** sur Google Colab (GPU T4 gratuit) avec le corpus complet — F1 > 0.97 attendu d'après la littérature sur Allociné
- **API REST FastAPI** : exposer les prédictions comme microservice, permettant l'intégration dans d'autres pipelines
- **Distillation** : entraîner un modèle étudiant (DistilCamemBERT) à partir de CamemBERT fine-tuné — latence réduite sans perte significative de performance

### Long terme — industrialisation

- **Active learning** : identifier les exemples à haute incertitude (confiance entre 0.45 et 0.55), les soumettre à annotation humaine, réentraîner de façon incrémentale
- **Serving GPU** : déployer CamemBERT sur HuggingFace Inference API ou AWS SageMaker pour une latence < 50 ms
- **Extensibilité multilingue** : remplacer CamemBERT par XLM-RoBERTa pour couvrir d'autres marchés francophones (Belgique, Suisse, Maroc)

---

## Conclusion

Ce projet démontre qu'un pipeline NLP rigoureux ne se mesure pas à la sophistication des modèles utilisés, mais à la qualité du raisonnement qui guide chaque choix. L'équivalence de performance entre LinearSVC et CamemBERT sur Allociné n'est pas un échec du transformer — c'est un résultat informatif sur la nature du problème.

Sur un corpus à forte polarisation lexicale et vocabulaire stable, un modèle linéaire bien optimisé atteint le plafond de performance atteignable sans annotation supplémentaire. CamemBERT apporterait une valeur différenciée sur des données hors distribution ou des formulations linguistiquement complexes — deux conditions rarement réunies dans un benchmark académique standard.

La vraie valeur ajoutée de ce projet réside dans l'ensemble du pipeline : optimisation du seuil sous contrainte métier asymétrique, calibration des probabilités, architecture de déploiement avec fallback, et réflexion critique sur les limites de chaque approche. C'est cette démarche — et non le score F1 final — qui caractérise un projet orienté production.

---

<div align="center">

**DJOKNONE Laurent · EKWANE Franck**
Master 2 Data Science — Modélisation Statistique | 2026

[Application](https://sentiment-allocine-nlp-8m5az5jann2bq53nxo9gms.streamlit.app/) · [GitHub](https://github.com/DjoknoneLaurent/sentiment-allocine-nlp)

</div>
