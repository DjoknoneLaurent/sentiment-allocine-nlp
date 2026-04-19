[# 🇫🇷 Sentiment Analysis – Allociné (French)

---

## Vue d'ensemble

Ce projet, réalisé par **DJOKNONE Laurent** et **EKWANE Franck**, étudiants en Master 2 Data Science et Modélisation Statistique (ISSEA-CEMAC), dans le cadre du cours **Text Mining & Web Mining** dirigé par le Professeur **Melatagia**, vise à analyser les sentiments exprimés dans les critiques de films françaises issues du corpus **Allociné**.

Nous avons conçu un pipeline complet allant de l’exploration des données à la mise en production d’une application web interactive.

---
## Stack technique
- NLP : CamemBERT, FlauBERT, DistilBERT
- API : FastAPI
- Interface : Streamlit
- Conteneurisation : Docker
- CI/CD : GitHub Actions
- Tests : pytest + pytest-cov

# Remplacer l'URL par la tienne après déploiement
APP_URL="[https://sentiment-allocine.streamlit.app](https://sentiment-allocine-nlp-8m5az5jann2bq53nxo9gms.streamlit.app/)"
GITHUB_URL="https://github.com/DjoknoneLaurent/sentiment-allocine-nlp"

cat > README.md << 'READMEEOF'
<div align="center">

# Analyse de Sentiment — Critiques Allociné

**Pipeline NLP complet** · Classification binaire en français · Déployé en production

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](APP_URL_PLACEHOLDER)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![HuggingFace](https://img.shields.io/badge/🤗-CamemBERT-yellow)](https://huggingface.co/camembert-base)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[**Démo live →**](APP_URL_PLACEHOLDER)

</div>

--- 
## Architecture du pipeline

1. **Données**
   - Corpus Allociné (160 000 critiques en français).
   - Chargement via HuggingFace Datasets.

2. **Prétraitement**
   - Nettoyage des textes (ponctuation, accents, stopwords).
   - Tokenisation pour les modèles Transformers.
   - Vectorisation TF‑IDF pour les modèles classiques.

3. **Modélisation**
   - **Baselines** :
     - Logistic Regression
     - Naive Bayes
     - LinearSVC (SVM linéaire)
   - **Transformers** :
     - CamemBERT fine‑tuné
     - DistilCamemBERT (optimisé CPU)

4. **Optimisation**
   - Recherche d’hyperparamètres avec Optuna.
   - Calibration des probabilités.
   - Ajustement du seuil de décision selon le critère métier (FN ×3, FP ×1).

5. **Évaluation**
   - Comparaison explicite des modèles (Accuracy, F1, ROC AUC).
   - Analyse des cas difficiles (sarcasme, négations).

6. **Déploiement**
   - Application Streamlit interactive.
   - Choix du modèle (rapide vs premium).
   - Visualisation des prédictions, probabilités et latence.
   - Hébergement sur Streamlit Cloud.

---

### Schéma simplifié

Données (Allociné, 160k critiques FR)  
↓  
Prétraitement (nettoyage, TF‑IDF / tokenisation)  
↓  
┌───────────────────────────┬─────────────────────────┐  
│ Baselines (LR, SVM, NB)   │ Transformers (CamemBERT) │  
└───────────────────────────┴─────────────────────────┘  
↓  
Optimisation (Optuna, seuil métier)  
↓  
Évaluation comparative  
↓  
Déploiement Streamlit (multi‑modèles)

## Vue d'ensemble

Ce projet transforme des critiques de films françaises en signal de sentiment (positif / négatif) à travers un pipeline MLOps complet : de l'exploration des données jusqu'au déploiement d'une application web accessible publiquement.

Le dataset utilisé est [Allociné](https://huggingface.co/datasets/allocine) (HuggingFace) — 160 000 critiques en français, parfaitement équilibré (50/50).

---


---

## Résultats

| Modèle | F1 | Accuracy | ROC AUC | Seuil | Temps inférence |
|--------|-----|----------|---------|-------|----------------|
| **LinearSVC** (TF-IDF) | **0.9424** | 0.9424 | 0.9874 | optimisé | < 5 ms |
| CamemBERT fine-tuné | 0.9407 | 0.9419 | 0.9857 | 0.2788 | ~500 ms (CPU) |
| LogisticRegression | 0.9380 | 0.9378 | 0.9840 | optimisé | < 5 ms |
| NaiveBayes | 0.9100 | 0.9100 | 0.9650 | optimisé | < 2 ms |

*Évaluation sur 20 000 exemples de test — jamais vus pendant l'entraînement.*

**Enseignement clé** : LinearSVC et CamemBERT sont statistiquement équivalents sur ce corpus. La valeur de CamemBERT se manifeste sur des formulations complexes (sarcasme, négations imbriquées), pas sur des critiques fortement polarisées.



## Structure des dossiers

## Structure des dossiers du projet
```
sentiment-allocine-nlp/
├── app/
│   ├── streamlit_app.py      # Interface Streamlit (application déployée)
│   ├── model_manager.py      # Gestion multi-modèles et fallback
│   └── utils.py              # Fonctions de preprocessing et utilitaires
│
├── notebooks/
│   ├── 01_eda_exploration.ipynb        # Analyse exploratoire des données
│   ├── 02_preprocessing_pipeline.ipynb # Pipeline de nettoyage et vectorisation
│   ├── 03_baseline_models.ipynb        # Modèles classiques (LR, SVM, NB)
│   ├── 04_transformer_finetuning.ipynb # Fine-tuning CamemBERT / DistilCamemBERT
│   └── 05_evaluation_analysis.ipynb    # Comparaison et analyse des résultats
│
├── models/
│   ├── baseline/             # Sauvegarde des modèles classiques (LinearSVC, LR, NB)
│   └── transformers/         # Configurations et checkpoints CamemBERT
│
├── reports/
│   ├── figures/              # Visualisations (EDA, benchmarks, métriques)
│   └── metrics/              # Résultats chiffrés (JSON, CSV)
│
├── src/
│   ├── data/                 # Scripts de téléchargement et preprocessing du corpus
│   └── training/             # Scripts d’entraînement et d’évaluation
│
├── .github/workflows/        # CI/CD (tests automatiques et intégration continue)
├── requirements.txt          # Dépendances du projet
├── deployment_config.json    # Stratégie multi-modèles pour le déploiement
└── README.md                 # Documentation principale
```


## Application

L'interface Streamlit permet de :

- coller une critique de film en français
- choisir entre modèle rapide (LinearSVC) et premium (CamemBERT)
- visualiser la prédiction, les probabilités et le niveau de confiance
- voir le seuil de décision appliqué et le temps de réponse

**→ [Tester l'application](APP_URL_PLACEHOLDER)**

---

## Stack technique

| Couche | Technologie |
|--------|------------|
| Langage | Python 3.10 |
| NLP classique | scikit-learn, TF-IDF, Optuna |
| Transformer | HuggingFace Transformers, CamemBERT |
| Interface | Streamlit |
| Tracking | MLflow |
| Déploiement | Streamlit Community Cloud |
| Versioning | Git / GitHub |


## Lancer en local

```bash
git clone GITHUB_URL_PLACEHOLDER
cd sentiment-allocine-nlp
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

L'application s'ouvre sur `http://localhost:8501`.

Pour régénérer les données et réentraîner les modèles :

```bash
python src/data/download_dataset.py   # Télécharge depuis HuggingFace
python src/data/preprocess.py          # Génère data/processed/
# Puis exécuter les notebooks dans l'ordre 01 → 05
```

---

## Pistes d'amélioration

- **GPU inference** : déployer CamemBERT sur Hugging Face Inference API pour réduire la latence à < 50ms
- **Active learning** : collecter les prédictions incertaines (confiance < 0.65) pour réentraîner
- **Multilingual** : étendre à d'autres langues romanes avec XLM-RoBERTa
- **API REST** : exposer les prédictions via FastAPI pour intégration dans d'autres services
- **Monitoring** : tracker la distribution des prédictions en production pour détecter le drift

---
(https://github.com/DjoknoneLaurent/sentiment-allocine-nlp)
