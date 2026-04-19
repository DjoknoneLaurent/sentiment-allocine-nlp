[# 🇫🇷 Sentiment Analysis – Allociné (French)

Classification binaire de critiques de films (positif/négatif) avec pipeline MLOps complète.

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

## Vue d'ensemble

Ce projet transforme des critiques de films françaises en signal de sentiment (positif / négatif) à travers un pipeline MLOps complet : de l'exploration des données jusqu'au déploiement d'une application web accessible publiquement.

Le dataset utilisé est [Allociné](https://huggingface.co/datasets/allocine) (HuggingFace) — 160 000 critiques en français, parfaitement équilibré (50/50).

---

## Objectif métier

La tâche n'est pas uniquement de maximiser l'accuracy. Un **critère de coût asymétrique** est intégré :
coût = 3 × FN + 1 × FP

Manquer un avis positif (faux négatif) est trois fois plus coûteux que signaler un faux positif. Ce critère guide l'optimisation du seuil de décision sur le jeu de validation — le seuil final n'est jamais 0.5 par défaut.

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

---

## Architecture
sentiment_allocine/
├── app/
│   ├── streamlit_app.py      # Interface utilisateur
│   ├── model_manager.py      # Abstraction multi-modèles + fallback
│   └── utils.py              # Preprocessing et utilitaires
├── notebooks/
│   ├── 01_eda_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_transformer_finetuning.ipynb
│   └── 05_evaluation_analysis.ipynb
├── models/
│   ├── baseline/             # LinearSVC sauvegardé
│   └── transformers/         # Config et seuils CamemBERT
├── reports/
│   ├── figures/              # Visualisations EDA et benchmark
│   └── metrics/              # Métriques JSON et CSV
├── src/
│   └── data/                 # Scripts de chargement et preprocessing
├── deployment_config.json    # Stratégie multi-modèles
└── requirements.txt

---

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

---

## Structure du projet
sentiment_allocine/
├── app/
│   ├── streamlit_app.py      # Interface utilisateur
│   ├── model_manager.py      # Abstraction multi-modèles + fallback
│   └── utils.py              # Preprocessing et utilitaires
├── notebooks/
│   ├── 01_eda_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_transformer_finetuning.ipynb
│   └── 05_evaluation_analysis.ipynb
├── models/
│   ├── baseline/             # LinearSVC sauvegardé
│   └── transformers/         # Config et seuils CamemBERT
├── reports/
│   ├── figures/              # Visualisations EDA et benchmark
│   └── metrics/              # Métriques JSON et CSV
├── src/
│   └── data/                 # Scripts de chargement et preprocessing
├── deployment_config.json    # Stratégie multi-modèles
└── requirements.txt

---

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
