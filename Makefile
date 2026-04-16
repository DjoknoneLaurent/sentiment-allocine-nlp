.PHONY: help install setup data train-baseline train-transformer evaluate api app test lint clean docker-build docker-run mlflow-ui

PYTHON := python
PIP := pip

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Installer les dependances
	$(PIP) install -r requirements.txt
	$(PYTHON) -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

setup: install ## Setup complet du projet
	mkdir -p mlruns logs data/raw data/processed models/production reports/figures reports/metrics

data: ## Telecharger et preparer les donnees
	$(PYTHON) src/data/download_dataset.py
	$(PYTHON) src/data/preprocess.py

train-baseline: ## Entrainer les modeles baseline
	$(PYTHON) src/models/train_baseline.py

train-transformer: ## Fine-tuner CamemBERT
	$(PYTHON) src/models/train_transformer.py

evaluate: ## Evaluer tous les modeles
	$(PYTHON) src/evaluation/evaluate_all.py

api: ## Lancer l'API FastAPI
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

app: ## Lancer l'interface Streamlit
	streamlit run app/streamlit_app.py

test: ## Lancer les tests
	pytest tests/ -v

lint: ## Linter le code
	ruff check src/ app/ tests/ --fix

clean: ## Nettoyer les fichiers temporaires
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

docker-build: ## Builder l'image Docker
	docker build -f docker/Dockerfile -t sentiment-allocine:latest .

docker-run: ## Lancer le conteneur Docker
	docker run -p 8000:8000 sentiment-allocine:latest

mlflow-ui: ## Ouvrir MLflow UI
	mlflow ui --backend-store-uri ./mlruns --port 5000
