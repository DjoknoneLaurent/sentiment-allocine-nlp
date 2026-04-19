# app/model_manager.py
# ============================================================
#  MODEL MANAGER — COUCHE D'ABSTRACTION MULTI-MODÈLES
#
#  Responsabilités :
#  1. Charger les modèles avec @st.cache_resource (une seule fois)
#  2. Exposer une interface predict() unique quel que soit le modèle
#  3. Gérer le seuil métier (jamais 0.5 par défaut)
#  4. Activer le fallback LinearSVC si CamemBERT échoue
#
#  Le reste de l'app (streamlit_app.py) ne sait pas lequel
#  des deux modèles tourne. Il appelle juste predict().
# ============================================================

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ── Résultat de prédiction standardisé ───────────────────────

@dataclass
class PredictionResult:
    """Résultat de prédiction normalisé — indépendant du modèle."""
    label:       int          # 0 = négatif, 1 = positif
    proba_pos:   float        # probabilité classe positive
    proba_neg:   float        # probabilité classe négative
    model_used:  str          # nom du modèle effectivement utilisé
    threshold:   float        # seuil appliqué
    latency_ms:  float        # temps d'inférence
    is_fallback: bool = False # True si le fallback a été activé
    warning:     str  = ""    # message si fallback ou anomalie


# ── Configuration des modèles ─────────────────────────────────

@dataclass
class ModelConfig:
    """Configuration d'un modèle chargée depuis deployment_config.json."""
    name:      str
    model_type: str          # "baseline" ou "transformer"
    threshold: float
    artifacts: dict = field(default_factory=dict)
    hf_name:   str  = ""


def _load_model_configs(config_path: Path) -> dict[str, ModelConfig]:
    """
    Lit deployment_config.json et extrait les configs des 3 modèles.
    Retourne un dict {role: ModelConfig}.
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {}

    configs = {}
    for role, data in cfg.get("models", {}).items():
        configs[role] = ModelConfig(
            name       = data.get("name", role),
            model_type = data.get("type", "baseline"),
            threshold  = float(data.get("threshold", 0.5)),
            artifacts  = data.get("artifacts", {}),
            hf_name    = data.get("hf_name", ""),
        )
    return configs


# ── Chargeurs de modèles ──────────────────────────────────────
# Ces fonctions sont décorées avec @st.cache_resource dans
# streamlit_app.py (pas ici) pour éviter une dépendance
# circulaire entre les modules.

def load_baseline_model(joblib_path: str):
    """
    Charge le modèle LinearSVC depuis son fichier joblib.
    Rapide (~100ms), toujours disponible.
    """
    import joblib
    path = Path(joblib_path)
    if not path.exists():
        raise FileNotFoundError(f"Modèle baseline introuvable : {path}")
    model = joblib.load(path)
    return model


def load_transformer_model(model_dir: str):
    """
    Charge CamemBERT depuis le dossier HuggingFace sauvegardé.
    Long (~30s sur CPU) — mis en cache par Streamlit après le 1er appel.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    import torch

    dir_path = Path(model_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Dossier transformer introuvable : {dir_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(dir_path), use_fast=True, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(dir_path), local_files_only=True
    )
    model.eval()  # désactive dropout pour l'inférence
    return tokenizer, model


# ── Fonctions de prédiction ───────────────────────────────────

def _predict_baseline(
    model,
    text: str,
    threshold: float,
) -> tuple[float, float]:
    """
    Prédiction avec LinearSVC calibré.
    Retourne (proba_positive, latency_ms).
    """
    from scipy.special import softmax as sp_softmax

    t0 = time.perf_counter()

    # Le modèle est un CalibratedClassifierCV → predict_proba disponible
    proba_arr = model.predict_proba([text])
    proba_pos = float(proba_arr[0][1])

    latency_ms = (time.perf_counter() - t0) * 1000
    return proba_pos, latency_ms


def _predict_transformer(
    tokenizer,
    model,
    text: str,
    threshold: float,
    max_length: int = 128,
) -> tuple[float, float]:
    """
    Prédiction avec CamemBERT.
    Retourne (proba_positive, latency_ms).
    """
    import torch
    from scipy.special import softmax as sp_softmax

    t0 = time.perf_counter()

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits    = outputs.logits.numpy()
    probas    = sp_softmax(logits, axis=1)[0]
    proba_pos = float(probas[1])

    latency_ms = (time.perf_counter() - t0) * 1000
    return proba_pos, latency_ms


# ── Interface publique principale ─────────────────────────────

class SentimentPredictor:
    """
    Interface unifiée pour les prédictions de sentiment.

    Usage :
        predictor = SentimentPredictor(config_path, project_root)
        result = predictor.predict(text, model_choice="fast")
    """

    def __init__(self, config_path: Path, project_root: Path):
        self.project_root = project_root
        self.configs      = _load_model_configs(config_path)
        self._baseline_model  = None
        self._transformer     = None   # (tokenizer, model)
        self._init_errors     = {}

    def _ensure_baseline_loaded(self) -> bool:
        """Charge le baseline si pas encore en mémoire. Retourne True si OK."""
        if self._baseline_model is not None:
            return True

        cfg = self.configs.get("fallback") or self.configs.get("main")
        if cfg is None:
            return False

        joblib_path = cfg.artifacts.get("joblib", "")
        if not joblib_path:
            # Chemin par défaut
            joblib_path = str(
                self.project_root / "models" / "baseline" / "best_model.joblib"
            )

        try:
            self._baseline_model = load_baseline_model(joblib_path)
            return True
        except Exception as e:
            self._init_errors["baseline"] = str(e)
            return False

    def _ensure_transformer_loaded(self) -> bool:
        """Charge CamemBERT si pas encore en mémoire. Retourne True si OK."""
        if self._transformer is not None:
            return True

        cfg = self.configs.get("premium")
        if cfg is None:
            return False

        model_dir = cfg.artifacts.get("model_dir", "")
        if not model_dir:
            model_dir = str(
                self.project_root / "models" / "transformers" / "camembert-base"
            )

        try:
            self._transformer = load_transformer_model(model_dir)
            return True
        except Exception as e:
            self._init_errors["transformer"] = str(e)
            return False

    def predict(self, text: str, model_choice: str = "fast") -> PredictionResult:
        """
        Prédit le sentiment d'un texte.

        Paramètres
        ----------
        text         : texte nettoyé (après clean_text)
        model_choice : "fast" → LinearSVC | "premium" → CamemBERT

        Stratégie de fallback :
        Si model_choice="premium" et CamemBERT échoue → LinearSVC activé
        Si LinearSVC échoue aussi → lève une exception (situation critique)
        """
        is_fallback = False
        warning     = ""

        # ── Choix du modèle et tentative ─────────────────────
        if model_choice == "premium":
            ok = self._ensure_transformer_loaded()
            if ok:
                cfg = self.configs.get("premium", ModelConfig(
                    name="CamemBERT-base", model_type="transformer",
                    threshold=0.2788
                ))
                try:
                    tok, mod   = self._transformer
                    proba_pos, latency = _predict_transformer(
                        tok, mod, text, cfg.threshold
                    )
                    model_used = cfg.name
                    threshold  = cfg.threshold
                except Exception as e:
                    # Fallback silencieux
                    is_fallback = True
                    warning = f"CamemBERT indisponible ({type(e).__name__}) — passage au modèle rapide."
                    model_choice = "fast"  # → bloc suivant
            else:
                is_fallback = True
                warning = "CamemBERT non chargé — passage au modèle rapide."
                model_choice = "fast"

        if model_choice == "fast" or is_fallback:
            ok = self._ensure_baseline_loaded()
            if not ok:
                raise RuntimeError(
                    "Impossible de charger LinearSVC. "
                    f"Erreur : {self._init_errors.get('baseline', 'inconnue')}"
                )
            cfg = self.configs.get("main") or self.configs.get("fallback")
            if cfg is None:
                cfg = ModelConfig(
                    name="LinearSVC", model_type="baseline", threshold=0.5
                )

            proba_pos, latency = _predict_baseline(
                self._baseline_model, text, cfg.threshold
            )
            model_used = cfg.name if not is_fallback else f"{cfg.name} (fallback)"
            threshold  = cfg.threshold

        # ── Application du seuil métier ───────────────────────
        label = 1 if proba_pos >= threshold else 0

        return PredictionResult(
            label       = label,
            proba_pos   = round(proba_pos, 4),
            proba_neg   = round(1.0 - proba_pos, 4),
            model_used  = model_used,
            threshold   = threshold,
            latency_ms  = round(latency, 1),
            is_fallback = is_fallback,
            warning     = warning,
        )

    @property
    def baseline_available(self) -> bool:
        return self._ensure_baseline_loaded()

    @property
    def transformer_available(self) -> bool:
        return self._ensure_transformer_loaded()

    @property
    def init_errors(self) -> dict:
        return self._init_errors
