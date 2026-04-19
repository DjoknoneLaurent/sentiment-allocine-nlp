# app/utils.py
# ============================================================
#  UTILITAIRES PARTAGÉS
#  Responsabilité unique : fonctions pures, sans état.
#  Importé par model_manager.py et streamlit_app.py.
# ============================================================

import re
import time
import json
from pathlib import Path
from typing import Any


# ── Nettoyage du texte ────────────────────────────────────────
# On applique le même nettoyage minimal que text_transformer
# (suppression HTML, URLs, emojis). On ne fait PAS de lowercase
# ni de suppression de ponctuation : CamemBERT en a besoin.

def clean_text(text: str) -> str:
    """
    Nettoyage minimal cohérent avec text_transformer du pipeline.
    Supprime HTML, URLs, emojis et espaces multiples.
    Ne touche pas à la casse ni à la ponctuation (Transformers).
    """
    if not isinstance(text, str):
        return ""

    # Balises HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Emojis (plage Unicode principale)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)
    # Espaces multiples
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Chargement de la config de déploiement ───────────────────

def load_deployment_config(config_path: Path) -> dict[str, Any]:
    """
    Charge deployment_config.json produit par le notebook 05.
    Retourne un dict vide en cas d'erreur (ne crashe pas l'app).
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"[utils] Erreur parsing deployment_config.json : {e}")
        return {}


# ── Formatage des résultats pour l'affichage ─────────────────

def format_prediction_result(
    label:        int,
    proba:        float,
    model_name:   str,
    latency_ms:   float,
    threshold:    float,
) -> dict[str, Any]:
    """
    Formate le résultat brut du modèle en dict d'affichage.
    Centralise toute la logique de présentation.
    """
    sentiment    = "positif" if label == 1 else "négatif"
    confidence   = proba if label == 1 else 1.0 - proba
    is_confident = confidence >= 0.75

    return {
        "label":        label,
        "sentiment":    sentiment,
        "proba_pos":    round(float(proba), 4),
        "proba_neg":    round(1.0 - float(proba), 4),
        "confidence":   round(float(confidence), 4),
        "is_confident": is_confident,
        "model_name":   model_name,
        "latency_ms":   round(latency_ms, 1),
        "threshold":    round(threshold, 4),
    }


# ── Timer contextuel ─────────────────────────────────────────

class Timer:
    """Mesure la durée d'une opération en millisecondes."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.ms = round((time.perf_counter() - self._start) * 1000, 1)


# ── Validation de l'input utilisateur ────────────────────────

def validate_input(text: str) -> tuple[bool, str]:
    """
    Valide le texte entré par l'utilisateur.
    Retourne (is_valid, error_message).
    """
    if not text or not text.strip():
        return False, "Le texte ne peut pas être vide."

    cleaned = clean_text(text)

    if len(cleaned) < 10:
        return False, "Le texte est trop court (minimum 10 caractères)."

    if len(cleaned) > 5000:
        return False, "Le texte est trop long (maximum 5000 caractères)."

    return True, ""
