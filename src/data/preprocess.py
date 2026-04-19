"""
Pipeline de prétraitement NLP pour les critiques Allociné.

Design : sklearn-compatible (fit/transform) pour être intégrable
dans une Pipeline scikit-learn ou utilisable standalone.

Deux modes :
  - mode "classical" : pour TF-IDF + modèles classiques
    (lowercase, suppression ponctuation, stopwords)
  - mode "transformer" : nettoyage minimal
    (CamemBERT gère lui-même la tokenisation)
"""
import re
import unicodedata
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Stopwords français minimalistes (on garde la négation : ne, pas, jamais...)
STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "en",
    "à", "au", "aux", "ce", "se", "on", "il", "elle", "ils", "elles",
    "je", "tu", "nous", "vous", "que", "qui", "quoi", "dont", "où",
    "mais", "ou", "donc", "or", "ni", "car", "par", "sur", "sous",
    "avec", "sans", "pour", "dans", "entre", "vers", "chez",
    "y", "en", "si", "plus", "très", "bien", "aussi",
}


def remove_html_tags(text: str) -> str:
    """Supprime les balises HTML résiduelles."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Supprime les URLs."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_emojis(text: str) -> str:
    """Supprime les emojis via la plage Unicode standard."""
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
    return emoji_pattern.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Remplace les espaces multiples/tabs/newlines par un espace simple."""
    return re.sub(r"\s+", " ", text).strip()


def remove_punctuation(text: str) -> str:
    """Supprime la ponctuation (mode classique uniquement)."""
    return re.sub(r"[^\w\s]", " ", text)


def remove_stopwords(text: str, stopwords: set = STOPWORDS_FR) -> str:
    """Supprime les stopwords français."""
    tokens = text.split()
    return " ".join(t for t in tokens if t.lower() not in stopwords)


def normalize_accents(text: str) -> str:
    """
    Convertit les caractères accentués en ASCII.
    ex: 'café' → 'cafe'
    NOTE: désactivé par défaut pour CamemBERT qui gère les accents.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesseur NLP sklearn-compatible.

    Parameters
    ----------
    mode : "classical" | "transformer"
        - classical : nettoyage complet pour TF-IDF
        - transformer : nettoyage minimal pour CamemBERT/FlauBERT
    remove_stopwords_flag : bool
        Activer la suppression des stopwords (mode classical seulement)
    """

    def __init__(
        self,
        mode: Literal["classical", "transformer"] = "transformer",
        remove_stopwords_flag: bool = False,
    ):
        self.mode = mode
        self.remove_stopwords_flag = remove_stopwords_flag

    def fit(self, X, y=None):
        """Rien à apprendre — requis par sklearn."""
        return self

    def transform(self, X, y=None):
        """
        Applique le pipeline de nettoyage sur une Series ou liste de textes.
        Retourne une liste de textes nettoyés.
        """
        if isinstance(X, pd.Series):
            return X.apply(self._clean).tolist()
        return [self._clean(text) for text in X]

    def _clean(self, text: str) -> str:
        """Applique séquentiellement les étapes de nettoyage."""
        if not isinstance(text, str):
            text = str(text)

        # Étapes communes aux deux modes
        text = remove_html_tags(text)
        text = remove_urls(text)
        text = remove_emojis(text)
        text = normalize_whitespace(text)

        if self.mode == "classical":
            # Nettoyage plus agressif pour TF-IDF
            text = text.lower()
            text = remove_punctuation(text)
            text = normalize_whitespace(text)
            if self.remove_stopwords_flag:
                text = remove_stopwords(text)

        # mode "transformer" : on s'arrête ici
        # CamemBERT tokenise lui-même, pas besoin de lowercase

        return text


def run_preprocessing() -> None:
    """
    Charge les données brutes, applique les deux modes de preprocessing,
    sauvegarde dans data/processed/.
    """
    preprocessor_classical = TextPreprocessor(
        mode="classical", remove_stopwords_flag=False
    )
    preprocessor_transformer = TextPreprocessor(mode="transformer")

    for split in ["train", "validation", "test"]:
        raw_path = RAW_DIR / f"{split}.parquet"

        if not raw_path.exists():
            logger.warning(f"Fichier brut absent : {raw_path} — skip")
            continue

        df = pd.read_parquet(raw_path)
        logger.info(f"[{split}] {len(df):,} exemples chargés")

        # Textes nettoyés pour les deux modes
        df["text_classical"] = preprocessor_classical.transform(df["text"])
        df["text_transformer"] = preprocessor_transformer.transform(df["text"])

        # Validation : pas de textes vides après nettoyage
        empty_classical = (df["text_classical"].str.strip() == "").sum()
        empty_transformer = (df["text_transformer"].str.strip() == "").sum()

        if empty_classical > 0:
            logger.warning(f"[{split}] {empty_classical} textes vides après nettoyage classique")
        if empty_transformer > 0:
            logger.warning(f"[{split}] {empty_transformer} textes vides après nettoyage transformer")

        # Sauvegarde
        out_path = PROCESSED_DIR / f"{split}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"[{split}] Sauvegardé → {out_path}")

    logger.info("✅ Preprocessing terminé")


if __name__ == "__main__":
    run_preprocessing()
