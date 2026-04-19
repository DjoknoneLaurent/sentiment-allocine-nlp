"""
Utilitaires partagés pour le chargement des données traitées.
Utilisé par les notebooks, les scripts d'entraînement et l'API.
"""
from pathlib import Path
from typing import Literal
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
LABEL_MAP    = {0: "négatif",  1: "positif"}
LABEL_MAP_EN = {0: "negative", 1: "positive"}

SplitName = Literal["train", "validation", "test"]
TextCol   = Literal["text", "text_classical", "text_transformer"]


def load_split(
    split: SplitName,
    text_col: TextCol = "text_transformer",
    processed_dir: Path | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Charge un split depuis data/processed/.

    Returns
    -------
    X : pd.Series  — textes
    y : pd.Series  — labels (0/1)
    """
    base = processed_dir or PROCESSED_DIR
    path = base / f"{split}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Split '{split}' introuvable : {path}\n"
            "→ Lance d'abord : python src/data/download_dataset.py && "
            "python src/data/preprocess.py"
        )

    df = pd.read_parquet(path)

    if text_col not in df.columns:
        available = [c for c in df.columns if c.startswith("text")]
        raise ValueError(
            f"Colonne '{text_col}' absente. Disponibles : {available}"
        )

    logger.info(
        f"[{split}] {len(df):,} exemples | "
        f"col='{text_col}' | "
        f"positif={df['label'].mean():.2%}"
    )
    return df[text_col], df["label"]


def load_all_splits(
    text_col: TextCol = "text_transformer",
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Charge les 3 splits d'un coup."""
    return {s: load_split(s, text_col) for s in ["train", "validation", "test"]}
