"""
Téléchargement et sauvegarde du dataset Allociné depuis HuggingFace.
Sauvegarde en Parquet (format colonne compressé, rapide et léger).
"""
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_allocine() -> dict[str, pd.DataFrame]:
    """
    Télécharge le dataset Allociné et le sauvegarde localement.
    Retourne un dict {split: DataFrame}.
    """
    logger.info("Chargement du dataset 'allocine' depuis HuggingFace...")

    dataset = load_dataset("allocine", trust_remote_code=True)

    splits = {}
    stats = {}

    for split_name in ["train", "validation", "test"]:
        df = dataset[split_name].to_pandas()

        # Renommage pour cohérence
        df = df.rename(columns={"review": "text", "label": "label"})

        # Validation minimale
        assert "text" in df.columns, f"Colonne 'text' absente dans {split_name}"
        assert "label" in df.columns, f"Colonne 'label' absente dans {split_name}"
        assert df["label"].nunique() == 2, "Attendu 2 classes uniquement"

        # Sauvegarde Parquet
        out_path = RAW_DIR / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)

        splits[split_name] = df
        stats[split_name] = {
            "n_samples": len(df),
            "n_positive": int((df["label"] == 1).sum()),
            "n_negative": int((df["label"] == 0).sum()),
            "balance_ratio": round((df["label"] == 1).mean(), 4),
        }

        logger.info(
            f"[{split_name}] {len(df):,} exemples | "
            f"pos={stats[split_name]['n_positive']:,} | "
            f"neg={stats[split_name]['n_negative']:,}"
        )

    # Sauvegarde des stats pour traçabilité
    stats_path = RAW_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Stats sauvegardées → {stats_path}")
    return splits


if __name__ == "__main__":
    splits = download_allocine()
    logger.info("✅ Dataset téléchargé et sauvegardé dans data/raw/")
