"""
Chargeur de configuration YAML centralise.
Utilise pydantic-settings pour la validation des variables d'environnement.
"""
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Variables d'environnement validees par Pydantic."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    model_name: str = Field(default="camembert-base")
    model_path: str = Field(default="./models/production/camembert_finetuned")
    max_length: int = Field(default=512)
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    mlflow_tracking_uri: str = Field(default="./mlruns")
    mlflow_experiment_name: str = Field(default="sentiment_allocine")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton des settings — charge une seule fois."""
    return Settings()


def load_yaml_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    """Charge la configuration YAML du projet."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable : {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_yaml_config()
settings = get_settings()
