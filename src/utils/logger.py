"""
Logging structure centralise — loguru.
Usage : from src.utils.logger import get_logger
        logger = get_logger(__name__)
"""
import sys
from pathlib import Path
from loguru import logger as _logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str):
    """Retourne un logger configure avec sortie console + fichier rotatif."""
    _logger.remove()

    _logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
               "<level>{message}</level>",
        level="DEBUG",
        colorize=True,
    )

    _logger.add(
        LOG_DIR / "sentiment_allocine.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    return _logger.bind(module=name)
