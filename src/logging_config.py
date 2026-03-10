import logging
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Set up logging configuration.

    Args:
        level (str): Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_file (str|Path, optional): Path to the log file. If None, logs will be printed to console.
    """

    logger = logging.getLogger("uk_used_car_price")
    logger.setLevel(level.upper())

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level.upper())
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level.upper())
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
