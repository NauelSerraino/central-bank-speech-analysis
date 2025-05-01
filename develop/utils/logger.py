import logging
import os
from datetime import datetime

# Get absolute path to log directory
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str, log_file: str = "pipeline.log", level: int = logging.INFO) -> logging.Logger:
    """
    Creates and returns a configured logger.

    Args:
        name (str): Name of the logger.
        log_file (str): Log file name (within logs/).
        level (int): Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        log_path = os.path.join(LOG_DIR, log_file)

        # File handler
        fh = logging.FileHandler(log_path, mode='a')
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
