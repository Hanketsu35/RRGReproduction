"""Logging utility for MOE-RRG training."""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "moe_rrg", log_dir: str = "logs",
                 level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with file and console output.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level

    Returns:
        Configured logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_dir / f"{name}_{timestamp}.log"
    )
    file_handler.setLevel(level)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
