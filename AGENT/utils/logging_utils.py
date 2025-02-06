# src/utils/logging_utils.py

import logging
import sys

def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s | %(name)s | %(levelname)s]: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
