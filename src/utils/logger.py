# type: ignore
import logging
import os
import sys
from config import ROOT_DIR, LOG_FILE, FORMAT

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


def check_log_location() -> None:
    os.makedirs(ROOT_DIR, exist_ok=True)


def log_file_handler(loc: str, formatter, logger: logging.Logger) -> logging.Logger:

    file_handler = logging.FileHandler(loc, encoding="utf-8")  # UTF-8 for emojis
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def console_log_handler(formatter, logger: logging.Logger):

    console_handler = logging.StreamHandler(sys.stdout)  # force stdout
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger


def setup_logger() -> logging.Logger:

    check_log_location()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(FORMAT)

    log_file_handler(f"{ROOT_DIR}/{LOG_FILE}", formatter, logger)
    console_log_handler(formatter, logger)

    return logger
