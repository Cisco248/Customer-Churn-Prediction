import logging
import os
from config import ROOT_DIR, LOG_FILE, FORMAT


def check_log_location() -> None:
    if ROOT_DIR / LOG_FILE == True:
        print("Already Exists!")

    return os.makedirs(ROOT_DIR, exist_ok=True)


def log_file_handler(
    loc: str,
    format,
    logger: logging.Logger,
) -> logging.Logger:

    file_handler = logging.FileHandler(loc)

    file_handler.setLevel(logging.DEBUG)

    file_handler.setFormatter(format)

    logger.addHandler(file_handler)

    return logger


def console_log_handler(format, logger: logging.Logger):

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.DEBUG)

    console_handler.setFormatter(format)

    logger.addHandler(console_handler)

    return logger


def setup_logger() -> logging.Logger:

    check_log_location()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(FORMAT)

    log_file_handler(LOG_FILE, formatter, logger)

    return logger
