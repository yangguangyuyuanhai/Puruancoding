import logging
import os
import pathlib

import helps.vairables as app_vars

logger_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class AppLogger:

    @classmethod
    def get_logger(cls, name):
        if not hasattr(cls, "initialized"):
            cls.init_logger_config()
        return logging.getLogger(name)

    @classmethod
    def init_logger_config(cls):
        log_base_dir = pathlib.Path(app_vars.APP_LOG_FILE).parent
        os.makedirs(log_base_dir, exist_ok=True)
        file_handler = logging.FileHandler(app_vars.APP_LOG_FILE)
        level = getattr(cls, "level", "info")

        logging.basicConfig(
            level=logger_levels.get(level, logging.INFO),
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logging.getLogger().addHandler(file_handler)

        if not hasattr(cls, "initialized"):
            setattr(cls, "initialized", True)

    @classmethod
    def set_logger_level(cls, level="info"):
        if not hasattr(cls, "level"):
            setattr(cls, "level", level)
        else:
            setattr(cls, "level", level)
        logging.getLogger().setLevel(logger_levels.get(level, logging.INFO))
