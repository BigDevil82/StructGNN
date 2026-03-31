import logging
from pathlib import Path


def setup_file_logger(name: str, log_file: Path, level: int) -> logging.Logger:
    """创建仅文件输出的 logger，重复调用时不会重复添加 handler。"""
    log_dir = log_file.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
