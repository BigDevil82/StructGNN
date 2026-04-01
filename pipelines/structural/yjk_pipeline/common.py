from __future__ import annotations

import logging
from pathlib import Path


def setup_file_logger(name: str, log_file: Path, level: int) -> logging.Logger:
    """创建仅文件输出的 logger，重复调用时不会重复添加 handler。"""
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


def clean_output_files(output_dir: Path, file_names: list[str], logger: logging.Logger) -> None:
    """清理旧输出文件。"""
    for file_name in file_names:
        target = output_dir / file_name
        if not target.exists():
            continue
        try:
            target.unlink()
            logger.debug("Removed old file: %s", target)
        except Exception as exc:
            logger.warning("Failed to remove %s: %s", target, exc)
