"""Logging configuration for the one-click full pipeline.

The full pipeline launches several Python subprocesses.  This module owns the
file logger shared by the orchestrator; ``full_pipeline.run_command`` relays
subprocess output into the same logger while keeping it visible in the
terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Any, Mapping


LOGGER_NAME = "rdb_prior.full_pipeline"
DEFAULT_LOG_PATH = "outputs/logs/full_pipeline.log"
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5
_VALID_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


@dataclass(frozen=True)
class PipelineLogSettings:
    """Resolved logging settings for one pipeline run."""

    enabled: bool
    level_name: str
    path: Path | None
    max_bytes: int = DEFAULT_MAX_BYTES
    backup_count: int = DEFAULT_BACKUP_COUNT

    @property
    def level(self) -> int:
        return _VALID_LEVELS[self.level_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "level": self.level_name,
            "file": str(self.path) if self.path is not None else None,
            "max_bytes": self.max_bytes,
            "backup_count": self.backup_count,
        }


def resolve_pipeline_log_settings(
    config: Mapping[str, Any],
    project_root: Path,
    log_file_override: Path | None = None,
    log_level_override: str | None = None,
    disabled_override: bool = False,
    environ: Mapping[str, str] | None = None,
) -> PipelineLogSettings:
    """Resolve settings using CLI > environment > YAML > defaults precedence."""

    env = os.environ if environ is None else environ
    logging_cfg = dict(config.get("logging", {}))
    enabled = _as_bool(logging_cfg.get("enabled", True)) and not disabled_override

    raw_level = (
        log_level_override
        or env.get("RDB_PRIOR_LOG_LEVEL")
        or logging_cfg.get("level")
        or "INFO"
    )
    level_name = str(raw_level).upper()
    if level_name not in _VALID_LEVELS:
        valid = ", ".join(_VALID_LEVELS)
        raise ValueError(f"Unsupported logging level {raw_level!r}; choose one of: {valid}.")

    raw_path: Any = (
        log_file_override
        if log_file_override is not None
        else env.get("RDB_PRIOR_LOG_FILE") or logging_cfg.get("file", DEFAULT_LOG_PATH)
    )
    path = _resolve_path(raw_path, project_root) if enabled and raw_path else None

    max_bytes = int(logging_cfg.get("max_bytes", DEFAULT_MAX_BYTES))
    backup_count = int(logging_cfg.get("backup_count", DEFAULT_BACKUP_COUNT))
    if max_bytes <= 0:
        raise ValueError("logging.max_bytes must be positive.")
    if backup_count < 0:
        raise ValueError("logging.backup_count cannot be negative.")

    return PipelineLogSettings(
        enabled=enabled,
        level_name=level_name,
        path=path,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )


def configure_pipeline_logger(settings: PipelineLogSettings) -> logging.Logger:
    """Configure and return the named pipeline logger.

    No console handler is installed here because the orchestrator already
    writes status and relayed child output to the terminal.  This avoids
    duplicate terminal lines and keeps tqdm progress rendering intact.
    """

    logger = logging.getLogger(LOGGER_NAME)
    _close_handlers(logger)
    logger.setLevel(settings.level)
    logger.propagate = False

    if settings.enabled and settings.path is not None:
        settings.path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            settings.path,
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
            encoding="utf-8",
        )
        handler.setLevel(settings.level)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def get_pipeline_logger() -> logging.Logger:
    """Return the shared logger without changing its configuration."""

    return logging.getLogger(LOGGER_NAME)


def close_pipeline_logger() -> None:
    """Flush and close handlers so the log is complete when the process exits."""

    _close_handlers(logging.getLogger(LOGGER_NAME))


def _close_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        try:
            handler.flush()
        finally:
            handler.close()
            logger.removeHandler(handler)


def _resolve_path(value: Any, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


__all__ = [
    "DEFAULT_BACKUP_COUNT",
    "DEFAULT_LOG_PATH",
    "DEFAULT_MAX_BYTES",
    "LOGGER_NAME",
    "PipelineLogSettings",
    "close_pipeline_logger",
    "configure_pipeline_logger",
    "get_pipeline_logger",
    "resolve_pipeline_log_settings",
]
