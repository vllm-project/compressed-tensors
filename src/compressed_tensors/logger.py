# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Logger configuration for Compressed Tensors.
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

from loguru import logger


__all__ = ["LoggerConfig", "configure_logger", "logger"]


# used by `support_log_once``
_logged_once = set()


def _parse_bool_env(value: Optional[str]) -> Optional[bool]:
    """
    Parse a boolean environment variable value.
    Returns None if the value is None, True/False otherwise.

    Accepts: "1", "true", "True", "TRUE", "yes", "Yes", "YES" as True
    Accepts: "0", "false", "False", "FALSE", "no", "No", "NO", "" as False
    """
    if value is None:
        return None

    value_lower = value.lower().strip()
    if value_lower in ("1", "true", "yes"):
        return True
    elif value_lower in ("0", "false", "no", ""):
        return False
    else:
        # For backwards compatibility, any other non-empty string is treated as True
        return bool(value)


@dataclass
class LoggerConfig:
    disabled: bool = False
    clear_loggers: bool = False
    console_log_level: Optional[str] = None
    log_file: Optional[str] = None
    log_file_level: Optional[str] = None


def configure_logger(config: Optional[LoggerConfig] = None):
    """
    Configure the logger for Compressed Tensors.
    This function sets up the console and file logging
    as per the specified or default parameters.

    Note: Environment variables take precedence over the function parameters.

    By default, this function does NOT clear existing loggers or add new handlers,
    making it safe to use in library code. Applications can opt-in to the previous
    behavior by setting COMPRESSED_TENSORS_CLEAR_LOGGERS=1 and
    COMPRESSED_TENSORS_LOG_LEVEL=INFO.

    :param config: The configuration for the logger to use.
    :type config: LoggerConfig
    """
    logger_config = config or LoggerConfig()

    # env vars get priority
    disabled_env = _parse_bool_env(os.getenv("COMPRESSED_TENSORS_LOG_DISABLED"))
    if disabled_env is not None:
        logger_config.disabled = disabled_env

    clear_loggers_env = _parse_bool_env(os.getenv("COMPRESSED_TENSORS_CLEAR_LOGGERS"))
    if clear_loggers_env is not None:
        logger_config.clear_loggers = clear_loggers_env

    if (console_log_level := os.getenv("COMPRESSED_TENSORS_LOG_LEVEL")) is not None:
        logger_config.console_log_level = console_log_level.upper()

    if (log_file := os.getenv("COMPRESSED_TENSORS_LOG_FILE")) is not None:
        logger_config.log_file = log_file

    if (log_file_level := os.getenv("COMPRESSED_TENSORS_LOG_FILE_LEVEL")) is not None:
        logger_config.log_file_level = log_file_level.upper()

    if logger_config.disabled:
        logger.disable("compressed_tensors")
        return

    logger.enable("compressed_tensors")

    if logger_config.clear_loggers:
        logger.remove()

    if logger_config.console_log_level:
        # log as a human readable string with the time, function, level, and message
        logger.add(
            sys.stdout,
            level=logger_config.console_log_level.upper(),
            format="{time} | {function} | {level} - {message}",
            filter=support_log_once,
        )

    if logger_config.log_file or logger_config.log_file_level:
        log_file = logger_config.log_file or "compressed_tensors.log"
        log_file_level = logger_config.log_file_level or "INFO"
        # log as json to the file for easier parsing
        logger.add(
            log_file,
            level=log_file_level.upper(),
            serialize=True,
            filter=support_log_once,
        )


def support_log_once(record: Dict[str, Any]) -> bool:
    """
    Support logging only once using `.bind(log_once=True)`

    ```
    logger.bind(log_once=False).info("This will log multiple times")
    logger.bind(log_once=False).info("This will log multiple times")
    logger.bind(log_once=True).info("This will only log once")
    logger.bind(log_once=True).info("This will only log once")  # skipped
    ```
    """
    log_once = record["extra"].get("log_once", False)
    level = getattr(record["level"], "name", "none")
    message = hash(str(level) + record["message"])

    if log_once and message in _logged_once:
        return False

    if log_once:
        _logged_once.add(message)

    return True


# invoke logger setup on import with default values enabling console logging with INFO
# and disabling file logging
configure_logger(config=LoggerConfig())
