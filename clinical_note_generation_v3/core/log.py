"""
Centralized structured logging helpers for the v3 pipeline.
"""

from __future__ import annotations

import logging
import warnings

import structlog


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="dotenv")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="dotenv")

_STRUCTLOG_CONFIGURED = False


def configure_structured_logging(*, enabled: bool, log_level: int = logging.INFO) -> None:
    """
    Configure structlog once. When disabled, preserve the existing stdlib logging behavior.
    """
    global _STRUCTLOG_CONFIGURED
    if not enabled or _STRUCTLOG_CONFIGURED:
        return

    processors = [
        structlog.threadlocal.merge_threadlocal_context,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(indent=2, sort_keys=True),
    ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=log_level, format="%(message)s")
    _STRUCTLOG_CONFIGURED = True


def get_logger(log_name: str = __name__):
    """
    Return a structlog-backed logger.
    """
    return structlog.wrap_logger(logging.getLogger(log_name))
