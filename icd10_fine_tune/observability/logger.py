"""
Structured Logger for Fine-Tuning Pipeline
==========================================

WHAT THIS MODULE DOES:
Provides structured JSON logging for the fine-tuning pipeline.
All training events, metrics, and errors are logged in a consistent format.

WHY WE NEED THIS:
1. **Debuggability**: When something fails at 3 AM, structured logs help
   pinpoint the issue quickly with searchable fields.

2. **Observability**: Log aggregation tools (ELK, Datadog) can parse JSON logs
   for dashboards and alerts.

3. **Reproducibility**: Logs capture all configuration, enabling reproduction
   of training runs.

HOW IT WORKS:
1. Configure Python logging with JSON formatter
2. Add context fields (run_id, epoch, step)
3. Provide convenience methods for common log patterns
4. Output to both console (human-readable) and file (JSON)
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

from icd10_fine_tune.config.settings import settings


class JSONFormatter(logging.Formatter):
    """
    Format log records as JSON.
    
    EDUCATIONAL NOTE - Why JSON Logging:
    Plain text logs like "Training step 100 loss=0.5" are hard to parse.
    JSON logs like {"step": 100, "loss": 0.5} can be:
    1. Searched with jq/grep
    2. Loaded into Pandas for analysis
    3. Ingested by log aggregation systems
    4. Filtered by any field
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_dict = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, "extra"):
            log_dict.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_dict)


class PrettyFormatter(logging.Formatter):
    """Human-readable formatter for console output."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format extra fields if present
        extra_str = ""
        if hasattr(record, "extra") and record.extra:
            extra_str = " | " + " ".join(f"{k}={v}" for k, v in record.extra.items())
        
        return f"{color}[{timestamp}] {record.levelname:<8}{self.RESET} {record.getMessage()}{extra_str}"


class TrainingLogger:
    """
    Structured logger for fine-tuning pipeline.
    
    Usage:
        logger = TrainingLogger("training")
        logger.info("Starting training", epoch=1, learning_rate=2e-4)
        logger.metric(step=100, loss=0.5, accuracy=0.85)
    """
    
    def __init__(
        self,
        name: str = "icd10_finetune",
        log_dir: Optional[Path] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files (defaults to settings.output_dir/logs)
            console_level: Minimum level for console output
            file_level: Minimum level for file output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(PrettyFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler (JSON)
        log_dir = log_dir or settings.get_logs_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)
        
        self._context: Dict[str, Any] = {}
        self.log_file = log_file
        
        self.info(f"Logger initialized. Log file: {log_file}")
    
    def set_context(self, **kwargs: Any) -> None:
        """
        Set persistent context fields that will be included in all logs.
        
        EDUCATIONAL NOTE - Log Context:
        Context fields like run_id, model_variant are included in every log.
        This makes it easy to filter logs for a specific training run.
        """
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context fields."""
        self._context = {}
    
    @contextmanager
    def context(self, **kwargs: Any):
        """
        Temporarily add context fields.
        
        Usage:
            with logger.context(epoch=1):
                logger.info("Training epoch")  # Includes epoch=1
            logger.info("Done")  # No longer includes epoch
        """
        old_context = self._context.copy()
        self._context.update(kwargs)
        try:
            yield
        finally:
            self._context = old_context
    
    def _log(
        self,
        level: int,
        message: str,
        **kwargs: Any
    ) -> None:
        """Internal logging method."""
        extra = {**self._context, **kwargs}
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "(unknown file)",
            0,
            message,
            (),
            None,
            extra=extra
        )
        record.extra = extra
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """
        Log error message.
        
        Args:
            message: Error message
            exc_info: If True, include exception traceback
            **kwargs: Additional fields to log
        """
        # Don't pass exc_info as a kwarg to avoid conflicts
        extra = {**self._context, **kwargs}
        record = self.logger.makeRecord(
            self.logger.name,
            logging.ERROR,
            "(unknown file)",
            0,
            message,
            (),
            sys.exc_info() if exc_info else None,
            extra=extra
        )
        record.extra = extra
        self.logger.handle(record)
    
    def metric(self, **metrics: Any) -> None:
        """
        Log training metrics.
        
        EDUCATIONAL NOTE - Metric Logging:
        Separating metric logs from info logs allows:
        1. Easy filtering: grep for "type":"metric"
        2. Time series analysis in visualization tools
        3. Automated alerting on metric thresholds
        """
        self._log(logging.INFO, "metric", type="metric", **metrics)
    
    def config(self, config: Dict[str, Any]) -> None:
        """Log training configuration."""
        self._log(logging.INFO, "Configuration logged", type="config", config=config)
    
    def epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Log epoch start."""
        self._log(
            logging.INFO,
            f"Epoch {epoch}/{total_epochs} started",
            type="epoch_start",
            epoch=epoch,
            total_epochs=total_epochs
        )
    
    def epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log epoch end with metrics."""
        self._log(
            logging.INFO,
            f"Epoch {epoch} completed",
            type="epoch_end",
            epoch=epoch,
            **metrics
        )
    
    def training_start(self, **config: Any) -> None:
        """Log training start."""
        self._log(
            logging.INFO,
            "Training started",
            type="training_start",
            **config
        )
    
    def training_end(self, **summary: Any) -> None:
        """Log training end."""
        self._log(
            logging.INFO,
            "Training completed",
            type="training_end",
            **summary
        )


# Global logger instance
_logger: Optional[TrainingLogger] = None


def get_logger() -> TrainingLogger:
    """
    Get or create the global logger instance.
    
    EDUCATIONAL NOTE - Singleton Logger:
    We use a single logger instance throughout the application to:
    1. Ensure consistent log file
    2. Share context across modules
    3. Avoid handler duplication
    """
    global _logger
    if _logger is None:
        _logger = TrainingLogger()
    return _logger
