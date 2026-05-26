"""
Structured logging configuration for the Nexus AI system.
Replaces all print/traceback usage with proper logging.
"""
import logging
import sys

def setup_logger(name: str = "nexus_ai", level: int = logging.INFO) -> logging.Logger:
    """
    Creates a configured logger with consistent formatting.
    
    Args:
        name: Logger name (used for hierarchical logging).
        level: Minimum log level to capture.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger

# Module-level logger for quick import
logger = setup_logger()
