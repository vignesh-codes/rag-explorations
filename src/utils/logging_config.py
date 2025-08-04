"""Simple logging setup."""

import logging

def setup_logging(level="INFO"):
    """Setup basic logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def get_logger(name):
    """Get a logger."""
    return logging.getLogger(name)