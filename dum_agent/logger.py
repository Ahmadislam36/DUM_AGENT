"""Step-by-step terminal logging for DUM Agent.

Level: env DUM_LOG_LEVEL = DEBUG | INFO (default) | WARNING
"""
import logging
import os
import sys

from . import config

_ROOT = "dum"


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever sys.stdout is *now* (works with Streamlit, pytest capture, etc.)."""

    def __init__(self):
        super().__init__()

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):  # StreamHandler.__init__ assigns it; ignore
        pass


def setup_logging() -> logging.Logger:
    root = logging.getLogger(_ROOT)
    if getattr(root, "_dum_configured", False):
        return root
    root.setLevel(getattr(logging, os.getenv("DUM_LOG_LEVEL", "INFO").upper(), logging.INFO))
    try:  # never crash on Windows consoles that cannot print some characters
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    handler = _StdoutHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-5s | %(name)-11s | %(message)s", "%H:%M:%S")
    )
    root.addHandler(handler)
    root.propagate = False
    root._dum_configured = True
    return root


def get_logger(area: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"{_ROOT}.{area}")


def log_startup() -> None:
    get_logger("app").info(
        "DUM Agent started | db=%s | deployment=%s | fast_routing=%s | reports=%s",
        config.DB_PATH, config.AZURE_OPENAI_DEPLOYMENT or "(not set)", config.FAST_ROUTING, config.REPORTS_DIR,
    )
