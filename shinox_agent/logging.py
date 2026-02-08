import logging
import os
import sys
from pythonjsonlogger.json import JsonFormatter


def setup_json_logging(service_name: str, level: str = None):
    """Configure structured JSON logging for a Shinox service."""
    level = level or os.getenv("LOG_LEVEL", "INFO")

    formatter = JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
        static_fields={"service": service_name},
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
