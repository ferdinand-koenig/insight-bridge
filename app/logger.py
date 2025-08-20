import logging
import os
from logging.handlers import RotatingFileHandler

import yaml

# Step 1: Read YAML config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Step 2: Map string level to logging constant
level_name = config.get("logging", {}).get("level", "INFO").upper()
level = getattr(logging, level_name, logging.INFO)

# Create a named logger for the project
logger = logging.getLogger("insight_bridge")
logger.setLevel(level)
logger.propagate = False  # prevent logs from bubbling to root logger

if not logger.handlers:
    # --- Common formatter (with filename:l.lineno) ---
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s - %(name)s - %(filename)s:l.%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- Console handler ---
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --- Rotating file handler ---
    os.makedirs("logs", exist_ok=True)
    fh = RotatingFileHandler(
        "logs/app.log",
        maxBytes=10*1024*1024,   # 10 MB per file
        backupCount=2           # keep last 2 logs + 1 current
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# Silence overly verbose libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)

# Example usage (can remove in production)
logger.info("Project-wide logger initialized")
logger.info(f"Logging level set to {logging.getLevelName(level)}")
