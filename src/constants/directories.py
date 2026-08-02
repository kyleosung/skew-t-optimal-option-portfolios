"""
Stores directories. Load from these without manual typing.
"""

import os
from pathlib import Path

REPO_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

DATA_DIR = Path(REPO_DIR.parent / "data")  # store data outside repo
CACHE_DIR = Path(REPO_DIR.parent / "cache")
MODEL_DIR = Path(REPO_DIR / "models")
IMG_DIR = Path(REPO_DIR / "images")

for dir_path in [DATA_DIR, CACHE_DIR, MODEL_DIR, IMG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(DATA_DIR)
    print(MODEL_DIR)
    print(IMG_DIR)
