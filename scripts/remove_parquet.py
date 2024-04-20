import os
from pathlib import Path
from typing import Final

from pydantic import DirectoryPath

DATASET_PATH: Final[DirectoryPath] = Path("dataset")

for f in DATASET_PATH.glob("**/*augmented*.parquet"):
    os.remove(f)
