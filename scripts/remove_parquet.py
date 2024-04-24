from pathlib import Path
from typing import Final

from pydantic import DirectoryPath

DATASET_PATH: Final[DirectoryPath] = Path("dataset")

for f in DATASET_PATH.glob("**/*augmented*.parquet"):
    print(f"Will remove: {f}")
    # os.remove(f)
