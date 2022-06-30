import os
from pathlib import Path

dataset = Path(".").resolve() / "dataset"

for f in dataset.glob("**/*.parquet"):
    s, name = f.stem.split(".")
    s = int(s) - 1
    os.rename(f, f.with_stem(f"{s}.{name}"))
