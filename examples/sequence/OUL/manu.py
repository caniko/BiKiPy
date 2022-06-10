import os
from pathlib import Path

for f in Path("dataset").glob("**/*.*"):
    f = Path(f)

    sequence = f.stem.split("-")[0]

    new_name = f"{sequence}.{f.stem[2:]}"

    os.rename(f, f.with_stem(new_name))
