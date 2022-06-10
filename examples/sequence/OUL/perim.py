import os
from pathlib import Path


for f in Path("dataset").glob("**/*perimeter*"):
    f = Path(f)

    os.rename(f, f.with_stem(f"{f.stem}-circle-{f.parent.stem}"))
