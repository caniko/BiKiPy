import pickle
from glob import glob, iglob
from pathlib import Path

from bikipy.behaviour.nort.trial import NortField

ROOT = Path(".").resolve().parent / "data" / "area_images"

for directory in iglob(ROOT / "**", recursive=True):
    nort_fields = [
        NortField.from_images(i, habit, novelty)
        for i, (habit, novelty) in enumerate(
            zip(glob(str(directory / "habit*")), glob(str(directory / "novel*"))),
            start=1,
        )
    ]
    with open(ROOT / directory / f"{directory.stem}_labels.pickle", "wb") as out_pickle:
        pickle.dump(nort_fields, out_pickle)
