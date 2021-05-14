import pickle
from pathlib import Path

EXAMPLE_DATA_DIR = Path(".").resolve().parent / "nort_belhaj_analysis"
IMAGE_DIR = EXAMPLE_DATA_DIR / "area_images"

with open(IMAGE_DIR / "annotations.pickle", "rb") as infile:
    (
        a1_1,
        a1_2,
        a1_3,
        a1_4,
        a2_1,
        a2_2,
        a2_3,
        a2_4,
        b1_1,
        b1_2,
        b1_3,
        b1_4,
        b2_1,
        b2_2,
        b2_3,
        b2_4,
    ) = pickle.load(infile)

with open(IMAGE_DIR / "annotations.pickle", "wb") as outfile:
    pickle.dump(
        {
            "before": {
                "t1": {1: a1_1, 2: a1_2, 3: a1_3, 4: a1_4},
                "t2": {1: a2_1, 2: a2_2, 3: a2_3, 4: a2_4},
            },
            "after": {
                "t1": {1: b1_1, 2: b1_2, 3: b1_3, 4: b1_4},
                "t2": {1: b2_1, 2: b2_2, 3: b2_3, 4: b2_4},
            },
        },
        outfile,
    )
