import pickle
from glob import glob
from pathlib import Path

from bikipy.behaviour.nort.trial import NortObjectField

ROOT = Path("C:/Users/Can/Projects/Neuroscience/bikipy/examples/data")
IMG_ROOT = ROOT / "images" / "nort" / "A"

T1 = IMG_ROOT / "before"
T2 = IMG_ROOT / "after"


nort_fields = []


for i, (habit, novelty) in enumerate(
    zip(glob(str(T2 / "habit*")), glob(str(T2 / "novel*"))), start=1
):
    nort_fields.append(NortObjectField.from_images(habit, novelty, label=i))


with open(ROOT / "b2_labels.pickle", "wb") as outpickle:
    pickle.dump(nort_fields, outpickle)
