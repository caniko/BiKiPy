from pathlib import Path

from bikipy.border.parallelogram.classes import ParallelogramBorder
from bikipy.border.triangular import TriangularBorder

ROOT = Path(
    "C:/Users/Can/Projects/Neuroscience/bikipy/examples/data/images/y_maze/master's"
)
path_to_image = ROOT / "before_63_masters.png"

print(ParallelogramBorder(guiding_image=path_to_image, label="A"))
print(ParallelogramBorder(guiding_image=path_to_image, label="B"))
print(ParallelogramBorder(guiding_image=path_to_image, label="C"))
print(TriangularBorder.from_image(guiding_image=path_to_image, label="X"))
