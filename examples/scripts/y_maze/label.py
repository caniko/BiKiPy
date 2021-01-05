from pathlib import Path

from bikipy.border.parallelogram.classes import ParallelogramBorder
from bikipy.border.triangular import TriangularBorder

ROOT = Path("C:/Users/Can/Projects/Neuroscience/bikipy/examples/data/images/y_maze/phd")
IMAGE_B = ROOT / "B"
path_to_image = IMAGE_B / "after_1.png"

print(ParallelogramBorder(guiding_image=path_to_image, semantic_label="A"))
print(ParallelogramBorder(guiding_image=path_to_image, semantic_label="B"))
print(ParallelogramBorder(guiding_image=path_to_image, semantic_label="C"))
print(TriangularBorder.from_image(guiding_image=path_to_image, semantic_label="X"))
