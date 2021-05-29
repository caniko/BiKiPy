from pathlib import Path

from bikipy.perimeter.parallelogram.classes import ParallelogramPerimeter
from bikipy.perimeter.triangular import TriangularPerimeter

ROOT = Path(
    "C:/Users/Can/Projects/Neuroscience/bikipy/examples/data/images/results/phd"
)
IMAGE_B = ROOT / "B"
path_to_image = IMAGE_B / "after_1.png"

print(ParallelogramPerimeter(inspect_image=path_to_image, semantic_label="A"))
print(ParallelogramPerimeter(inspect_image=path_to_image, semantic_label="B"))
print(ParallelogramPerimeter(inspect_image=path_to_image, semantic_label="C"))
print(TriangularPerimeter.from_image(inspect_image=path_to_image, semantic_label="X"))
