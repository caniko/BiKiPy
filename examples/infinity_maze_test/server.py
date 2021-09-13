from pathlib import Path

from bikipy.behaviour.infinity_maze.trial import InfinityMaze
from bikipy.perimeter.base import PolygonalPerimeter

ROOT = Path(".").resolve()

perimeters = PolygonalPerimeter.from_coco(
    ROOT / "coco_annotations_2021-09-01-02-19-41.json",
    inspect_image="./maze_example.png",
)

maze = InfinityMaze(
    **perimeters,
    delay_timings=(10, 40),
    delay_timings_trial_count=2,
    inspect_image="./maze_example.png",
    fps=30,
    save_root=ROOT / "results",
)

maze.application()
