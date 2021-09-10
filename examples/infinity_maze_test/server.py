import asyncio

from bikipy.behaviour.infinity_maze.trial import InfinityMaze
from bikipy.perimeter.base import PolygonalPerimeter

perimeters = PolygonalPerimeter.from_coco(
    "./coco_annotations_2021-09-01-02-19-41.json",
    inspect_image="./maze_example.png"
)

maze = InfinityMaze(
    **perimeters,
    delay_timings=(10, 40),
    delay_timings_trial_count=2,
    inspect_image="./maze_example.png"
)
event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(event_loop)
event_loop.run_until_complete(maze.live_localize())
