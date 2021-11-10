from pathlib import Path

from bikipy.perimeter.base import Perimeter


FILE_ROOT = Path(__file__).parent


perimeter_object = Perimeter.from_coco(
    FILE_ROOT / "perimeter.json", image_root=FILE_ROOT, single_obj_return=True
)
perimeter_object.plot()
