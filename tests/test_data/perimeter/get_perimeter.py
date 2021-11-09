from bikipy.perimeter.base import Perimeter


perimeter_object = Perimeter.from_coco(
    "perimeter.json", image_root=".", single_obj_return=True
)
