from pathlib import Path

from bikipy.perimeter.polygon.makesense import (
    init_polygon_from_makesense_coco_polygon,
    init_polygon_from_makesense_csv_rectangle,
)
from bikipy.utils.collection_utils import get_first

ANNOTATION_DIR = Path(__file__).parent / "annotations"

rectangle_perimeter_rectangle_test_object = get_first(
    init_polygon_from_makesense_csv_rectangle(
        ANNOTATION_DIR / "rectangle_annotation.csv", image_root=ANNOTATION_DIR, meters_per_pixel=1.0
    ).values()
)
polygon_rectangle_perimeter_coco_test_object = get_first(
    init_polygon_from_makesense_coco_polygon(
        ANNOTATION_DIR / "polygon_coco_annotation.json",
        image_root=ANNOTATION_DIR,
        single_obj_return=True,
        meters_per_pixel=1.0,
    ).values()
)


def test_rectangle_read():
    assert rectangle_perimeter_rectangle_test_object


def test_polygon_read():
    assert polygon_rectangle_perimeter_coco_test_object
