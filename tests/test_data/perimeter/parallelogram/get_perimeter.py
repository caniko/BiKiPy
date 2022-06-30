from pathlib import Path

from bikipy.perimeter.polygon.makesense import (
    init_polygon_from_makesense_coco_polygon,
    init_polygon_from_makesense_csv_rectangle,
)

ANNOTATION_DIR = Path(__file__).parent / "annotations"


rectangle_perimeter_rectangle_test_object = next(
    init_polygon_from_makesense_csv_rectangle(
        ANNOTATION_DIR / "rectangle_annotation.csv", image_root=ANNOTATION_DIR
    ).values()
)

rectangle_perimeter_coco_test_object = init_polygon_from_makesense_coco_polygon(
    ANNOTATION_DIR / "polygon_coco_annotation.json",
    image_root=ANNOTATION_DIR,
    single_obj_return=True,
)
