from pathlib import Path

from bikipy.perimeter.base import Perimeter

ANNOTATION_DIR = Path(__file__).parent / "annotations"


parallelogram_perimeter_rectangle_test_object = Perimeter.from_makesense_ai(
    ANNOTATION_DIR / "rectangle_annotation.csv", image_root=ANNOTATION_DIR
)

parallelogram_perimeter_coco_test_object = Perimeter.from_polygon_coco(
    ANNOTATION_DIR / "polygon_coco_annotation.json",
    image_root=ANNOTATION_DIR,
    single_obj_return=True,
)
