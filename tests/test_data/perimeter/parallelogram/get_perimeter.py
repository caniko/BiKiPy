from pathlib import Path

from bikipy.perimeter.polygon.base import PolygonPerimeter

ANNOTATION_DIR = Path(__file__).parent / "annotations"


parallelogram_perimeter_rectangle_test_object = PolygonPerimeter.from_makesense_ai(
    ANNOTATION_DIR / "rectangle_annotation.csv", image_root=ANNOTATION_DIR
)

parallelogram_perimeter_coco_test_object = PolygonPerimeter.from_makesense_coco_polygon(
    ANNOTATION_DIR / "polygon_coco_annotation.json",
    image_root=ANNOTATION_DIR,
    single_obj_return=True,
)
