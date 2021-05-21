"""
Script used to update border class instances inside pickle file
after the respective border class has been updated
"""

import pickle
from pathlib import Path

from bikipy.behaviour.nort.trial import NortObjectField
from bikipy.border.base import GenericPolygonalBorder

NORT_EXAMPLE_DIR = Path(".").resolve().parent
IMAGE_DIR = NORT_EXAMPLE_DIR / "area_images"

ANNOTATION_B1_T1 = IMAGE_DIR / "B1" / "b1_labels.pickle"
ANNOTATION_B1_T2 = IMAGE_DIR / "B2" / "b2_labels.pickle"

BORDER_DISTANCE = 0.03 * 224 / 0.4


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "bikipy.preferance.gaze":
            renamed_module = "bikipy.border.base"
        if module == "bikipy.behaviour.nort.experiment":
            renamed_module = "bikipy.behaviour.nort.trial"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


with open(ANNOTATIONS_PATH, "rb") as infile:
    gen_poly_seq = renamed_load(infile)

for gen_poly in gen_poly_seq:
    for area_type, value in gen_poly.items():
        gen_poly_seq[area_type] = GenericPolygonalBorder(
            value.perimeter_corners, border_distance=BORDER_DISTANCE
        )

# for i, gen_poly in enumerate(gen_poly_seq):
#     gen_poly.constant_object.border_distance = BORDER_DISTANCE
#     gen_poly.novel_object.border_distance = BORDER_DISTANCE
#     gen_poly.variable_object.border_distance = BORDER_DISTANCE
#
#     gen_poly_seq[i] = NortObjectField(
#         gen_poly.constant_object,
#         gen_poly.novel_object,
#         gen_poly.variable_object,
#         label=gen_poly.label
#     )

with open(ANNOTATIONS_PATH, "wb") as outfile:
    pickle.dump(gen_poly_seq, outfile)
