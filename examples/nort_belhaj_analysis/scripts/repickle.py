"""
Script used to update perimeter class instances inside pickle file
after the respective perimeter class has been updated
"""

import pickle
from pathlib import Path

from bikipy.behaviour.object_recognition.nort.trial import NortField
from bikipy.perimeter.base import PolygonPerimeter
from bikipy.plugins.belhaj import round_vs_apparatus_to_general_nort_fields

NORT_EXAMPLE_DIR = Path(".").resolve().parent
IMAGE_DIR = NORT_EXAMPLE_DIR / "data" / "area_images"

A_PICKLE_PATH = IMAGE_DIR / "A_annotations.pickle"
B_PICKLE_PATHS = (
    IMAGE_DIR / "B1" / "b1_labels.pickle",
    IMAGE_DIR / "B2" / "b2_labels.pickle",
)

NEW_A_PICKLE_PATHS = (
    IMAGE_DIR / "A1" / "a1_labels_repickled.pickle",
    IMAGE_DIR / "A2" / "a2_labels_repickled.pickle",
)


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "bikipy.preferance.gaze":
            renamed_module = "bikipy.perimeter.base"
        elif module == "bikipy.behaviour.nort.experiment":
            renamed_module = "bikipy.behaviour.nort.trial"
        elif "bikipy.perimeter" in module:
            renamed_module = module.replace("bikipy.perimeter", "bikipy.perimeter")

        if name == "NortObjectField":
            name = "NortField"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def deserialise_generic(obj):
    return {
        "corners": obj.sides,
    }


for annotation_obj_path in B_PICKLE_PATHS:
    with open(annotation_obj_path, "rb") as infile:
        gen_poly_seq = renamed_load(infile)

    for i, gen_poly in enumerate(gen_poly_seq):
        gen_poly_seq[i] = NortField(
            label=int(gen_poly.label),
            constant_object_perimeter=PolygonPerimeter.init_polygon(
                inspect_image=annotation_obj_path.parent / f"training_{i+1}.png",
                **deserialise_generic(gen_poly.constant_object),
            ),
            variable_object_perimeter=PolygonPerimeter.init_polygon(
                inspect_image=annotation_obj_path.parent / f"training_{i+1}.png",
                **deserialise_generic(gen_poly.variable_object),
            ),
            novel_object_perimeter=PolygonPerimeter.init_polygon(
                inspect_image=annotation_obj_path.parent / f"novel_{i+1}.png",
                **deserialise_generic(gen_poly.novel_object),
            ),
        )

    with open(
        annotation_obj_path.with_stem(f"{annotation_obj_path.stem}_repickled"), "wb"
    ) as outfile:
        pickle.dump(gen_poly_seq, outfile)

with open(A_PICKLE_PATH, "rb") as infile:
    round_vs_field_apparatus = renamed_load(infile)

round_keys = [f"round_{num}" for num in range(len(round_vs_field_apparatus))]
round_vs_field_vs_apparatus = {
    rem_round: round_vs_apparatus_to_general_nort_fields(
        field_apparatus, convert_from_legacy=True
    )
    for rem_round, field_apparatus in zip(round_keys, round_vs_field_apparatus.values())
}

for (round_number, apparatuses), path in zip(
    round_vs_field_vs_apparatus.items(), NEW_A_PICKLE_PATHS
):
    for app_id, apparatus in enumerate(apparatuses):
        apparatuses[app_id].constant_object_perimeter.inspect_image = (
            path.parent / f"training_{apparatus.label}.png"
        )
        apparatuses[app_id].variable_object_perimeter.inspect_image = (
            path.parent / f"training_{apparatus.label}.png"
        )
        apparatuses[app_id].novel_object_perimeter.inspect_image = (
            path.parent / f"novel_{apparatus.label}.png"
        )
        if apparatuses[app_id].novelty_constant_object_perimeter:
            apparatuses[app_id].novelty_constant_object_perimeter.inspect_image = (
                path.parent / f"novel_{apparatus.label}.png"
            )
    with open(path, "wb") as outfile:
        pickle.dump(apparatuses, outfile)
