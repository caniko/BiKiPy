from pathlib import Path
import pickle

from bikipy.border.base import GenericPolygonalBorder


WORKING_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data")
ANNOTATIONS_PATH = str(WORKING_DIR / "python_data" / "annotations" / "all.pickle")


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "bikipy.preferance.gaze":
            renamed_module = "bikipy.border.base"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


with open(ANNOTATIONS_PATH, "rb") as infile:
    (
        a1_1,
        a1_2,
        a1_3,
        a1_4,
        a2_1,
        a2_2,
        a2_3,
        a2_4,
        b1_1,
        b1_2,
        b1_3,
        b1_4,
        b2_1,
        b2_2,
        b2_3,
        b2_4,
    ) = renamed_load(infile)

# with open(IMPORTED_DLC_FILES, "rb") as infile:
#     id_dlc_0, id_dlc_1 = pickle.load(infile)

b1_1["A"] = GenericPolygonalBorder(
    b1_1["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b1_2["A"] = GenericPolygonalBorder(
    b1_2["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b1_3["A"] = GenericPolygonalBorder(
    b1_3["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b1_4["A"] = GenericPolygonalBorder(
    b1_4["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b2_1["A"] = GenericPolygonalBorder(
    b2_1["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b2_2["A"] = GenericPolygonalBorder(
    b2_2["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b2_3["A"] = GenericPolygonalBorder(
    b2_3["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b2_4["A"] = GenericPolygonalBorder(
    b2_4["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a1_1["A"] = GenericPolygonalBorder(
    a1_1["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a1_2["A"] = GenericPolygonalBorder(
    a1_2["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a1_3["A"] = GenericPolygonalBorder(
    a1_3["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a1_4["A"] = GenericPolygonalBorder(
    a1_4["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a2_1["A"] = GenericPolygonalBorder(
    a2_1["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a2_2["A"] = GenericPolygonalBorder(
    a2_2["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a2_3["A"] = GenericPolygonalBorder(
    a2_3["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a2_4["A"] = GenericPolygonalBorder(
    a2_4["A"]._NortObject__sides, border_distance=BORDER_DISTANCE
)

b1_1["B"] = GenericPolygonalBorder(
    b1_1["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b1_2["B"] = GenericPolygonalBorder(
    b1_2["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b1_3["B"] = GenericPolygonalBorder(
    b1_3["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b1_4["B"] = GenericPolygonalBorder(
    b1_4["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b2_1["B"] = GenericPolygonalBorder(
    b2_1["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b2_2["B"] = GenericPolygonalBorder(
    b2_2["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b2_3["B"] = GenericPolygonalBorder(
    b2_3["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
b2_4["B"] = GenericPolygonalBorder(
    b2_4["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a1_1["B"] = GenericPolygonalBorder(
    a1_1["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a1_2["B"] = GenericPolygonalBorder(
    a1_2["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a1_3["B"] = GenericPolygonalBorder(
    a1_3["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a1_4["B"] = GenericPolygonalBorder(
    a1_4["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a2_1["B"] = GenericPolygonalBorder(
    a2_1["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a2_2["B"] = GenericPolygonalBorder(
    a2_2["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a2_3["B"] = GenericPolygonalBorder(
    a2_3["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)
a2_4["B"] = GenericPolygonalBorder(
    a2_4["B"]._NortObject__sides, border_distance=BORDER_DISTANCE
)

with open(ANNOTATIONS_PATH, "wb") as outfile:
    pickle.dump(
        (
            a1_1,
            a1_2,
            a1_3,
            a1_4,
            a2_1,
            a2_2,
            a2_3,
            a2_4,
            b1_1,
            b1_2,
            b1_3,
            b1_4,
            b2_1,
            b2_2,
            b2_3,
            b2_4,
        ),
        outfile,
    )
