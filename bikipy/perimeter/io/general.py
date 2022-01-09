from typing import Optional, Iterable

from pydantic import DirectoryPath

from bikipy.perimeter.base import PerimeterSet


def defer_perimeter_set_from_multi_row_reference(
    reference_perimeters: Iterable,
    reference_perimeter_image_name: str,
    image_name_to_reference_data: dict,
    image_root: Optional[DirectoryPath] = None,
):
    perimeter_set = PerimeterSet(
        perimeters=tuple(reference_perimeters),
    )
    result = {perimeter_set.inspect_image_path.name: perimeter_set}
    for image_name, reference in image_name_to_reference_data.items():
        if image_name == reference_perimeter_image_name:
            continue
        result[image_name] = perimeter_set.change_reference(
            new_reference=reference, new_inspect_image_path=image_root / image_name
        )

    return result
