from functools import cached_property, lru_cache
from logging import getLogger
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from pydantic import DirectoryPath, FilePath, computed_field, validate_call

from bikipy.core.typing import Label
from bikipy.ingress.name_parser import PluginFileStemParseLastIsLabel
from bikipy.ingress.plugin.core.plugin_scope import PluginScope
from bikipy.ingress.plugin.perimeter.base import AbstractPerimeterPlugin
from bikipy.ingress.plugin.perimeter.constant import LABEL_TO_TRIAL_SHEET_NAME
from bikipy.perimeter.base import SinglePerimeter, perimeter_set_from_makesense
from bikipy.utils.collection_utils import get_first_key_in_dict
from bikipy.utils.image import axis_frame_imshow, read_image_from_path
from bikipy.utils.makesense import (
    SHAPE_TO_MAKESENSE_TYPE,
    first_image_name_from_makesense,
)

logger = getLogger(__name__)


class SinglePerimeterPluginFileStemParse(PluginFileStemParseLastIsLabel):
    def __pop_split_till_empty__(self) -> None:
        super().__pop_split_till_empty__()
        self.shape = self.split.pop()


# TODO: Manual radius readings from settings.yaml read.
class PluginSinglePerimeter(AbstractPerimeterPlugin):
    label_prefix: Optional[str]
    label_suffix: Optional[str]

    plugin_file_stem_parser = SinglePerimeterPluginFileStemParse

    ingress_key = "perimeter"
    code_key = "perimeter"
    default_trial_argument_key = "label_to_perimeter"
    human_readable_index = "Perimeter"

    @computed_field
    @cached_property
    def image_name(self):
        return first_image_name_from_makesense(self.data_path, SHAPE_TO_MAKESENSE_TYPE[self.stem_info.shape])

    def trialwise_and_metadata(
        self, trial_id: Label, naive: bool = False
    ) -> dict[str, SinglePerimeter] | SinglePerimeter:
        self._assert_correct_scope_trialwise_metadata()

        result = {}
        for label, perimeter in self.perimeter_mapper(trial_id).items():
            if (
                PluginScope.METADATA in self.ingress.definition_single_perimeter
                and LABEL_TO_TRIAL_SHEET_NAME in self.ingress.metadata_sheet_names
            ):
                label = self.ingress.metadata_perimeter_label_sheet.loc[trial_id, label]
            if self.label_prefix:
                label = f"{self.label_prefix}{label}"
            if self.label_suffix:
                label = f"{label}{self.label_suffix}"

            result[label] = _perimeter_with_label(perimeter, label)

            if self.stem_info.label not in self.ingress.ingress_defined_perimeters:
                self.ingress.ingress_defined_perimeters[self.stem_info.label] = {}
            self.ingress.ingress_defined_perimeters[self.stem_info.label][label] = perimeter

        if naive:
            return next(iter(result.values()))
        return result

    @computed_field
    @property
    def globally_defined(self) -> dict[str, SinglePerimeter]:
        self._assert_correct_scope_global()

        result = {}
        for label, perimeter in self.perimeter_mapper().items():
            if self.label_prefix:
                label = f"{self.label_prefix}_{label}"
            if self.label_suffix:
                label = f"{label}_{self.label_suffix}"

            perimeter = _perimeter_with_label(perimeter, label)

            result[label] = perimeter
            self.ingress.ingress_defined_perimeters[label] = perimeter

        return result


@validate_call
def inspect_annotations(annotation_path: FilePath, image_directory: Optional[DirectoryPath] = None) -> None:
    _, shape, annotation_label = annotation_path.stem.split("-")
    image_name_to_perimeter_set = perimeter_set_from_makesense(annotation_path, shape)

    # Check if image is in current directory
    if not image_directory:
        image_directory = Path(".").resolve()
        if not (image_directory / get_first_key_in_dict(image_name_to_perimeter_set)).exists():
            msg = "Can not inspect annotation without the image coupled to it being provided"
            raise ValueError(msg)

    fig, axes = plt.subplots(len(image_name_to_perimeter_set))
    if len(image_name_to_perimeter_set) == 1:
        axes = [axes]

    for ax, (image_name, perimeter_set) in zip(axes, image_name_to_perimeter_set.items()):
        axis_frame_imshow(ax, read_image_from_path(image_directory / image_name))
        perimeter_set.plot(manual_ax=ax, coordinates_as_pixels=True)

    plt.show()


@lru_cache
def _perimeter_with_label(perimeter: SinglePerimeter, new_label: str) -> SinglePerimeter:
    if new_label == perimeter.label:
        return perimeter
    return perimeter.copy(update={"label": new_label})
