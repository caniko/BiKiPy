from functools import cached_property, lru_cache
from logging import getLogger
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import DirectoryPath, FilePath, PositiveInt, validate_arguments

from bikipy.core.typing import TrialId
from bikipy.ingress.plugin.base import BasePluginFile, HasReferenceMixin
from bikipy.perimeter.base import (
    SinglePerimeter,
    StringPerimeterShapes,
    perimeter_set_from_makesense,
)
from bikipy.utils.collection_utils import get_first_key_in_dict
from bikipy.utils.image import axis_frame_imshow, read_image_from_path
from bikipy.utils.makesense import (
    SHAPE_TO_MAKESENSE_TYPE,
    first_image_name_from_makesense,
)

logger = getLogger(__name__)


# TODO: Manual radius readings from settings.yaml read.
class PluginSinglePerimeter(BasePluginFile, HasReferenceMixin):
    manual_shape: Optional[StringPerimeterShapes] = None
    warn_missing_re_reference_file: bool = False

    plural_entries = True

    ingress_key = "perimeter"
    code_key = "perimeter"
    bikipy_trial_key = "label_to_perimeter"
    human_readable_index = "Perimeter"

    @property
    def shape(self) -> str:
        return self._info[1]

    @property
    def init_args(self) -> list:
        if len(self._info) == 3:
            return []
        return self._info[2:-1]

    @property
    def label(self) -> str:
        return self._info[-1]

    @cached_property
    def image_name(self):
        return first_image_name_from_makesense(self.data_path, SHAPE_TO_MAKESENSE_TYPE[self.shape])

    @property
    def perimeter_settings(self) -> dict[str, SinglePerimeter]:
        return self.ingress.settings["perimeter"]

    @cached_property
    def label_to_trial_label_df(self) -> pd.DataFrame:
        if self.perimeter_settings["perimeter_names_in_metadata"]:
            return _open_label_to_trial_label_df(self.ingress.metadata_path)

    def perimeter_mapper(self, trial_id: Optional[str | PositiveInt] = None) -> dict[str, SinglePerimeter]:
        image_name_to_perimeter_set = perimeter_set_from_makesense(
            self.data_path,
            self.manual_shape or self.shape,
            init_args=self.init_args,
            meters_per_pixel=self.ingress.get_meter_per_pixel(trial_id),
            reference_point_array=self.reference_point,
            inspect_arg=self.ingress.inspect_directory_path,
        )

        result = {}
        for image_name, perimeter_set in image_name_to_perimeter_set.items():
            for label, perimeter in perimeter_set.label_to_perimeter.items():
                for field, value in self.perimeter_settings["fields"]["defined"].items():
                    if value is not None:
                        perimeter.__setattr__(field, value)

                result[label] = perimeter

        return result

    def trialwise_and_metadata(
        self, trial_id: TrialId, naive: bool = False
    ) -> dict[str, SinglePerimeter] | SinglePerimeter:
        result = {}
        for label, perimeter in self.perimeter_mapper(trial_id).items():
            if self.label_to_trial_label_df is not None:
                label = self.label_to_trial_label_df.loc[trial_id, label]
            if p := self.perimeter_settings["label_prefix"]:
                label = f"{p}_{label}"
            if s := self.perimeter_settings["label_suffix"]:
                label = f"{label}_{s}"

            result[label] = _perimeter_with_label(perimeter, label)

            if self.label not in self.ingress.ingress_defined_perimeters:
                self.ingress.ingress_defined_perimeters[self.label] = {}
            self.ingress.ingress_defined_perimeters[self.label][label] = perimeter

        if naive:
            return next(iter(result.values()))
        return result

    @property
    def globally_defined(self) -> dict[str, SinglePerimeter]:
        result = {}
        for label, perimeter in self.perimeter_mapper().items():
            if p := self.perimeter_settings["label_prefix"]:
                label = f"{p}_{label}"
            if s := self.perimeter_settings["label_suffix"]:
                label = f"{label}_{s}"

            perimeter = _perimeter_with_label(perimeter, label)

            result[label] = perimeter
            self.ingress.ingress_defined_perimeters[label] = perimeter

        return result


@validate_arguments
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
        ax.invert_yaxis()

        perimeter_set.plot(manual_ax=ax, inspect_pixels=True)

    plt.show()


@lru_cache
def _perimeter_with_label(perimeter: SinglePerimeter, new_label: str) -> SinglePerimeter:
    if new_label == perimeter.label:
        return perimeter
    return perimeter.copy(update={"label": new_label})


@lru_cache
def _open_label_to_trial_label_df(metadata_path: FilePath):
    return pd.read_excel(metadata_path, sheet_name="perimeter_label", index_col=0)
