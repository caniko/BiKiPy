from copy import copy
from functools import cached_property, lru_cache
from logging import getLogger
from typing import Any, Optional

import pandas as pd
from pydantic import FilePath, PositiveInt, Field

from bikipy.ingress.plugin.base import BasePluginFile, HasReferenceMixin, Plugin
from bikipy.perimeter.base import (
    AnyPerimeter,
    StringPerimeterShapes,
    perimeter_set_from_makesense,
)
from bikipy.utils.collection_utils import get_first_value_in_dict
from bikipy.utils.io.makesense import (
    image_name_from_makesense,
    SHAPE_TO_MAKESENSE_TYPE,
)

logger = getLogger(__name__)


class PluginPerimeter(BasePluginFile, HasReferenceMixin):
    manual_shape: Optional[StringPerimeterShapes] = None
    warn_missing_re_reference_file: bool = False

    ingress_key = "perimeter_definition_strategy"
    code_key = "perimeter"
    bikipy_trial_key = "label_to_perimeter"
    human_readable_index = "Perimeter"

    @property
    def shape(self) -> str:
        return self._info[1]

    @property
    def label(self) -> str:
        return self._info[2]

    @cached_property
    def image_name(self):
        return image_name_from_makesense(self.data_path, SHAPE_TO_MAKESENSE_TYPE[self.shape])

    @property
    def perimeter_settings(self) -> dict[str, AnyPerimeter]:
        return self.ingress.settings["perimeter"]

    @cached_property
    def label_to_trial_label_df(self) -> pd.DataFrame:
        if self.perimeter_settings["perimeter_names_in_metadata"]:
            return _open_label_to_trial_label_df(self.ingress.metadata_path)

    @cached_property
    def perimeter_mapper(self) -> dict[str, AnyPerimeter]:
        image_name_to_perimeter_set = perimeter_set_from_makesense(
            self.data_path,
            self.manual_shape or self.shape,
            meters_per_pixel=self.ingress.get_meter_per_pixel(),
            reference_point_array=self.reference_point,
        )

        result = {}
        for image_name, perimeter_set in image_name_to_perimeter_set.items():
            for label, perimeter in perimeter_set.label_to_perimeter.items():
                for field, value in self.perimeter_settings["fields"]["defined"].items():
                    if value is not None:
                        perimeter.__setattr__(field, value)

                result[label] = perimeter

        return result

    @cached_property
    def trialwise_globally_defined(self):
        result = {}
        for label, perimeter in self.perimeter_mapper.items():
            if p := self.perimeter_settings["label_prefix"]:
                label = f"{p}_{label}"
            if s := self.perimeter_settings["label_suffix"]:
                label = f"{label}_{s}"

            result[label] = _perimeter_with_label(perimeter, label)

        return result

    @property
    def trialwise(self) -> dict[str, AnyPerimeter]:
        return self.trialwise_globally_defined

    def metadata(self, trial_id: str | PositiveInt) -> dict[str, AnyPerimeter]:
        result = {}
        for label, perimeter in self.perimeter_mapper.items():
            if self.label_to_trial_label_df is not None:
                label = self.label_to_trial_label_df.loc[trial_id, label]
            if p := self.perimeter_settings["label_prefix"]:
                label = f"{p}_{label}"
            if s := self.perimeter_settings["label_suffix"]:
                label = f"{label}_{s}"

            result[label] = _perimeter_with_label(perimeter, label)

        return result

    @property
    def globally_defined(self) -> dict[str, AnyPerimeter]:
        return self.trialwise_globally_defined


def perimeter_file_path_to_data_object(file_path: FilePath, trial_id: str | PositiveInt, ingress: Any, *args, **kwargs):
    return PluginPerimeter(data_path=file_path, ingress=ingress, trial_id=trial_id)


@lru_cache
def _perimeter_with_label(perimeter: AnyPerimeter, new_label: str) -> AnyPerimeter:
    if new_label == perimeter.label:
        return perimeter
    return perimeter.copy(update={"label": new_label})


@lru_cache
def _open_label_to_trial_label_df(metadata_path: FilePath):
    return pd.read_excel(metadata_path, sheet_name="perimeter_label", index_col=0)
