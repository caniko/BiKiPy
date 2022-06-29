from copy import copy, deepcopy
from functools import cached_property, lru_cache
from logging import getLogger
from typing import Any, Optional

import pandas as pd
from pydantic import FilePath, PositiveInt, validator

from bikipy.ingress.plugin.base import BasePluginFile
from bikipy.perimeter.base import (
    AnyPerimeter,
    StringPerimeterShapes,
    perimeter_set_from_makesense,
)

logger = getLogger(__name__)


class PluginPerimeter(BasePluginFile):
    ingress: Any
    trial_id: str | PositiveInt

    manual_shape: Optional[StringPerimeterShapes] = None

    data_label = "perimeter"

    @property
    def shape(self) -> str:
        return self._info[1]

    @property
    def label(self) -> str:
        return self._info[2]

    @property
    def perimeter_settings(self) -> dict:
        return self.ingress.settings["perimeter"]

    @cached_property
    def label_to_perimeter_from_first_makesense(self) -> dict[str, AnyPerimeter]:
        image_name_to_perimeter_set = perimeter_set_from_makesense(
            self.data_path,
            self.manual_shape or self.shape,
            meters_per_pixel=self.ingress.get_meter_per_pixel(self.trial_id),
        )

        perimeter_set = tuple(image_name_to_perimeter_set.values())[0]

        for perimeter in perimeter_set.all_perimeters:
            for field, value in self.perimeter_settings["fields"]["defined"].items():
                if value is not None:
                    perimeter.__setattr__(field, value)

        result = {}
        for label, perimeter in perimeter_set.label_to_perimeter.items():
            new_label = copy(label)
            if self.label_to_trial_label_df is not None:
                new_label = self.label_to_trial_label_df.loc[self.trial_id, label]
            if p := self.perimeter_settings["label_prefix"]:
                new_label = f"{p}_{new_label}"
            if s := self.perimeter_settings["label_suffix"]:
                new_label = f"{new_label}_{s}"

            if new_label != label:
                result[new_label] = _perimeter_with_label(perimeter, new_label)
            elif result:
                result[label] = perimeter

        return result or perimeter_set.label_to_perimeter

    @cached_property
    def label_to_trial_label_df(self) -> pd.DataFrame:
        if self.perimeter_settings["perimeter_names_in_metadata"]:
            return _open_label_to_trial_label_df(self.ingress.metadata_path)


def perimeter_file_path_to_value(file_path: FilePath, trial_id: str | PositiveInt, ingress: Any, *args, **kwargs):
    return PluginPerimeter(
        data_path=file_path, ingress=ingress, trial_id=trial_id
    ).label_to_perimeter_from_first_makesense


@lru_cache
def _perimeter_with_label(perimeter: AnyPerimeter, new_label: str) -> AnyPerimeter:
    return perimeter.copy(update={"label": new_label})


@lru_cache
def _open_label_to_trial_label_df(metadata_path: FilePath):
    return pd.read_excel(metadata_path, sheet_name="perimeter_label", index_col=0)
