from copy import copy
from functools import cached_property, lru_cache
from logging import getLogger
from typing import Any, Optional

import pandas as pd
from pydantic import FilePath, PositiveInt, Field

from bikipy.core.base_class import BaseBikipy
from bikipy.core.typing import NDArrayFp64
from bikipy.ingress.plugin.base import BasePluginFile
from bikipy.perimeter.base import (
    AnyPerimeter,
    StringPerimeterShapes,
    perimeter_set_from_makesense,
)
from bikipy.utils.io.makesense import get_only_point_from_makesense, image_name_to_point_from_makesense

logger = getLogger(__name__)


class PluginPerimeterMixin(BaseBikipy):
    manual_reference: Optional[NDArrayFp64] = Field(
        description="Override the perimeter detection with values defined outside model"
    )

    @cached_property
    def reference_point(self) -> pd.DataFrame | None:
        if self.manual_reference is not None:
            return self.manual_reference
        if (path_to_reference_file := self.data_path.parent / f"reference-{self.label}.csv").exists():
            return get_only_point_from_makesense(path_to_reference_file)

    @cached_property
    def image_name_to_re_referencing_point(self) -> dict[str, NDArrayFp64]:
        if (path_to_re_reference_file := self.data_path.parent / f"re_reference-{self.label}.csv").exists():
            assert self.reference_point is not None, "Reference point must be defined for re-referencing"
            return image_name_to_point_from_makesense(path_to_re_reference_file)
        return {}


class PluginPerimeter(BasePluginFile, PluginPerimeterMixin):
    ingress: Any = Field(description="Bikipy ingress object to access project metadata relevant for defining perimeter")
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
    def perimeter_mapper_from_first_makesense(self) -> dict[str, AnyPerimeter]:
        def map_perimeter(new_perimeter: AnyPerimeter) -> None:
            match self.perimeter_settings["perimeter_mapper_key"]:
                case "label":
                    result[key] = new_perimeter
                case "image-label":
                    if image_name not in result:
                        result[image_name] = {}
                    result[image_name][key] = new_perimeter
                case _:
                    msg = f"{self.perimeter_settings['perimeter_mapper_key']} is an unsupported map key for perimeters"
                    raise ValueError(msg)

        image_name_to_perimeter_set = perimeter_set_from_makesense(
            self.data_path,
            self.manual_shape or self.shape,
            meters_per_pixel=self.ingress.get_meter_per_pixel(self.trial_id),
            reference_point_array=self.reference_point,
        )

        if self.reference_point is not None:
            if not self.image_name_to_re_referencing_point:
                logger.warning(f"Dataset {self.label}: Defines reference point, yet no re-reference data was detected")
            if len(image_name_to_perimeter_set) > 1:
                logger.warning(f"Dataset {self.label}: The same reference is being applied to several images")

        result = {}
        for image_name, perimeter_set in image_name_to_perimeter_set.items():
            for label, perimeter in perimeter_set.label_to_perimeter.items():
                for field, value in self.perimeter_settings["fields"]["defined"].items():
                    if value is not None:
                        perimeter.__setattr__(field, value)

                key = copy(label)
                new_label = copy(label)
                if self.label_to_trial_label_df is not None:
                    new_label = self.label_to_trial_label_df.loc[self.trial_id, label]
                if p := self.perimeter_settings["label_prefix"]:
                    new_label = f"{p}_{new_label}"
                if s := self.perimeter_settings["label_suffix"]:
                    new_label = f"{new_label}_{s}"

                map_perimeter(_perimeter_with_label(perimeter, new_label))

                for new_image_name, new_reference in self.image_name_to_re_referencing_point.items():
                    result[key.replace(image_name, new_image_name)] = result[key].change_reference(new_reference)

        return result

    @cached_property
    def label_to_trial_label_df(self) -> pd.DataFrame:
        if self.perimeter_settings["perimeter_names_in_metadata"]:
            return _open_label_to_trial_label_df(self.ingress.metadata_path)

    @property
    def get_only_perimeter(self) -> AnyPerimeter:
        assert len(self.perimeter_mapper_from_first_makesense) == 1
        return next(self.perimeter_mapper_from_first_makesense.values())


def perimeter_file_path_to_value(file_path: FilePath, trial_id: str | PositiveInt, ingress: Any, *args, **kwargs):
    perimeter_mapper = PluginPerimeter(
        data_path=file_path, ingress=ingress, trial_id=trial_id
    ).perimeter_mapper_from_first_makesense

    match ingress.settings["perimeter"]["perimeter_mapper_key"]:
        case "label":
            pass
        case "image-label":
            key = f"{image_name}-{label}"
        case _:
            msg = f"{self.perimeter_settings['perimeter_mapper_key']} is an unsupported map key for perimeters"
            raise ValueError(msg)


@lru_cache
def _perimeter_with_label(perimeter: AnyPerimeter, new_label: str) -> AnyPerimeter:
    if new_label == perimeter.label:
        return perimeter
    return perimeter.copy(update={"label": new_label})


@lru_cache
def _open_label_to_trial_label_df(metadata_path: FilePath):
    return pd.read_excel(metadata_path, sheet_name="perimeter_label", index_col=0)
